"""Tests for rung 1: the head, the loss, the training loop, and C2's power.

The two tests that matter most here are the last two: they demonstrate that the
shuffle-future detector actually has **power** - a deliberately leaky pipeline
is caught red-handed, and an honest pipeline with genuine skill collapses to
noise exactly as it should. A causality test that cannot catch a planted leak
would be theatre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from rrc.backtest.ledger import weekly_execution_schedule
from rrc.backtest.metrics import TRADING_DAYS_PER_YEAR, excess_sharpe
from rrc.eval.causality import collapse_verdict, permute_rows_after
from rrc.models.experts import LinearPortfolioHead, build_logit_mask, count_parameters
from rrc.models.losses import balance_penalty, decision_loss
from rrc.optim.fold_data import SegmentPack
from rrc.optim.toy import finite_difference_check, make_toy_fold
from rrc.optim.train_loop import (
    TrainConfig,
    annualised_validation_sharpe,
    conditioning_edge,
    portfolio_returns,
    train_single_head,
)

CASH = 2  # the toys below use columns [A, B, CASH]


# --------------------------------------------------------------------------- #
# The head
# --------------------------------------------------------------------------- #
def test_head_outputs_valid_portfolios():
    head = LinearPortfolioHead(n_state_features=8, n_assets=4)
    states = torch.randn(6, 8, dtype=torch.float64)
    mask = torch.zeros(6, 4, dtype=torch.float64)

    weights = head(states, mask)

    assert weights.shape == (6, 4)
    assert bool((weights > 0).all())
    torch.testing.assert_close(
        weights.sum(dim=1), torch.ones(6, dtype=torch.float64)
    )


def test_head_gives_exactly_zero_weight_to_masked_assets():
    """The -1e9 mask must underflow to a weight of exactly 0.0, not merely small."""
    head = LinearPortfolioHead(n_state_features=5, n_assets=3)
    states = torch.randn(4, 5, dtype=torch.float64)

    availability = torch.tensor([[1, 0], [1, 1], [0, 1], [1, 1]])
    mask = build_logit_mask(availability)

    weights = head(states, mask)

    weights = weights.detach()
    assert float(weights[0, 1]) == 0.0   # asset B unavailable on day 0
    assert float(weights[2, 0]) == 0.0   # asset A unavailable on day 2
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(4, dtype=torch.float64))


def test_logit_mask_never_masks_cash():
    availability = torch.tensor([[0, 0], [1, 0]])
    mask = build_logit_mask(availability)

    assert mask.shape == (2, 3)
    assert float(mask[0, 2]) == 0.0  # cash column open even when everything else shut
    assert float(mask[0, 0]) < -1e8


def test_parameter_count_matches_the_arithmetic():
    head = LinearPortfolioHead(n_state_features=300, n_assets=12)
    assert count_parameters(head) == 300 * 12 + 12  # the plan's 3612 example


# --------------------------------------------------------------------------- #
# The loss
# --------------------------------------------------------------------------- #
def test_loss_with_zero_lambdas_is_negative_sharpe_alone():
    generator = torch.Generator().manual_seed(0)
    net_returns = 0.01 * torch.randn(300, generator=generator, dtype=torch.float64)
    risk_free = torch.full((300,), 1e-4, dtype=torch.float64)

    parts = decision_loss(net_returns, risk_free, lambda_cvar=0.0)

    torch.testing.assert_close(parts.total, parts.negative_sharpe)
    assert float(parts.cvar_term) == 0.0
    assert float(parts.balance_term) == 0.0


def test_cvar_term_adds_a_positive_tail_penalty():
    generator = torch.Generator().manual_seed(1)
    net_returns = 0.01 * torch.randn(300, generator=generator, dtype=torch.float64)
    risk_free = torch.zeros(300, dtype=torch.float64)

    without_tail = decision_loss(net_returns, risk_free, lambda_cvar=0.0)
    with_tail = decision_loss(net_returns, risk_free, lambda_cvar=1.0)

    assert float(with_tail.cvar_term) > 0
    assert float(with_tail.total) > float(without_tail.total)


def test_balance_penalty_behaviour():
    """Zero for one expert, zero for even use, positive for lopsided use."""
    single = torch.tensor([5.0], dtype=torch.float64)
    even = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64)
    lopsided = torch.tensor([8.0, 0.5, 0.5], dtype=torch.float64)

    assert float(balance_penalty(single)) == 0.0
    assert float(balance_penalty(even)) == pytest.approx(0.0, abs=1e-12)
    assert float(balance_penalty(lopsided)) > 0.1


# --------------------------------------------------------------------------- #
# The Section-10 toy protocol: gradients, then overfitting
# --------------------------------------------------------------------------- #
def test_finite_differences_match_autograd_through_the_whole_stack():
    """B7 extended: head -> softmax -> ledger -> loss, checked end to end."""
    toy = make_toy_fold(seed=0)
    torch.manual_seed(0)
    head = LinearPortfolioHead(toy.pack.states.shape[1], toy.pack.logit_mask.shape[1])

    def loss_now() -> torch.Tensor:
        returns = portfolio_returns(head, toy.pack)
        return decision_loss(returns, toy.pack.risk_free, lambda_cvar=1.0).total

    loss_now().backward()
    worst_error, largest_gradient = finite_difference_check(
        loss_now, head.linear.weight, n_samples=12, step=1e-6, seed=0
    )

    assert largest_gradient > 0, "no gradient reached the weights at all"
    assert worst_error < 1e-5, (
        f"autograd and finite differences disagree by {worst_error:.2e}"
    )


def test_the_head_can_overfit_the_planted_toy():
    """If gradient descent through the ledger cannot crack a 40-day toy with a
    planted regime, the plumbing is broken - the plan's day-8 test."""
    toy = make_toy_fold(seed=0)
    config = TrainConfig(max_epochs=300, patience=300, lambda_cvar=0.0, seed=0)

    # Validating on the training pack itself is deliberate here: the question
    # is pure capacity-to-fit, not generalisation.
    result = train_single_head(toy.pack, toy.pack, config)

    final_train_sharpe = result.history["train_sharpe_ann"].iloc[-1]
    assert final_train_sharpe > 5.0, (
        f"toy overfit only reached annualised Sharpe {final_train_sharpe:.2f}"
    )
    assert not result.history.isna().any().any()


def test_early_stopping_restores_the_best_epoch():
    """The returned head must actually be the best-validation head, not the last."""
    toy_train = make_toy_fold(seed=0)
    toy_val = make_toy_fold(seed=99)  # different noise -> imperfect transfer
    config = TrainConfig(max_epochs=200, patience=10, lambda_cvar=0.0, seed=0)

    result = train_single_head(toy_train.pack, toy_val.pack, config)

    revalidated = annualised_validation_sharpe(result.head, toy_val.pack)
    assert revalidated == pytest.approx(result.best_val_sharpe, abs=1e-9)
    assert result.best_epoch <= len(result.history) - 1


# --------------------------------------------------------------------------- #
# C2's power: it catches a planted leak, and honest skill collapses
# --------------------------------------------------------------------------- #
# A miniature "pipeline": block-regime returns, and a state builder that is
# either honest (trailing 5-day means - causal) or leaky (the NEXT 5 days'
# means - blatant future information). The C2 procedure rebuilds the states
# from the permuted returns, exactly as the real pipeline would.

def _block_regime_returns(n_days: int, seed: int) -> pd.DataFrame:
    """Two risky assets whose +/- drifts swap every 25 days, plus cash."""
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days, name="date")

    block = (np.arange(n_days) // 25) % 2  # 0: A wins, 1: B wins
    drift_a = np.where(block == 0, 0.008, -0.008)
    drift_b = -drift_a

    frame = pd.DataFrame(
        {
            "A": drift_a + generator.normal(0, 0.005, n_days),
            "B": drift_b + generator.normal(0, 0.005, n_days),
            "CASH": 0.0001,
        },
        index=dates,
    )
    return frame


def _build_states(returns: pd.DataFrame, leaky: bool) -> np.ndarray:
    """The pipeline's state builder - recomputed from whatever data it is given.

    Honest features: today's returns and TRAILING 5-day means (causal).
    Leaky variant adds the mean of days t+2..t+6 - exactly the window the
    portfolio signed at t will be earning over. The planted crime.

    Features are scaled to be O(1), mirroring what the real pipeline's
    z-scoring does (E12): raw returns are ~0.008, and a linear head cannot
    build the logit gaps a decisive softmax needs from inputs that small.
    """
    feature_scale = 100.0  # raw drifts are ~0.008 -> features become O(1)
    risky = returns[["A", "B"]]
    trailing_mean = risky.rolling(5).mean().fillna(0.0)

    features = [risky.to_numpy(), trailing_mean.to_numpy()]
    if leaky:
        forward_mean = (
            risky.iloc[::-1].rolling(5).mean().iloc[::-1].shift(-2).fillna(0.0)
        )
        features.append(forward_mean.to_numpy())
    return np.hstack(features) * feature_scale


def _make_pack(
    returns: pd.DataFrame, states: np.ndarray, start: int, end: int, name: str
) -> SegmentPack:
    """Package a [start, end) day window of the mini pipeline."""
    window_dates = returns.index[start:end]
    signal_relative, execution_relative = weekly_execution_schedule(window_dates)

    return SegmentPack(
        name=name,
        dates=window_dates,
        states=torch.as_tensor(states[start + signal_relative], dtype=torch.float64),
        logit_mask=torch.zeros(len(signal_relative), 3, dtype=torch.float64),
        execution_positions=execution_relative,
        returns=torch.as_tensor(returns.iloc[start:end].to_numpy().copy(), dtype=torch.float64),
        risk_free=torch.as_tensor(
            returns["CASH"].iloc[start:end].to_numpy().copy(), dtype=torch.float64
        ),
        cash_index=CASH,
    )


def _c2_run(leaky: bool) -> tuple[dict, list[float]]:
    """Train the mini pipeline, then measure conditioning edges on real and
    shuffled futures.

    The C2 quantity is the **conditioning edge** - the head's Sharpe minus its
    own static counterpart's, within the same world. Raw Sharpe does not
    collapse under shuffling for a strategy with a static tilt (shuffling
    preserves drift), so the edge is what separates timing skill (and leaked
    timing) from mere exposure.
    """
    n_days, cut = 600, 400
    returns = _block_regime_returns(n_days, seed=5)
    states = _build_states(returns, leaky=leaky)

    train_pack = _make_pack(returns, states, 10, cut, "train")
    config = TrainConfig(max_epochs=150, patience=150, lambda_cvar=0.0, seed=0)
    result = train_single_head(train_pack, train_pack, config)

    evaluation_window = (cut + 5, n_days - 5)
    actual_pack = _make_pack(returns, states, *evaluation_window, "eval")
    actual = conditioning_edge(result.head, actual_pack)

    permuted_edges = []
    for permutation_seed in range(4):
        shuffled = permute_rows_after(
            returns, cut, seed=permutation_seed, columns=["A", "B"]
        )
        rebuilt_states = _build_states(shuffled, leaky=leaky)  # the pipeline re-runs
        shuffled_pack = _make_pack(shuffled, rebuilt_states, *evaluation_window, "perm")
        permuted_edges.append(conditioning_edge(result.head, shuffled_pack)["edge"])
    return actual, permuted_edges


def test_c2_catches_a_planted_leak():
    """A pipeline whose state contains the future must KEEP timing shuffled
    data - because rebuilding it re-leaks the shuffled future. C2 sees that.

    The leaked timing is weaker on shuffled data than on real data (the
    forward-mean feature averages days drawn from both regimes, shrinking its
    magnitude), but its signature is unmistakable: the edge stays positive in
    EVERY shuffled world. Honest edges scatter around zero; leaked edges do
    not scatter - they survive systematically. Observed here: every edge above
    1.0 with a mean of ~1.7, against a noise band of 0.6.
    """
    actual, permuted_edges = _c2_run(leaky=True)
    verdict = collapse_verdict(actual["edge"], permuted_edges, noise_band=0.6)

    assert actual["edge"] > 2.0           # the leak looks like timing brilliance
    assert min(permuted_edges) > 0.5      # survives in every shuffled world
    assert verdict.mean_permuted > 1.0    # systematically, not by luck
    assert not verdict.collapsed          # the detector fires


def test_c2_lets_honest_skill_collapse():
    """Genuine timing skill exploits real temporal structure, which shuffling
    destroys - so the edge collapses even though drift survives.

    The noise band is set by sampling arithmetic, not by what passes: the
    ~185-day evaluation window is ~0.73 years, so a single annualised Sharpe
    (or edge) estimate has a standard deviation near sqrt(1/0.73) ~ 1.17, and
    the mean of 4 permutations has a standard error near 0.6. A 2-SE band is
    therefore ~1.25. (A small negative bias is also expected: the dynamic head
    trades on rebuilt noise features and pays costs the static counterpart
    does not.) A leak does not hide inside this band - see the companion test,
    where every single edge exceeds it.
    """
    actual, permuted_edges = _c2_run(leaky=False)
    verdict = collapse_verdict(actual["edge"], permuted_edges, noise_band=1.25)

    assert actual["edge"] > 2.0           # the block regime is genuinely timeable
    assert verdict.collapsed              # and the edge evaporates when shuffled
    assert abs(verdict.mean_permuted) < actual["edge"] / 2


def test_c2_does_not_accuse_a_static_strategy():
    """The lesson the real etf10 run taught, pinned as a regression test.

    A head with zero weights emits the same (uniform) portfolio every week - a
    purely static strategy. On drifting data its RAW Sharpe survives shuffling
    almost perfectly (drift is order-invariant), which under a raw-Sharpe C2
    would scream "leak" at a strategy that cannot possibly be leaking. Its
    conditioning EDGE, by contrast, is ~0 in every world, and the verdict is
    correctly clean.
    """
    n_days, cut = 600, 400
    # Give the whole market a shared positive drift so raw Sharpe is high.
    returns = _block_regime_returns(n_days, seed=7)
    returns[["A", "B"]] += 0.004
    states = _build_states(returns, leaky=False)

    static_head = LinearPortfolioHead(states.shape[1], 3)
    with torch.no_grad():
        static_head.linear.weight.zero_()
        static_head.linear.bias.zero_()

    evaluation_window = (cut + 5, n_days - 5)
    actual_pack = _make_pack(returns, states, *evaluation_window, "eval")
    actual = conditioning_edge(static_head, actual_pack)

    permuted_edges, permuted_raw = [], []
    for permutation_seed in range(4):
        shuffled = permute_rows_after(returns, cut, seed=permutation_seed,
                                      columns=["A", "B"])
        rebuilt = _build_states(shuffled, leaky=False)
        pack = _make_pack(shuffled, rebuilt, *evaluation_window, "perm")
        result = conditioning_edge(static_head, pack)
        permuted_edges.append(result["edge"])
        permuted_raw.append(result["dynamic"])

    # Raw Sharpe survives the shuffle (this is what fooled the naive test)...
    assert np.mean(permuted_raw) > 1.0
    # ...but the edge is zero everywhere, so the verdict is clean.
    verdict = collapse_verdict(actual["edge"], permuted_edges, noise_band=0.6)
    assert abs(actual["edge"]) < 1e-9
    assert verdict.collapsed


# --------------------------------------------------------------------------- #
# The permutation helper itself
# --------------------------------------------------------------------------- #
def test_permutation_leaves_the_past_untouched_and_shuffles_the_future():
    frame = _block_regime_returns(100, seed=0)
    permuted = permute_rows_after(frame, cut_position=59, seed=1, columns=["A", "B"])

    pd.testing.assert_frame_equal(frame.iloc[:60], permuted.iloc[:60])
    assert not np.allclose(
        frame.iloc[60:][["A", "B"]].to_numpy(),
        permuted.iloc[60:][["A", "B"]].to_numpy(),
    )
    # Same multiset of days, different order.
    np.testing.assert_allclose(
        np.sort(frame.iloc[60:]["A"].to_numpy()),
        np.sort(permuted.iloc[60:]["A"].to_numpy()),
    )
    # The cash column was excluded and stays in real time order.
    np.testing.assert_array_equal(
        frame.iloc[60:]["CASH"].to_numpy(), permuted.iloc[60:]["CASH"].to_numpy()
    )


def test_collapse_verdict_arithmetic():
    verdict = collapse_verdict(2.5, [0.1, -0.3, 0.2], noise_band=0.6)
    assert verdict.collapsed
    assert verdict.mean_permuted == pytest.approx(0.0, abs=1e-12)

    fired = collapse_verdict(2.5, [2.2, 2.6, 2.4], noise_band=0.6)
    assert not fired.collapsed
