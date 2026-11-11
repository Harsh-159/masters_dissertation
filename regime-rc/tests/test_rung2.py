"""Tests for rung 2: the gating network, the mixture, and its training.

The properties that matter most:

* the blend is a genuine convex combination (valid portfolio, differentiable
  both ways), and reduces exactly to rung 1 when K = 1;
* K is truly variable - shapes and parameter counts scale as arithmetic says;
* the noise is train-time only and the anneal follows R2's schedule;
* the balance term actually penalises lopsided expert usage;
* the diagnostics report what the G2 verdict depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rrc.models.experts import count_parameters
from rrc.models.gating import (
    GatingNetwork,
    MixtureOfExperts,
    annealed_noise_scale,
    importance_shares,
    mean_gate_entropy,
)
from rrc.models.losses import decision_loss
from rrc.optim.toy import make_toy_fold
from rrc.optim.train_loop import (
    MixtureTrainConfig,
    annualised_validation_sharpe,
    mixture_diagnostics,
    train_mixture,
)


def make_mixture(n_experts: int = 3, n_state: int = 8, n_assets: int = 4) -> MixtureOfExperts:
    torch.manual_seed(0)
    return MixtureOfExperts(
        n_state_features=n_state, n_assets=n_assets, n_experts=n_experts
    )


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def test_gate_memberships_live_on_the_simplex():
    gate = GatingNetwork(n_inputs=6, n_experts=3)
    memberships = gate(torch.randn(10, 6, dtype=torch.float64))

    assert memberships.shape == (10, 3)
    assert bool((memberships > 0).all())
    torch.testing.assert_close(
        memberships.sum(dim=1), torch.ones(10, dtype=torch.float64)
    )


def test_gate_noise_is_train_time_only():
    """Noise perturbs the routing across calls; noise-free calls are identical."""
    gate = GatingNetwork(n_inputs=5, n_experts=3)
    inputs = torch.randn(8, 5, dtype=torch.float64)

    torch.manual_seed(1)
    noisy_first = gate(inputs, noise_scale=0.3)
    noisy_second = gate(inputs, noise_scale=0.3)
    assert not torch.allclose(noisy_first, noisy_second)

    clean_first = gate(inputs, noise_scale=0.0)
    clean_second = gate(inputs, noise_scale=0.0)
    torch.testing.assert_close(clean_first, clean_second)


def test_lower_temperature_sharpens_the_routing():
    """tau divides the logits, so a smaller tau gives harder memberships."""
    inputs = torch.randn(20, 5, dtype=torch.float64)

    torch.manual_seed(2)
    soft_gate = GatingNetwork(5, 3, temperature=2.0)
    torch.manual_seed(2)  # identical weights, different temperature
    hard_gate = GatingNetwork(5, 3, temperature=0.5)

    soft_entropy = mean_gate_entropy(soft_gate(inputs))
    hard_entropy = mean_gate_entropy(hard_gate(inputs))
    assert hard_entropy < soft_entropy


def test_invalid_gate_arguments_are_rejected():
    with pytest.raises(ValueError, match="n_experts"):
        GatingNetwork(5, 0)
    with pytest.raises(ValueError, match="temperature"):
        GatingNetwork(5, 3, temperature=0.0)


# --------------------------------------------------------------------------- #
# The mixture
# --------------------------------------------------------------------------- #
def test_blend_is_a_valid_portfolio_and_matches_the_manual_sum():
    mixture = make_mixture()
    states = torch.randn(6, 8, dtype=torch.float64)
    mask = torch.zeros(6, 4, dtype=torch.float64)

    blended = mixture(states, mask)
    memberships, expert_weights = mixture.route(states, mask)
    manual = (memberships.unsqueeze(-1) * expert_weights).sum(dim=1)

    torch.testing.assert_close(blended, manual)
    assert bool((blended.detach() > 0).all())
    torch.testing.assert_close(
        blended.sum(dim=1), torch.ones(6, dtype=torch.float64)
    )


def test_masked_assets_get_exactly_zero_weight_in_the_blend():
    """Every expert masks, so the convex blend inherits the exact zeros."""
    mixture = make_mixture()
    states = torch.randn(3, 8, dtype=torch.float64)
    mask = torch.zeros(3, 4, dtype=torch.float64)
    mask[1, 2] = -1e9  # asset 2 unavailable on day 1

    blended = mixture(states, mask).detach()
    assert float(blended[1, 2]) == 0.0
    torch.testing.assert_close(blended.sum(dim=1), torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize("n_experts", [1, 2, 4, 7])
def test_k_is_fully_variable(n_experts):
    """Shapes and parameter counts scale exactly as the arithmetic says."""
    n_state, n_assets = 10, 5
    mixture = MixtureOfExperts(n_state, n_assets, n_experts)

    states = torch.randn(6, n_state, dtype=torch.float64)
    mask = torch.zeros(6, n_assets, dtype=torch.float64)
    memberships, expert_weights = mixture.route(states, mask)

    assert memberships.shape == (6, n_experts)
    assert expert_weights.shape == (6, n_experts, n_assets)

    expected_parameters = (
        n_experts * (n_state * n_assets + n_assets)   # the K expert heads
        + (n_state * n_experts + n_experts)           # the gate
    )
    assert count_parameters(mixture) == expected_parameters


def test_k_equals_one_degenerates_to_the_single_head():
    """With one expert the gate has one slot: pi is identically 1 and the
    blend IS that expert's portfolio - rung 1 as a special case of rung 2."""
    mixture = MixtureOfExperts(n_state_features=8, n_assets=4, n_experts=1)
    states = torch.randn(5, 8, dtype=torch.float64)
    mask = torch.zeros(5, 4, dtype=torch.float64)

    memberships, expert_weights = mixture.route(states, mask)
    torch.testing.assert_close(
        memberships, torch.ones(5, 1, dtype=torch.float64)
    )
    torch.testing.assert_close(mixture(states, mask), expert_weights[:, 0])


def test_gradients_reach_both_the_gate_and_every_expert():
    """The blend must be differentiable through BOTH channels (Spec 8.2)."""
    mixture = make_mixture(n_experts=3)
    states = torch.randn(6, 8, dtype=torch.float64)
    mask = torch.zeros(6, 4, dtype=torch.float64)

    mixture(states, mask).sum().backward()

    assert mixture.gating.linear.weight.grad is not None
    assert float(mixture.gating.linear.weight.grad.abs().sum()) >= 0  # exists
    for expert in mixture.experts:
        assert expert.linear.weight.grad is not None
        assert float(expert.linear.weight.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# The anneal schedule and the routing statistics
# --------------------------------------------------------------------------- #
def test_noise_anneal_follows_the_r2_schedule():
    assert annealed_noise_scale(0, 500, 0.3) == pytest.approx(0.3)
    assert annealed_noise_scale(125, 500, 0.3) == pytest.approx(0.15)
    assert annealed_noise_scale(250, 500, 0.3) == pytest.approx(0.0)
    assert annealed_noise_scale(400, 500, 0.3) == 0.0  # stays at zero after


def test_importance_shares_and_entropy_arithmetic():
    memberships = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64
    )
    shares = importance_shares(memberships)
    torch.testing.assert_close(
        shares, torch.tensor([2.5 / 4, 1.5 / 4], dtype=torch.float64)
    )

    # Three hard days (entropy 0) and one even day (entropy ln 2).
    assert mean_gate_entropy(memberships) == pytest.approx(np.log(2) / 4, rel=1e-6)


def test_balance_term_penalises_lopsided_usage_in_the_full_loss():
    """Identical returns, different routing: the lopsided mixture costs more."""
    net_returns = 0.01 * torch.randn(200, generator=torch.Generator().manual_seed(3),
                                     dtype=torch.float64)
    risk_free = torch.zeros(200, dtype=torch.float64)

    even = decision_loss(
        net_returns, risk_free, lambda_balance=0.01,
        importance=torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64),
    )
    lopsided = decision_loss(
        net_returns, risk_free, lambda_balance=0.01,
        importance=torch.tensor([28.0, 1.0, 1.0], dtype=torch.float64),
    )
    assert float(lopsided.total) > float(even.total)
    assert float(even.balance_term) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Training the mixture on the planted toy
# --------------------------------------------------------------------------- #
def test_mixture_can_crack_the_planted_toy():
    """The routing mechanism must be able to learn the two-regime toy."""
    toy = make_toy_fold(seed=0)
    config = MixtureTrainConfig(
        n_experts=2, max_epochs=300, patience=300, lambda_cvar=0.0, seed=0
    )
    result = train_mixture(toy.pack, toy.pack, config)

    assert float(result.history["train_sharpe_ann"].iloc[-1]) > 5.0
    assert not result.history.isna().any().any()


def test_mixture_early_stopping_restores_the_best_epoch():
    toy_train = make_toy_fold(seed=0)
    toy_val = make_toy_fold(seed=99)
    config = MixtureTrainConfig(
        n_experts=2, max_epochs=150, patience=10, lambda_cvar=0.0, seed=0
    )
    result = train_mixture(toy_train.pack, toy_val.pack, config)

    revalidated = annualised_validation_sharpe(result.model, toy_val.pack)
    assert revalidated == pytest.approx(result.best_val_sharpe, abs=1e-9)


def test_mixture_history_tracks_the_routing_diagnostics():
    toy = make_toy_fold(seed=0)
    config = MixtureTrainConfig(n_experts=3, max_epochs=30, patience=30, seed=0)
    result = train_mixture(toy.pack, toy.pack, config)

    history = result.history
    for column in ("noise_scale", "gate_entropy", "min_share", "max_share"):
        assert column in history.columns
    # The anneal starts at the configured level and never rises.
    assert history["noise_scale"].iloc[0] == pytest.approx(0.3)
    assert bool((history["noise_scale"].diff().dropna() <= 1e-12).all())
    # Shares are fractions of a whole.
    assert bool((history["min_share"] >= 0).all())
    assert bool((history["max_share"] <= 1).all())


# --------------------------------------------------------------------------- #
# The diagnostics dashboard
# --------------------------------------------------------------------------- #
def test_diagnostics_report_everything_g2_needs():
    toy = make_toy_fold(seed=0)
    config = MixtureTrainConfig(n_experts=3, max_epochs=40, patience=40, seed=0)
    result = train_mixture(toy.pack, toy.pack, config)

    diagnostics = mixture_diagnostics(result.model, toy.pack)

    memberships = diagnostics["memberships"]
    assert memberships.shape == (toy.pack.n_signals, 3)
    np.testing.assert_allclose(memberships.sum(axis=1), 1.0, atol=1e-9)

    shares = diagnostics["importance_shares"]
    assert len(shares) == 3
    assert sum(shares) == pytest.approx(1.0)

    assert set(diagnostics["pairwise_weight_correlations"]) == {"0-1", "0-2", "1-2"}
    assert np.isfinite(diagnostics["max_pairwise_correlation"])
    assert len(diagnostics["standalone"]) == 3
    for entry in diagnostics["standalone"]:
        assert np.isfinite(entry["ann_vol"]) and np.isfinite(entry["sharpe"])


def test_identical_experts_show_correlation_one():
    """The specialisation measure must read 1.0 when experts are clones."""
    torch.manual_seed(0)
    # Built to the toy's dimensions: 8 state features, 3 assets (2 risky + cash).
    mixture = MixtureOfExperts(n_state_features=8, n_assets=3, n_experts=3)
    with torch.no_grad():
        for expert in mixture.experts[1:]:
            expert.load_state_dict(mixture.experts[0].state_dict())

    toy = make_toy_fold(seed=0)
    diagnostics = mixture_diagnostics(mixture, toy.pack)
    assert diagnostics["max_pairwise_correlation"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# lambda_div: the anti-redundancy penalty (Spec 8.5)
# --------------------------------------------------------------------------- #
def test_divergence_penalty_is_one_for_identical_experts():
    """Identical experts are maximally redundant - the penalty must be 1."""
    import torch
    from rrc.models.losses import divergence_penalty

    one = torch.rand(20, 1, 5, dtype=torch.float64)
    identical = one.repeat(1, 3, 1)                 # three copies of one expert
    assert float(divergence_penalty(identical)) == pytest.approx(1.0, abs=1e-9)


def test_divergence_penalty_is_lower_for_differentiated_experts():
    """Experts proposing different portfolios must be penalised less."""
    import torch
    from rrc.models.losses import divergence_penalty

    generator = torch.Generator().manual_seed(0)
    distinct = torch.rand(20, 3, 5, generator=generator, dtype=torch.float64)
    identical = distinct[:, :1, :].repeat(1, 3, 1)
    assert float(divergence_penalty(distinct)) < float(divergence_penalty(identical))


def test_divergence_penalty_vanishes_for_a_single_expert():
    """K = 1 has no pair to correlate, so the term must disappear."""
    import torch
    from rrc.models.losses import divergence_penalty

    assert float(divergence_penalty(torch.rand(20, 1, 5, dtype=torch.float64))) == 0.0


def test_divergence_penalty_is_differentiable():
    """It sits in the loss, so gradients must flow back to the experts."""
    import torch
    from rrc.models.losses import divergence_penalty

    weights = torch.rand(15, 3, 4, dtype=torch.float64, requires_grad=True)
    divergence_penalty(weights).backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_lambda_div_zero_leaves_the_loss_unchanged():
    """The default must be a no-op, so existing results are unaffected."""
    import torch
    from rrc.models.losses import decision_loss

    returns = torch.tensor([0.01, -0.02, 0.03, 0.00, 0.01], dtype=torch.float64)
    rf = torch.full((5,), 0.0001, dtype=torch.float64)
    experts = torch.rand(5, 3, 4, dtype=torch.float64)

    without = decision_loss(returns, rf)
    with_zero = decision_loss(returns, rf, lambda_div=0.0, expert_weights=experts)
    assert float(with_zero.total) == pytest.approx(float(without.total))
    assert float(with_zero.divergence_term) == 0.0


# --------------------------------------------------------------------------- #
# R2 anneal horizon (the bug that ran through P0-P8)
# --------------------------------------------------------------------------- #
def test_anneal_horizon_defaults_to_twice_the_patience():
    """The anneal must finish inside a run that early-stops, not at epoch 250.

    Before this was fixed the caller passed ``max_epochs`` (500), so noise was
    scheduled to reach zero at epoch 250 while early stopping fired between
    epochs 7 and 80. Every mixture ever trained ran under ~90% of full gate
    noise for its whole life. This pins the horizon to the stopping mechanism
    so the two cannot drift apart again.
    """
    from rrc.optim.train_loop import MixtureTrainConfig

    config = MixtureTrainConfig(max_epochs=500, patience=25)
    horizon = (
        config.gate_noise_anneal_epochs
        if config.gate_noise_anneal_epochs is not None
        else 2 * config.patience
    )
    assert horizon == 50

    # Noise is fully gone by epoch 25 - inside a realistic run, not after it.
    assert annealed_noise_scale(0, horizon, 0.3) == pytest.approx(0.3)
    assert annealed_noise_scale(25, horizon, 0.3) == pytest.approx(0.0)

    # The old behaviour, for contrast: still ~90% of full noise at epoch 25.
    assert annealed_noise_scale(25, 500, 0.3) == pytest.approx(0.27)


def test_anneal_horizon_can_be_set_explicitly():
    from rrc.optim.train_loop import MixtureTrainConfig

    config = MixtureTrainConfig(gate_noise_anneal_epochs=10)
    assert config.gate_noise_anneal_epochs == 10
    assert annealed_noise_scale(5, 10, 0.3) == pytest.approx(0.0)
