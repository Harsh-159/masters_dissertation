"""The training loop: full-span gradient descent through the ledger.

What one epoch is
-----------------
There are no mini-batches anywhere in this project (A3): the Sharpe objective
couples every day through its denominator, so a batch smaller than the whole
training fold silently changes what is being optimised. One epoch is therefore:

1. forward: head reads the reservoir states at every signal day in the
   training window and emits the full weight schedule;
2. ledger: the closed-form engine turns that schedule into ~1,500-4,000 daily
   net returns in a handful of tensor operations;
3. loss: negative excess Sharpe plus the tail term, on the whole stream;
4. backward + clipped Adam step on the head's few thousand parameters.

The whole epoch is milliseconds - the payoff of cached states (D3) and the
closed-form ledger. Five hundred epochs cost seconds, which is what makes the
ablation ladder's ~thousand training runs a laptop job.

Early stopping (R3)
-------------------
After every epoch the current head is scored on the validation year - a fresh
per-segment run, exactly how the baselines were scored. Training stops when
validation excess Sharpe has not improved for 25 consecutive epochs, and the
parameters from the best epoch are restored. Validation is thereby the
*selection* segment: it chooses when to stop (and, across runs, which
lambda_cvar to keep), which is precisely why its rows are labelled ``val`` in
the registry and why the deflated-Sharpe trial count is defined over them.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from rrc.backtest.ledger import run_ledger
from rrc.backtest.metrics import TRADING_DAYS_PER_YEAR, excess_sharpe_torch, summarise
from rrc.models.experts import LinearPortfolioHead
from rrc.models.losses import LossParts, decision_loss
from rrc.optim.fold_data import SegmentPack
from rrc.utils.seeds import set_all_seeds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters of one training run - the plan's R3/Table 3 defaults.

    Everything here is part of the experiment's identity (it enters the
    cfg_hash), except ``seed``, which is a registry column of its own so that
    the same configuration across seeds counts as one trial, not five.
    """

    learning_rate: float = 3e-3
    max_epochs: int = 500
    patience: int = 25
    gradient_clip_norm: float = 5.0
    lambda_cvar: float = 1.0
    lambda_balance: float = 0.0   # 0 at rung 1; the term only exists for K > 1
    lambda_turn: float = 0.0      # explicit turnover pressure; 0 preserves E9
    # L2 penalty on the head's weights. The classical reservoir-computing readout
    # is RIDGE regression precisely because reservoir states are high-dimensional
    # and collinear, so an unregularised readout can memorise the training fold.
    # Measured on this project it closes the train-validation gap but does NOT
    # improve validation Sharpe (see the P8 capacity diagnostic), so it stays 0
    # by default and exists as a searchable knob rather than a recommended one.
    weight_decay: float = 0.0
    seed: int = 0


@dataclass
class TrainResult:
    """Everything a caller needs after a training run.

    Attributes
    ----------
    head
        The trained head, with the best-validation-epoch parameters restored.
    history
        One row per epoch: loss parts, train Sharpe, validation Sharpe.
    best_epoch
        Epoch whose parameters were kept.
    best_val_sharpe
        Annualised validation excess Sharpe at that epoch.
    stopped_early
        Whether patience ran out before ``max_epochs``.
    """

    head: LinearPortfolioHead
    history: pd.DataFrame
    best_epoch: int
    best_val_sharpe: float
    stopped_early: bool
    config: TrainConfig = field(default=None)  # type: ignore[assignment]


def model_ledger_result(head, pack: SegmentPack):
    """Run a model over a segment and return the full ledger result.

    This IS the loss's forward pass: weights from the model, returns from the
    ledger, gradients flowing straight through both. Returning the whole
    :class:`LedgerResult` (not just the net returns) lets the caller also read
    the per-rebalance turnover, which the optional ``lambda_turn`` term needs.
    Models that understand packs (every mixture variant) are called through
    ``weights_for_pack``; the rung-1 head keeps its plain (states, mask) call.
    """
    if hasattr(head, "weights_for_pack"):
        weights = head.weights_for_pack(pack)
    else:
        weights = head(pack.states, pack.logit_mask)
    return run_ledger(
        targets=weights,
        exec_days=pack.execution_positions,
        simple_returns=pack.returns,
        cash_index=pack.cash_index,
    )


def portfolio_returns(head, pack: SegmentPack) -> torch.Tensor:
    """The daily net-return stream of a model over a segment.

    A thin wrapper over :func:`model_ledger_result` for the many callers
    (validation scoring, the toy protocol) that only need the return stream.
    """
    return model_ledger_result(head, pack).net_returns


def annualised_validation_sharpe(head: LinearPortfolioHead, pack: SegmentPack) -> float:
    """Annualised excess Sharpe of the head on a segment, without gradients."""
    with torch.no_grad():
        net_returns = portfolio_returns(head, pack)
        per_period = excess_sharpe_torch(net_returns, pack.risk_free)
    return float(per_period) * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def train_single_head(
    train_pack: SegmentPack,
    val_pack: SegmentPack,
    config: TrainConfig,
) -> TrainResult:
    """Train one linear head on one fold, with early stopping on validation.

    Parameters
    ----------
    train_pack
        The embargoed, usable-intersected training window.
    val_pack
        The validation year - the selection segment.
    config
        Hyperparameters; ``config.seed`` fixes the head initialisation, which
        is the run's only source of randomness (the reservoir is deterministic).

    Returns
    -------
    TrainResult
        The trained head (best epoch restored) plus the full history.
    """
    set_all_seeds(config.seed)

    n_state_features = train_pack.states.shape[1]
    n_assets = train_pack.logit_mask.shape[1]
    head = LinearPortfolioHead(n_state_features, n_assets)

    optimiser = torch.optim.Adam(
        head.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_sharpe = -np.inf
    best_epoch = -1
    best_parameters: dict | None = None
    epochs_without_improvement = 0
    history_rows: list[dict] = []
    stopped_early = False

    for epoch in range(config.max_epochs):
        # ---- one full-span step -----------------------------------------
        optimiser.zero_grad()
        ledger_result = model_ledger_result(head, train_pack)
        loss_parts: LossParts = decision_loss(
            ledger_result.net_returns,
            train_pack.risk_free,
            lambda_cvar=config.lambda_cvar,
            lambda_balance=config.lambda_balance,
            lambda_turn=config.lambda_turn,
            turnover=ledger_result.turnover,
        )
        loss_parts.total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip_norm)
        optimiser.step()

        # ---- validation, and the early-stopping decision ----------------
        val_sharpe = annualised_validation_sharpe(head, val_pack)
        train_sharpe = -float(loss_parts.negative_sharpe.detach()) * float(
            np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        history_rows.append(
            {
                "epoch": epoch,
                **loss_parts.as_floats(),
                "train_sharpe_ann": train_sharpe,
                "val_sharpe_ann": val_sharpe,
            }
        )

        if not np.isfinite(float(loss_parts.total.detach())):
            raise FloatingPointError(
                f"Loss became non-finite at epoch {epoch} - this is exactly the "
                "instability G1 exists to catch; check the input scaling and "
                "the learning rate."
            )

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_epoch = epoch
            best_parameters = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_early = True
                break

    if best_parameters is not None:
        head.load_state_dict(best_parameters)

    history = pd.DataFrame(history_rows)
    logger.info(
        "trained %d epochs (best %d, val Sharpe %.3f, early stop: %s)",
        len(history), best_epoch, best_val_sharpe, stopped_early,
    )
    return TrainResult(
        head=head,
        history=history,
        best_epoch=best_epoch,
        best_val_sharpe=best_val_sharpe,
        stopped_early=stopped_early,
        config=config,
    )


def conditioning_edge(head: LinearPortfolioHead, pack: SegmentPack) -> dict[str, float]:
    """The head's timing skill: its Sharpe minus its static counterpart's.

    Why this exists (a lesson the C2 test taught)
    ---------------------------------------------
    Shuffling future days destroys temporal *ordering* but preserves the
    *marginal distribution* of returns - so a constant-weight portfolio keeps
    essentially its whole Sharpe on shuffled data. Buy-and-hold SPY "survives
    the shuffle" perfectly, and no one would call that leakage. What leakage
    actually corrupts is the *conditional* part of a strategy: the decisions
    that change with the state. So the quantity that must collapse under C2 is
    not the raw Sharpe but the **edge over the strategy's own static
    counterpart** - a portfolio holding the head's average weights, rebalanced
    on the same days, through the same ledger, in the same world.

    The static counterpart is computed within-world: on shuffled data it is the
    average of the head's weights *on that shuffled data*, so the comparison is
    always internally consistent.

    Returns
    -------
    dict
        ``dynamic`` and ``static`` annualised excess Sharpes, and ``edge`` -
        their difference, the number C2's verdict rests on.
    """
    with torch.no_grad():
        weights = head(pack.states, pack.logit_mask)

        # The average allocation, held constantly. Averaging rows of a simplex
        # stays on the simplex; re-masking guards the late-entrant case where
        # an asset is unavailable at some signals, and renormalising restores
        # the budget after any masking.
        average_weights = weights.mean(dim=0, keepdim=True).expand_as(weights).clone()
        average_weights = average_weights * (pack.logit_mask > -1.0).to(weights.dtype)
        average_weights = average_weights / average_weights.sum(dim=1, keepdim=True)

        dynamic_returns = run_ledger(
            weights, pack.execution_positions, pack.returns, pack.cash_index
        ).net_returns
        static_returns = run_ledger(
            average_weights, pack.execution_positions, pack.returns, pack.cash_index
        ).net_returns

        annualisation = float(np.sqrt(TRADING_DAYS_PER_YEAR))
        dynamic_sharpe = float(excess_sharpe_torch(dynamic_returns, pack.risk_free)) * annualisation
        static_sharpe = float(excess_sharpe_torch(static_returns, pack.risk_free)) * annualisation

    return {
        "dynamic": dynamic_sharpe,
        "static": static_sharpe,
        "edge": dynamic_sharpe - static_sharpe,
    }


# --------------------------------------------------------------------------- #
# Rung 2+: training the mixture of experts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MixtureTrainConfig:
    """Hyperparameters of one mixture training run.

    Extends the rung-1 settings with the mixture-specific knobs. Everything
    except ``seed`` enters the experiment's cfg_hash; ``n_experts`` (K) is fully
    variable and read from config - changing K is a one-line YAML edit and a
    new cfg_hash, never a code change.
    """

    n_experts: int = 3
    learning_rate: float = 3e-3
    max_epochs: int = 500
    patience: int = 25
    gradient_clip_norm: float = 5.0
    lambda_cvar: float = 1.0
    lambda_balance: float = 0.01      # the anti-collapse pressure (Table 3)
    lambda_turn: float = 0.0          # explicit turnover pressure; 0 preserves E9
    # Anti-redundancy (Spec 8.5). Table 3 says to switch this on only if G2
    # finds near-identical experts - which it did on sector11 - so it is a
    # searched knob there rather than a permanent default.
    lambda_div: float = 0.0
    # Penalty on sustained underwater depth. CVaR sees bad DAYS; drawdown
    # sees bad MONTHS, which is the shape of a crisis. 0 by default.
    lambda_drawdown: float = 0.0
    gate_temperature: float = 1.0     # tau_g
    gate_noise_initial: float = 0.3   # sigma_g at epoch 0, annealed to 0 (R2)
    # The span the R2 noise anneal is measured over. None means "twice the
    # early-stopping patience", which is the fix for a bug that affected every
    # mixture trained before this field existed: the anneal used to be measured
    # against max_epochs (500), finishing at epoch 250, while early stopping
    # fires between epochs 7 and 80. Noise therefore never reached zero and the
    # gate never committed to a routing. Tying the horizon to patience keeps the
    # schedule inside a realistic run however max_epochs is set.
    gate_noise_anneal_epochs: int | None = None
    weight_decay: float = 0.0         # L2 on the heads; see TrainConfig
    seed: int = 0


@dataclass
class MixtureTrainResult:
    """Everything a caller needs after a mixture training run."""

    model: "MixtureOfExperts"
    history: pd.DataFrame
    best_epoch: int
    best_val_sharpe: float
    stopped_early: bool
    config: MixtureTrainConfig


def train_mixture(
    train_pack: SegmentPack,
    val_pack: SegmentPack,
    config: MixtureTrainConfig,
    model=None,
) -> MixtureTrainResult:
    """Train a K-expert mixture on one fold, with early stopping on validation.

    Identical discipline to the single head - full-span batches, Adam on the
    trainable parts only, clipped gradients, best-epoch restoration - plus the
    two mixture mechanics:

    * the gate's logit noise follows the R2 anneal (``annealed_noise_scale``),
      so routing stays exploratory while the experts differentiate and then
      hardens as the noise fades;
    * the loss carries the balance term, fed by the per-expert importance of
      THIS epoch's memberships, which is what keeps every expert receiving
      gradient.

    Validation is always noise-free: selection never sees the regulariser.
    """
    from rrc.models.gating import (  # local import to avoid a module cycle
        MixtureOfExperts,
        annealed_noise_scale,
        importance_shares,
        mean_gate_entropy,
    )

    set_all_seeds(config.seed)

    # The caller may hand in a pre-built variant (a SignalRoutedMixture or a
    # FixedRoutingMixture); the default is the plain rung-2 mixture. Either
    # way the model must be constructed AFTER set_all_seeds, so callers pass a
    # factory, not an instance, when they need a variant.
    if model is None:
        model = MixtureOfExperts(
            n_state_features=train_pack.states.shape[1],
            n_assets=train_pack.logit_mask.shape[1],
            n_experts=config.n_experts,
            gate_temperature=config.gate_temperature,
        )
    elif callable(model) and not isinstance(model, torch.nn.Module):
        model = model()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_sharpe = -np.inf
    best_epoch = -1
    best_parameters: dict | None = None
    epochs_without_improvement = 0
    history_rows: list[dict] = []
    stopped_early = False

    # R2's anneal is measured against a horizon a real run actually reaches,
    # not against max_epochs - see MixtureTrainConfig.gate_noise_anneal_epochs.
    anneal_horizon = (
        config.gate_noise_anneal_epochs
        if config.gate_noise_anneal_epochs is not None
        else 2 * config.patience
    )

    for epoch in range(config.max_epochs):
        noise_scale = annealed_noise_scale(
            epoch, anneal_horizon, config.gate_noise_initial
        )

        # ---- one full-span step, with train-time gate noise ---------------
        optimiser.zero_grad()
        memberships, expert_weights = model.route_pack(
            train_pack, noise_scale=noise_scale
        )
        blended = torch.einsum("jk,jkn->jn", memberships, expert_weights)
        ledger_result = run_ledger(
            blended, train_pack.execution_positions, train_pack.returns,
            train_pack.cash_index,
        )
        loss_parts = decision_loss(
            ledger_result.net_returns,
            train_pack.risk_free,
            lambda_cvar=config.lambda_cvar,
            lambda_balance=config.lambda_balance,
            importance=memberships.sum(dim=0),
            lambda_turn=config.lambda_turn,
            turnover=ledger_result.turnover,
            lambda_div=config.lambda_div,
            expert_weights=expert_weights,
            lambda_drawdown=config.lambda_drawdown,
        )
        loss_parts.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimiser.step()

        # ---- noise-free diagnostics and the early-stopping decision -------
        val_sharpe = annualised_validation_sharpe(model, val_pack)
        with torch.no_grad():
            clean_memberships, _ = model.route_pack(train_pack, noise_scale=0.0)
        shares = importance_shares(clean_memberships).detach().numpy()

        history_rows.append(
            {
                "epoch": epoch,
                **loss_parts.as_floats(),
                "train_sharpe_ann": -float(loss_parts.negative_sharpe.detach())
                * float(np.sqrt(TRADING_DAYS_PER_YEAR)),
                "val_sharpe_ann": val_sharpe,
                "noise_scale": noise_scale,
                "gate_entropy": mean_gate_entropy(clean_memberships),
                "min_share": float(shares.min()),
                "max_share": float(shares.max()),
            }
        )

        if not np.isfinite(float(loss_parts.total.detach())):
            raise FloatingPointError(
                f"Mixture loss became non-finite at epoch {epoch}."
            )

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_epoch = epoch
            best_parameters = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_early = True
                break

    if best_parameters is not None:
        model.load_state_dict(best_parameters)

    history = pd.DataFrame(history_rows)
    logger.info(
        "mixture K=%d: %d epochs (best %d, val Sharpe %.3f, min share %.3f)",
        config.n_experts, len(history), best_epoch, best_val_sharpe,
        float(history["min_share"].iloc[min(best_epoch, len(history) - 1)]),
    )
    return MixtureTrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_sharpe=best_val_sharpe,
        stopped_early=stopped_early,
        config=config,
    )


def mixture_diagnostics(model, pack: SegmentPack) -> dict:
    """The routing internals of a trained mixture on one segment, noise-free.

    Everything gate G2's verdict and the F6/F7 figures need:

    * per-day memberships ``pi`` and their mean entropy;
    * per-expert importance shares (the aliveness measure);
    * every expert's own proposed portfolio schedule;
    * pairwise correlations between the experts' weight schedules - the
      specialisation measure ("below 0.9" is G2's threshold);
    * each expert's standalone net-return stream through the same ledger,
      whose realised-volatility ranking later feeds R1's risk-aversion
      assignment at rung 5.
    """
    from rrc.models.gating import importance_shares, mean_gate_entropy

    with torch.no_grad():
        memberships, expert_weights = model.route_pack(pack, noise_scale=0.0)

        standalone: list[dict] = []
        for expert_index in range(expert_weights.shape[1]):
            own_returns = run_ledger(
                expert_weights[:, expert_index],
                pack.execution_positions,
                pack.returns,
                pack.cash_index,
            ).net_returns
            standalone.append(
                {
                    "expert": expert_index,
                    "ann_vol": float(own_returns.std(unbiased=True))
                    * float(np.sqrt(TRADING_DAYS_PER_YEAR)),
                    "sharpe": float(excess_sharpe_torch(own_returns, pack.risk_free))
                    * float(np.sqrt(TRADING_DAYS_PER_YEAR)),
                }
            )

    n_experts = expert_weights.shape[1]
    flattened = expert_weights.reshape(pack.n_signals, n_experts, -1)
    pairwise: dict[str, float] = {}
    for first in range(n_experts):
        for second in range(first + 1, n_experts):
            series_a = flattened[:, first].reshape(-1).numpy()
            series_b = flattened[:, second].reshape(-1).numpy()
            pairwise[f"{first}-{second}"] = float(np.corrcoef(series_a, series_b)[0, 1])

    return {
        # Under E7 every execution day is its signal day plus one, so the
        # membership rows are indexed by the signal dates they were decided on.
        "memberships": pd.DataFrame(
            memberships.numpy(),
            index=pack.dates[pack.execution_positions - 1],
            columns=[f"expert_{k}" for k in range(n_experts)],
        ),
        "importance_shares": importance_shares(memberships).numpy().tolist(),
        "gate_entropy": mean_gate_entropy(memberships),
        "expert_weights": expert_weights.numpy(),
        "pairwise_weight_correlations": pairwise,
        "max_pairwise_correlation": max(pairwise.values()) if pairwise else float("nan"),
        "standalone": standalone,
    }


def evaluate_on_pack(head: LinearPortfolioHead, pack: SegmentPack) -> dict:
    """Full Section-13 evaluation of a trained head on one segment.

    Used for the registry rows: runs the ledger with diagnostics on so turnover
    and holdings statistics are available, then summarises with the same
    functions every baseline used.
    """
    with torch.no_grad():
        if hasattr(head, "weights_for_pack"):
            weights = head.weights_for_pack(pack)
        else:
            weights = head(pack.states, pack.logit_mask)
        result = run_ledger(
            targets=weights,
            exec_days=pack.execution_positions,
            simple_returns=pack.returns,
            cash_index=pack.cash_index,
            return_diagnostics=True,
        )

    metrics = summarise(
        result.net_returns.numpy(),
        pack.risk_free.numpy(),
        turnover=result.turnover.numpy(),
        held_weights=result.held_weights.numpy(),
        cash_index=pack.cash_index,
    )
    mean_cash_weight = float(weights[:, pack.cash_index].mean())
    return {
        "metrics": metrics,
        "net_returns": pd.Series(result.net_returns.numpy(), index=pack.dates),
        "mean_cash_weight": mean_cash_weight,
    }
