"""The gating network and the mixture of experts - the hinge of the method.

The idea in one paragraph
-------------------------
Instead of one head proposing one portfolio, **K expert heads** each propose a
portfolio from the same frozen reservoir state, and a small **gating network**
reads the state and produces a soft membership ``pi_t`` over the experts (a
probability vector). The traded portfolio is the membership-weighted blend::

    w_t = sum_k pi_{t,k} * w_t^(k)

This single equation is what makes the whole method trainable: the blend is
differentiable in the gating (through ``pi``) *and* in the experts (through
``w^(k)``), so one outcome-based gradient trains both. It is also, structurally,
a mixture of experts - with the twist that the training signal is realised
portfolio quality, which is what pushes the experts to specialise into
*decision regimes* rather than into anything statistical.

What a "regime" is here
-----------------------
Nothing is labelled. ``pi_t`` is a soft assignment over K initially meaningless
slots; the slots acquire meaning only through training, because each expert's
gradient is scaled by how much the gate used it (``d w / d w^(k) = pi_k``). The
gate routes certain states to certain experts, those experts get better on
those states, better experts earn more routing there - a self-reinforcing loop
that produces specialisation. The loop's failure mode - one expert absorbing
everything while the rest starve - is held off by two mechanisms:

* **train-time logit noise** (R2): Gaussian noise on the gate logits, annealed
  linearly to zero over the first half of training, stops the gate from
  freezing into a hard, near-one-hot routing in the first few epochs before the
  experts have differentiated;
* **the balance loss** (Spec 8.4): a penalty on uneven total expert usage,
  already built into the loss stack at P5 and switched on here.

Evaluation always uses noise-free logits: the noise is a training regulariser,
never part of the deployed decision.

K is a first-class knob
-----------------------
``n_experts`` is a constructor argument read from config, nowhere hard-coded.
K = 3 is the default; the K-ablation ({2, 4}) at P10 and any later
experimentation change one YAML line. K = 1 degenerates exactly to rung 1's
single head (the gate has one slot and pi is identically 1), which is a useful
consistency check and is tested.
"""

from __future__ import annotations

import torch
from torch import nn

from rrc.models.experts import LinearPortfolioHead


class GatingNetwork(nn.Module):
    """A linear map from a conditioning input to a soft expert membership.

    Parameters
    ----------
    n_inputs
        Width of the conditioning input. At rung 2 this is the reservoir state
        alone; at rung 3 the signal vector is concatenated on, so the width is
        whatever the caller feeds - the module does not care what the inputs
        mean.
    n_experts
        K, the number of experts to route over.
    temperature
        ``tau_g``. Logits are divided by it before the softmax: a higher
        temperature gives softer, more blended memberships; a lower one gives
        harder routing. Default 1.0 (searched 0.5-2 in the P8 outer search).
    """

    def __init__(self, n_inputs: int, n_experts: int, temperature: float = 1.0) -> None:
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be at least 1, got {n_experts}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        self.n_inputs = n_inputs
        self.n_experts = n_experts
        self.temperature = float(temperature)
        self.linear = nn.Linear(n_inputs, n_experts).double()

    def forward(self, inputs: torch.Tensor, noise_scale: float = 0.0) -> torch.Tensor:
        """Produce the membership matrix ``pi``.

        Parameters
        ----------
        inputs
            ``(J, n_inputs)`` conditioning inputs at the signal days.
        noise_scale
            Standard deviation ``sigma_g`` of Gaussian noise added to the raw
            logits. Nonzero only during training (R2); evaluation passes 0.

        Returns
        -------
        torch.Tensor
            ``(J, n_experts)`` rows on the simplex: non-negative, summing to 1.
        """
        logits = self.linear(inputs)
        if noise_scale > 0:
            logits = logits + noise_scale * torch.randn_like(logits)
        return torch.softmax(logits / self.temperature, dim=-1)


class MixtureOfExperts(nn.Module):
    """K expert heads blended by a gating network - the rung-2+ model.

    The forward signature deliberately matches the single head's
    ``(states, logit_mask)`` call, so every evaluation utility built at P5
    (``evaluate_on_pack``, ``conditioning_edge``) works on the mixture
    unchanged. The extra arguments have training-time or later-rung defaults.

    Parameters
    ----------
    n_state_features
        Width of the reservoir state the experts read.
    n_assets
        Number of output assets including cash.
    n_experts
        K. Fully variable; read from config.
    gating_n_inputs
        Width of the gating's conditioning input. Defaults to
        ``n_state_features`` (rung 2: the gate reads the state alone). Rung 3
        passes a larger width and supplies ``gating_inputs`` at call time.
    gate_temperature
        ``tau_g`` for the gating softmax.
    """

    def __init__(
        self,
        n_state_features: int,
        n_assets: int,
        n_experts: int,
        gating_n_inputs: int | None = None,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_state_features = n_state_features
        self.n_assets = n_assets
        self.n_experts = n_experts

        self.experts = nn.ModuleList(
            [LinearPortfolioHead(n_state_features, n_assets) for _ in range(n_experts)]
        )
        self.gating = GatingNetwork(
            n_inputs=gating_n_inputs or n_state_features,
            n_experts=n_experts,
            temperature=gate_temperature,
        )

    def forward(
        self,
        states: torch.Tensor,
        logit_mask: torch.Tensor,
        gating_inputs: torch.Tensor | None = None,
        noise_scale: float = 0.0,
    ) -> torch.Tensor:
        """Blend the experts' portfolios by the gate's memberships.

        Parameters
        ----------
        states
            ``(J, n_state_features)`` reservoir states at the signal days.
        logit_mask
            ``(J, n_assets)`` availability mask, applied inside every expert.
        gating_inputs
            Conditioning input for the gate. ``None`` (the rung-2 case) means
            the gate reads the reservoir state itself.
        noise_scale
            Train-time gate noise; 0 for evaluation.

        Returns
        -------
        torch.Tensor
            ``(J, n_assets)`` blended weights. A convex combination of simplex
            rows is itself on the simplex, so the blend is long-only and fully
            invested by construction - no renormalisation step needed.
        """
        memberships, expert_weights = self.route(
            states, logit_mask, gating_inputs, noise_scale
        )
        # (J, K) memberships blending (J, K, N) expert portfolios -> (J, N).
        return torch.einsum("jk,jkn->jn", memberships, expert_weights)

    def route(
        self,
        states: torch.Tensor,
        logit_mask: torch.Tensor,
        gating_inputs: torch.Tensor | None = None,
        noise_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the routing internals - what the diagnostics dashboard reads.

        Returns
        -------
        memberships
            ``(J, K)`` the gate's soft membership ``pi``.
        expert_weights
            ``(J, K, N)`` each expert's own proposed portfolio.
        """
        conditioning = states if gating_inputs is None else gating_inputs
        memberships = self.gating(conditioning, noise_scale=noise_scale)
        expert_weights = torch.stack(
            [expert(states, logit_mask) for expert in self.experts], dim=1
        )
        return memberships, expert_weights


    def route_pack(self, pack, noise_scale: float = 0.0):
        """Route a :class:`SegmentPack` - the one entry point every trainer uses.

        Having a single pack-level routing method is what makes the R6
        "identical loss, schedule, seeds" guarantee *literal code reuse*: the
        rung-2 mixture, the rung-3 signal-routed variants, and the rung-4
        fixed-routing baselines all train through exactly the same loop, which
        calls exactly this method. The base mixture ignores the pack's signals.
        """
        return self.route(pack.states, pack.logit_mask, noise_scale=noise_scale)

    def weights_for_pack(self, pack, noise_scale: float = 0.0) -> torch.Tensor:
        """Blended portfolio weights for a pack - evaluation's entry point."""
        memberships, expert_weights = self.route_pack(pack, noise_scale)
        return torch.einsum("jk,jkn->jn", memberships, expert_weights)


class SignalRoutedMixture(MixtureOfExperts):
    """The rung-3 arms: the market signals join the gate, the experts, or both.

    The three arms and the question they answer (RQ3 / E11)
    -------------------------------------------------------
    ============  ==================  ==================  =====================
    arm           gate reads          experts read        extra parameters
    ============  ==================  ==================  =====================
    3a sent_gate  [state ; signals]   state               K * d        (tiny)
    3b sent_exp   state               [state ; signals]   K * d * N    (small)
    3c sent_both  [state ; signals]   [state ; signals]   both         (ceiling)
    ============  ==================  ==================  =====================

    Feeding the signals only to the *gate* (3a) lets them influence WHICH
    expert is in charge but not WHAT any expert does; only to the *experts*
    (3b) the reverse. Exact capacity equality between the two is impossible
    (A8: K*d vs K*d*N weights), so the parameter counts are logged exactly and
    3c serves as the ceiling reference - the contrast is information routing at
    near-equal capacity, stated honestly rather than pretended equal.

    Parameters
    ----------
    n_signals
        Width ``d`` of the market-signal vector.
    signals_to_gate, signals_to_experts
        Which parts of the model read the signals. Both False would just be the
        base mixture, so it is rejected.
    """

    def __init__(
        self,
        n_state_features: int,
        n_assets: int,
        n_experts: int,
        n_signals: int,
        signals_to_gate: bool,
        signals_to_experts: bool,
        gate_temperature: float = 1.0,
    ) -> None:
        if not (signals_to_gate or signals_to_experts):
            raise ValueError(
                "At least one of signals_to_gate / signals_to_experts must be "
                "True; the no-signal case is the base MixtureOfExperts."
            )
        expert_input_width = n_state_features + (n_signals if signals_to_experts else 0)
        gate_input_width = n_state_features + (n_signals if signals_to_gate else 0)

        # The base constructor sizes the experts from its first argument, so we
        # hand it the widened expert input; the true state width is kept for
        # documentation and error messages.
        super().__init__(
            n_state_features=expert_input_width,
            n_assets=n_assets,
            n_experts=n_experts,
            gating_n_inputs=gate_input_width,
            gate_temperature=gate_temperature,
        )
        self.true_state_width = n_state_features
        self.n_market_signals = n_signals
        self.signals_to_gate = signals_to_gate
        self.signals_to_experts = signals_to_experts

    def route_pack(self, pack, noise_scale: float = 0.0):
        """Compose the arm's inputs from the pack, then route as usual."""
        if pack.signals is None:
            raise ValueError(
                f"{type(self).__name__} needs the pack to carry market signals."
            )
        with_signals = torch.cat([pack.states, pack.signals], dim=1)

        expert_inputs = with_signals if self.signals_to_experts else pack.states
        gate_inputs = with_signals if self.signals_to_gate else pack.states

        memberships = self.gating(gate_inputs, noise_scale=noise_scale)
        expert_weights = torch.stack(
            [expert(expert_inputs, pack.logit_mask) for expert in self.experts], dim=1
        )
        return memberships, expert_weights


class FixedRoutingMixture(nn.Module):
    """The rung-4 baseline: a statistical gate's pi, with trainable experts.

    The R6 equal-conditions rule made concrete: the learned gate is REPLACED by
    an externally supplied membership series (the filtered HMM's, or the VIX
    terciles'), carried on the pack as ``fixed_memberships``, and the experts
    are retrained from scratch under that fixed routing - same loss, same
    schedule, same seeds, same code path. Anything less would compare a trained
    system against a crippled one and the comparison would be worthless.

    There is no gate module at all, so the parameter count is exactly the
    experts' - recorded in the registry so the capacity difference against the
    learned system (which carries a gate) is stated, not hidden.
    """

    def __init__(self, n_state_features: int, n_assets: int, n_experts: int) -> None:
        super().__init__()
        self.n_state_features = n_state_features
        self.n_assets = n_assets
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [LinearPortfolioHead(n_state_features, n_assets) for _ in range(n_experts)]
        )

    def route_pack(self, pack, noise_scale: float = 0.0):
        """Return the pack's fixed memberships and the experts' proposals.

        ``noise_scale`` is accepted for training-loop compatibility and ignored:
        the routing is not learned, so there is nothing to regularise.
        """
        if pack.fixed_memberships is None:
            raise ValueError(
                "FixedRoutingMixture needs the pack to carry fixed_memberships "
                "- build the pack through the rung-4 gate first."
            )
        if pack.fixed_memberships.shape != (pack.n_signals, self.n_experts):
            raise ValueError(
                f"fixed_memberships is {tuple(pack.fixed_memberships.shape)}, "
                f"expected ({pack.n_signals}, {self.n_experts})."
            )
        expert_weights = torch.stack(
            [expert(pack.states, pack.logit_mask) for expert in self.experts], dim=1
        )
        return pack.fixed_memberships, expert_weights

    def weights_for_pack(self, pack, noise_scale: float = 0.0) -> torch.Tensor:
        """Blended weights under the fixed routing."""
        memberships, expert_weights = self.route_pack(pack, noise_scale)
        return torch.einsum("jk,jkn->jn", memberships, expert_weights)


def annealed_noise_scale(epoch: int, horizon_epochs: int, initial_scale: float) -> float:
    """The R2 noise schedule: linear from ``initial_scale`` to 0 over the first
    half of ``horizon_epochs``, then exactly 0.

    The point of the anneal: early on, noise keeps the routing exploratory so
    no expert is starved before the experts have had a chance to differentiate;
    later, the noise is gone and the gate commits to what it has learned.

    The horizon this is measured against
    ------------------------------------
    ``horizon_epochs`` is the span the schedule is *defined over*, and choosing
    it badly silently disables the second half of R2's intent.

    This project got that wrong for its entire P0-P8 history: the caller passed
    ``max_epochs`` (500), so the anneal was scheduled to finish at epoch 250,
    while early stopping actually fired between epochs 7 and 80. Every mixture
    ever trained therefore ran under roughly 90% of full gate noise for its
    whole life and never committed to a routing.

    The fix is at the call site, not here - see
    ``MixtureTrainConfig.gate_noise_anneal_epochs``, which defaults to twice the
    early-stopping patience so the schedule completes inside a realistic run.
    The parameter is named ``horizon_epochs`` rather than ``max_epochs`` so the
    distinction cannot be lost again.

    Parameters
    ----------
    epoch
        Current epoch (0-based).
    horizon_epochs
        The span the schedule is defined over. Noise reaches zero at
        ``horizon_epochs / 2``.
    initial_scale
        ``sigma_g`` at epoch 0 (default 0.3 per R2).

    Returns
    -------
    float
        The noise standard deviation for this epoch.
    """
    if horizon_epochs < 1:
        raise ValueError(f"horizon_epochs must be at least 1, got {horizon_epochs}.")
    half_life = horizon_epochs / 2.0
    remaining = max(0.0, 1.0 - epoch / half_life)
    return initial_scale * remaining


def importance_shares(memberships: torch.Tensor) -> torch.Tensor:
    """Each expert's share of the total gate mass over a span.

    ``Imp_k = sum_t pi_{t,k}``, normalised to fractions. This is the aliveness
    measure behind gate G2's "every expert holds at least 5 percent" criterion,
    and the quantity the balance loss regularises.
    """
    importance = memberships.sum(dim=0)
    return importance / importance.sum()


def mean_gate_entropy(memberships: torch.Tensor) -> float:
    """Average per-day entropy of the memberships, in nats.

    The saturation diagnostic (A2). Near ``log K`` means the gate is blending
    everything evenly (it has learned no routing); near 0 means hard one-hot
    routing (at risk of collapse). Somewhere in between is where a healthy,
    specialised mixture lives. Logged for every epoch and written to the
    registry for every run.
    """
    clamped = memberships.clamp_min(1e-12)
    entropy_per_day = -(clamped * clamped.log()).sum(dim=1)
    return float(entropy_per_day.mean().detach())
