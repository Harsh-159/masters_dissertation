"""The portfolio head: from reservoir state to portfolio weights.

What a head is
--------------
The reservoir is frozen; the head is the small trainable map that reads the
reservoir state on a signal day and proposes a portfolio. At rung 1 there is
exactly one head (K = 1) and no gating - the entire trainable system is one
linear layer. That is deliberate (locked decision D9): a linear readout on a
reservoir is the classical reservoir-computing recipe, it trains in seconds,
and when something goes wrong the diagnostics are interpretable. Depth is added
later only if linear demonstrably saturates.

From logits to weights, and why softmax
---------------------------------------
The head produces one logit per asset (cash included) and passes them through a
softmax. The softmax guarantees, by construction, that the output is a valid
long-only, fully-invested portfolio: every weight is positive and they sum to
one. No projection step, no constraint solver - feasibility is baked into the
output layer, and gradients flow through it smoothly.

Availability masking (locked decision R4)
-----------------------------------------
Not every asset exists on every day - XLRE only launches in October 2015. The
availability mask is an *additive* mask on the logits: unavailable assets get
``-1e9`` added, so after the softmax their weight is exactly zero (``exp(-1e9)``
underflows to 0.0, not to something merely small). Cash is never masked. The
same constant is used here as in the data layer's mask so there is exactly one
definition of "unavailable" in the project.

Precision note
--------------
The head runs in float64, matching the ledger. The model is tiny (a few
thousand parameters), so the cost is nil, and the finite-difference gradient
check can then be held to a tight tolerance instead of being blurred by
float32 rounding.
"""

from __future__ import annotations

import torch
from torch import nn

from rrc.data.availability import UNAVAILABLE_LOGIT_PENALTY


class LinearPortfolioHead(nn.Module):
    """A single linear layer from reservoir state to portfolio weights.

    Parameters
    ----------
    n_state_features
        Dimensionality of the reservoir state (``N_r``, e.g. 300).
    n_assets
        Number of output assets **including the cash column**.
    """

    def __init__(self, n_state_features: int, n_assets: int) -> None:
        super().__init__()
        self.n_state_features = n_state_features
        self.n_assets = n_assets
        # double(): the whole decision path runs in float64 (see module note).
        self.linear = nn.Linear(n_state_features, n_assets).double()

    def forward(
        self, states: torch.Tensor, logit_mask: torch.Tensor
    ) -> torch.Tensor:
        """Map states to portfolio weights.

        Parameters
        ----------
        states
            ``(J, n_state_features)`` reservoir states at the signal days.
        logit_mask
            ``(J, n_assets)`` additive mask: ``0.0`` where the asset is
            tradeable, ``-1e9`` where it is not. Cash is always ``0.0``.

        Returns
        -------
        torch.Tensor
            ``(J, n_assets)`` weights: non-negative, rows summing to 1, with
            exactly zero weight on masked assets.
        """
        if states.ndim != 2 or states.shape[1] != self.n_state_features:
            raise ValueError(
                f"states must be (J, {self.n_state_features}), got {tuple(states.shape)}."
            )
        if logit_mask.shape != (states.shape[0], self.n_assets):
            raise ValueError(
                f"logit_mask must be ({states.shape[0]}, {self.n_assets}), "
                f"got {tuple(logit_mask.shape)}."
            )

        logits = self.linear(states) + logit_mask
        return torch.softmax(logits, dim=-1)


def build_logit_mask(
    availability_rows: torch.Tensor, cash_last: bool = True
) -> torch.Tensor:
    """Turn 0/1 availability rows into an additive logit mask with a cash column.

    Parameters
    ----------
    availability_rows
        ``(J, n_risky)`` availability of the risky assets at each signal day
        (1 = tradeable, 0 = not).
    cash_last
        Where the cash column sits. The project convention is cash as the last
        column, matching the accounting panel's layout.

    Returns
    -------
    torch.Tensor
        ``(J, n_risky + 1)`` float64 mask: ``0.0`` where tradeable (and always
        for cash), ``UNAVAILABLE_LOGIT_PENALTY`` where not.
    """
    if not cash_last:
        raise NotImplementedError("The project convention is cash as the last column.")

    risky_mask = (1 - availability_rows).to(torch.float64) * UNAVAILABLE_LOGIT_PENALTY
    cash_column = torch.zeros(availability_rows.shape[0], 1, dtype=torch.float64)
    return torch.cat([risky_mask, cash_column], dim=1)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters - the registry's ``n_params`` column.

    Recorded for every run because the equal-capacity argument at rung 3
    (sentiment into the gating versus into the experts, A8) rests on being able
    to state parameter counts exactly rather than approximately.
    """
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
