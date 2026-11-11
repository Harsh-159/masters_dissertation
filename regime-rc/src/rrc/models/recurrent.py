"""The LSTM comparison arm: a *trained* temporal encoder instead of a frozen one.

What this exists to test
------------------------
Decision D2 freezes the temporal feature space: a deterministic Simple Cycle
Reservoir, never trained, with only small readout heads learned on top. The
obvious challenge is whether a *trained* recurrent network would do better, and
that is the question this arm answers. An LSTM replaces the reservoir and
nothing else - same decision loss, same ledger, same folds, seeds, costs and
availability masking - so the single variable is **frozen encoder versus trained
encoder**.

The expectation, stated in advance
----------------------------------
The LSTM is expected to **lose**, and saying so before running it is the point
(the plan applies the same discipline to its evolve-the-heads comparison). The
grounds are quantitative: a fold carries only ~300-450 weekly training
decisions, the existing heads already run at 30-40 parameters per decision, and
this project has repeatedly measured that patterns fitted on a training fold do
not transfer to the next year - weight decay, lambda_cvar and a drawdown penalty
all closed training-side gaps without improving validation. A trained recurrent
encoder adds an order of magnitude more parameters into exactly that regime.

If it loses, decision D2 stops being an assumption and becomes a measurement:
the reservoir wins *because* there is not enough data to train a recurrent net.
If it wins, that is a finding worth having, and the pre-registered expectation
is what makes either outcome honest.

Windowed inputs, and why
------------------------
The reservoir is driven day by day across the whole sample and read at the
weekly signal days. Backpropagating an LSTM through ~6,900 daily steps is both
expensive and badly conditioned, so this arm uses the standard sequence-model
setup instead: at each signal day the LSTM consumes the trailing ``lookback``
days and its final hidden state feeds the same portfolio head. The window ends
*at* the signal day and never extends past it, so the causality rule is the same
one every other component obeys.

This is a real difference from the reservoir - bounded memory versus fading
memory - and it is stated rather than hidden. At the operating leak rate the
reservoir's effective memory is short too, so a one-quarter window is a
reasonable match; ``lookback`` is a config knob if that is ever challenged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch
from torch import nn

from rrc.models.experts import LinearPortfolioHead


class LSTMPortfolioModel(nn.Module):
    """An LSTM temporal encoder feeding the project's standard portfolio head.

    The output layer is deliberately the SAME :class:`LinearPortfolioHead` the
    reservoir arms use: availability-masked logits through a softmax, so the
    portfolio is long-only and fully invested by construction. Only the encoder
    differs.

    Parameters
    ----------
    n_inputs
        Width of the driving series (the reservoir-input universe).
    n_assets
        Number of output assets, cash column included.
    hidden_size
        LSTM hidden width. Kept small (32 by default) because a training fold
        holds only a few hundred weekly decisions; a large network here would be
        a strawman that loses for the wrong reason.
    """

    def __init__(
        self,
        n_inputs: int,
        n_assets: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        # float64 throughout, matching the ledger and the reservoir heads.
        #
        # torch applies `dropout` BETWEEN stacked layers only, so it is silently
        # a no-op at num_layers=1 and torch warns about it. Passing 0.0 there
        # keeps the warning out of the logs and makes the no-op explicit rather
        # than something a reader has to know about torch to spot.
        self.lstm = nn.LSTM(
            n_inputs,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        ).double()
        self.head = LinearPortfolioHead(hidden_size, n_assets)

    def forward(self, windows: torch.Tensor, logit_mask: torch.Tensor) -> torch.Tensor:
        """Map trailing input windows to portfolio weights.

        Parameters
        ----------
        windows
            ``(J, lookback, n_inputs)`` trailing windows, one per signal day.
        logit_mask
            ``(J, n_assets)`` additive availability mask.

        Returns
        -------
        torch.Tensor
            ``(J, n_assets)`` weights, non-negative and summing to 1.
        """
        if windows.ndim != 3 or windows.shape[-1] != self.n_inputs:
            raise ValueError(
                f"windows must be (J, lookback, {self.n_inputs}), "
                f"got {tuple(windows.shape)}."
            )
        outputs, _ = self.lstm(windows)
        final_hidden = outputs[:, -1, :]          # the state at the signal day
        return self.head(final_hidden, logit_mask)

    def route_pack(self, pack, noise_scale: float = 0.0):
        """Present the LSTM as a one-expert mixture, for the shared train loop.

        The training loop drives every mixture variant through ``route_pack``.
        Returning an all-ones membership over a single expert lets the LSTM use
        that identical loop - same loss, same schedule, same early stopping -
        which is what makes the comparison equal-conditions rather than merely
        similar. ``noise_scale`` is accepted and ignored: there is no gate here.
        """
        weights = self.weights_for_pack(pack)
        memberships = torch.ones(weights.shape[0], 1, dtype=weights.dtype)
        return memberships, weights.unsqueeze(1)     # (J, 1), (J, 1, N)

    def weights_for_pack(self, pack, noise_scale: float = 0.0) -> torch.Tensor:
        """Portfolio weights for a pack carrying ``input_windows``."""
        if getattr(pack, "input_windows", None) is None:
            raise ValueError(
                "LSTMPortfolioModel needs the pack to carry input_windows - "
                "build it with attach_input_windows() first."
            )
        return self(pack.input_windows, pack.logit_mask)


def build_input_windows(
    pack,
    reservoir_panel: pd.DataFrame,
    artifacts,
    lookback: int,
) -> torch.Tensor:
    """Trailing input windows ending at each of a pack's signal days.

    The window for signal day ``t`` is days ``[t - lookback + 1, t]`` inclusive:
    it ends **at** the signal day and never reaches past it, which is the same
    causality rule the reservoir, the signals and the ledger all obey. Windows
    that would start before the panel begins are left-padded with the first
    available row, which only ever affects the earliest folds.

    Parameters
    ----------
    pack
        A :class:`~rrc.optim.fold_data.SegmentPack`.
    reservoir_panel
        The universe's z-scored reservoir-input panel - the same driving series
        the reservoir consumes, so both encoders see identical data.
    artifacts
        The universe's artifacts, for the calendar.
    lookback
        Window length in trading days.

    Returns
    -------
    torch.Tensor
        ``(J, lookback, n_inputs)`` float64.
    """
    values = reservoir_panel.to_numpy()
    windows = np.empty((pack.n_signals, lookback, values.shape[1]), dtype=np.float64)

    for j in range(pack.n_signals):
        # Under E7 the signal day is the execution day minus one.
        signal_date = pack.dates[pack.execution_positions[j] - 1]
        end = int(artifacts.dates.get_loc(signal_date))
        start = end - lookback + 1
        if start >= 0:
            windows[j] = values[start : end + 1]
        else:
            # Left-pad with the first available row rather than with zeros: a
            # zero is a meaningful z-score (an average day) and would amount to
            # inventing observations.
            pad = np.repeat(values[:1], -start, axis=0)
            windows[j] = np.vstack([pad, values[: end + 1]])

    if not np.isfinite(windows).all():
        raise ValueError(
            "Non-finite values in the LSTM input windows; the window reaches "
            "into the panel's z-score warm-up."
        )
    return torch.as_tensor(windows, dtype=torch.float64)


def attach_input_windows(packs: dict, reservoir_panel, artifacts, lookback: int) -> dict:
    """Return a copy of a fold's packs with ``input_windows`` attached."""
    return {
        name: dataclasses.replace(
            pack,
            input_windows=build_input_windows(pack, reservoir_panel, artifacts, lookback),
        )
        for name, pack in packs.items()
    }
