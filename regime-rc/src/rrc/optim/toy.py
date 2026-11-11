"""The synthetic toy fold - find broken plumbing on day 8, not day 30.

The idea (Section 10's gradient-check protocol)
-----------------------------------------------
Before any real training, the whole trainable stack - head, softmax, ledger,
loss - is exercised on a tiny synthetic problem whose right answer is known:

* **Two risky assets plus cash over 40 trading days, with a planted regime.**
  In the first half asset A goes up strongly and B goes down; in the second
  half they swap. The "reservoir state" is a small feature vector whose first
  component says which half you are in (plus pure-noise features to make sure
  the head can ignore distractors).
* A linear head that reads that state can learn "first half: hold A; second
  half: hold B" and achieve an absurd Sharpe. If gradient descent through the
  ledger cannot find that mapping, the plumbing is broken - and it is broken
  on a 40-day toy that trains in a second, not discovered after days of real
  runs producing quietly wrong numbers.
* The same toy drives the **finite-difference gradient check**: for a sample of
  head parameters, nudge the parameter, recompute the loss, and compare the
  slope against autograd. This extends P2's B7 (which checked the ledger alone)
  to the entire model-to-loss chain.

Nothing here touches real data, and the toy is deterministic given its seed, so
tests and the run script exercise identical objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from rrc.backtest.ledger import weekly_execution_schedule
from rrc.optim.fold_data import SegmentPack

# Toy dimensions - small enough to reason about completely.
N_DAYS = 40
N_RISKY = 2
N_STATE_FEATURES = 8
CASH_INDEX = 2          # columns: [A, B, CASH]

# Planted structure: strong, clean drifts so the right policy is unambiguous.
WINNING_DRIFT = 0.015   # +1.5% a day for the in-regime asset
LOSING_DRIFT = -0.010   # -1.0% a day for the out-of-regime asset
NOISE_SCALE = 0.003     # small enough that the drift dominates
DAILY_RISK_FREE = 0.0001


@dataclass(frozen=True)
class ToyFold:
    """The toy problem, packaged like a real fold.

    Attributes
    ----------
    pack
        A :class:`SegmentPack`, so the real training loop runs on it unchanged.
    regime_switch_day
        The day the planted regime flips (A stops winning, B starts).
    """

    pack: SegmentPack
    regime_switch_day: int


def make_toy_fold(seed: int = 0) -> ToyFold:
    """Build the planted-regime toy fold.

    Parameters
    ----------
    seed
        Seed for the noise; the planted structure itself is deterministic.

    Returns
    -------
    ToyFold
    """
    generator = np.random.default_rng(seed)
    switch_day = N_DAYS // 2
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS, name="date")

    # ---- returns with the planted regime ---------------------------------
    returns = np.empty((N_DAYS, N_RISKY + 1), dtype=np.float64)
    noise = generator.normal(0.0, NOISE_SCALE, size=(N_DAYS, N_RISKY))

    returns[:switch_day, 0] = WINNING_DRIFT + noise[:switch_day, 0]   # A wins early
    returns[:switch_day, 1] = LOSING_DRIFT + noise[:switch_day, 1]
    returns[switch_day:, 0] = LOSING_DRIFT + noise[switch_day:, 0]    # then B wins
    returns[switch_day:, 1] = WINNING_DRIFT + noise[switch_day:, 1]
    returns[:, CASH_INDEX] = DAILY_RISK_FREE

    # ---- the "state": one honest regime feature + noise features ---------
    # Feature 0 is +1 before the switch and -1 after: exactly the kind of
    # slow, regime-describing signal a real reservoir state carries. The other
    # features are noise the head must learn to ignore.
    states = generator.normal(0.0, 1.0, size=(N_DAYS, N_STATE_FEATURES))
    states[:, 0] = np.where(np.arange(N_DAYS) < switch_day, 1.0, -1.0)

    # ---- weekly schedule, exactly as on real data -------------------------
    signal_relative, execution_relative = weekly_execution_schedule(dates)

    pack = SegmentPack(
        name="toy",
        dates=dates,
        states=torch.as_tensor(states[signal_relative], dtype=torch.float64),
        logit_mask=torch.zeros(
            len(signal_relative), N_RISKY + 1, dtype=torch.float64
        ),  # everything always tradeable in the toy
        execution_positions=execution_relative,
        returns=torch.as_tensor(returns, dtype=torch.float64),
        risk_free=torch.as_tensor(returns[:, CASH_INDEX].copy(), dtype=torch.float64),
        cash_index=CASH_INDEX,
    )
    return ToyFold(pack=pack, regime_switch_day=switch_day)


def finite_difference_check(
    loss_of_parameters,
    parameters: torch.Tensor,
    n_samples: int = 12,
    step: float = 1e-6,
    seed: int = 0,
) -> tuple[float, float]:
    """Compare autograd against central finite differences on sampled entries.

    Parameters
    ----------
    loss_of_parameters
        Callable evaluating the scalar loss from the CURRENT parameter values
        (the parameters are mutated in place between calls).
    parameters
        The parameter tensor to probe (e.g. the head's weight matrix), with
        ``requires_grad=True`` and ``.grad`` already populated by a backward
        pass.
    n_samples
        How many randomly chosen entries to probe.
    step
        Finite-difference step.
    seed
        Which entries get probed.

    Returns
    -------
    max_absolute_error, max_gradient_magnitude
        The worst disagreement found, and the scale of the gradients probed
        (for interpreting the error as a relative quantity).
    """
    generator = np.random.default_rng(seed)
    flat_gradient = parameters.grad.detach().reshape(-1)
    n_entries = parameters.numel()
    probe_indices = generator.choice(n_entries, size=min(n_samples, n_entries), replace=False)

    worst_error = 0.0
    largest_gradient = 0.0

    for flat_index in probe_indices:
        with torch.no_grad():
            flat_view = parameters.reshape(-1)
            original_value = float(flat_view[flat_index])

            flat_view[flat_index] = original_value + step
            loss_up = float(loss_of_parameters())
            flat_view[flat_index] = original_value - step
            loss_down = float(loss_of_parameters())
            flat_view[flat_index] = original_value  # restore exactly

        finite_difference_slope = (loss_up - loss_down) / (2.0 * step)
        autograd_slope = float(flat_gradient[flat_index])

        worst_error = max(worst_error, abs(finite_difference_slope - autograd_slope))
        largest_gradient = max(largest_gradient, abs(autograd_slope))

    return worst_error, largest_gradient
