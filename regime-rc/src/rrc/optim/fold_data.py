"""Assembling walk-forward fold data into ready-to-train packs.

What a "pack" is
----------------
Training the head needs, for one window of days: the reservoir states at the
weekly signal days, the availability mask at those days, the execution-day
positions, the daily return panel to run the ledger over, and the risk-free
series for the loss. A :class:`SegmentPack` bundles exactly that, as tensors,
with every alignment question settled once - so the training loop and the
evaluation code contain no indexing arithmetic at all, just model calls.

The alignment rules, stated once
--------------------------------
* A window's days are a contiguous slice of the universe calendar. The weekly
  schedule (signal on the last trading day of each week, execute the next
  trading day - E7) is built **within** the window.
* The model reads the reservoir state at the **absolute** calendar position of
  each signal day: states were computed once over the whole sample (P3) and are
  simply indexed here, never recomputed.
* The training window is intersected with the states' usable mask, so the
  z-score warm-up and the 250-day washout are automatically excluded. For
  validation and test segments this is a no-op (washout is years past), and an
  assertion confirms it rather than assumes it.
* The ledger panel has its pre-inception NaNs replaced by zero, with the same
  safety argument proven at P4: no strategy can hold weight on an asset before
  its first available day, so a zero return on a zero holding changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rrc.backtest.ledger import weekly_execution_schedule
from rrc.eval.slicing import (
    Fold,
    assert_no_holdout,
    segment_dates,
    train_dates,
    truncate_before_holdout,
)
from rrc.models.experts import build_logit_mask
from rrc.reservoir.cache import CachedStates

CASH_COLUMN = "CASH"


@dataclass(frozen=True)
class UniverseArtifacts:
    """The per-universe data a training run consumes, loaded once.

    Everything is truncated before the holdout at load time, so 2025+ rows are
    not merely unevaluated - they are absent from memory.
    """

    universe: str
    returns_simple: pd.DataFrame     # risky + CASH, the accounting panel
    ledger_panel: pd.DataFrame       # same, with pre-inception NaN -> 0.0
    availability: pd.DataFrame       # risky assets only, 0/1
    risk_free: pd.Series
    signals: pd.DataFrame            # the market-signal matrix (lagged, z-scored)
    dates: pd.DatetimeIndex
    columns: list[str]
    cash_index: int


@dataclass(frozen=True)
class SegmentPack:
    """One window of days, ready for a forward pass and a ledger run.

    Attributes
    ----------
    name
        ``"train"`` / ``"val"`` / ``"test"`` - for logging.
    dates
        The window's trading days.
    states
        ``(J, N_r)`` float64 reservoir states at the signal days.
    logit_mask
        ``(J, N)`` additive availability mask at the signal days (cash never
        masked).
    execution_positions
        ``(J,)`` window-relative day positions the weights execute at.
    returns
        ``(D, N)`` float64 simple-return panel over the window (cash included),
        for the ledger.
    risk_free
        ``(D,)`` float64 daily risk-free returns over the window.
    cash_index
        Column of the cash asset.
    """

    name: str
    dates: pd.DatetimeIndex
    states: torch.Tensor
    logit_mask: torch.Tensor
    execution_positions: np.ndarray
    returns: torch.Tensor
    risk_free: torch.Tensor
    cash_index: int
    # The market signals at the signal days, for the rung-3 arms. Always
    # present and finite; models that do not read signals simply ignore them.
    signals: torch.Tensor | None = None
    # Externally fixed gate memberships at the signal days - the rung-4
    # mechanism (R6): a statistical gate supplies pi, the experts retrain
    # under it. None everywhere else.
    fixed_memberships: torch.Tensor | None = None
    # Trailing raw-input windows at the signal days, ``(J, lookback, n_inputs)``.
    # Only the LSTM comparison arm needs these: it encodes the driving series
    # itself instead of reading precomputed reservoir states. Attached by
    # ``rrc.models.recurrent.attach_input_windows``; None for every other arm.
    input_windows: torch.Tensor | None = None

    @property
    def n_signals(self) -> int:
        """Number of weekly decisions in the window."""
        return self.states.shape[0]

    @property
    def n_days(self) -> int:
        """Number of trading days in the window."""
        return self.returns.shape[0]


def load_universe_artifacts(processed_dir: Path, universe: str) -> UniverseArtifacts:
    """Load and validate one universe's artifacts for training.

    Mirrors the baseline runner's loading rules exactly - same truncation, same
    inception-NaN reasoning - so models and baselines consume literally the
    same numbers.
    """
    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(processed_dir / f"{universe}_{name}.parquet")

    returns_simple = truncate_before_holdout(read("returns_simple"))
    availability = truncate_before_holdout(read("availability"))
    risk_free = truncate_before_holdout(read("risk_free"))["r_f"]
    signals = truncate_before_holdout(read("signals"))
    assert_no_holdout(returns_simple.index, context=f"{universe} artifacts")

    risky_columns = [c for c in returns_simple.columns if c != CASH_COLUMN]

    # A return must exist wherever an asset was available today AND yesterday;
    # the only legitimate NaNs are pre-inception and the inception day itself
    # (a price with no prior close). Proven before the zero-fill, as at P4.
    available = availability[risky_columns].to_numpy() == 1
    available_yesterday = np.vstack([available[:1] & False, available[:-1]])
    finite = np.isfinite(returns_simple[risky_columns].to_numpy())
    holes = available & available_yesterday & ~finite
    if holes.any():
        raise ValueError(
            f"{universe}: {int(holes.sum())} return hole(s) inside an asset's "
            "active life; the data layer's guarantee is broken."
        )

    return UniverseArtifacts(
        universe=universe,
        returns_simple=returns_simple,
        ledger_panel=returns_simple.fillna(0.0),
        availability=availability[risky_columns],
        risk_free=risk_free,
        signals=signals,
        dates=returns_simple.index,
        columns=list(returns_simple.columns),
        cash_index=list(returns_simple.columns).index(CASH_COLUMN),
    )


def assert_state_alignment(
    artifacts: UniverseArtifacts, cached_states: CachedStates
) -> None:
    """Prove the artifact calendar is a prefix of the states calendar.

    Position ``i`` in the artifacts must be the same trading day as position
    ``i`` in the cached states, or every state lookup in this module is silently
    off by some shift. Compared at nanosecond resolution because parquet and the
    state cache store timestamps at different units.

    Raises
    ------
    ValueError
        If the calendars disagree - which happens exactly when the data was
        rebuilt without rebuilding the state cache.
    """
    n_artifact_days = len(artifacts.dates)
    if len(cached_states.dates) < n_artifact_days:
        raise ValueError(
            f"Cached states cover {len(cached_states.dates)} days but the "
            f"artifacts have {n_artifact_days} - rebuild the state cache "
            "(scripts/run_reservoir.py) after rebuilding the data."
        )
    states_prefix = cached_states.dates[:n_artifact_days].as_unit("ns")
    artifact_days = artifacts.dates.as_unit("ns")
    if not states_prefix.equals(artifact_days):
        raise ValueError(
            "The cached states' calendar does not match the artifact calendar "
            "- rebuild the state cache after any data rebuild."
        )


def build_segment_pack(
    name: str,
    window_days: pd.DatetimeIndex,
    artifacts: UniverseArtifacts,
    cached_states: CachedStates,
) -> SegmentPack:
    """Assemble one window into a pack.

    Parameters
    ----------
    name
        Label for logging (``"train"`` / ``"val"`` / ``"test"``).
    window_days
        The window's trading days (already usable-intersected for training).
    artifacts
        The universe's loaded data.
    cached_states
        The frozen reservoir's cached state path.

    Returns
    -------
    SegmentPack
    """
    if len(window_days) < 30:
        raise ValueError(f"{name} window has only {len(window_days)} days.")

    # Weekly schedule inside the window; positions are window-relative.
    signal_relative, execution_relative = weekly_execution_schedule(window_days)
    if len(signal_relative) == 0:
        raise ValueError(f"{name} window yields no weekly signals.")

    # Absolute calendar positions of the signal days, for state/mask lookup.
    # States are indexed by positions computed from the artifact calendar, so
    # the two calendars must agree positionally. The states file covers the
    # FULL sample (it was built before the holdout truncation), while the
    # artifacts stop at 2024-12-31 - so the requirement is that the artifact
    # calendar is exactly the states calendar's prefix.
    assert_state_alignment(artifacts, cached_states)
    window_start_position = int(artifacts.dates.get_loc(window_days[0]))
    signal_absolute = signal_relative + window_start_position

    # States at the signal days - must be inside the usable region.
    if not cached_states.usable[signal_absolute].all():
        n_bad = int((~cached_states.usable[signal_absolute]).sum())
        raise ValueError(
            f"{name}: {n_bad} signal day(s) fall outside the usable state "
            "region (warm-up/washout); the window was not usable-intersected."
        )
    states = torch.as_tensor(
        cached_states.states[signal_absolute], dtype=torch.float64
    )

    availability_rows = torch.as_tensor(
        artifacts.availability.to_numpy()[signal_absolute], dtype=torch.int64
    )
    logit_mask = build_logit_mask(availability_rows)

    signal_values = torch.as_tensor(
        artifacts.signals.to_numpy()[signal_absolute].copy(), dtype=torch.float64
    )
    if not torch.isfinite(signal_values).all():
        raise ValueError(
            f"{name}: non-finite market signals at some signal days - the "
            "window starts before the signal warm-up, which the fold schedule "
            "should make impossible."
        )

    returns = torch.as_tensor(
        artifacts.ledger_panel.loc[window_days].to_numpy().copy(), dtype=torch.float64
    )
    risk_free = torch.as_tensor(
        artifacts.risk_free.loc[window_days].to_numpy().copy(), dtype=torch.float64
    )
    if not torch.isfinite(risk_free).all():
        raise ValueError(f"{name}: non-finite risk-free values inside the window.")

    return SegmentPack(
        name=name,
        dates=window_days,
        states=states,
        logit_mask=logit_mask,
        execution_positions=execution_relative,
        returns=returns,
        risk_free=risk_free,
        cash_index=artifacts.cash_index,
        signals=signal_values,
    )


def build_fold_packs(
    artifacts: UniverseArtifacts,
    cached_states: CachedStates,
    fold: Fold,
) -> dict[str, SegmentPack]:
    """Build the train / validation / test packs for one fold.

    * **train**: universe start to the December before the validation year,
      minus the 5-day embargo, intersected with the usable state region (which
      trims the z-score warm-up and washout off the front).
    * **val**: the validation calendar year (used for early stopping - it is
      the selection segment, and is recorded as such in the registry).
    * **test**: the test calendar year (never influences training).
    """
    usable_dates = artifacts.dates[cached_states.usable[: len(artifacts.dates)]]

    embargoed_train = train_dates(artifacts.dates, fold)
    train_window = embargoed_train.intersection(usable_dates)
    validation_window = segment_dates(artifacts.dates, fold.val_year)
    test_window = segment_dates(artifacts.dates, fold.test_year)

    return {
        "train": build_segment_pack("train", train_window, artifacts, cached_states),
        "val": build_segment_pack("val", validation_window, artifacts, cached_states),
        "test": build_segment_pack("test", test_window, artifacts, cached_states),
    }
