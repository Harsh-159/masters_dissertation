"""The C2 shuffle-future test: the project's strongest leak detector.

The logic, in plain words
-------------------------
Take everything after a cut date and shuffle the days into a random order. Two
things are now true about the shuffled world:

* **Genuine skill disappears.** Whatever temporal structure a strategy was
  really exploiting - momentum, regimes, volatility clustering - has been
  destroyed, so an honest strategy's performance after the cut collapses to
  noise around zero.
* **Structural leaks survive.** If the pipeline somehow feeds tomorrow's
  information into today's decision (a same-bar join, an off-by-one, a state
  built with future rows), then rebuilding the pipeline on the shuffled data
  re-leaks the *shuffled* future - and the strategy keeps "working" on data
  where working is impossible.

So the verdict is one number: the mean performance across several independent
shuffles. Near zero - clean. Systematically positive - a leak, somewhere, with
certainty. This is why the review calls C2 thesis-critical: it is the one test
whose failure cannot be argued with.

The test has real statistical power both ways, and the unit tests demonstrate
it: a deliberately leaky pipeline is caught (shuffled performance stays high),
and an honest pipeline with real skill collapses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def permute_rows_after(
    frame: pd.DataFrame,
    cut_position: int,
    seed: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Shuffle a frame's rows strictly after a positional cut.

    Whole rows move together (the same day-reordering is applied to every
    selected column), so each day's cross-section stays intact while the time
    ordering after the cut is destroyed. Rows at or before the cut are
    untouched - bit-identical, which the caller can and should verify.

    Parameters
    ----------
    frame
        The panel to permute (e.g. daily returns, dates on the index).
    cut_position
        Positional index of the last untouched row.
    seed
        Seed for the permutation, so every shuffle is reproducible.
    columns
        If given, only these columns are permuted (e.g. the risky assets,
        leaving the cash/risk-free column in real time order). Others keep
        their original values.

    Returns
    -------
    pandas.DataFrame
        A copy with the post-cut rows of the selected columns shuffled.
    """
    if not 0 <= cut_position < len(frame):
        raise ValueError(
            f"cut_position {cut_position} outside the frame's {len(frame)} rows."
        )

    generator = np.random.default_rng(seed)
    n_after = len(frame) - (cut_position + 1)
    if n_after < 2:
        raise ValueError("Fewer than two rows after the cut; nothing to shuffle.")

    shuffled_order = generator.permutation(n_after)

    permuted = frame.copy()
    target_columns = list(frame.columns) if columns is None else columns
    tail = permuted.iloc[cut_position + 1 :][target_columns].to_numpy()
    permuted.iloc[
        cut_position + 1 :, [permuted.columns.get_loc(c) for c in target_columns]
    ] = tail[shuffled_order]
    return permuted


@dataclass(frozen=True)
class CollapseVerdict:
    """The outcome of a shuffle-future run.

    Attributes
    ----------
    actual_sharpe
        The strategy's annualised excess Sharpe on the real (unshuffled) data.
    permuted_sharpes
        One Sharpe per independent shuffle.
    mean_permuted
        Their mean - the number the verdict rests on.
    noise_band
        The absolute band within which the mean counts as "collapsed".
    collapsed
        True if the mean permuted Sharpe lies inside the band.
    """

    actual_sharpe: float
    permuted_sharpes: tuple[float, ...]
    mean_permuted: float
    noise_band: float
    collapsed: bool


def collapse_verdict(
    actual_sharpe: float,
    permuted_sharpes: list[float],
    noise_band: float,
) -> CollapseVerdict:
    """Judge whether shuffled-future performance collapsed to noise.

    The band is deliberately generous: a single year's Sharpe estimate has a
    sampling standard deviation of roughly 1.0, so even a handful of shuffles
    averages to something meaningfully inside +/-0.6 when the truth is zero. A
    leak does not produce a mean of 0.4 - it produces a mean comparable to the
    strategy's apparent skill, far outside any reasonable band.

    Parameters
    ----------
    actual_sharpe
        Annualised excess Sharpe on the real data (reported for contrast).
    permuted_sharpes
        The per-shuffle Sharpes.
    noise_band
        The |mean| threshold for declaring collapse.

    Returns
    -------
    CollapseVerdict
    """
    if not permuted_sharpes:
        raise ValueError("At least one permuted Sharpe is required.")

    mean_permuted = float(np.mean(permuted_sharpes))
    return CollapseVerdict(
        actual_sharpe=float(actual_sharpe),
        permuted_sharpes=tuple(float(s) for s in permuted_sharpes),
        mean_permuted=mean_permuted,
        noise_band=float(noise_band),
        collapsed=abs(mean_permuted) <= noise_band,
    )
