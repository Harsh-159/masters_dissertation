"""checks the cleaned panels.

two kinds. the hard ones stop the run, because if any of them fail the data is
wrong and everything downstream is built on sand. the soft ones only warn, since
a huge return in october 2008 is real and deleting it would be worse than keeping
it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config, Universe
from .processor import Panels
from .universe import membership_spans

log = logging.getLogger(__name__)


class DataError(AssertionError):
    """a hard check failed, so the panel is not safe to use."""


def validate(cfg: Config, panels: Panels, universe: Universe) -> None:
    # the dates come first and on their own. every other check looks rows up by
    # date, so on a duplicated index they fail in confusing ways instead of
    # telling you the index is the problem
    broken_index = _index_is_sane(panels)
    if broken_index:
        raise DataError(f"{panels.name}: " + "; ".join(broken_index))

    errors: list[str] = []
    errors += _no_inf(panels)
    errors += _prices_positive(panels)
    errors += _no_invented_history(panels)
    errors += _gaps_were_filled(cfg, panels)
    errors += _returns_use_two_prices(panels)

    if errors:
        raise DataError(f"{panels.name} failed {len(errors)} check(s): " + "; ".join(errors))

    if cfg.outlier_sigma is None and cfg.stale_run_length is None:
        log.info("%s: no validate block in the config, skipping the soft warnings", panels.name)

    _warn_outliers(cfg, panels)
    _warn_stale_prices(cfg, panels)
    _warn_short_history(cfg, panels)
    _warn_membership_gaps(panels, universe)
    log.info("%s passed all checks", panels.name)


# hard checks


def _index_is_sane(p: Panels) -> list[str]:
    out = []
    for label, frame in (("prices", p.prices), ("returns", p.returns)):
        if frame.index.has_duplicates:
            out.append(f"{label} has duplicate dates")
        if not frame.index.is_monotonic_increasing:
            out.append(f"{label} dates are not in order")
    return out


def _no_inf(p: Panels) -> list[str]:
    out = []
    for label, frame in (("prices", p.prices), ("returns", p.returns)):
        count = int(np.isinf(frame.to_numpy(dtype="float64", na_value=np.nan)).sum())
        if count:
            out.append(f"{label} has {count} inf value(s)")
    return out


def _prices_positive(p: Panels) -> list[str]:
    # log of zero is -inf and log of a negative is undefined, so a bad price here
    # turns into a nonsense return rather than an error
    count = int((p.prices <= 0).to_numpy().sum())
    return [f"prices has {count} value(s) at or below zero"] if count else []


def _no_invented_history(p: Panels) -> list[str]:
    """nothing may exist before an asset's first real price. catches a backfill."""
    out = []
    for ticker in p.tickers:
        first = p.raw_aligned[ticker].first_valid_index()
        if first is None:
            continue
        before = p.prices.loc[:first, ticker].iloc[:-1]
        if before.notna().any():
            out.append(f"{ticker} has {int(before.notna().sum())} price(s) before it existed")
    return out


def _gaps_were_filled(cfg: Config, p: Panels) -> list[str]:
    """a short gap surviving inside an asset's life means ffill did not run."""
    out = []
    for ticker in p.tickers:
        real = p.raw_aligned[ticker]
        first, last = real.first_valid_index(), real.last_valid_index()
        if first is None:
            continue
        gap = _longest_run(p.prices.loc[first:last, ticker].isna().to_numpy())
        if 0 < gap <= cfg.ffill_limit:
            out.append(f"{ticker} still has a {gap} day gap, under the limit of {cfg.ffill_limit}")
    return out


def _returns_use_two_prices(p: Panels) -> list[str]:
    """the first return per asset must be nan, and the rest need a price either side."""
    out = []
    for ticker in p.tickers:
        prices, returns = p.prices[ticker], p.returns[ticker]
        first = prices.first_valid_index()
        if first is None:
            continue
        if not np.isnan(returns.loc[first]):
            out.append(f"{ticker} has a return on its first day, which cannot be real")
        present = prices.notna()
        unbacked = returns.notna() & ~(present & present.shift(1))
        if unbacked.any():
            out.append(f"{ticker} has {int(unbacked.sum())} return(s) with no price behind them")
    return out


# soft checks


def _warn_outliers(cfg: Config, p: Panels) -> None:
    if cfg.outlier_sigma is None or cfg.extreme_return_abs is None:
        return
    total = 0
    for ticker in p.tickers:
        returns = p.returns[ticker].dropna()
        if returns.empty:
            continue
        spread = returns.std(ddof=0)
        z = (returns - returns.mean()) / spread if spread > 0 else pd.Series(0.0, index=returns.index)
        total += int(((returns.abs() > cfg.extreme_return_abs) | (z.abs() > cfg.outlier_sigma)).sum())
    if total:
        log.warning("%s: %d extreme return(s), worth a look but left in place", p.name, total)


def _warn_stale_prices(cfg: Config, p: Panels) -> None:
    if cfg.stale_run_length is None:
        return
    # checked against raw_aligned, not the filled panel, otherwise our own gap
    # filling would show up as a frozen price feed
    flagged = []
    for ticker in p.tickers:
        values = p.raw_aligned[ticker].dropna().to_numpy()
        run = 1
        for i in range(1, len(values)):
            run = run + 1 if values[i] == values[i - 1] else 1
            if run >= cfg.stale_run_length:
                flagged.append(ticker)
                break
    if flagged:
        log.warning("%s: price never moved for %d days or more in %s", p.name,
                    cfg.stale_run_length, ", ".join(flagged))


def _warn_short_history(cfg: Config, p: Panels) -> None:
    for ticker in p.tickers:
        count = int(p.raw_aligned[ticker].notna().sum())
        if count < cfg.min_history_days:
            log.warning("%s: %s only has %d days, under the %d we want",
                        p.name, ticker, count, cfg.min_history_days)


def _warn_membership_gaps(p: Panels, universe: Universe) -> None:
    """where the dow data does not cover the years a ticker was actually a member.

    this is the survivorship problem in the open. old gm went bankrupt in 2009 and
    yahoo only has the company that relisted in 2010, so none of its dow years are
    covered. free data cannot fix that, but it can admit it.
    """
    for ticker, (starts, ends) in membership_spans(universe).items():
        if ticker not in p.raw_aligned.columns:
            continue
        real = p.raw_aligned[ticker].dropna()
        covered = real.loc[starts:ends] if ends is not None else real.loc[starts:]
        if covered.empty:
            log.warning("%s: %s has no data for its %s to %s membership", p.name, ticker,
                        starts.date(), ends.date() if ends is not None else "now")


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best
