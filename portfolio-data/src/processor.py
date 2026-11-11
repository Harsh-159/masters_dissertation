"""turns per-ticker histories into a price panel and a log return panel."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Panels:
    """cleaned data for one universe.

    raw_aligned is the panel before any gap filling, so we can always tell a real
    observation apart from one we carried forward.
    """

    name: str
    prices: pd.DataFrame
    returns: pd.DataFrame
    raw_aligned: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)


class DataProcessor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def process(self, name: str, prices: dict[str, pd.Series], tickers) -> Panels:
        columns = list(tickers)

        sliced = {}
        for ticker in columns:
            series = prices.get(ticker)
            if series is None or series.empty:
                continue
            window = series.loc[pd.Timestamp(self.cfg.start):pd.Timestamp(self.cfg.end)]
            if not window.empty:
                sliced[ticker] = window

        # the calendar is the days something actually traded. generating business
        # days instead would invent good friday and thanksgiving, and then we would
        # have to fill rows that never existed
        index = pd.DatetimeIndex([], name="date")
        for series in sliced.values():
            index = index.union(series.index)
        index = index.sort_values()
        index.name = "date"

        raw_aligned = pd.DataFrame(index=index, columns=columns, dtype="float64")
        for ticker in columns:
            if ticker in sliced:
                raw_aligned[ticker] = sliced[ticker].reindex(index)

        # forward fill only. filling backwards would copy a 2015 price into 2007,
        # which hands the model information it could not have had. ffill cannot
        # touch leading gaps anyway, so assets that started late stay empty until
        # they existed
        filled = raw_aligned.ffill(limit=self.cfg.ffill_limit)

        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.log(filled).diff()

        panels = Panels(name=name, prices=filled, returns=returns, raw_aligned=raw_aligned)
        self._log(panels)
        return panels

    def _log(self, panels: Panels) -> None:
        prices = panels.prices
        if prices.empty:
            log.warning("%s: empty panel", panels.name)
            return
        log.info("%s: %d rows x %d assets (%s to %s)", panels.name, *prices.shape,
                 prices.index.min().date(), prices.index.max().date())
        for ticker in panels.tickers:
            real = panels.raw_aligned[ticker].dropna()
            if real.empty:
                log.warning("  %-5s no data in the window", ticker)
                continue
            log.info("  %-5s %s to %s, %d obs", ticker, real.index.min().date(),
                     real.index.max().date(), len(real))
