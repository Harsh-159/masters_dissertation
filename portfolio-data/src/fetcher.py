"""downloads each ticker's full history once and caches it to parquet.

the cache holds everything yahoo has, not just the study window. that way
changing the dates in config.yaml never triggers a download, and after the first
run the whole pipeline works offline.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import Config

log = logging.getLogger(__name__)


class EmptyDownload(RuntimeError):
    """yahoo returned nothing usable. worth retrying, it is often a rate limit."""


class DataFetcher:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.raw_dir = cfg.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.raw_dir / "_cache_manifest.json"

    def fetch_universe(
        self, tickers: tuple[str, ...] | list[str], force_refresh: bool = False
    ) -> dict[str, pd.Series]:
        """price series for every ticker we could get. missing ones are left out."""
        prices = {}
        for ticker in tickers:
            series = self.fetch_one(ticker, force_refresh)
            if series is not None and not series.empty:
                prices[ticker] = series
        missing = [t for t in tickers if t not in prices]
        log.info("got %d of %d tickers%s", len(prices), len(tickers),
                 f", missing {', '.join(missing)}" if missing else "")
        return prices

    def fetch_one(self, ticker: str, force_refresh: bool = False) -> pd.Series | None:
        """one ticker's full history, from cache if we have it. never raises."""
        cache_path = self.raw_dir / f"{ticker}.parquet"

        if not force_refresh:
            if cache_path.is_file():
                series = pd.read_parquet(cache_path).iloc[:, 0]
                series.name = ticker
                log.info("%s from cache, %d rows", ticker, len(series))
                return series
            # tickers like UTX and KFT no longer exist, so once we know a symbol
            # is dead we skip it instead of burning three retries every run
            if self._manifest().get(ticker, {}).get("failed"):
                log.info("%s skipped, known to be dead. set force_refresh to retry", ticker)
                return None

        try:
            series = self._download(ticker)
        except Exception as exc:
            log.error("%s failed: %s", ticker, exc)
            self._record_failure(ticker, str(exc))
            return None

        frame = series.to_frame(name=ticker)
        frame.index.name = "date"
        frame.to_parquet(cache_path)
        log.info("%s downloaded, %d rows (%s to %s)", ticker, len(series),
                 series.index.min().date(), series.index.max().date())
        return series

    def _download(self, ticker: str) -> pd.Series:
        delay = self.cfg.retry_backoff_seconds
        for attempt in range(1, self.cfg.retry_attempts + 1):
            time.sleep(self.cfg.polite_delay_seconds)
            try:
                return self._download_once(ticker)
            except EmptyDownload:
                if attempt == self.cfg.retry_attempts:
                    raise
                time.sleep(delay)
                delay *= 2
        raise EmptyDownload(ticker)

    def _download_once(self, ticker: str) -> pd.Series:
        raw = yf.download(ticker, period="max", auto_adjust=False,
                          progress=False, actions=False, threads=False)
        series = _column(raw, ticker, "Adj Close")

        # a few symbols come back with no Adj Close, so let yahoo do the split and
        # dividend adjustment for us instead
        if series is None and self.cfg.auto_adjust_fallback:
            raw = yf.download(ticker, period="max", auto_adjust=True,
                              progress=False, actions=False, threads=False)
            series = _column(raw, ticker, "Close")

        if series is None or series.empty:
            raise EmptyDownload(f"no price data for {ticker}")
        return _clean_series(series, ticker)

    def _manifest(self) -> dict:
        if not self.manifest_path.is_file():
            return {}
        try:
            return json.loads(self.manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("cache manifest unreadable, carrying on as if empty")
            return {}

    def _record_failure(self, ticker: str, error: str) -> None:
        manifest = self._manifest()
        manifest[ticker] = {
            "failed": True,
            "error": error,
            "when": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _column(df: pd.DataFrame | None, ticker: str, field: str) -> pd.Series | None:
    # yfinance has changed its column shape more than once, so handle both the
    # flat columns and the (field, ticker) pairs it returns now
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if field not in set(df.columns.get_level_values(0)):
            return None
        sub = df[field]
        if isinstance(sub, pd.DataFrame):
            return sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
        return sub
    return df[field] if field in df.columns else None


def _clean_series(series: pd.Series, ticker: str) -> pd.Series:
    """real observations only, sorted, no duplicates, no timezone.

    filling gaps and lining assets up is the processor's job. what we cache here
    is exactly what yahoo gave us.
    """
    out = series.copy()
    index = pd.to_datetime(out.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    # name it and drop any freq here so a fresh download and a cache read give
    # back the same thing. yahoo calls the index "Date", parquet stores it as
    # "date", and parquet does not keep a freq. a freq would be wrong anyway,
    # trading days are not evenly spaced
    out.index.name = "date"
    out.index.freq = None
    out.name = ticker
    return out
