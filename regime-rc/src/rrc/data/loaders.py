"""Price loading, caching, and construction of the canonical panels.

Caching strategy
----------------
Each ticker's **entire available history** is pulled once and cached to
``data/raw/<TICKER>.parquet``. The requested study window is then sliced from
the cache in memory. Two consequences, both deliberate:

* changing the study window never triggers a re-download, so the walk-forward
  folds and the holdout can be re-cut freely;
* after the first pull the whole pipeline runs **offline**, which is what lets
  the test suite run against cached fixtures rather than live pulls (yfinance
  rate-limits, and its schema drifts).

Both the **raw** (unadjusted) close and the **adjusted** close are cached. The
adjusted close is the price basis for everything - it is corrected for splits
and dividends, so a 2-for-1 split does not appear as a -50% return - but the
raw close is kept so that any future question about adjustment can be answered
from the cache rather than by re-pulling.

The three panels this module produces
-------------------------------------
1. **Adjusted price panel** - the canonical artifact; everything else derives
   from it.
2. **Log-return panel** ``ln(P_t / P_{t-1})`` - additive over time and better
   behaved as reservoir input.
3. **Simple-return panel** ``P_t / P_{t-1} - 1`` **with the cash column** - used
   for portfolio accounting, because a portfolio's return is the weighted sum of
   its assets' *simple* returns. Mixing the two conventions is a subtle and very
   real error, so they are produced separately and named unambiguously.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rrc.data.calendar import normalise_index

logger = logging.getLogger(__name__)

# Column names inside each cached per-ticker parquet file.
RAW_CLOSE = "close_raw"
ADJUSTED_CLOSE = "close_adj"


@dataclass(frozen=True)
class LoadOutcome:
    """Record of how one ticker's history was obtained.

    Attributes
    ----------
    ticker
        The symbol requested.
    source
        ``yfinance``, ``stooq``, or ``cache``.
    n_observations
        Rows in the cached full history.
    first_date, last_date
        Range of the cached full history, as ISO date strings.
    error
        Failure message, or ``None`` on success.
    """

    ticker: str
    source: str
    n_observations: int
    first_date: str | None
    last_date: str | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when usable price history was obtained."""
        return self.error is None and self.n_observations > 0


class PriceLoader:
    """Fetch, cache, and serve per-ticker price histories."""

    def __init__(self, raw_dir: Path, polite_delay_seconds: float = 0.4, max_attempts: int = 3):
        """Initialise the loader.

        Parameters
        ----------
        raw_dir
            Directory for the per-ticker parquet cache.
        polite_delay_seconds
            Pause between network calls, to stay well inside rate limits.
        max_attempts
            Attempts per ticker before falling back to Stooq.
        """
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.polite_delay_seconds = polite_delay_seconds
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_many(
        self, tickers: list[str], force_refresh: bool = False
    ) -> tuple[dict[str, pd.DataFrame], list[LoadOutcome]]:
        """Load full histories for many tickers.

        Parameters
        ----------
        tickers
            Symbols to load.
        force_refresh
            Ignore the cache and re-download.

        Returns
        -------
        histories, outcomes
            ``histories`` maps ticker to a frame with ``close_raw`` and
            ``close_adj``; ``outcomes`` has one record per requested ticker.
        """
        histories: dict[str, pd.DataFrame] = {}
        outcomes: list[LoadOutcome] = []

        for ticker in tickers:
            frame, outcome = self.load_one(ticker, force_refresh=force_refresh)
            outcomes.append(outcome)
            if frame is not None and not frame.empty:
                histories[ticker] = frame

        n_ok = sum(outcome.succeeded for outcome in outcomes)
        logger.info("Loaded %d/%d tickers.", n_ok, len(tickers))
        return histories, outcomes

    def load_one(
        self, ticker: str, force_refresh: bool = False
    ) -> tuple[pd.DataFrame | None, LoadOutcome]:
        """Load one ticker's full history, from cache when available.

        Never raises on a failed ticker: the failure is logged and returned in
        the outcome, so one bad symbol cannot abort a data build.
        """
        cache_path = self.raw_dir / f"{ticker}.parquet"

        if cache_path.is_file() and not force_refresh:
            frame = pd.read_parquet(cache_path)
            return frame, LoadOutcome(
                ticker=ticker,
                source="cache",
                n_observations=len(frame),
                first_date=_date_string(frame.index.min()),
                last_date=_date_string(frame.index.max()),
            )

        frame, source, error = self._download(ticker)
        if frame is None:
            logger.error("Could not obtain %s: %s", ticker, error)
            return None, LoadOutcome(ticker, source, 0, None, None, error)

        frame.to_parquet(cache_path)
        logger.info(
            "Downloaded %s from %s: %d rows (%s to %s).",
            ticker, source, len(frame),
            _date_string(frame.index.min()), _date_string(frame.index.max()),
        )
        return frame, LoadOutcome(
            ticker=ticker,
            source=source,
            n_observations=len(frame),
            first_date=_date_string(frame.index.min()),
            last_date=_date_string(frame.index.max()),
        )

    # ------------------------------------------------------------------ #
    # Downloading
    # ------------------------------------------------------------------ #
    def _download(self, ticker: str) -> tuple[pd.DataFrame | None, str, str | None]:
        """Try yfinance with retries, then fall back to Stooq."""
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                time.sleep(self.polite_delay_seconds)
                frame = self._download_from_yfinance(ticker)
                if frame is not None and not frame.empty:
                    return frame, "yfinance", None
                last_error = "yfinance returned no rows"
            except Exception as exc:  # noqa: BLE001 - report, retry, then fall back
                last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s: yfinance attempt %d/%d failed (%s).",
                           ticker, attempt, self.max_attempts, last_error)

        try:
            frame = self._download_from_stooq(ticker)
            if frame is not None and not frame.empty:
                logger.warning("%s: served from the Stooq fallback.", ticker)
                return frame, "stooq", None
        except Exception as exc:  # noqa: BLE001
            last_error = f"stooq fallback failed: {type(exc).__name__}: {exc}"

        return None, "none", last_error

    def _download_from_yfinance(self, ticker: str) -> pd.DataFrame | None:
        """Pull full history from yfinance, keeping raw and adjusted closes.

        ``auto_adjust=False`` is requested specifically because it returns *both*
        ``Close`` (raw) and ``Adj Close`` (split- and dividend-adjusted) in one
        call, which is exactly the pair we cache.
        """
        import yfinance as yf

        response = yf.download(
            ticker, period="max", auto_adjust=False,
            progress=False, actions=False, threads=False,
        )
        if response is None or response.empty:
            return None

        adjusted = _extract_column(response, ticker, ("Adj Close",))
        raw = _extract_column(response, ticker, ("Close",))

        if adjusted is None:
            # Rare, but it happens. Fall back to the auto-adjusted close, which
            # is already corporate-action adjusted.
            logger.warning("%s: no 'Adj Close'; using auto_adjust=True Close.", ticker)
            response = yf.download(
                ticker, period="max", auto_adjust=True,
                progress=False, actions=False, threads=False,
            )
            adjusted = _extract_column(response, ticker, ("Close",))
            raw = adjusted
        if adjusted is None:
            return None

        return _assemble_history(raw_close=raw, adjusted_close=adjusted)

    def _download_from_stooq(self, ticker: str) -> pd.DataFrame | None:
        """Pull from Stooq as a fallback source.

        Stooq serves split-adjusted closes but does not publish a separate
        unadjusted series, so both cached columns carry the same values. That is
        recorded honestly rather than faked: the outcome's ``source`` field says
        ``stooq``, and the data report shows it.
        """
        from pandas_datareader import data as pdr

        response = pdr.DataReader(ticker, "stooq")
        if response is None or response.empty or "Close" not in response.columns:
            return None

        close = response["Close"].sort_index()
        return _assemble_history(raw_close=close, adjusted_close=close)


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def adjusted_prices(histories: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Extract the adjusted-close series from cached per-ticker frames."""
    return {
        ticker: normalise_index(frame[ADJUSTED_CLOSE].dropna())
        for ticker, frame in histories.items()
    }


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ``ln(P_t / P_{t-1})``, computed from the price panel.

    The first observation of each asset, and any cell whose predecessor is
    missing, yields NaN - never 0. A fabricated 0 would assert "this asset did
    not move today", which is a claim the data does not support.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(prices).diff()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns ``P_t / P_{t-1} - 1``, for portfolio accounting.

    Portfolio return is the weighted sum of *simple* asset returns, so the
    accounting panel must use these and not log returns.
    """
    return prices.pct_change(fill_method=None)


def append_cash_column(
    returns: pd.DataFrame,
    risk_free_daily: pd.Series,
    cash_column: str = "CASH",
) -> pd.DataFrame:
    """Append the cash asset to an accounting return panel.

    Cash earns the daily risk-free rate and has zero variance. Because the loss
    is computed on returns *in excess of the same* ``r_f``, an all-cash portfolio
    scores exactly zero excess Sharpe: cash is a refuge, not a free lunch. Without
    this alignment the optimiser discovers that holding cash forever has a huge
    raw Sharpe and never learns anything else.

    Parameters
    ----------
    returns
        Simple-return panel for the risky assets.
    risk_free_daily
        Daily simple risk-free return, indexed by trading day.
    cash_column
        Name for the appended column.

    Returns
    -------
    pandas.DataFrame
        The panel with the cash column appended as the last column.
    """
    if cash_column in returns.columns:
        raise ValueError(f"Column {cash_column!r} already exists in the panel.")

    with_cash = returns.copy()
    with_cash[cash_column] = risk_free_daily.reindex(returns.index).astype("float64")
    return with_cash


def rolling_zscore(
    frame: pd.DataFrame, window: int, min_periods: int
) -> pd.DataFrame:
    """Standardise each column with a trailing window ending at ``t``.

    Used to prepare the reservoir input: raw log returns are of order 0.01, which
    would drive ``tanh`` in its near-linear region and waste the reservoir's
    nonlinearity.

    The window **ends at t and includes t**, which is causal: the return for day
    ``t`` is known at the close of day ``t``. No shift is applied here, unlike
    the published macro signals in :mod:`rrc.data.signals`, which are not known
    until after their reference date.
    """
    rolling = frame.rolling(window=window, min_periods=min_periods)
    return (frame - rolling.mean()) / rolling.std(ddof=0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_column(
    frame: pd.DataFrame | None, ticker: str, candidates: tuple[str, ...]
) -> pd.Series | None:
    """Pull a named column out of a yfinance frame of any known shape.

    yfinance returns flat columns in some versions and a ``(field, ticker)``
    MultiIndex in others; both are handled so a library upgrade cannot silently
    change what gets cached.
    """
    if frame is None or frame.empty:
        return None

    if isinstance(frame.columns, pd.MultiIndex):
        available_fields = set(frame.columns.get_level_values(0))
        for field in candidates:
            if field in available_fields:
                section = frame[field]
                if isinstance(section, pd.DataFrame):
                    return section[ticker] if ticker in section.columns else section.iloc[:, 0]
                return section
        return None

    for field in candidates:
        if field in frame.columns:
            return frame[field]
    return None


def _assemble_history(raw_close: pd.Series | None, adjusted_close: pd.Series) -> pd.DataFrame:
    """Build the two-column cached frame from raw and adjusted close series."""
    adjusted = normalise_index(adjusted_close.dropna())
    raw = normalise_index(raw_close.dropna()) if raw_close is not None else adjusted

    frame = pd.DataFrame(
        {RAW_CLOSE: raw.reindex(adjusted.index), ADJUSTED_CLOSE: adjusted}
    )
    frame.index.name = "date"
    return frame


def _date_string(value) -> str | None:
    """Format a timestamp as ``YYYY-MM-DD``, or ``None`` when absent."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")
