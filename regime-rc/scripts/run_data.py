"""Build the canonical data artifacts for every universe.

Run once with network access; everything afterwards runs offline from the
parquet caches::

    python scripts/run_data.py                 # build from cache where possible
    python scripts/run_data.py --force-refresh # re-pull every source
    python scripts/run_data.py --universe etf10

Artifacts written per universe, into ``data/processed/``:

======================================  ====================================================
``<u>_prices_adj.parquet``              adjusted close panel (risky assets, forward-filled)
``<u>_prices_observed.parquet``         real observations only, no filling
``<u>_returns_log.parquet``             ln(P_t / P_{t-1}); reservoir/return basis
``<u>_returns_simple.parquet``          P_t/P_{t-1} - 1 **plus the CASH column**; accounting
``<u>_reservoir_input.parquet``         z-scored log returns, reservoir-input assets only
``<u>_signals.parquet``                 standardised market signals, publication-lagged
``<u>_availability.parquet``            A[t, i] in {0, 1}
``<u>_coverage.csv``                    per-asset availability summary (figure F1's source)
======================================  ====================================================

plus ``data/processed/data_manifest.json`` recording every source, its date
range, and the exact parameters used, so a rebuild is verifiable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.data.availability import (  # noqa: E402
    assert_no_prices_before_inception,
    availability_summary,
    build_availability_matrix,
)
from rrc.data.calendar import align_panel, build_trading_calendar  # noqa: E402
from rrc.data.fred import build_risk_free_daily, fetch_fred_series  # noqa: E402
from rrc.data.loaders import (  # noqa: E402
    PriceLoader,
    adjusted_prices,
    append_cash_column,
    log_returns,
    rolling_zscore,
    simple_returns,
)
from rrc.data.signals import (  # noqa: E402
    SentimentUnavailableError,
    build_signal_matrix,
    fetch_news_sentiment,
)
from rrc.utils.io import load_yaml, utc_timestamp  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_data")


def build_universe(
    universe_name: str,
    universe_config: dict,
    config: dict,
    raw_dir: Path,
    processed_dir: Path,
    macro_series: dict[str, pd.Series],
    force_refresh: bool,
) -> dict:
    """Build and write every artifact for one universe. Returns a manifest entry."""
    logger.info("=== universe: %s ===", universe_name)

    tickers = list(universe_config["tickers"])
    reservoir_tickers = list(universe_config["reservoir_input"])
    start = pd.Timestamp(universe_config["start"])
    end = pd.Timestamp(config["end"])
    ffill_limit = int(config["clean"]["ffill_limit"])

    # ---- 1. Prices -------------------------------------------------------
    loader = PriceLoader(raw_dir)
    histories, load_outcomes = loader.load_many(tickers, force_refresh=force_refresh)
    missing = [outcome.ticker for outcome in load_outcomes if not outcome.succeeded]
    if missing:
        raise RuntimeError(
            f"Universe {universe_name} is missing price data for {missing}. "
            "Both core universes must be complete; investigate before continuing."
        )

    price_series = adjusted_prices(histories)
    calendar = build_trading_calendar(price_series, start, end)
    filled_prices, observed_prices = align_panel(price_series, calendar, tickers, ffill_limit)
    logger.info(
        "Calendar: %d trading days, %s to %s.",
        len(calendar), calendar.min().date(), calendar.max().date(),
    )

    # ---- 2. Availability, and the anti-fabrication assertion -------------
    availability = build_availability_matrix(observed_prices)
    assert_no_prices_before_inception(filled_prices, availability)
    coverage = availability_summary(availability)

    # ---- 3. Risk-free rate and the return panels -------------------------
    risk_free = build_risk_free_daily(
        short_rate_id=config["fred"]["short_rate"],
        raw_dir=raw_dir,
        calendar=calendar,
        shift_days=int(config["signals"]["shift_days"]),
        force_refresh=force_refresh,
    )

    returns_log = log_returns(filled_prices)
    returns_simple = append_cash_column(
        simple_returns(filled_prices), risk_free, cash_column=config["cash"]["column"]
    )

    # Reservoir input: z-scored log returns, restricted to assets alive since
    # the universe start, cash excluded (decision R4 and causality test C3).
    reservoir_input = rolling_zscore(
        returns_log[reservoir_tickers],
        window=int(config["returns"]["reservoir_z_window"]),
        min_periods=int(config["returns"]["reservoir_z_min_periods"]),
    )

    # ---- 4. Signals ------------------------------------------------------
    signals = build_signal_matrix(
        calendar=calendar,
        raw_series=macro_series,
        shift_days=int(config["signals"]["shift_days"]),
        sentiment_shift_days=int(config["sentiment"]["shift_business_days"]),
        z_window=int(config["signals"]["z_window"]),
        z_min_periods=int(config["signals"]["z_min_periods"]),
    )

    # ---- 5. Write --------------------------------------------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "prices_adj": filled_prices,
        "prices_observed": observed_prices,
        "returns_log": returns_log,
        "returns_simple": returns_simple,
        "reservoir_input": reservoir_input,
        "signals": signals,
        "availability": availability,
    }
    written: dict[str, str] = {}
    for name, frame in artifacts.items():
        path = processed_dir / f"{universe_name}_{name}.parquet"
        frame.to_parquet(path)
        written[name] = path.name

    coverage_path = processed_dir / f"{universe_name}_coverage.csv"
    coverage.to_csv(coverage_path)
    written["coverage"] = coverage_path.name

    risk_free_path = processed_dir / f"{universe_name}_risk_free.parquet"
    risk_free.to_frame().to_parquet(risk_free_path)
    written["risk_free"] = risk_free_path.name

    logger.info(
        "Wrote %d artifacts. Panel %d x %d; reservoir input %d x %d.",
        len(written), *filled_prices.shape, *reservoir_input.shape,
    )
    for ticker in tickers:
        row = coverage.loc[ticker]
        logger.info(
            "  %-5s available %s to %s  (%5.1f%% of window)",
            ticker, row["first_available"], row["last_available"],
            100 * row["fraction_of_window"],
        )

    return {
        "tickers": tickers,
        "reservoir_input_tickers": reservoir_tickers,
        "window": {"start": str(start.date()), "end": str(end.date())},
        "trading_days": int(len(calendar)),
        "calendar_first": str(calendar.min().date()),
        "calendar_last": str(calendar.max().date()),
        "price_sources": {o.ticker: o.source for o in load_outcomes},
        "price_history_start": {o.ticker: o.first_date for o in load_outcomes},
        "files": written,
    }


def fetch_macro_series(config: dict, raw_dir: Path, force_refresh: bool) -> tuple[dict, dict]:
    """Fetch every raw macro/market input once, shared across universes."""
    fred_config = config["fred"]

    short_rate = fetch_fred_series(fred_config["short_rate"], raw_dir, force_refresh)
    long_rate = fetch_fred_series(fred_config["long_rate"], raw_dir, force_refresh)
    volatility = fetch_fred_series(fred_config["volatility"], raw_dir, force_refresh)
    credit = fetch_fred_series(fred_config["credit_spread"], raw_dir, force_refresh)

    # Term slope is a derived spread; align on the union of both indices so a
    # missing print in either series does not silently drop a day.
    combined = pd.concat([long_rate, short_rate], axis=1).ffill()
    term_slope = (combined[fred_config["long_rate"]] - combined[fred_config["short_rate"]]).dropna()
    term_slope.name = "term_slope"

    raw_series = {
        "vix": volatility,
        "term_slope": term_slope,
        "short_rate": short_rate,
        "credit_spread": credit,
    }
    provenance = {
        "vix": fred_config["volatility"],
        "term_slope": f"{fred_config['long_rate']} - {fred_config['short_rate']}",
        "short_rate": fred_config["short_rate"],
        "credit_spread": fred_config["credit_spread"],
    }

    # Sentiment is fetched separately: it is the one source that is neither
    # FRED nor guaranteed stable, so its failure must be visible, not fatal to
    # the core panels. It is only consumed from rung 3 onwards.
    try:
        raw_series["sentiment"] = fetch_news_sentiment(
            config["sentiment"]["url"], raw_dir, force_refresh
        )
        provenance["sentiment"] = config["sentiment"]["url"]
    except SentimentUnavailableError as error:
        logger.error("SENTIMENT UNAVAILABLE: %s", error)
        logger.error(
            "Continuing without it. The sentiment column will be all-NaN and "
            "rungs 3a/3b/3c CANNOT be run until this is resolved."
        )
        provenance["sentiment"] = f"UNAVAILABLE: {error}"

    return raw_series, provenance


def main(argv: list[str] | None = None) -> int:
    """Build the data layer. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Build the canonical data artifacts.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars "
             "(etf10 from 2006, sector11 from 1999), so they are run and "
             "reported independently and never aggregated.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Re-pull every source.")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

    selected = (
        [args.universe]
    )
    for name in selected:
        if name not in config["universes"]:
            logger.error("Unknown universe %r; known: %s", name, list(config["universes"]))
            return 2

    macro_series, provenance = fetch_macro_series(config, raw_dir, args.force_refresh)

    manifest: dict = {
        "generated_at": utc_timestamp(),
        "config_file": str(Path(args.config).name),
        "end_date": config["end"],
        "parameters": {
            "ffill_limit": config["clean"]["ffill_limit"],
            "signal_shift_days": config["signals"]["shift_days"],
            "sentiment_shift_business_days": config["sentiment"]["shift_business_days"],
            "z_window": config["signals"]["z_window"],
            "z_min_periods": config["signals"]["z_min_periods"],
            "cash_column": config["cash"]["column"],
        },
        "signal_provenance": provenance,
        "universes": {},
    }

    for name in selected:
        manifest["universes"][name] = build_universe(
            universe_name=name,
            universe_config=config["universes"][name],
            config=config,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            macro_series=macro_series,
            force_refresh=args.force_refresh,
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    logger.info("=============== DATA BUILD COMPLETE ===============")
    for name, entry in manifest["universes"].items():
        logger.info(
            "  %-9s %d assets | %d trading days | %s to %s",
            name, len(entry["tickers"]), entry["trading_days"],
            entry["calendar_first"], entry["calendar_last"],
        )
    logger.info("  manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
