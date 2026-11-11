"""Run the rung-0 baselines over one universe's walk-forward folds.

    python scripts/run_baselines.py --universe etf10
    python scripts/run_baselines.py --universe etf10

What this establishes (phase P4)
--------------------------------
The humility line: the numbers every later rung must beat, computed **before**
any learned system exists, under the exact conventions the models will face -
the same weekly signal->execute timing, the same 5 bp cost function, the same
ledger, the same metrics, the same fold calendar. A model comparison is only
meaningful if the thing being compared against was given the same rules.

How the evaluation is structured
--------------------------------
* Everything after 2024-12-31 is refused on load: the holdout firewall
  (``truncate_before_holdout`` + ``assert_no_holdout``) makes touching 2025+
  a hard error rather than a discipline to remember.
* Each calendar year that appears as a validation or test segment is run as an
  **independent segment**: the strategy starts the year in cash, trades in at
  the first weekly execution day, and runs to the year's end. This mirrors
  exactly how the trained models will be evaluated per fold - so the entry
  cost and the drift state are identical on both sides of every future
  comparison, and no baseline enjoys a continuity advantage.
* Consecutive folds share years (fold k's test year is fold k+1's validation
  year). Baselines are deterministic, so each year is computed once and the
  result recorded under every (fold, split) role it plays.
* Registry rows are appended **before** any table or figure is written - the
  P0 discipline: the audit trail comes first.

Outputs
-------
* registry rows: one per (universe, arm, fold, split), rung "0"
* ``results/runs/rung0/<universe>_<arm>.parquet`` - daily net-return streams
* ``results/runs/rung0/table_v0_<universe>.csv`` - the rung-0 table with
  bootstrap confidence intervals on the excess Sharpe
* ``results/figures/rung0_test_equity_<universe>.png`` - growth of 1.0 over
  the concatenated test years
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.backtest.ledger import (  # noqa: E402
    LEDGER_DTYPE,
    run_ledger,
    weekly_execution_schedule,
)
from rrc.backtest.metrics import summarise  # noqa: E402
from rrc.baselines.markowitz import mean_variance_schedule  # noqa: E402
from rrc.baselines.naive import (  # noqa: E402
    buy_and_hold_schedule,
    inverse_volatility_schedule,
    one_over_n_schedule,
)
from rrc.data.loaders import PriceLoader, adjusted_prices  # noqa: E402
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import (  # noqa: E402
    assert_no_holdout,
    folds_for,
    segment_dates,
    ordered_test_years,
    truncate_before_holdout,
    year_roles,
)
from rrc.eval.stats import sharpe_confidence_interval  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("baselines")

CASH_COLUMN = "CASH"
ARM_ORDER = ["one_over_n", "buy_hold_spy", "inverse_vol", "mean_variance"]

# Colour per arm, kept identical across universes so figures stay comparable.
ARM_COLOURS = {
    "one_over_n": "#0072B2",
    "buy_hold_spy": "#E69F00",
    "inverse_vol": "#009E73",
    "mean_variance": "#D55E00",
}


# --------------------------------------------------------------------------- #
# Data loading (with the holdout firewall applied immediately)
# --------------------------------------------------------------------------- #
def load_universe_data(processed_dir: Path, universe: str) -> dict:
    """Load every artifact a baseline needs, truncated before the holdout.

    The truncation happens *here*, at load time, so no code downstream can even
    see a 2025+ row. ``assert_no_holdout`` then double-checks the result -
    belt and braces, because this is the one discipline that cannot be walked
    back once broken.
    """
    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(processed_dir / f"{universe}_{name}.parquet")

    returns_simple = truncate_before_holdout(read("returns_simple"))
    availability = truncate_before_holdout(read("availability"))
    risk_free = truncate_before_holdout(read("risk_free"))["r_f"]

    assert_no_holdout(returns_simple.index, context=f"{universe} returns")

    risky_columns = [c for c in returns_simple.columns if c != CASH_COLUMN]
    data = {
        "returns_simple": returns_simple,                  # risky + CASH
        "risky_returns": returns_simple[risky_columns],    # estimation basis
        "availability": availability[risky_columns],
        "risk_free": risk_free,
        "dates": returns_simple.index,
        "columns": list(returns_simple.columns),
        "cash_index": list(returns_simple.columns).index(CASH_COLUMN),
    }

    # The ledger multiplies (1 + return) cumulatively, so a NaN would poison
    # everything after it. NaNs are legitimate in exactly two places: before an
    # asset's inception, and ON the inception day itself - a price exists that
    # day but no prior close, so no return can. In both cases every strategy
    # necessarily holds weight 0 there (buying would have required an earlier
    # execution day, when availability was still 0), so replacing those NaNs
    # with 0.0 is harmless: growth factor 1 on a zero holding.
    #
    # The assertion proves that premise: wherever an asset was available today
    # AND yesterday, its return must be finite. A violation would mean a hole
    # inside an asset's life - a genuine data-layer failure.
    available_mask = data["availability"].to_numpy() == 1
    available_yesterday = np.vstack([available_mask[:1] & False, available_mask[:-1]])
    risky_matrix = data["risky_returns"].to_numpy()
    bad = available_mask & available_yesterday & ~np.isfinite(risky_matrix)
    if bad.any():
        raise ValueError(
            f"{universe}: {int(bad.sum())} non-finite return(s) inside an "
            "asset's active life - the data layer's ffill guarantee is broken."
        )
    data["ledger_panel"] = returns_simple.fillna(0.0)
    return data


def load_spy_benchmark_panel(universe_data: dict, raw_dir: Path) -> pd.DataFrame:
    """Build the two-column [SPY, CASH] panel for the buy-and-hold benchmark.

    SPY is the market line for **both** universes, including sector11 where it
    is not a tradeable asset - it answers "did anything here beat just holding
    the S&P?". Its full history is already in the raw price cache from P1, so
    this is a cache read, never a download.
    """
    loader = PriceLoader(raw_dir)
    histories, outcomes = loader.load_many(["SPY"])
    if not outcomes[0].succeeded:
        raise RuntimeError("SPY history unavailable from the raw cache.")

    spy_prices = adjusted_prices(histories)["SPY"]
    spy_returns = spy_prices.pct_change(fill_method=None)

    panel = pd.DataFrame(index=universe_data["dates"])
    panel["SPY"] = spy_returns.reindex(universe_data["dates"])
    panel[CASH_COLUMN] = universe_data["risk_free"]

    # SPY trades on every US equity day, so its reindexed returns must be
    # complete everywhere a segment could be evaluated; a gap would make the
    # market benchmark silently wrong. The CASH column is checked separately
    # because it has one legitimate NaN: the very first calendar row, where the
    # one-day publication shift of r_f has no yesterday to draw from. That row
    # predates every evaluated segment, so filling it with 0 is harmless.
    evaluated = panel.index >= pd.Timestamp("2006-01-01")
    if panel.loc[evaluated, "SPY"].isna().any():
        n_gaps = int(panel.loc[evaluated, "SPY"].isna().sum())
        raise ValueError(f"SPY benchmark returns have {n_gaps} gap(s) after 2006.")
    cash_gaps = panel[CASH_COLUMN].isna()
    if cash_gaps.sum() > 1 or (cash_gaps.sum() == 1 and not cash_gaps.iloc[0]):
        raise ValueError(
            "The risk-free column has unexpected gaps beyond the first row."
        )
    return panel.fillna(0.0)


# --------------------------------------------------------------------------- #
# Running one arm over one calendar-year segment
# --------------------------------------------------------------------------- #
def run_arm_over_year(
    arm: str,
    arm_config: dict,
    universe_data: dict,
    spy_panel: pd.DataFrame,
    year: int,
    cost_per_unit: float,
) -> dict:
    """Run one baseline over one year, starting from cash, and summarise it.

    Returns a dict with the daily net-return series (indexed by date), the
    risk-free series over the same days, and the Section-13 metric summary.
    """
    dates = universe_data["dates"]
    year_days = segment_dates(dates, year)
    if len(year_days) < 60:
        raise ValueError(f"Year {year} has only {len(year_days)} trading days.")

    # Segment positions in the full calendar; estimators receive absolute
    # positions so trailing windows can reach back before the segment start
    # (history that genuinely existed at the time - causal, not leaky).
    segment_start = int(dates.get_loc(year_days[0]))
    signal_relative, execution_relative = weekly_execution_schedule(year_days)
    signal_absolute = signal_relative + segment_start

    if arm == "buy_hold_spy":
        panel = spy_panel
        columns = list(panel.columns)
        targets, execution_days = buy_and_hold_schedule(
            asset_index=columns.index("SPY"),
            first_execution_position=int(execution_relative[0]),
            n_columns=len(columns),
        )
        cash_index = columns.index(CASH_COLUMN)
    else:
        panel = universe_data["ledger_panel"]
        cash_index = universe_data["cash_index"]
        n_columns = len(universe_data["columns"])

        if arm == "one_over_n":
            targets, execution_days = one_over_n_schedule(
                availability=universe_data["availability"],
                signal_positions=signal_absolute,
                execution_positions=execution_relative,
                n_columns=n_columns,
                cash_index=cash_index,
            )
        elif arm == "inverse_vol":
            targets, execution_days = inverse_volatility_schedule(
                risky_returns=universe_data["risky_returns"],
                availability=universe_data["availability"],
                signal_positions=signal_absolute,
                execution_positions=execution_relative,
                n_columns=n_columns,
                cash_index=cash_index,
                window=arm_config["window"],
                min_observations=arm_config["min_observations"],
            )
        elif arm == "mean_variance":
            targets, execution_days = mean_variance_schedule(
                risky_returns=universe_data["risky_returns"],
                risk_free=universe_data["risk_free"],
                availability=universe_data["availability"],
                dates=dates,
                signal_positions=signal_absolute,
                execution_positions=execution_relative,
                n_columns=n_columns,
                cash_index=cash_index,
                lookback=arm_config["lookback"],
                min_observations=arm_config["min_observations"],
                risk_aversion=arm_config["risk_aversion"],
            )
        else:
            raise ValueError(f"Unknown arm {arm!r}.")

    segment_returns = torch.as_tensor(
        panel.loc[year_days].to_numpy().copy(), dtype=LEDGER_DTYPE
    )
    result = run_ledger(
        torch.as_tensor(targets, dtype=LEDGER_DTYPE),
        execution_days,
        segment_returns,
        cash_index,
        cost_per_unit=cost_per_unit,
        return_diagnostics=True,
    )

    net_returns = pd.Series(result.net_returns.numpy(), index=year_days)
    risk_free_segment = universe_data["risk_free"].loc[year_days]

    metric_summary = summarise(
        net_returns.to_numpy(),
        risk_free_segment.to_numpy(),
        turnover=result.turnover.numpy(),
        held_weights=result.held_weights.numpy(),
        cash_index=cash_index,
    )
    return {
        "net_returns": net_returns,
        "risk_free": risk_free_segment,
        "metrics": metric_summary,
    }


# --------------------------------------------------------------------------- #
# One universe end to end
# --------------------------------------------------------------------------- #
def run_universe(universe: str, config: dict, registry_path: Path | None = None) -> pd.DataFrame:
    """Evaluate all four baselines on one universe; return the rung-0 table."""
    logger.info("=== %s ===", universe)
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    cost_per_unit = float(config["evaluation"]["cost_per_unit"])

    universe_data = load_universe_data(processed_dir, universe)
    spy_panel = load_spy_benchmark_panel(universe_data, PROJECT_ROOT / "data" / "raw")

    folds = folds_for(universe)
    roles_by_year = year_roles(folds)
    years_in_test = ordered_test_years(folds)
    logger.info(
        "%d folds; validation %d..%d, test %d..%d; holdout untouched.",
        len(folds), folds[0].val_year, folds[-1].val_year,
        years_in_test[0], years_in_test[-1],
    )

    table_rows = []
    streams_per_arm: dict[str, pd.Series] = {}

    for arm in ARM_ORDER:
        arm_config = dict(config["arms"][arm])
        arm_config.pop("description", None)
        arm_identity = {
            "universe": universe, "rung": "0", "arm": arm,
            "cost_per_unit": cost_per_unit, **arm_config,
        }
        arm_hash = config_hash(arm_identity)

        # ---- run every year this arm is scored on, once -------------------
        yearly_runs = {
            year: run_arm_over_year(
                arm, arm_config, universe_data, spy_panel, year, cost_per_unit
            )
            for year in sorted(roles_by_year)
        }

        # ---- registry first: the audit trail precedes every artifact ------
        for year, roles in sorted(roles_by_year.items()):
            for fold_name, split in roles:
                append_row(
                    RegistryRow(
                        universe=universe, rung="0", arm=arm, seed=0,
                        fold=fold_name, split=split, n_params=0,
                        cfg_hash=arm_hash,
                        note=f"baseline year {year}; per-segment fresh start; "
                             "deterministic, no selection performed",
                        **yearly_runs[year]["metrics"],
                    ),
                    registry_path or (PROJECT_ROOT / "results" / "registry.csv"),
                )

        # ---- aggregate over the concatenated test years -------------------
        test_stream = pd.concat(
            [yearly_runs[year]["net_returns"] for year in years_in_test]
        )
        test_risk_free = pd.concat(
            [yearly_runs[year]["risk_free"] for year in years_in_test]
        )
        streams_per_arm[arm] = test_stream

        aggregate = summarise(test_stream.to_numpy(), test_risk_free.to_numpy())
        interval = sharpe_confidence_interval(
            test_stream.to_numpy(), test_risk_free.to_numpy(),
            level=config["evaluation"]["bootstrap"]["level"],
            n_resamples=config["evaluation"]["bootstrap"]["n_resamples"],
            mean_block_length=config["evaluation"]["bootstrap"]["mean_block_length"],
            seed=config["evaluation"]["bootstrap"]["seed"],
        )
        table_rows.append({
            "arm": arm,
            "ann_return": aggregate["ann_return"],
            "ann_vol": aggregate["ann_vol"],
            "excess_sharpe": interval.point,
            "sharpe_ci_low": interval.low,
            "sharpe_ci_high": interval.high,
            "sortino": aggregate["sortino"],
            "max_drawdown": aggregate["max_drawdown"],
            "cvar_95": aggregate["cvar_95"],
        })
        logger.info(
            "  %-14s Sharpe %6.3f  [%6.3f, %6.3f]   vol %5.1f%%   maxDD %6.1f%%",
            arm, interval.point, interval.low, interval.high,
            100 * aggregate["ann_vol"], 100 * aggregate["max_drawdown"],
        )

        runs_dir.mkdir(parents=True, exist_ok=True)
        test_stream.to_frame(name="net_return").to_parquet(
            runs_dir / f"{universe}_{arm}.parquet"
        )

    table = pd.DataFrame(table_rows).set_index("arm")
    table.to_csv(runs_dir / f"table_v0_{universe}.csv")

    _sanity_check_orderings(universe, table)
    _plot_test_equity(universe, streams_per_arm, years_in_test, figures_dir)
    return table


def _sanity_check_orderings(universe: str, table: pd.DataFrame) -> None:
    """The plan's P4 exit check: do the baselines order the way finance says?

    Inverse-volatility deliberately overweights the calm assets, so its realised
    volatility must come out below equal weight's. If it does not, either the
    estimator or the accounting is wrong, and no later result should be trusted
    until that is understood - hence a hard failure, not a warning.
    """
    risk_parity_vol = table.loc["inverse_vol", "ann_vol"]
    equal_weight_vol = table.loc["one_over_n", "ann_vol"]
    if risk_parity_vol >= equal_weight_vol:
        raise AssertionError(
            f"{universe}: inverse-vol volatility ({risk_parity_vol:.4f}) is not "
            f"below 1/N's ({equal_weight_vol:.4f}) - the risk-parity tilt is "
            "not doing what it must."
        )
    logger.info(
        "  sanity: inverse-vol vol %.2f%% < 1/N vol %.2f%%  (ordering holds)",
        100 * risk_parity_vol, 100 * equal_weight_vol,
    )


def _plot_test_equity(
    universe: str,
    streams_per_arm: dict[str, pd.Series],
    years_in_test: list[int],
    figures_dir: Path,
) -> None:
    """Growth of 1.0 over the concatenated test years, one line per baseline."""
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for arm in ARM_ORDER:
        stream = streams_per_arm[arm]
        axis.plot(
            stream.index, (1.0 + stream).cumprod(),
            lw=1.3, label=arm, color=ARM_COLOURS[arm],
        )
    axis.set_yscale("log")
    axis.axhline(1.0, color="grey", lw=0.8, ls="--")
    axis.set_title(
        f"Rung 0 - {universe} baselines over the test years "
        f"({years_in_test[0]}-{years_in_test[-1]}), net of 5 bp costs "
        "(log scale)",
        fontsize=10, loc="left",
    )
    axis.set_ylabel("growth of 1.0 (log)")
    axis.set_xlabel("date")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"rung0_test_equity_{universe}.png"
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    logger.info("  equity figure: %s", path.name)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run the rung-0 baselines.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baselines.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars "
             "(etf10 from 2006, sector11 from 1999), so they are run and "
             "reported independently and never aggregated.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    universes = [args.universe]

    for universe in universes:
        table = run_universe(universe, config)
        logger.info("\n%s\n", table.round(3).to_string())

    logger.info("=== RUNG 0 ESTABLISHED - the humility line is on the record ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
