"""Figures of record from the walk-forward streams: F4, F5, F5-money.

    python scripts/make_walkforward_figures.py --universe sector11

Reads only ``results/runs/walkforward/<universe>/*.parquet`` - the daily streams
``run_walkforward.py`` saved - so figures regenerate in seconds without any
retraining. Nothing is recomputed from a model here.

The three figures
-----------------
**F4 - cumulative Sharpe through each year.** For every arm, the excess Sharpe
of the stream *from inception up to the end of each test year*. This is not a
per-year Sharpe and not an average of per-year Sharpes: point 2011 is the single
Sharpe an investor who started in 2007 and stopped at the end of 2011 would have
realised. It answers the question a per-year chart cannot - "what would this
strategy's track record have looked like at any point along the way?" - and it
shows how long a run of good years it takes to recover from a bad one.

Reading it: an arm that ends high but spent years underwater is a different
proposition from one that got there smoothly, and only this view distinguishes
them. The early years are noisy by construction (a Sharpe on one year of data
has an enormous standard error), so the left edge should be read as
uninformative rather than as a ranking.

**F5 - log equity curves.** The conventional view. Log scale, because a 50% loss
and a 100% gain are the same distance on it.

**F5-money - growth of GBP 10,000, linear.** The same information on the scale
people actually think in. It was promised at P8 and it is the one panel a
non-specialist reader will look at first. Linear scale exaggerates late
divergence, which is exactly why it is shown BESIDE the log version and never
instead of it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.eval.stats import annualised_excess_sharpe  # noqa: E402
from run_walkforward import in_ladder_order  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("figures")

STARTING_CAPITAL = 10_000

# Baselines drawn dashed and grey-ish so the learned arms read as the subject
# and the humility line reads as context.
BASELINE_PREFIX = "r0_"


def load_streams(stream_dir: Path) -> dict[str, pd.DataFrame]:
    """Load each arm's stream under its REAL name.

    Filenames are path-safe and therefore lossy - "ours K=1 (rung 1)" becomes
    "ours_K=1_rung_1", and no amount of substitution recovers the parentheses.
    ``arms.json`` carries the exact mapping so figure labels and the money table
    match the summary table character for character. Falling back to guessing
    from the filename is what previously made the two tables fail to join.
    """
    if not stream_dir.exists():
        raise SystemExit(f"No streams at {stream_dir}; run run_walkforward.py first.")
    manifest_path = stream_dir / "arms.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        manifest.get(path.stem, path.stem.replace("_", " ")): pd.read_parquet(path)
        for path in sorted(stream_dir.glob("*.parquet"))
    }


def cumulative_sharpe_by_year(frame: pd.DataFrame) -> pd.Series:
    """Excess Sharpe of the stream from inception through the end of each year."""
    excess = (frame["net_return"] - frame["risk_free"]).to_numpy()
    years = frame["year"].to_numpy()
    points = {}
    for year in sorted(pd.unique(years)):
        through = excess[years <= year]
        if through.size >= 30:                 # a Sharpe on less is meaningless
            points[int(year)] = annualised_excess_sharpe(through)
    return pd.Series(points)


def equity_curve(frame: pd.DataFrame) -> pd.Series:
    """Compounded growth of one unit, by day."""
    return (1.0 + frame["net_return"]).cumprod()


def style_for(arm: str) -> dict:
    """Baselines dashed and thin; learned arms solid and prominent."""
    if arm.startswith(BASELINE_PREFIX):
        return {"linestyle": "--", "linewidth": 1.2, "alpha": 0.75}
    return {"linestyle": "-", "linewidth": 1.9}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward figures of record.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    streams = load_streams(data_root / "results" / "runs" / "walkforward" / args.universe)
    figures_dir = data_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded %d arms", len(streams))

    # Legend follows the LADDER order, not the scoreboard: the point of these
    # figures is the argument (baselines -> challenger -> our rungs), and
    # sorting by result would reorder the panels between universes and hide
    # which arms are peers.
    finals = {arm: annualised_excess_sharpe(
        (f["net_return"] - f["risk_free"]).to_numpy()) for arm, f in streams.items()}
    order = in_ladder_order(streams)
    colours = plt.cm.tab20(np.linspace(0, 1, len(order)))

    # ---- F4: cumulative Sharpe through each year --------------------------
    figure, axis = plt.subplots(figsize=(11, 6.5))
    table = {}
    for colour, arm in zip(colours, order):
        series = cumulative_sharpe_by_year(streams[arm])
        table[arm] = series
        axis.plot(series.index, series.to_numpy(), color=colour,
                  label=f"{arm}  ({finals[arm]:+.2f})", **style_for(arm))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("through end of test year")
    axis.set_ylabel("excess Sharpe since inception")
    axis.set_title(
        f"{args.universe}: cumulative out-of-sample Sharpe through each year\n"
        "(each point is one Sharpe on all data so far, not a per-year figure)"
    )
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures_dir / f"F4_cumulative_sharpe_{args.universe}.png", dpi=150)
    plt.close(figure)

    pd.DataFrame(table).to_csv(
        data_root / "results" / "runs" / "walkforward"
        / f"cumulative_sharpe_{args.universe}.csv"
    )

    # ---- F5: log equity curves --------------------------------------------
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for colour, arm in zip(colours, order):
        frame = streams[arm]
        axis.plot(np.arange(len(frame)), equity_curve(frame).to_numpy(),
                  color=colour, label=arm, **style_for(arm))
    axis.set_yscale("log")
    axis.set_xlabel("trading day of the concatenated out-of-sample stream")
    axis.set_ylabel("growth of 1 unit (log scale)")
    axis.set_title(f"{args.universe}: out-of-sample equity curves, net of 5 bp costs")
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(figures_dir / f"F5_equity_log_{args.universe}.png", dpi=150)
    plt.close(figure)

    # ---- F5-money: growth of GBP 10,000, linear ---------------------------
    figure, axis = plt.subplots(figsize=(11, 6.5))
    summary = []
    for colour, arm in zip(colours, order):
        frame = streams[arm]
        money = STARTING_CAPITAL * equity_curve(frame).to_numpy()
        axis.plot(np.arange(len(frame)), money, color=colour,
                  label=f"{arm}  (GBP {money[-1]:,.0f})", **style_for(arm))
        summary.append({"arm": arm, "final_value": money[-1],
                        "sharpe": finals[arm]})
    axis.axhline(STARTING_CAPITAL, color="black", linewidth=0.8, linestyle=":")
    axis.set_xlabel("trading day of the concatenated out-of-sample stream")
    axis.set_ylabel("portfolio value (GBP), linear scale")
    axis.set_title(
        f"{args.universe}: growth of GBP 10,000 out of sample, net of costs\n"
        "(linear scale exaggerates late divergence - read beside the log panel)"
    )
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures_dir / f"F5_money_{args.universe}.png", dpi=150)
    plt.close(figure)

    money_frame = pd.DataFrame(summary).set_index("arm")
    money_frame = money_frame.reindex(in_ladder_order(money_frame.index)).reset_index()
    money_frame.to_csv(
        data_root / "results" / "runs" / "walkforward" / f"money_{args.universe}.csv",
        index=False,
    )

    logger.info("")
    logger.info("=== growth of GBP 10,000 ===")
    for row in money_frame.itertuples():
        logger.info("  %-26s GBP %9s   (Sharpe %+.3f)",
                    row.arm, f"{row.final_value:,.0f}", row.sharpe)
    logger.info("")
    logger.info("Wrote 3 figures to %s", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
