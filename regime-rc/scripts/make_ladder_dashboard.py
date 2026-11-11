"""The ladder dashboard: growth PLUS the risk metrics that judge "best".

    python scripts/make_ladder_dashboard.py            # both universes

Why the growth line alone is not enough
---------------------------------------
Two lines can end at the same number with completely different journeys: one
calm, one with a 40% hole in the middle; one holding steady, one churning the
whole book weekly and bleeding costs. This dashboard puts the judgement axes
next to the growth curve, per method, all computed on the SAME common
out-of-sample window as the ladder preview:

======================  =====================================================
excess Sharpe           the project's own objective - reward per unit risk
max drawdown            the deepest peak-to-trough loss (the pain axis)
annualised volatility   how rough the ride was (partly redundant with the two
                        above, kept because it is the standard axis)
annualised turnover     how much trading it took - the cost-awareness theme;
                        read from the REGISTRY's test rows (a return stream
                        cannot yield turnover; the weight schedules can), mean
                        over seeds and the preview folds
Calmar                  annualised return / |max drawdown| - included because
                        the plan's Section 13 lists it, flagged as noisy on a
                        three-year window
======================  =====================================================

Same quarantine as the other ladder visuals: reads streams + registry only,
writes to results/diagnostics/ladder/, no F-number, superseded by P9's tables
of record.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.metrics import (  # noqa: E402
    annualised_return,
    annualised_volatility,
    excess_sharpe,
    max_drawdown,
)
from rrc.eval.registry import read_registry  # noqa: E402
from make_ladder_preview import (  # noqa: E402
    FALLBACK_STYLE,
    LINE_STYLES,
    STARTING_MONEY,
    copy_rung0_streams,
    load_streams,
    populate_model_streams,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("dashboard")

OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "ladder"
PREVIEW_FOLDS = ["F01","F02","F03","F04","F05","F06","F07","F08","F09","F10"] 


def registry_turnover(universe: str) -> dict[tuple[str, str], float]:
    """Mean annualised one-way turnover per (rung, arm) from the registry.

    Test rows only, preview folds only, averaged over seeds (and over any
    duplicate append-only re-runs, which is exactly what a mean does).
    """
    frame = read_registry()
    rows = frame[
        (frame["universe"] == universe)
        & (frame["split"] == "test")
        & (frame["fold"].isin(PREVIEW_FOLDS))
    ]
    # The registry is append-only, so corrected re-runs COEXIST with older
    # rows (e.g. rung-0 rows written before turnover annualisation was made
    # calendar-based). Latest-wins per (rung, arm, fold, seed) is the honest
    # read policy for current values; the history stays in the file.
    rows = rows.sort_values("timestamp").drop_duplicates(
        subset=["rung", "arm", "fold", "seed"], keep="last"
    )
    turnover: dict[tuple[str, str], float] = {}
    for (rung, arm), group in rows.groupby(["rung", "arm"]):
        values = pd.to_numeric(group["ann_turnover"], errors="coerce").dropna()
        if len(values):
            turnover[(str(rung), str(arm))] = float(values.mean())
    return turnover


def stream_metrics(
    streams: dict[tuple[str, str], pd.Series], risk_free: pd.Series
) -> pd.DataFrame:
    """Sharpe / vol / drawdown / Calmar per method, on the common window."""
    common_start = max(s.index.min() for s in streams.values())
    common_end = min(s.index.max() for s in streams.values())

    rows = []
    for key, series in streams.items():
        window = series.loc[common_start:common_end]
        rates = risk_free.reindex(window.index).ffill().fillna(0.0)

        drawdown = max_drawdown(window.to_numpy())
        yearly_return = annualised_return(window.to_numpy())
        rows.append(
            {
                "key": key,
                "excess_sharpe": excess_sharpe(window.to_numpy(), rates.to_numpy()),
                "ann_vol": annualised_volatility(window.to_numpy()),
                "max_drawdown": drawdown,
                "calmar": (yearly_return / abs(drawdown)) if drawdown < -1e-9 else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("key")
    table.attrs["window"] = (common_start.date(), common_end.date())
    return table


def plot_dashboard(
    label: str,
    streams: dict[tuple[str, str], pd.Series],
    metrics: pd.DataFrame,
    turnover: dict[tuple[str, str], float],
    output_stem: str,
    turnover_note: str = "turnover from registry test rows, mean over seeds and folds",
) -> Path:
    """Growth curve on top; five metric bar panels below, same colours."""
    ordered_keys = sorted(
        metrics.index, key=lambda key: LINE_STYLES.get(key, FALLBACK_STYLE)["rank"]
    )
    colours = [LINE_STYLES.get(k, FALLBACK_STYLE)["colour"] for k in ordered_keys]
    labels = [
        LINE_STYLES.get(k, FALLBACK_STYLE)["label"] or f"r{k[0]} {k[1]}"
        for k in ordered_keys
    ]
    window_start, window_end = metrics.attrs["window"]

    figure = plt.figure(figsize=(15, 11))
    grid = figure.add_gridspec(
        2, 5, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.55
    )

    # ---- top: the growth curve (context) ---------------------------------
    growth_axis = figure.add_subplot(grid[0, :])
    for key in ordered_keys:
        style = LINE_STYLES.get(key, FALLBACK_STYLE)
        window = streams[key].loc[str(window_start) : str(window_end)]
        money = STARTING_MONEY * (1.0 + window).cumprod()
        growth_axis.plot(
            money.index, money, color=style["colour"], lw=style["width"],
            ls=style["dash"], label=style["label"],
            zorder=3 if style["width"] > 1.2 else 2,
        )
    growth_axis.axhline(STARTING_MONEY, color="grey", lw=0.8, ls=":")
    growth_axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    growth_axis.set_ylabel(f"growth of {STARTING_MONEY:,.0f}")
    growth_axis.grid(alpha=0.25)
    growth_axis.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left")
    growth_axis.set_title(
        f"Ladder dashboard - {label}: growth AND the judgement axes, common "
        f"out-of-sample window {window_start} to {window_end}\n"
        f"(seed 0 streams; {turnover_note}; net of 5 bp costs) - "
        "INTERIM VISUAL, superseded by P9",
        fontsize=10, loc="left",
    )

    # ---- bottom: the metric panels ---------------------------------------
    panels = [
        ("excess_sharpe", "excess Sharpe\n(higher better)", "{:.2f}", False),
        ("max_drawdown", "max drawdown\n(closer to 0 better)", "{:.0%}", False),
        ("ann_vol", "annualised vol\n(lower better)", "{:.0%}", False),
        ("turnover", "ann. one-way turnover\n(lower = cheaper to run)", "{:.1f}x", False),
        ("calmar", "Calmar = return/|DD|\n(higher better; noisy on 3y)", "{:.2f}", False),
    ]
    positions = np.arange(len(ordered_keys))[::-1]  # base at top, increments below

    for column, (metric_name, title, value_format, _) in enumerate(panels):
        axis = figure.add_subplot(grid[1, column])
        if metric_name == "turnover":
            values = [turnover.get(k, np.nan) for k in ordered_keys]
        else:
            values = [metrics.loc[[k], metric_name].iloc[0] for k in ordered_keys]

        axis.barh(positions, values, color=colours, alpha=0.9, height=0.7)
        for position, value in zip(positions, values):
            if np.isfinite(value):
                axis.text(
                    value, position, " " + value_format.format(value),
                    va="center", fontsize=6.5,
                    ha="left" if value >= 0 else "right",
                )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels if column == 0 else [], fontsize=7.5)
        axis.set_title(title, fontsize=8.5)
        axis.grid(alpha=0.25, axis="x")
        axis.axvline(0, color="grey", lw=0.8)
        axis.margins(x=0.25)

    path = OUTPUT_DIR / f"{output_stem}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ladder dashboard with risk metrics.")
    parser.add_argument("--universe", default="all", help="etf10, sector11, or all.")
    args = parser.parse_args(argv)

    universes = ["etf10", "sector11"] if args.universe == "all" else [args.universe]
    for universe in universes:
        copy_rung0_streams(universe)
        populate_model_streams(universe, repopulate=False)  # cache hit normally
        streams = load_streams(universe)
        risk_free = pd.read_parquet(
            PROJECT_ROOT / "data" / "processed" / f"{universe}_risk_free.parquet"
        )["r_f"]
        metrics = stream_metrics(streams, risk_free)
        turnover = registry_turnover(universe)
        path = plot_dashboard(
            universe, streams, metrics, turnover,
            output_stem=f"ladder_dashboard_{universe}",
        )
        logger.info("[%s] dashboard -> %s", universe, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
