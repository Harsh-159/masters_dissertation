"""The combined 50/50 dashboard: growth + judgement axes for the blended book.

    python scripts/make_combined_dashboard.py

Same layout as the per-universe dashboards, applied to the 50/50 daily
rebalanced portfolio of each method's etf10 and sector11 strategies on the
matched 2015-2017 window (see make_combined_ladder for why that window).

How each panel is computed for a BLEND
--------------------------------------
* Sharpe / drawdown / volatility / Calmar: directly from the blended daily
  return stream - these are genuine properties of the combined book.
* Turnover: a return stream cannot yield turnover, so the blend's figure is
  the capital-weighted average of the two sleeves' turnovers
  (0.5 x etf10 + 0.5 x sector11). The etf10 sleeve's number comes from the
  registry (folds F01-F03 - the same years as the window); the sector sleeve's
  comes from the sidecar metadata written when the matched F09-F11 streams
  were populated (those trainings are registry-free by design). The tiny
  daily flow that rebalances the 50/50 split itself is second-order and
  ignored - stated here so nobody mistakes the figure for exact.

Quarantine unchanged: diagnostics output, no registry writes, no F-number.
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.eval.registry import read_registry  # noqa: E402
from make_ladder_preview import copy_rung0_streams, load_streams, populate_model_streams  # noqa: E402
from make_combined_ladder import (  # noqa: E402
    MATCHED_SECTOR_FOLDS,
    MATCHED_STORE,
    blend_streams,
    load_matched_streams,
    populate_matched_sector_streams,
)
from make_ladder_dashboard import plot_dashboard, stream_metrics  # noqa: E402
from make_ladder_preview import LINE_STYLES, FALLBACK_STYLE, STARTING_MONEY  # noqa: E402
from rrc.backtest.metrics import annualised_return  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("combined-dashboard")

ETF_FOLDS = ["F01", "F02", "F03"]  # the etf10 folds whose test years match


def registry_turnover_for(universe: str, folds: list[str]) -> dict[tuple[str, str], float]:
    """Latest-wins mean annualised turnover per (rung, arm) on given folds."""
    frame = read_registry()
    rows = frame[
        (frame["universe"] == universe)
        & (frame["split"] == "test")
        & (frame["fold"].isin(folds))
    ]
    rows = rows.sort_values("timestamp").drop_duplicates(
        subset=["rung", "arm", "fold", "seed"], keep="last"
    )
    turnover: dict[tuple[str, str], float] = {}
    for (rung, arm), group in rows.groupby(["rung", "arm"]):
        values = pd.to_numeric(group["ann_turnover"], errors="coerce").dropna()
        if len(values):
            turnover[(str(rung), str(arm))] = float(values.mean())
    return turnover


def sidecar_turnover() -> dict[tuple[str, str], float]:
    """The matched sector streams' turnover, from their population metadata."""
    turnover: dict[tuple[str, str], float] = {}
    for path in sorted(MATCHED_STORE.glob("r*_*.meta.json")):
        rung, arm = path.name.replace(".meta.json", "").split("_", 1)
        turnover[(rung.lstrip("r"), arm)] = float(
            json.loads(path.read_text())["ann_turnover_mean"]
        )
    return turnover


def blended_turnover(keys) -> dict[tuple[str, str], float]:
    """0.5 x etf10 sleeve + 0.5 x sector11 sleeve, where both sides exist."""
    etf_side = registry_turnover_for("etf10", ETF_FOLDS)
    sector_side = {
        # rung-0 sector sleeve: registry rows on the matched folds
        **registry_turnover_for("sector11", MATCHED_SECTOR_FOLDS),
        # model sleeves: the sidecars written at stream population
        **sidecar_turnover(),
    }
    blended: dict[tuple[str, str], float] = {}
    for key in keys:
        if key in etf_side and key in sector_side:
            blended[key] = 0.5 * etf_side[key] + 0.5 * sector_side[key]
    return blended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combined 50/50 ladder dashboard.")
    parser.add_argument("--repopulate", action="store_true")
    args = parser.parse_args(argv)

    copy_rung0_streams("etf10")
    copy_rung0_streams("sector11")
    populate_model_streams("etf10", repopulate=False)
    populate_matched_sector_streams(repopulate=args.repopulate)

    blended = blend_streams(load_streams("etf10"), load_matched_streams())
    risk_free = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "etf10_risk_free.parquet"
    )["r_f"]  # both universes share the same DGS3MO-derived series

    metrics = stream_metrics(blended, risk_free)
    turnover = blended_turnover(list(blended))

    # ---- the numbers behind the picture, as a CSV -------------------------
    window_start, window_end = metrics.attrs["window"]
    csv_rows = []
    for key in sorted(blended, key=lambda k: LINE_STYLES.get(k, FALLBACK_STYLE)["rank"]):
        window = blended[key].loc[str(window_start) : str(window_end)]
        style = LINE_STYLES.get(key, FALLBACK_STYLE)
        csv_rows.append(
            {
                "rung": key[0],
                "arm": key[1],
                "label": style["label"] or f"r{key[0]} {key[1]}",
                "window_start": window_start,
                "window_end": window_end,
                "final_value_of_10000": round(
                    float(STARTING_MONEY * (1.0 + window).cumprod().iloc[-1]), 2
                ),
                "ann_return": round(annualised_return(window.to_numpy()), 4),
                "excess_sharpe": round(float(metrics.loc[[key], "excess_sharpe"].iloc[0]), 4),
                "ann_vol": round(float(metrics.loc[[key], "ann_vol"].iloc[0]), 4),
                "max_drawdown": round(float(metrics.loc[[key], "max_drawdown"].iloc[0]), 4),
                "calmar": round(float(metrics.loc[[key], "calmar"].iloc[0]), 4),
                "ann_turnover": round(turnover[key], 3) if key in turnover else None,
            }
        )
    csv_path = (
        PROJECT_ROOT / "results" / "diagnostics" / "ladder"
        / "ladder_combined_50_50_metrics.csv"
    )
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info("metrics CSV -> %s", csv_path)

    path = plot_dashboard(
        label="combined 50/50 (etf10 + sector11 sleeves, daily-rebalanced)",
        streams=blended,
        metrics=metrics,
        turnover=turnover,
        output_stem="ladder_dashboard_combined_50_50",
        turnover_note=(
            "turnover = 0.5 x etf10 sleeve (registry F01-F03) + 0.5 x sector11 "
            "sleeve (matched-stream sidecars / registry F09-F11)"
        ),
    )
    logger.info("%d blended lines -> %s", len(blended), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
