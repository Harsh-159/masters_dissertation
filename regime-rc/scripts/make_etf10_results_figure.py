"""A proper etf10 results figure from the REGISTRY (not the seed-0 streams).

    python scripts/make_etf10_results_figure.py

Why this exists
---------------
The ladder *dashboard* plots the seed-0 stream files and cannot show a
multi-seed result. This script reads the registry - the results of record that
`run_rung3.py --universe etf10` wrote - and draws the ladder on **all five
judgement axes** over the full 2015-2024 test era, with an error bar showing the
**spread across the 5 head seeds** for the trained arms.

The five panels
---------------
* excess Sharpe          - higher is better (the headline risk-adjusted return)
* max drawdown           - closer to 0 is better (worst peak-to-trough)
* annualised volatility  - lower is better
* ann. one-way turnover  - lower is cheaper to run
* Calmar (= ann return / |max drawdown|) - higher is better

Convention (stated on the figure)
---------------------------------
Each arm's per-(fold, seed) test metric is read from the registry. Per seed we
average over the ten test folds (2015..2024) to get that seed's full-era number;
the bar is the mean over the 5 seeds and the whisker is +/- 1 sd over the 5
seeds. Baselines are deterministic (seed 0 only), so they are reference bars
without a seed whisker. Calmar is formed from each seed's fold-averaged return
and drawdown (more stable than averaging noisy per-fold Calmars).

This is the "mean of yearly metrics" aggregation and a PRE-FREEZE checkpoint
(default lambda, unsearched reservoir) - not a figure of record. P9 adds
stationary-bootstrap CIs, the frozen config, and rungs 1 and 5.

Scope note
----------
Rung 1 (single head) is omitted: at full multi-seed it exists only on fold F01
(it is `run_rung1.py`'s deliverable), so it has no comparable full-era number.
"""

from __future__ import annotations

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

from rrc.eval.registry import read_registry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("etf10-results")

OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "ladder"

# Colour + label + draw order per arm. Baselines muted grey; models in the
# project's stable palette (matches the ladder dashboard).
STYLE = {
    ("0", "buy_hold_spy"): dict(rank=0, colour="#4D4D4D", label="r0 buy&hold SPY",   baseline=True),
    ("0", "one_over_n"):   dict(rank=1, colour="#7F7F7F", label="r0 1/N",            baseline=True),
    ("0", "inverse_vol"):  dict(rank=2, colour="#9E9E9E", label="r0 inverse-vol",    baseline=True),
    ("0", "mean_variance"):dict(rank=3, colour="#BDBDBD", label="r0 Ledoit-Wolf MV", baseline=True),
    ("2", "moe_k3"):       dict(rank=4, colour="#009E73", label="r2 mixture K=3",    baseline=False),
    ("3a", "sent_gate"):   dict(rank=5, colour="#E69F00", label="r3a signals->gate", baseline=False),
    ("3b", "sent_experts"):dict(rank=6, colour="#D55E00", label="r3b signals->experts", baseline=False),
    ("3c", "sent_both"):   dict(rank=7, colour="#CC79A7", label="r3c signals->both", baseline=False),
    ("4", "hmm_gate"):     dict(rank=8, colour="#56B4E9", label="r4 HMM gate",       baseline=False),
    ("4", "vix_gate"):     dict(rank=9, colour="#F0E442", label="r4 VIX gate",       baseline=False),
}

# (column, panel title, whether lower is better) - drives titles and value signs.
PANELS = [
    ("excess_sharpe", "excess Sharpe\n(higher better)", False),
    ("max_drawdown",  "max drawdown\n(closer to 0 better)", False),
    ("ann_vol",       "annualised vol\n(lower better)", True),
    ("ann_turnover",  "ann. one-way turnover\n(lower = cheaper)", True),
    ("calmar",        "Calmar = return/|DD|\n(higher better)", False),
]


def collect() -> pd.DataFrame:
    """Return one row per arm with point + seed-sd for each of the five metrics."""
    frame = read_registry()
    rows = frame[(frame["universe"] == "etf10") & (frame["split"] == "test")].copy()
    for column in ("excess_sharpe", "max_drawdown", "ann_vol", "ann_turnover", "ann_return"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["excess_sharpe"])
    # One record per (rung, arm, fold, seed); a fold rerun under two cfg_hashes
    # (e.g. rung-2 F01 from P6 and P7) keeps the latest.
    rows = rows.sort_values("timestamp").drop_duplicates(
        subset=["rung", "arm", "fold", "seed"], keep="last"
    )

    records = []
    for (rung, arm), style in STYLE.items():
        arm_rows = rows[(rows["rung"] == rung) & (rows["arm"] == arm)]
        if arm_rows.empty:
            logger.warning("no rows for r%s %s; skipping.", rung, arm)
            continue

        record = {
            "rung": rung, "arm": arm, "label": style["label"], "colour": style["colour"],
            "rank": style["rank"], "baseline": style["baseline"],
            "n_folds": int(arm_rows["fold"].nunique()),
            "n_seeds": int(arm_rows["seed"].nunique()),
        }

        # Per-seed, fold-averaged value for each of the four registry metrics.
        for column in ("excess_sharpe", "max_drawdown", "ann_vol", "ann_turnover"):
            per_seed = arm_rows.groupby("seed")[column].mean()
            record[column] = float(per_seed.mean())
            record[f"{column}_sd"] = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0

        # Calmar from each seed's fold-averaged return and drawdown (stable).
        ret_by_seed = arm_rows.groupby("seed")["ann_return"].mean()
        dd_by_seed = arm_rows.groupby("seed")["max_drawdown"].mean().abs()
        calmar_by_seed = ret_by_seed / dd_by_seed.replace(0.0, np.nan)
        record["calmar"] = float(calmar_by_seed.mean())
        record["calmar_sd"] = float(calmar_by_seed.std(ddof=1)) if len(calmar_by_seed) > 1 else 0.0

        records.append(record)

    return pd.DataFrame(records).sort_values("rank")


def plot(data: pd.DataFrame) -> Path:
    n_arms = len(data)
    y = np.arange(n_arms)[::-1]  # first arm at the top

    figure, axes = plt.subplots(1, len(PANELS), figsize=(19, 0.62 * n_arms + 2.2), sharey=True)

    for axis, (column, title, _lower_better) in zip(axes, PANELS):
        values = data[column].to_numpy()
        sds = data[f"{column}_sd"].to_numpy()
        for yi, (_, row), value, sd in zip(y, data.iterrows(), values, sds):
            err = None if row["baseline"] else float(sd)
            axis.barh(
                yi, value, color=row["colour"],
                xerr=err, capsize=3 if err else 0,
                edgecolor="white", height=0.72,
                error_kw=dict(ecolor="#333333", lw=1.0),
            )
            offset = (axis.get_xlim()[1] - axis.get_xlim()[0]) * 0.02
            ha = "left" if value >= 0 else "right"
            axis.text(value + (offset if value >= 0 else -offset), yi,
                      f"{value:.2f}", va="center", ha=ha, fontsize=7.8)
        axis.axvline(0, color="black", lw=0.7)
        axis.set_title(title, fontsize=9.5)
        axis.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(data["label"], fontsize=9)

    figure.suptitle(
        "etf10 ladder - full out-of-sample era 2015-2024, from the registry (5 head seeds; "
        "models: bar = mean, whisker = ±1 sd over seeds)\n"
        "PRE-FREEZE checkpoint: default lambda_cvar=1.0, unsearched reservoir; rung 1 omitted "
        "(F01-only at multi-seed) - NOT a figure of record (P9 adds bootstrap CIs + frozen config)",
        fontsize=10.5, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "etf10_ladder_results_2015_2024.png"
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    data = collect()
    if data.empty:
        raise RuntimeError("No etf10 test rows in the registry to plot.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "etf10_ladder_results_2015_2024.csv"
    export_columns = [
        "rung", "arm", "label", "n_folds", "n_seeds",
        "excess_sharpe", "excess_sharpe_sd", "max_drawdown", "max_drawdown_sd",
        "ann_vol", "ann_vol_sd", "ann_turnover", "ann_turnover_sd",
        "calmar", "calmar_sd",
    ]
    data[export_columns].round(4).to_csv(csv_path, index=False)

    png_path = plot(data)
    logger.info("wrote:\n  %s\n  %s", png_path, csv_path)
    for _, r in data.iterrows():
        logger.info(
            "  %-22s Sharpe %6.3f | maxDD %6.3f | vol %5.3f | turn %6.2f | Calmar %6.3f  [%d folds, %d seeds]",
            r["label"], r["excess_sharpe"], r["max_drawdown"], r["ann_vol"],
            r["ann_turnover"], r["calmar"], r["n_folds"], r["n_seeds"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
