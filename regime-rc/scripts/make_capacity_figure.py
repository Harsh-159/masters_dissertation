"""F10: out-of-sample performance versus the number of experts K.

    python scripts/make_capacity_figure.py

Redraws in a second from ``results/runs/capacity/capacity_<universe>.csv``.

What the figure argues
----------------------
Two panels, one per universe, sharing a structure:

* the **dashed** line is validation Sharpe - what capacity K could be made to
  FIT, at the reservoir configuration searched specifically for that K;
* the **solid** line is the honest number - the same configuration scored on the
  full walk-forward test split, which nothing in the selection ever saw;
* the **shaded band** is the seed spread, so a reader can tell immediately
  whether a wiggle in the curve is a result or noise;
* the **grey line** is the best rung-0 baseline on that universe, because the
  question is never "which K is best" in isolation but "does any K earn its
  keep against doing nothing clever".

The vertical distance between the dashed and solid lines is the generalisation
gap. If it widens with K, extra experts are buying fit rather than out-of-sample
performance - which is the capacity argument the rest of the project makes from
train/validation gaps, seed spreads and the loss-term nulls.

Why the searched hyperparameters are annotated
----------------------------------------------
Each point carries the reservoir width chosen for that K. A curve where the
searched width climbs with K is evidence that the two interact, and that any
earlier K comparison at fixed reservoir settings was measuring the wrong thing.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("capacity-figure")

# Best rung-0 baseline per universe, from the walk-forward summary. Quoted
# rather than recomputed so the reference line is literally the number in the
# results table.
BEST_BASELINE = {
    "sector11": ("inverse-vol", 0.511),
    "etf10": ("buy-and-hold SPY", 0.688),
}
VALIDATION_COLOUR = "#8d99ae"
TEST_COLOUR = "#c1440e"


def panel(axis, frame: pd.DataFrame, universe: str) -> None:
    k = frame["K"].to_numpy()

    axis.plot(k, frame["val_sharpe"], "--o", color=VALIDATION_COLOUR, markersize=5,
              label="validation (what K can FIT)")
    axis.fill_between(k, frame["val_sharpe"] - frame["val_sd"],
                      frame["val_sharpe"] + frame["val_sd"],
                      color=VALIDATION_COLOUR, alpha=0.15)

    axis.plot(k, frame["test_sharpe"], "-o", color=TEST_COLOUR, linewidth=2.2,
              markersize=6, label="test, full walk-forward (the honest number)")
    axis.fill_between(k, frame["test_sharpe"] - frame["test_sd"],
                      frame["test_sharpe"] + frame["test_sd"],
                      color=TEST_COLOUR, alpha=0.18)

    name, value = BEST_BASELINE[universe]
    axis.axhline(value, color="black", linewidth=1.3, linestyle=":",
                 label=f"best baseline: {name} ({value:+.3f})")

    # Annotate the reservoir width the inner search picked for each K - if it
    # climbs, K and the reservoir interact.
    for _, row in frame.iterrows():
        axis.annotate(f"N={int(row['n_reservoir'])}",
                      (row["K"], row["test_sharpe"]), textcoords="offset points",
                      xytext=(0, -16), ha="center", fontsize=7, color="#555555")

    axis.set_xlabel("K  (number of experts)")
    axis.set_ylabel("concatenated excess Sharpe")
    axis.set_xticks(k)
    axis.set_title(
        f"{universe}\nreservoir searched separately inside each K",
        fontsize=11,
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="best")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capacity curve figure.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    root = Path(args.data_root).expanduser().resolve()
    capacity = root / "results" / "runs" / "capacity"

    available = [u for u in ("sector11", "etf10")
                 if (capacity / f"capacity_{u}.csv").exists()]
    if not available:
        raise SystemExit(f"No capacity CSVs in {capacity}; run run_capacity_curve.py first.")

    figure, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6),
                               squeeze=False)
    for axis, universe in zip(axes[0], available):
        frame = pd.read_csv(capacity / f"capacity_{universe}.csv").sort_values("K")
        panel(axis, frame, universe)

        gap_low, gap_high = frame["gap"].iloc[0], frame["gap"].iloc[-1]
        logger.info("%s: generalisation gap  K=%d %.3f  ->  K=%d %.3f",
                    universe, int(frame["K"].iloc[0]), gap_low,
                    int(frame["K"].iloc[-1]), gap_high)
        best = frame.loc[frame["test_sharpe"].idxmax()]
        logger.info("  best TEST Sharpe at K=%d (%.3f); best VALIDATION at K=%d (%.3f)",
                    int(best["K"]), best["test_sharpe"],
                    int(frame.loc[frame["val_sharpe"].idxmax(), "K"]),
                    frame["val_sharpe"].max())

    figure.suptitle(
        "Out-of-sample performance versus mixture capacity\n"
        "each K given its own reservoir search, then scored on the full walk-forward",
        fontsize=13,
    )
    figure.tight_layout()
    out = root / "results" / "figures" / "F10_capacity_curve.png"
    figure.savefig(out, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
