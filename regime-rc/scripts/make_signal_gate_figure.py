"""M3-companion: does giving the learned gate the market signals move it onto risk?

    python scripts/make_signal_gate_figure.py

M3 (make_meeting_figures.py) showed three schemes - HMM, VIX and the learned
gate that reads ONLY the reservoir state - and made the point that the
statistical gates track RISK (realised volatility) while the learned gate tracks
DIRECTION (5-day return), each nearly blind to the other's variable.

That left one question open: the learned gate never saw VIX, so was direction a
CHOICE or just all it could see? run_regime_interpretation_signals.py answered
it by adding a fourth arm - the same gate given [state ; signals], i.e. handed
VIX and four macro signals directly (the 3a routing; experts unchanged).

This figure draws all four arms in the M3 style so the answer is visible: the
signal-gate bar sits right beside the state-gate bar at nearly the same height
on every row. Giving the gate VIX does NOT move it onto realised volatility -
it stays a direction detector. What it DOES take up is a slow macro level
(short rate), not risk, so short_rate is kept in the panel here (M3 dropped it)
because it is where the new arm visibly changes.

Reads  results/runs/regimes_signals/separation_{universe}.csv  (3-seed-mean eta
squared, written by the companion script). Writes the PNG beside it and also
into results/meeting_figures/ so it lives next to M1-M6.
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
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("m3b")

UNIVERSES = ["sector11", "etf10"]
UNIVERSE_TITLE = {"sector11": "sector11  (18 folds, 2007-2024)",
                  "etf10": "etf10  (10 folds, 2015-2024)"}

# Reader-friendly variable names; the two axes that carry the argument are
# shouted in capitals exactly as M3 does.
LABELS = {"realised_vol_21d": "REALISED VOLATILITY (risk)", "dispersion": "dispersion",
          "avg_correlation": "avg correlation", "momentum_63d": "momentum (63d)",
          "return_5d": "RECENT DIRECTION (5d)", "vix": "VIX level (a gate input now)",
          "credit_spread": "credit spread", "sentiment": "news sentiment",
          "short_rate": "short rate (a level)", "term_slope": "term slope"}

# term_slope stays tiny for every arm (<3.4%) and only adds a row without adding
# argument; short_rate is KEPT because it is where the signal-gate visibly moves.
DROP_FROM_PANEL = ["term_slope"]

# Two blues for the statistical (RISK) schemes, two warms for the learned
# (DIRECTION) schemes, so the eye groups them without reading the legend. The
# signal-gate (gold) sits beside the state-gate (rust): near-equal heights are
# the whole point.
ARMS = [
    ("hmm",            "HMM  (statistical)",          "#3d5a80"),
    ("vix",            "VIX  (statistical)",          "#98c1d9"),
    ("learned_state",  "learned gate  (state only)",  "#c1440e"),
    ("learned_signal", "learned gate  + VIX/signals", "#f4a261"),
]


def main(argv: list[str] | None = None) -> int:
    root = PROJECT_ROOT
    in_dir = root / "results" / "runs" / "regimes_signals"
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))

    for axis, universe in zip(axes, UNIVERSES):
        frame = pd.read_csv(in_dir / f"separation_{universe}.csv", index_col=0) * 100
        frame = frame.drop(index=DROP_FROM_PANEL, errors="ignore")
        # Sort rows so the RISK variables (what the statistical gates own) sit at
        # the top and DIRECTION at the bottom - same ordering logic as M3.
        frame = frame.loc[frame[["hmm", "vix"]].max(axis=1).sort_values().index]
        positions = np.arange(len(frame))
        height = 0.20
        offsets = [-1.5 * height, -0.5 * height, 0.5 * height, 1.5 * height]

        for offset, (col, label, colour) in zip(offsets, ARMS):
            axis.barh(positions + offset, frame[col], height, color=colour,
                      label=label, edgecolor="white", linewidth=0.4)

        axis.set_yticks(positions)
        axis.set_yticklabels([LABELS.get(n, n) for n in frame.index], fontsize=9)
        axis.set_xlabel("% of the variable explained by the labelling  (eta squared, 3-seed mean)")
        axis.set_title(UNIVERSE_TITLE[universe], fontsize=11)
        axis.grid(axis="x", alpha=0.3)
        axis.set_axisbelow(True)
        if universe == "sector11":
            axis.legend(fontsize=9, loc="lower right", framealpha=0.95)

    figure.suptitle(
        "Giving the learned gate VIX does NOT turn it into a risk detector\n"
        "the gold bar (gate + VIX) sits beside the rust bar (state only): still tall on DIRECTION, "
        "still at the floor on VOLATILITY, while the statistical gates own risk",
        fontsize=13.5, y=1.02)

    figure.tight_layout()
    for out_dir in (in_dir, root / "results" / "meeting_figures"):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "M3b_gate_with_signals.png"
        figure.savefig(path, dpi=170, bbox_inches="tight")
        logger.info("  wrote %s", path)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
