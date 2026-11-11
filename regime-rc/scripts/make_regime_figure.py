"""F9: what each gating scheme tracks - the regime interpretability figure.

    python scripts/make_regime_figure.py --universe sector11

Reads only the CSVs ``run_regime_interpretation.py`` wrote, so it redraws in a
second without retraining anything.

What the figure says
--------------------
Two panels, one argument.

**Left - separation.** A grouped bar per market-state variable showing how much
of that variable each scheme's labelling explains (eta squared). This is the
headline: the statistical gates load on volatility, dispersion, correlation and
VIX; the learned gate loads on recent direction and nothing else. Each is nearly
blind to the other's variable, which is the mechanism behind AMI ~ 0.

**Right - profile.** The learned gate's three regimes as standardised means, so a
reader can see what the regimes MEAN: the market just rose, went sideways, or
just fell. Direction states, not risk states.

Why eta squared and not a correlation
--------------------------------------
The regime label is categorical with no natural ordering - "regime 2" is not
twice "regime 1" - so a correlation is undefined. Eta squared is the one-way
ANOVA effect size and is exactly the right quantity: the fraction of a
variable's variance explained by group membership, invariant to how the groups
happen to be numbered.
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
logger = logging.getLogger("regime-figure")

# Plain-English names; the reader should not have to decode variable names.
LABELS = {
    "realised_vol_21d": "realised volatility (21d)",
    "dispersion": "cross-sectional dispersion",
    "avg_correlation": "average pairwise correlation",
    "momentum_63d": "momentum (63d)",
    "return_5d": "recent direction (5d return)",
    "vix": "VIX level",
    "credit_spread": "credit spread",
    "sentiment": "news sentiment",
    "term_slope": "term slope",
    "short_rate": "short rate",
}
SCHEMES = {"learned": "learned gate (decision-defined)",
           "hmm": "HMM gate (statistical)",
           "vix": "VIX terciles (statistical)"}
COLOURS = {"learned": "#c1440e", "hmm": "#3d5a80", "vix": "#98c1d9"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regime interpretability figure.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    root = Path(args.data_root).expanduser().resolve()
    regimes = root / "results" / "runs" / "regimes"
    separation = pd.read_csv(regimes / f"separation_{args.universe}.csv", index_col=0) * 100
    profile = pd.read_csv(regimes / f"profile_learned_{args.universe}.csv", index_col=0)

    # Order by what the STATISTICAL schemes track, so the learned gate's single
    # spike at the bottom is the visual punchline rather than being buried.
    separation = separation.loc[separation[["hmm", "vix"]].max(axis=1).sort_values().index]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.7, 1]}
    )

    # ---- left: what each scheme tracks ------------------------------------
    positions = np.arange(len(separation))
    height = 0.26
    for offset, scheme in zip((-height, 0.0, height), ("hmm", "vix", "learned")):
        left.barh(positions + offset, separation[scheme], height,
                  label=SCHEMES[scheme], color=COLOURS[scheme])
    left.set_yticks(positions)
    left.set_yticklabels([LABELS.get(name, name) for name in separation.index])
    left.set_xlabel("% of the variable explained by the regime labelling (eta squared)")
    left.set_title(f"{args.universe}: what each gating scheme tracks\n"
                   "statistical gates find RISK states; the learned gate finds DIRECTION")
    left.legend(fontsize=9, loc="lower right")
    left.grid(axis="x", alpha=0.3)

    # ---- right: what the learned regimes mean -----------------------------
    ordered = [n for n in separation.index[::-1]]
    block = profile.reindex(ordered)
    limit = float(np.nanmax(np.abs(block.to_numpy())))
    image = right.imshow(block.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit,
                         aspect="auto")
    right.set_xticks(range(block.shape[1]))
    right.set_xticklabels([f"regime {c}" for c in block.columns])
    right.set_yticks(range(len(ordered)))
    right.set_yticklabels([LABELS.get(n, n) for n in ordered], fontsize=9)
    for row in range(block.shape[0]):
        for column in range(block.shape[1]):
            value = block.to_numpy()[row, column]
            if np.isfinite(value):
                right.text(column, row, f"{value:+.2f}", ha="center", va="center",
                           fontsize=8,
                           color="white" if abs(value) > 0.6 * limit else "black")
    right.set_title("what the learned regimes MEAN\n(standard deviations from average)")
    figure.colorbar(image, ax=right, shrink=0.75, label="std devs from sample mean")

    figure.tight_layout()
    out = root / "results" / "figures" / f"F9_regime_interpretation_{args.universe}.png"
    figure.savefig(out, dpi=150)
    plt.close(figure)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
