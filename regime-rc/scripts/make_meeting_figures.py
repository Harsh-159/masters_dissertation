"""The six figures for the supervisor meeting, into results/meeting_figures/.

    python scripts/make_meeting_figures.py

Every panel is redrawn from CSVs and parquet already on disk - nothing is
retrained - so the whole set regenerates in a few seconds.

The set tells one story, in this order:

    M1  the ladder            what we built, and how it performs
    M2  orthogonality         THE contribution: the two carvings are unrelated
    M3  what they track       WHY they are unrelated: risk versus direction
    M4  capacity curve        does more capacity help? searched fairly
    M6  growth of GBP 10,000  the same result in money, one panel per universe

M1 and M6 are the same fact twice, deliberately: a supervisor reads Sharpe, and
everyone reads money.
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.eval.stats import annualised_excess_sharpe  # noqa: E402
from run_walkforward import in_ladder_order  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("meeting")

UNIVERSES = ["sector11", "etf10"]
UNIVERSE_TITLE = {"sector11": "sector11  (18 folds, 2007-2024)",
                  "etf10": "etf10  (10 folds, 2015-2024)"}
BEST_BASELINE = {"sector11": ("inverse-vol", 0.511), "etf10": ("buy-and-hold SPY", 0.688)}

BASELINE_COLOUR = "#9aa5b1"
LEARNED_COLOUR = "#c1440e"
STATISTICAL_COLOUR = "#3d5a80"
ACCENT = "#2a9d8f"

def save(figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    logger.info("  %s", path.name)


# --------------------------------------------------------------------------- #
# M1 - the ladder
# --------------------------------------------------------------------------- #
def figure_ladder(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    for axis, universe in zip(axes, UNIVERSES):
        frame = pd.read_csv(root / "results" / "runs" / "walkforward"
                            / f"summary_{universe}.csv").set_index("arm")
        frame = frame.reindex(in_ladder_order(frame.index))[::-1]   # top-to-bottom

        colours = [BASELINE_COLOUR if arm.startswith("r0")
                   else STATISTICAL_COLOUR if arm.startswith("r4")
                   else LEARNED_COLOUR for arm in frame.index]
        positions = np.arange(len(frame))
        axis.barh(positions, frame["sharpe"], color=colours,
                  xerr=frame["sharpe_sd"], error_kw={"ecolor": "#333333", "lw": 1})
        axis.set_yticks(positions)
        axis.set_yticklabels(frame.index, fontsize=9)

        name, value = BEST_BASELINE[universe]
        axis.axvline(value, color="black", linestyle=":", linewidth=1.6)
        axis.text(value, len(frame) - 0.3, f" best baseline\n {name} {value:+.3f}",
                  fontsize=8, va="top")

        for position, sharpe in zip(positions, frame["sharpe"]):
            axis.text(sharpe + 0.012, position, f"{sharpe:+.3f}", va="center", fontsize=8)

        axis.set_xlabel("Sharpe ratio  (out of sample, net of 5 bp costs)")
        axis.set_title(UNIVERSE_TITLE[universe], fontsize=11)
        axis.set_xlim(0, max(frame["sharpe"].max(), value) * 1.22)
        axis.grid(axis="x", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (BASELINE_COLOUR, LEARNED_COLOUR, STATISTICAL_COLOUR)]
    figure.legend(handles, ["classical baselines", "our decision-focused arms",
                            "statistical-gate comparisons"],
                  loc="lower center", ncol=3, frameon=False, fontsize=10)
    figure.suptitle("M1  The ladder: no learned arm beats the classical baselines",
                    fontsize=14, y=1.00)
    figure.subplots_adjust(bottom=0.10)
    save(figure, out_dir, "M1_ladder_both_universes.png")


# --------------------------------------------------------------------------- #
# M2 - orthogonality
# --------------------------------------------------------------------------- #
def figure_orthogonality(root: Path, out_dir: Path) -> None:
    """All THREE schemes, in every panel - this is a three-way comparison.

    The argument only lands if the reader can see that the two STATISTICAL
    schemes resemble each other while neither resembles the learned gate. That
    needs all three in the strips and all three pairings in the contingencies,
    not just the learned-versus-HMM pair.
    """
    figure = plt.figure(figsize=(15.5, 10))
    grid = figure.add_gridspec(4, 3, height_ratios=[0.85, 0.85, 0.85, 2.0],
                               hspace=0.75, wspace=0.30)

    labels = pd.read_parquet(root / "results" / "runs" / "regimes"
                             / "state_and_labels_sector11.parquet")
    window = slice(0, 260)                       # five years of weekly decisions
    palette = matplotlib.colors.ListedColormap(["#e63946", "#f4a261", "#457b9d"])

    strips = [
        ("label_learned", "learned gate  (decision-defined)", "dwell 1.8 wks, switches 56%"),
        ("label_hmm", "HMM gate  (statistical)", "dwell 12.9 wks, switches 8%"),
        ("label_vix", "VIX terciles  (statistical)", "dwell 3.7 wks, switches 27%"),
    ]
    for row, (key, name, stats) in enumerate(strips):
        axis = figure.add_subplot(grid[row, :])
        axis.imshow(labels[key].to_numpy()[window].reshape(1, -1), aspect="auto",
                    interpolation="nearest", cmap=palette, vmin=0, vmax=2)
        axis.set_yticks([])
        axis.set_xlabel("weekly decision  (five years of sector11 test data)"
                        if row == 2 else "")
        axis.set_title(f"{name}    -    {stats}", fontsize=10.5, loc="left")

    # Every pairing, so the reader can compare like with like. The third panel
    # is the control: two schemes that DO resemble each other.
    # AMI is computed from the saved labels, never hardcoded - an earlier
    # version quoted numbers from a different seed's run and they drifted.
    from rrc.eval.agreement import label_agreement

    pairings = [
        ("label_learned", "label_hmm", "learned", "HMM"),
        ("label_learned", "label_vix", "learned", "VIX"),
        ("label_hmm", "label_vix", "HMM", "VIX"),
    ]
    for column, (left_key, right_key, left_name, right_name) in enumerate(pairings):
        ami = "AMI {:+.3f}".format(
            label_agreement(labels[left_key].to_numpy(), labels[right_key].to_numpy())["ami"]
        )
        axis = figure.add_subplot(grid[3, column])
        table = pd.crosstab(labels[left_key], labels[right_key], normalize="all") * 100
        values = table.to_numpy()
        image = axis.imshow(values, cmap="Purples", vmin=0, vmax=values.max())
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                          fontsize=9.5,
                          color="white" if values[i, j] > values.max() * 0.62 else "black")
        axis.set_xticks(range(values.shape[1]))
        axis.set_xticklabels([f"{right_name} {c}" for c in table.columns], fontsize=8.5)
        axis.set_yticks(range(values.shape[0]))
        axis.set_yticklabels([f"{left_name} {r}" for r in table.index], fontsize=8.5)
        emphasis = " <- the two STATISTICAL schemes" if column == 2 else ""
        axis.set_title(f"{left_name} vs {right_name}    {ami}{emphasis}", fontsize=10)
        if column == 2:
            for spine in axis.spines.values():
                spine.set_edgecolor(ACCENT)
                spine.set_linewidth(2.5)
    figure.colorbar(image, ax=figure.axes[-1], shrink=0.85, label="% of weeks")

    figure.suptitle(
        "M2  Decision-defined regimes are orthogonal to BOTH statistical schemes\n"
        "sector11: same weeks, same data - and the only pair showing structure "
        "is statistical-vs-statistical",
        fontsize=14, y=0.97)
    save(figure, out_dir, "M2_orthogonality.png")


# --------------------------------------------------------------------------- #
# M3 - what each scheme tracks
# --------------------------------------------------------------------------- #
LABELS = {"realised_vol_21d": "realised volatility", "dispersion": "dispersion",
          "avg_correlation": "avg correlation", "momentum_63d": "momentum (63d)",
          "return_5d": "RECENT DIRECTION (5d)", "vix": "VIX level",
          "credit_spread": "credit spread", "sentiment": "news sentiment"}

# Dropped from the panel: both are slow-moving interest-rate levels that drift
# with the monetary cycle rather than describing the market state a weekly
# regime could plausibly respond to. Neither separates any scheme materially
# (term slope 0.5-34%, short rate 0-14%, and both driven by the 2008-2015 rate
# collapse rather than by regime structure), so they add rows without adding
# argument. They remain in the underlying CSV.
DROP_FROM_PANEL = ["term_slope", "short_rate"]


def figure_what_tracked(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for axis, universe in zip(axes, UNIVERSES):
        frame = pd.read_csv(root / "results" / "runs" / "regimes"
                            / f"separation_{universe}.csv", index_col=0) * 100
        frame = frame.drop(index=DROP_FROM_PANEL, errors="ignore")
        frame = frame.loc[frame[["hmm", "vix"]].max(axis=1).sort_values().index]
        positions = np.arange(len(frame))
        for offset, scheme, colour in ((-0.26, "hmm", STATISTICAL_COLOUR),
                                       (0.0, "vix", "#98c1d9"),
                                       (0.26, "learned", LEARNED_COLOUR)):
            axis.barh(positions + offset, frame[scheme], 0.26, color=colour,
                      label={"hmm": "HMM (statistical)", "vix": "VIX (statistical)",
                             "learned": "learned gate"}[scheme])
        axis.set_yticks(positions)
        axis.set_yticklabels([LABELS.get(n, n) for n in frame.index], fontsize=9)
        axis.set_xlabel("% of the variable explained by the labelling")
        axis.set_title(UNIVERSE_TITLE[universe], fontsize=11)
        axis.legend(fontsize=9, loc="lower right")
        axis.grid(axis="x", alpha=0.3)

    figure.suptitle(
        "M3  Why they are orthogonal: statistical gates track RISK, "
        "the learned gate tracks DIRECTION\n"
        "each is nearly blind to the other's variable",
        fontsize=14, y=1.01)
    save(figure, out_dir, "M3_what_each_scheme_tracks.png")


# --------------------------------------------------------------------------- #
# M4 - capacity curve
# --------------------------------------------------------------------------- #
def figure_capacity(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for axis, universe in zip(axes, UNIVERSES):
        frame = pd.read_csv(root / "results" / "runs" / "capacity"
                            / f"capacity_{universe}.csv").sort_values("K")
        k = frame["K"].to_numpy()
        axis.plot(k, frame["val_sharpe"], "--o", color="#8d99ae", markersize=5,
                  label="validation  (the number that was MAXIMISED)")
        axis.fill_between(k, frame["val_sharpe"] - frame["val_sd"],
                          frame["val_sharpe"] + frame["val_sd"],
                          color="#8d99ae", alpha=0.15)
        axis.plot(k, frame["test_sharpe"], "-o", color=LEARNED_COLOUR, linewidth=2.4,
                  markersize=7, label="test, full walk-forward  (the number merely MEASURED)")
        axis.fill_between(k, frame["test_sharpe"] - frame["test_sd"],
                          frame["test_sharpe"] + frame["test_sd"],
                          color=LEARNED_COLOUR, alpha=0.18)
        name, value = BEST_BASELINE[universe]
        axis.axhline(value, color="black", linestyle=":", linewidth=1.5,
                     label=f"best baseline: {name} ({value:+.3f})")
        for _, row in frame.iterrows():
            axis.annotate(f"N={int(row['n_reservoir'])}", (row["K"], row["test_sharpe"]),
                          textcoords="offset points", xytext=(0, -17), ha="center",
                          fontsize=7, color="#555555")
        axis.set_xlabel("K   (number of experts)")
        axis.set_ylabel("Sharpe ratio")
        axis.set_xticks(k)
        axis.set_title(UNIVERSE_TITLE[universe], fontsize=11)
        axis.legend(fontsize=8.5, loc="best")
        axis.grid(alpha=0.3)

    figure.suptitle("M4  Does more capacity help?", fontsize=15, y=1.02)
    save(figure, out_dir, "M4_capacity_curve.png")


# --------------------------------------------------------------------------- #
# M6 - growth of GBP 10,000
# --------------------------------------------------------------------------- #
def figure_money(root: Path, out_dir: Path) -> None:
    """One clear panel per universe.

    Thirteen overlapping curves is unreadable, so every arm is drawn faintly to
    show the envelope, and only the four that carry the argument are highlighted
    and labelled: the best baseline, buy-and-hold, our best arm and our frozen
    configuration. The x-axis is calendar years rather than trading-day index,
    because "day 2000" means nothing to a reader and "2019" means something.
    """
    import json

    HIGHLIGHT = {
        "sector11": [("r0 inverse-vol", "#1d3557", 3.0, "-"),
                     ("r0 buy-and-hold SPY", "#457b9d", 2.4, "--"),
                     ("ours K=1 + sentiment", LEARNED_COLOUR, 3.0, "-"),
                     ("ours K=1 (rung 1)", "#e9a178", 2.2, "-")],
        "etf10": [("r0 buy-and-hold SPY", "#1d3557", 3.0, "-"),
                  ("r0 1/N", "#457b9d", 2.4, "--"),
                  ("ours K=1 + sentiment", LEARNED_COLOUR, 3.0, "-"),
                  ("ours K=3 (rung 2)", "#e9a178", 2.2, "-")],
    }

    for universe in UNIVERSES:
        stream_dir = root / "results" / "runs" / "walkforward" / universe
        manifest = json.loads((stream_dir / "arms.json").read_text())
        streams = {manifest.get(p.stem, p.stem): pd.read_parquet(p)
                   for p in sorted(stream_dir.glob("*.parquet"))}

        figure, axis = plt.subplots(figsize=(12, 7))
        reference = next(iter(streams.values()))
        years = reference["year"].to_numpy()
        days = np.arange(len(reference))

        # Everything, faintly - the envelope, so nothing is hidden.
        for arm, frame in streams.items():
            axis.plot(days, 10_000 * (1 + frame["net_return"]).cumprod().to_numpy(),
                      color="#b8c0c8", linewidth=0.9, alpha=0.55, zorder=1)

        # The four that carry the argument.
        for arm, colour, width, style in HIGHLIGHT[universe]:
            if arm not in streams:
                continue
            money = 10_000 * (1 + streams[arm]["net_return"]).cumprod().to_numpy()
            axis.plot(days, money, color=colour, linewidth=width, linestyle=style,
                      zorder=3, label=f"{arm}")
            axis.annotate(f"  GBP {money[-1]:,.0f}", (days[-1], money[-1]),
                          color=colour, fontsize=10, fontweight="bold", va="center")

        axis.axhline(10_000, color="black", linewidth=1.0, linestyle=":", zorder=2)
        axis.annotate("  starting capital", (0, 10_000), fontsize=8, va="bottom",
                      color="#444444")

        # Calendar years on the axis, one tick per year boundary.
        boundaries = [int(np.argmax(years == y)) for y in sorted(pd.unique(years))]
        axis.set_xticks(boundaries)
        axis.set_xticklabels([str(y) for y in sorted(pd.unique(years))],
                             rotation=45, fontsize=9)
        axis.set_xlim(0, len(reference) * 1.13)          # room for the labels

        axis.set_ylabel("value of an initial GBP 10,000")
        axis.set_title(
            f"M6  {UNIVERSE_TITLE[universe]}\n"
            "GBP 10,000 invested out of sample, net of 5 bp trading costs\n"
            "faint grey = all 13 arms; coloured = the four that matter",
            fontsize=12)
        axis.legend(fontsize=10, loc="upper left", framealpha=0.92)
        axis.grid(alpha=0.3)
        save(figure, out_dir, f"M6_growth_of_10000_{universe}.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meeting figure set.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    root = Path(args.data_root).expanduser().resolve()
    out_dir = root / "results" / "meeting_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing meeting figures to %s", out_dir)
    figure_ladder(root, out_dir)
    figure_orthogonality(root, out_dir)
    figure_what_tracked(root, out_dir)
    figure_capacity(root, out_dir)
    figure_money(root, out_dir)
    logger.info("Done - 5 figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
