"""Publication-quality versions of the two headline results figures.

Writes to results/analysis_figures/ as both PDF (vector, for the report) and
PNG at 300 dpi (for slides):

    fig_ladder_sharpe.{pdf,png}   the 13-arm ladder on both universes
    fig_growth_10k.{pdf,png}      growth of GBP 10,000 out of sample

Encoding, and why
-----------------
Three groups, two categorical hues plus recessive grey:

    grey    classical baselines - the reference the argument is measured against,
            not a series with its own identity
    orange  our decision-focused arms - the subject of the thesis
    blue    alternative learned comparisons (LSTM encoder, statistical gates)

blue/orange is the standard colour-vision-safe pair: worst adjacent separation
is dE 24.7 under protanopia against a threshold of 8, and the two differ in
lightness so the figures survive greyscale printing. Within a hue, identity is
carried by line style rather than a second shade, so nothing depends on
distinguishing two similar colours.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "walkforward"
OUT = ROOT / "results" / "analysis_figures"

GREY_BAR = "#b8b8b6"
GREY_LINE = "#d5d5d2"
ORANGE = "#eb6834"
BLUE = "#2a78d6"
INK = "#111111"
INK_SOFT = "#52514e"
RULE = "#e2e2df"

UNIVERSES = {
    "sector11": "sector11  ·  18 folds, 2007–2024",
    "etf10": "etf10  ·  10 folds, 2015–2024",
}

# grouped so colour and position agree; ladder progression kept inside "ours"
LADDER = [
    ("r0 1/N", "base"),
    ("r0 buy-and-hold SPY", "base"),
    ("r0 inverse-vol", "base"),
    ("r0 Ledoit-Wolf MV", "base"),
    ("K=1 (rung 1)", "ours"),
    ("K=3 (rung 2)", "ours"),
    ("3a sent-gate (K=3)", "ours"),
    ("3b sent-experts (K=3)", "ours"),
    ("3c sent-both (K=3)", "ours"),
    ("K=1 + sentiment", "ours"),
    ("LSTM", "alt"),
    ("r4 HMM gate", "alt"),
    ("r4 VIX gate", "alt"),
]

PRETTY = {
    "r0 1/N": "1/N",
    "r0 buy-and-hold SPY": "buy-and-hold SPY",
    "r0 inverse-vol": "inverse-volatility",
    "r0 Ledoit-Wolf MV": "Ledoit–Wolf MV",
    "K=1 (rung 1)": "ours, K=1  (rung 1)",
    "K=3 (rung 2)": "ours, K=3  (rung 2)",
    "3a sent-gate (K=3)": "3a  sentiment into gate",
    "3b sent-experts (K=3)": "3b  sentiment into experts",
    "3c sent-both (K=3)": "3c  sentiment into both",
    "K=1 + sentiment": "ours, K=1 + sentiment",
    "LSTM": "LSTM (trained encoder)",
    "r4 HMM gate": "HMM gate",
    "r4 VIX gate": "VIX gate",
}

FILL = {"base": GREY_BAR, "ours": ORANGE, "alt": BLUE}

# The best capacity on the TEST curve, shown under the ladder as a reference.
# It is drawn hollow, below a rule, because it is not a selectable
# configuration: the K was read off the test split, and on sector11 validation
# points the other way (r = -0.477). Solid bars are what a procedure could have
# chosen; this one is what capacity could have bought.
BEST_K = {"sector11": 2, "etf10": 5}


def best_capacity(universe: str) -> tuple[float, float, int]:
    """Concatenated test Sharpe at the best-capacity K, over five seeds.

    Read from the state-only column of the VIX-on-best-K run, which re-scored
    the capacity winner at seeds 0-4. The capacity CSV itself carries the same
    configuration at three seeds (0.505 / 0.411); five is quoted so the row is
    on the same seed count as the rest of the ladder.
    """
    k = BEST_K[universe]
    frame = pd.read_csv(RUNS.parent / "vix_bestk" / f"vix_bestk_{universe}_K{k}.csv")
    return float(frame["state"].mean()), float(frame["state"].std(ddof=1)), k


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SOFT,
        "axes.titlesize": 9,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "grid.color": RULE,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def load_summary(universe: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS / f"summary_{universe}.csv")
    df["arm"] = df["arm"].str.strip()
    return df.set_index("arm")


def ladder_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3))

    for ax, (universe, subtitle) in zip(axes, UNIVERSES.items()):
        summary = load_summary(universe)
        rows = [(PRETTY[a], summary.loc[a, "sharpe"], summary.loc[a, "sharpe_sd"], g)
                for a, g in LADDER]
        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]
        errors = [r[2] for r in rows]
        groups = [r[3] for r in rows]

        ref_value, ref_sd, ref_k = best_capacity(universe)
        labels.append(f"best capacity  (K={ref_k})")
        values.append(ref_value)
        errors.append(ref_sd)
        groups.append("ref")

        y = list(range(len(rows) + 1))[::-1]  # first row at the top
        ax.barh(y[:-1], values[:-1], height=0.62,
                color=[FILL[g] for g in groups[:-1]], linewidth=0)
        # hollow, so it cannot be read as another competitor in the ladder
        ax.barh(y[-1:], values[-1:], height=0.62, facecolor="none",
                edgecolor=ORANGE, hatch="////", linewidth=0.9)
        ax.errorbar(values, y, xerr=errors, fmt="none",
                    ecolor=INK_SOFT, elinewidth=0.8, capsize=2, capthick=0.8)
        ax.axhline(0.5, color=RULE, linewidth=0.8)

        best = summary.loc[[a for a, g in LADDER if g == "base"], "sharpe"].max()
        ax.axvline(best, color=INK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)

        xmax = max(values) * 1.32
        for yi, v in zip(y, values):
            ax.text(xmax * 0.985, yi, f"{v:.3f}",
                    va="center", ha="right", fontsize=7.2, color=INK_SOFT)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.8)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Sharpe ratio  (out of sample)")
        ax.set_title(subtitle, loc="left", color=INK, pad=8)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FILL[k]) for k in ("base", "ours", "alt")]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=ORANGE,
                                 hatch="////", linewidth=0.9))
    fig.legend(handles,
               ["classical baselines", "decision-focused arms (ours)",
                "learned comparisons", "best capacity"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.045),
               fontsize=8, handlelength=1.1, handleheight=0.9, columnspacing=1.6)

    fig.subplots_adjust(wspace=0.85)
    save(fig, "fig_ladder_sharpe")


def growth_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))

    for ax, (universe, subtitle) in zip(axes, UNIVERSES.items()):
        folder = RUNS / universe
        names = json.loads((folder / "arms.json").read_text())
        summary = load_summary(universe)

        curves = {}
        for stem, label in names.items():
            stream = pd.read_parquet(folder / f"{stem}.parquet")
            curves[label.strip()] = 10_000 * (1 + stream["net_return"]).cumprod()

        panel = pd.DataFrame(curves)
        best_base = panel[[_match(names, a) for a, g in LADDER
                           if g == "base"]].iloc[-1].idxmax()
        best_ours = panel[[_match(names, a) for a, g in LADDER
                           if g == "ours"]].iloc[-1].idxmax()

        focus = {best_base: ("best baseline · " + _short(best_base), BLUE),
                 best_ours: ("ours · " + _short(best_ours), ORANGE)}

        # everything else becomes one shaded min-max band: a range reads as
        # context, ten overlapping lines read as noise
        rest = panel.drop(columns=list(focus))
        weekly = slice(None, None, 5)  # 4,530 daily points is finer than print resolves
        ax.fill_between(rest.index[weekly], rest.min(axis=1)[weekly],
                        rest.max(axis=1)[weekly],
                        color=GREY_LINE, alpha=0.55, linewidth=0, zorder=1)

        ends = []
        for label, (pretty, colour) in focus.items():
            series = panel[label]
            ax.plot(series.index[weekly], series.values[weekly], color=colour,
                    linewidth=1.3, zorder=3, label=pretty, solid_joinstyle="round")
            ends.append((series.iloc[-1], colour, len(series) - 1))

        # nudge the end labels apart: on a log axis a fixed ratio is a fixed gap
        ends.sort(key=lambda e: -e[0])
        placed: list[float] = []
        for value, colour, xi in ends:
            at = value
            for prior in placed:
                at = min(at, prior / 1.085)
            placed.append(at)
            ax.annotate(f"£{value:,.0f}", xy=(xi, at),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=7.2, color=colour, va="center", fontweight="bold")

        ax.axhline(10_000, color=INK_SOFT, linewidth=0.7, zorder=0)
        ax.set_yscale("log")
        lo, hi = panel.to_numpy().min(), panel.to_numpy().max()
        ticks = [t for t in (5_000, 10_000, 20_000, 30_000, 40_000, 60_000)
                 if lo * 0.92 <= t <= hi * 1.25]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"£{t // 1000}k" for t in ticks])

        years = stream["year"].to_numpy()
        marks = sorted({int(y) for y in years})[::3]
        ax.set_xticks([int((years == y).argmax()) for y in marks])
        ax.set_xticklabels([str(y) for y in marks])

        ax.set_title(subtitle, loc="left", color=INK, pad=6)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.margins(x=0.14)
        ax.fill_between([], [], [], color=GREY_LINE, alpha=0.55,
                        label="range of the other 11 arms")
        ax.legend(loc="upper left", fontsize=7.2, labelspacing=0.35,
                  handlelength=1.6, borderpad=0.2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("value of an initial £10,000  (log scale)")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "fig_growth_10k")


def _short(label: str) -> str:
    """arms.json labels, trimmed for a direct label on the plot."""
    return (label.replace("r0 ", "").replace("ours ", "")
                 .replace(" (rung 1)", "").replace(" (rung 2)", "").strip())


def _match(names: dict, arm: str) -> str:
    """summary.csv and arms.json spell a few arms differently."""
    wanted = arm.replace("ours ", "").strip()
    for label in names.values():
        if label.replace("ours ", "").strip() == wanted:
            return label.strip()
    return arm



def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


REGIMES = ROOT / "results" / "runs" / "regimes"
AQUA = "#199e70"
SCHEMES = {"learned": ("our gate", ORANGE),
           "hmm": ("HMM", BLUE),
           "vix": ("VIX terciles", AQUA)}

# fixed order across panels so the two are directly comparable; the variable
# that separates the schemes goes first
VARIABLES = [
    ("return_5d", "recent direction, 5d"),
    ("realised_vol_21d", "realised volatility"),
    ("vix", "VIX level"),
    ("dispersion", "dispersion"),
    ("credit_spread", "credit spread"),
    ("avg_correlation", "average correlation"),
    ("momentum_63d", "momentum, 63d"),
    ("sentiment", "news sentiment"),
    ("term_slope", "term slope"),
    ("short_rate", "short rate"),
]


def variables_figure() -> None:
    """What each labelling actually separates - eta squared per variable."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), sharex=True)

    for ax, (universe, subtitle) in zip(axes, UNIVERSES.items()):
        sep = pd.read_csv(REGIMES / f"separation_{universe}.csv", index_col=0)
        rows = [v for v, _ in VARIABLES]
        offsets = {"learned": 0.26, "hmm": 0.0, "vix": -0.26}

        for scheme, (name, colour) in SCHEMES.items():
            ax.barh([len(rows) - 1 - i + offsets[scheme] for i in range(len(rows))],
                    [sep.loc[v, scheme] * 100 for v in rows],
                    height=0.24, color=colour, linewidth=0, label=name)

        ax.set_yticks(range(len(rows))[::-1])
        ax.set_yticklabels([label for _, label in VARIABLES], fontsize=7.6)
        ax.set_xlabel("variance explained by the labelling  (η², %)")
        ax.set_title(subtitle, loc="left", color=INK, pad=6)
        ax.axhspan(len(rows) - 1.5, len(rows) - 0.5, color="#f4f4f2", zorder=0)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[0].legend(loc="lower right", fontsize=7.6, labelspacing=0.35)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_regime_variables")


def ribbons_figure() -> None:
    """Five years of weekly labels, one row per scheme."""
    full = pd.read_parquet(REGIMES / "state_and_labels_sector11.parquet")
    labels = full.tail(260)          # what is drawn

    order_by = {"learned": "return_5d", "hmm": "realised_vol_21d", "vix": "vix"}
    meaning = {"learned": ("state 1", "state 2", "state 3"),
               "hmm": ("calm", "stressed", "crisis"),
               "vix": ("low VIX", "mid", "high VIX")}

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 3.4))
    for ax, (scheme, (name, colour)) in zip(axes, SCHEMES.items()):
        col = f"label_{scheme}"
        # regimes within a scheme are ordered, so an ordinal ramp is the right
        # encoding - light to dark along the variable the scheme actually tracks
        rank = labels.groupby(col)[order_by[scheme]].mean().rank().astype(int) - 1
        ramp = _ramp(colour)

        run_start, values = 0, labels[col].to_numpy()
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[run_start]:
                ax.axvspan(run_start, i, color=ramp[rank[values[run_start]]], linewidth=0)
                run_start = i

        every = full[col].to_numpy()          # statistics use the full test period
        changed = every[1:] != every[:-1]
        switches = changed.mean()
        dwell = len(every) / max(changed.sum(), 1)
        ax.set_title(f"{name}   ·   mean dwell {dwell:.1f} weeks   ·   switches {switches:.0%}",
                     loc="left", color=INK, fontsize=8.5, pad=4)
        ax.set_xlim(0, len(values))
        ax.set_yticks([])
        ax.set_xticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        for step, text in zip(ramp, meaning[scheme]):
            ax.plot([], [], color=step, linewidth=6, label=text)
        ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7,
                  labelspacing=0.3, handlelength=1.0, borderpad=0.15)

    axes[-1].set_xlabel("260 consecutive weekly decisions, sector11 test period",
                        fontsize=8)
    fig.subplots_adjust(hspace=1.0, right=0.83)
    save(fig, "fig_regime_ribbons")


def capacity_figure() -> None:
    """The capacity curve, and the anti-correlation that is the actual finding."""
    cap = pd.read_csv(RUNS.parent / "capacity" / "capacity_sector11.csv")
    best = load_summary("sector11").loc[
        [a for a, g in LADDER if g == "base"], "sharpe"].max()
    r = cap["val_sharpe"].corr(cap["test_sharpe"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.2),
                                 gridspec_kw={"width_ratios": [1.6, 1]})

    ax.plot(cap["K"], cap["val_sharpe"], color=INK_SOFT, linewidth=1.2,
            linestyle=(0, (4, 2)), marker="o", markersize=3.5,
            label="validation  (maximised)")
    ax.plot(cap["K"], cap["test_sharpe"], color=ORANGE, linewidth=1.5,
            marker="o", markersize=4, label="test  (measured)")
    ax.axhline(best, color=INK, linewidth=0.9, linestyle=(0, (1, 2)))
    ax.text(0.99, best, f"best baseline {best:.3f} ", fontsize=7, color=INK,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("K   (number of experts)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("sector11  ·  reservoir searched inside each K", loc="left",
                 color=INK, pad=6)
    ax.set_xticks(cap["K"])
    ax.legend(loc="lower right", fontsize=7.4, labelspacing=0.35)
    ax.grid(True)
    ax.set_axisbelow(True)

    bx.scatter(cap["val_sharpe"], cap["test_sharpe"], s=26, color=ORANGE,
               zorder=3, linewidth=0)
    for _, row in cap.iterrows():
        bx.annotate(f"K={int(row['K'])}", (row["val_sharpe"], row["test_sharpe"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=6.6,
                    color=INK_SOFT)
    bx.set_xlabel("validation Sharpe")
    bx.set_ylabel("test Sharpe")
    bx.set_title(f"selection points the wrong way   r = {r:+.3f}", loc="left",
                 color=INK, pad=6)
    bx.grid(True)
    bx.set_axisbelow(True)

    for panel in (ax, bx):
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)

    fig.subplots_adjust(wspace=0.34)
    save(fig, "fig_capacity_curve")


def agreement_figure() -> None:
    """Where the three labellings put the same weeks.

    A co-occurrence table is a magnitude, so it gets a single-hue sequential
    ramp - not the categorical colours used elsewhere. Meaning is carried by
    the AMI in each panel title, and by which panel shows a diagonal.
    """
    labels = pd.read_parquet(REGIMES / "state_and_labels_sector11.parquet")
    order_by = {"learned": "return_5d", "hmm": "realised_vol_21d", "vix": "vix"}
    names = {"learned": ("state 1", "state 2", "state 3"),
             "hmm": ("calm", "stressed", "crisis"),
             "vix": ("low", "mid", "high")}

    ranked = {}
    for scheme, variable in order_by.items():
        col = f"label_{scheme}"
        rank = labels.groupby(col)[variable].mean().rank().astype(int) - 1
        ranked[scheme] = labels[col].map(rank)

    pairs = [("learned", "hmm"), ("learned", "vix"), ("hmm", "vix")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.85))

    for ax, (a, b) in zip(axes, pairs):
        table = pd.crosstab(ranked[a], ranked[b], normalize=True) * 100
        table = table.reindex(index=range(3), columns=range(3), fill_value=0.0)
        ami = adjusted_mutual_info_score(ranked[a], ranked[b])

        ax.imshow(table.to_numpy(), cmap="Greys", vmin=0, vmax=40)
        for i in range(3):
            for j in range(3):
                v = table.iat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="#FFFFFF" if v > 22 else INK)

        control = (a, b) == ("hmm", "vix")
        show = {"learned": "our gate", "hmm": "HMM", "vix": "VIX"}
        ax.set_title(f"{show[a]} vs {show[b]}      AMI {ami:+.3f}", loc="left", pad=8,
                     color=INK, fontsize=9, fontweight="bold" if control else "normal")
        ax.set_xticks(range(3), names[b], fontsize=7.5)
        ax.set_yticks(range(3), names[a], fontsize=7.5)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_edgecolor(INK if control else RULE)
            side.set_linewidth(1.6 if control else 0.8)

    axes[2].text(1.0, -0.28, "positive control: two schemes built differently that still agree",
                 transform=axes[2].transAxes, ha="right", fontsize=7.5, color=INK_SOFT)
    axes[0].text(0.0, -0.28, "cells are % of weeks",
                 transform=axes[0].transAxes, ha="left", fontsize=7.5, color=INK_SOFT)

    fig.subplots_adjust(wspace=0.5)
    save(fig, "fig_regime_agreement")


def _ramp(hex_colour: str) -> list[str]:
    """Three ordinal steps of one hue, light to dark."""
    base = tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return ["#%02x%02x%02x" % tuple(int(c + (255 - c) * t) for c in base)
            for t in (0.72, 0.38, 0.0)]


if __name__ == "__main__":
    style()
    ladder_figure()
    growth_figure()
    variables_figure()
    ribbons_figure()
    capacity_figure()
    agreement_figure()
