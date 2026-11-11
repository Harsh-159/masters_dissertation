"""Exploratory data diagnostics - for understanding, NOT results of record.

    python scripts/make_data_diagnostics.py               # both universes
    python scripts/make_data_diagnostics.py --universe etf10

Deliberately isolated from the dissertation's figure pipeline
-------------------------------------------------------------
Everything here is quarantined so it can never be confused with, or collide
with, a result:

* it writes to ``results/diagnostics/``, never ``results/figures/``;
* figures are numbered **D1-D10**, never F1-F12;
* it is **not** registered in ``FIGURE_BUILDERS``, so ``make figures`` ignores it;
* it writes nothing to the registry, so it cannot affect the deflated-Sharpe
  trial count;
* it only ever *reads* ``data/processed``.

Nothing in the dissertation should cite these. They exist so that a human can
look at the data before trusting a model built on it - and so that the
motivation for the regime thesis can be seen rather than asserted.

The ten diagnostics
-------------------
=====  ===================================================================
D1     cumulative growth from 1.0 (linear and log)
D2     correlation heatmap of daily log returns
D3     return distributions against a fitted normal (fat tails)
D4     drawdown / underwater curves
D5     rolling volatility (regimes become visible)
D6     rolling mean cross-asset correlation (why regimes matter)
D7     the five gating signals over time (what the gate will see)
D8     signal correlation matrix (are the signals independent?)
D9     growth with causal VIX-tercile regime shading
D10    summary statistics table
=====  ===================================================================
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
from scipy import stats as scipy_stats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DIAGNOSTICS_DIR = PROJECT_ROOT / "results" / "diagnostics"

TRADING_DAYS_PER_YEAR = 252
VOLATILITY_WINDOW = 63       # about one quarter
CORRELATION_WINDOW = 63

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("diagnostics")

# A colour-blind-safe qualitative palette (Okabe-Ito), so up to 11 assets stay
# distinguishable without seaborn as a dependency.
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#000000", "#8C564B", "#7F7F7F", "#17BECF",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_universe(universe: str) -> dict[str, pd.DataFrame]:
    """Load every processed artifact for a universe."""
    artifacts = {}
    for name in (
        "prices_adj", "returns_log", "returns_simple",
        "signals", "availability", "risk_free",
    ):
        path = PROCESSED_DIR / f"{universe}_{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} not found. Run scripts/run_data.py first.")
        artifacts[name] = pd.read_parquet(path)
    return artifacts


def style_axis(axis, title: str, ylabel: str = "", xlabel: str = "") -> None:
    """Apply a consistent, uncluttered style to an axis."""
    axis.set_title(title, fontsize=11, loc="left")
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)


def save(figure, output_dir: Path, name: str) -> Path:
    """Save a figure and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    logger.info("  %s", path.name)
    return path


# --------------------------------------------------------------------------- #
# D1 - cumulative growth
# --------------------------------------------------------------------------- #
def d1_cumulative_growth(prices: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """Growth of 1.0, each asset normalised to its own first real price.

    Lines begin at inception, so XLRE starts in 2015 rather than being back
    filled to the start of the chart. The log panel makes percentage moves
    comparable across assets that ended up at very different levels.
    """
    normalised = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        first_valid = prices[ticker].first_valid_index()
        if first_valid is not None:
            normalised[ticker] = prices[ticker] / prices[ticker].loc[first_valid]

    figure, (linear_axis, log_axis) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for position, ticker in enumerate(normalised.columns):
        colour = PALETTE[position % len(PALETTE)]
        linear_axis.plot(normalised.index, normalised[ticker], lw=1.1, color=colour, label=ticker)
        log_axis.plot(normalised.index, normalised[ticker], lw=1.1, color=colour)

    linear_axis.axhline(1.0, color="grey", ls="--", lw=0.8)
    log_axis.axhline(1.0, color="grey", ls="--", lw=0.8)
    log_axis.set_yscale("log")

    style_axis(linear_axis, f"D1 · {universe} — growth of 1.0 (each line starts at its inception)",
               "growth (linear)")
    style_axis(log_axis, "same series, log scale", "growth (log)", "date")
    linear_axis.legend(ncol=6, fontsize=8, frameon=False, loc="upper left")
    figure.tight_layout()
    return save(figure, output_dir, "D1_cumulative_growth")


# --------------------------------------------------------------------------- #
# D2 - correlation heatmap
# --------------------------------------------------------------------------- #
def d2_correlation_heatmap(returns: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """Correlation of daily log returns - the diversification structure.

    Read it as: how much genuinely independent risk is on offer? A block of
    dark red means those assets are effectively one bet wearing several names.
    """
    correlation = returns.corr(min_periods=60)

    figure, axis = plt.subplots(figsize=(1 + 0.62 * len(correlation), 0.9 + 0.55 * len(correlation)))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)

    axis.set_xticks(range(len(correlation)))
    axis.set_yticks(range(len(correlation)))
    axis.set_xticklabels(correlation.columns, rotation=90, fontsize=8)
    axis.set_yticklabels(correlation.columns, fontsize=8)

    for row in range(len(correlation)):
        for column in range(len(correlation)):
            value = correlation.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=6.5, color="white" if abs(value) > 0.55 else "black")

    axis.set_title(f"D2 · {universe} — daily log-return correlation", fontsize=11, loc="left")
    figure.colorbar(image, ax=axis, shrink=0.75, label="correlation")
    figure.tight_layout()
    return save(figure, output_dir, "D2_correlation_heatmap")


# --------------------------------------------------------------------------- #
# D3 - return distributions
# --------------------------------------------------------------------------- #
def d3_return_distributions(returns: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """Each asset's return histogram against a fitted normal curve.

    The gap between the bars and the red curve at the extremes is the reason
    this project optimises CVaR rather than variance: a normal distribution
    badly understates how often very large losses actually occur.
    """
    tickers = [t for t in returns.columns if returns[t].notna().sum() > 100]
    n_columns = min(4, len(tickers))
    n_rows = int(np.ceil(len(tickers) / n_columns))

    figure, axes = plt.subplots(n_rows, n_columns, figsize=(3.6 * n_columns, 2.7 * n_rows),
                                squeeze=False)

    for position, ticker in enumerate(tickers):
        axis = axes[position // n_columns][position % n_columns]
        series = returns[ticker].dropna()

        axis.hist(series, bins=90, density=True, color="#0072B2", alpha=0.55)
        grid = np.linspace(series.min(), series.max(), 300)
        axis.plot(grid, scipy_stats.norm.pdf(grid, series.mean(), series.std(ddof=1)),
                  color="#D55E00", lw=1.3, ls="--")

        excess_kurtosis = scipy_stats.kurtosis(series, fisher=True, bias=False)
        skewness = scipy_stats.skew(series, bias=False)
        axis.set_title(ticker, fontsize=9, loc="left")
        axis.text(0.02, 0.95, f"skew {skewness:+.2f}\nex.kurt {excess_kurtosis:+.1f}",
                  transform=axis.transAxes, va="top", fontsize=7,
                  bbox={"boxstyle": "round", "fc": "white", "ec": "grey", "alpha": 0.85})
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.2)

    for empty in range(len(tickers), n_rows * n_columns):
        axes[empty // n_columns][empty % n_columns].axis("off")

    figure.suptitle(
        f"D3 · {universe} — daily log returns vs a fitted normal (dashed). "
        "Excess kurtosis >> 0 means fat tails.",
        fontsize=11, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return save(figure, output_dir, "D3_return_distributions")


# --------------------------------------------------------------------------- #
# D4 - drawdowns
# --------------------------------------------------------------------------- #
def d4_drawdowns(prices: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """Underwater curves: percentage below each asset's running peak."""
    figure, axis = plt.subplots(figsize=(13, 5.5))

    for position, ticker in enumerate(prices.columns):
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue
        drawdown = series / series.cummax() - 1.0
        axis.plot(drawdown.index, drawdown, lw=0.9,
                  color=PALETTE[position % len(PALETTE)], label=ticker)

    axis.axhline(0.0, color="grey", lw=0.8)
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    style_axis(axis, f"D4 · {universe} — drawdown from running peak", "drawdown", "date")
    axis.legend(ncol=6, fontsize=8, frameon=False, loc="lower left")
    figure.tight_layout()
    return save(figure, output_dir, "D4_drawdowns")


# --------------------------------------------------------------------------- #
# D5 - rolling volatility
# --------------------------------------------------------------------------- #
def d5_rolling_volatility(returns: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """Annualised rolling volatility - the most direct visual evidence of regimes.

    Calm stretches punctuated by violent ones is exactly the structure a
    regime-switching model is designed to exploit. If this chart were flat,
    the premise of the dissertation would be in trouble.
    """
    rolling_volatility = (
        returns.rolling(VOLATILITY_WINDOW).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    figure, axis = plt.subplots(figsize=(13, 5.5))
    for position, ticker in enumerate(rolling_volatility.columns):
        axis.plot(rolling_volatility.index, rolling_volatility[ticker], lw=0.8,
                  color=PALETTE[position % len(PALETTE)], alpha=0.75, label=ticker)

    cross_sectional_median = rolling_volatility.median(axis=1)
    axis.plot(cross_sectional_median.index, cross_sectional_median, lw=2.2, color="black",
              label="median across assets")

    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    style_axis(axis,
               f"D5 · {universe} — {VOLATILITY_WINDOW}-day rolling annualised volatility",
               "annualised volatility", "date")
    axis.legend(ncol=6, fontsize=8, frameon=False, loc="upper left")
    figure.tight_layout()
    return save(figure, output_dir, "D5_rolling_volatility")


# --------------------------------------------------------------------------- #
# D6 - rolling cross-asset correlation
# --------------------------------------------------------------------------- #
def rolling_mean_pairwise_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """Average pairwise correlation across all assets, on a rolling window.

    Computed directly rather than via ``rolling().corr()`` so that windows
    containing not-yet-launched assets simply use the assets that exist, which
    keeps the early sector history usable.
    """
    values = returns.to_numpy()
    n_days, _ = values.shape
    output = np.full(n_days, np.nan)

    for end in range(window, n_days + 1):
        block = values[end - window : end]
        usable_columns = ~np.isnan(block).any(axis=0)
        if usable_columns.sum() < 2:
            continue
        correlation = np.corrcoef(block[:, usable_columns], rowvar=False)
        upper_triangle = np.triu_indices(usable_columns.sum(), k=1)
        output[end - 1] = np.nanmean(correlation[upper_triangle])

    return pd.Series(output, index=returns.index, name="mean_pairwise_correlation")


def d6_rolling_correlation(returns: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """The single most thesis-relevant diagnostic.

    Average pairwise correlation is not stable - it *spikes during crises*.
    Diversification therefore fails exactly when it is needed most, which is
    precisely the failure a regime-aware allocator is meant to anticipate. A
    fixed allocation cannot respond to this chart; a gated one can.
    """
    mean_correlation = rolling_mean_pairwise_correlation(returns, CORRELATION_WINDOW)

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(mean_correlation.index, mean_correlation, lw=1.1, color="#0072B2")
    axis.fill_between(mean_correlation.index, 0, mean_correlation, alpha=0.15, color="#0072B2")

    overall_mean = mean_correlation.mean()
    axis.axhline(overall_mean, color="#D55E00", ls="--", lw=1.0,
                 label=f"full-sample mean = {overall_mean:.2f}")

    style_axis(axis,
               f"D6 · {universe} — {CORRELATION_WINDOW}-day mean pairwise correlation "
               "(diversification disappears in crises)",
               "mean pairwise correlation", "date")
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    return save(figure, output_dir, "D6_rolling_correlation")


# --------------------------------------------------------------------------- #
# D7 - the gating signals
# --------------------------------------------------------------------------- #
def d7_signals(signals: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """The five standardised signals the gating network will actually see.

    These are already publication-lagged and z-scored, so this is literally the
    gate's input view. Worth studying: the spikes and inversions you can see by
    eye are the raw material the model has to identify regimes from.
    """
    available = [c for c in signals.columns if signals[c].notna().any()]
    figure, axes = plt.subplots(len(available), 1, figsize=(13, 1.9 * len(available)),
                                sharex=True, squeeze=False)

    for position, column in enumerate(available):
        axis = axes[position][0]
        series = signals[column]
        axis.plot(series.index, series, lw=0.9, color=PALETTE[position % len(PALETTE)])
        axis.axhline(0.0, color="grey", lw=0.7, ls="--")
        axis.fill_between(series.index, 0, series, alpha=0.18,
                          color=PALETTE[position % len(PALETTE)])
        style_axis(axis, f"{column}  (z-score, publication-lagged)", "z")

    axes[-1][0].set_xlabel("date", fontsize=9)
    figure.suptitle(
        f"D7 · {universe} — the gating signal matrix (what the gate sees)",
        fontsize=11, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return save(figure, output_dir, "D7_signals")


# --------------------------------------------------------------------------- #
# D8 - signal correlation
# --------------------------------------------------------------------------- #
def d8_signal_correlation(signals: pd.DataFrame, output_dir: Path, universe: str) -> Path:
    """How independent are the five signals?

    If two signals correlate at 0.9 the gate has less genuinely independent
    information than the column count suggests - useful to know before
    interpreting which signal the gate "chose" to rely on.
    """
    correlation = signals.corr(min_periods=100)

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)

    axis.set_xticks(range(len(correlation)))
    axis.set_yticks(range(len(correlation)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(correlation.columns, fontsize=8)

    for row in range(len(correlation)):
        for column in range(len(correlation)):
            value = correlation.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=8, color="white" if abs(value) > 0.55 else "black")

    axis.set_title(f"D8 · {universe} — signal correlation", fontsize=11, loc="left")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return save(figure, output_dir, "D8_signal_correlation")


# --------------------------------------------------------------------------- #
# D9 - VIX-tercile regime shading
# --------------------------------------------------------------------------- #
def d9_vix_regime_shading(
    prices: pd.DataFrame, signals: pd.DataFrame, output_dir: Path, universe: str
) -> Path:
    """Equal-weight growth, shaded by causal VIX terciles.

    This previews the rung-4 *statistical gating* baseline that the learned gate
    must beat. Terciles are computed on an **expanding** window (data up to t-1
    only), exactly as the baseline will compute them - not on the full sample,
    which would be look-ahead and would make the regimes look far cleaner than
    they can legitimately be made to look.
    """
    vix = signals["vix"].dropna()
    if vix.empty:
        raise ValueError("VIX signal is unavailable; cannot draw D9.")

    lower_tercile = vix.expanding(min_periods=252).quantile(1 / 3)
    upper_tercile = vix.expanding(min_periods=252).quantile(2 / 3)

    regime = pd.Series("mid", index=vix.index, dtype=object)
    regime[vix <= lower_tercile] = "low"
    regime[vix >= upper_tercile] = "high"
    regime[lower_tercile.isna()] = "warmup"

    equal_weight_growth = (
        prices.divide(prices.apply(lambda column: column.loc[column.first_valid_index()]))
        .mean(axis=1)
    )

    figure, axis = plt.subplots(figsize=(13, 5.5))
    axis.plot(equal_weight_growth.index, equal_weight_growth, lw=1.3, color="black",
              label="equal-weight growth of 1.0")

    shading = {"low": "#009E73", "mid": "#F0E442", "high": "#D55E00"}
    for label, colour in shading.items():
        mask = (regime == label).reindex(equal_weight_growth.index, fill_value=False)
        axis.fill_between(equal_weight_growth.index, 0, equal_weight_growth.max() * 1.05,
                          where=mask.to_numpy(), color=colour, alpha=0.16, step="mid",
                          label=f"VIX tercile: {label}")

    axis.set_ylim(0, equal_weight_growth.max() * 1.05)
    style_axis(axis,
               f"D9 · {universe} — equal-weight growth shaded by CAUSAL VIX terciles "
               "(previews the rung-4 statistical gate)",
               "growth of 1.0", "date")
    axis.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
    figure.tight_layout()
    return save(figure, output_dir, "D9_vix_regime_shading")


# --------------------------------------------------------------------------- #
# D10 - summary statistics
# --------------------------------------------------------------------------- #
def d10_summary_table(
    prices: pd.DataFrame, returns: pd.DataFrame, risk_free: pd.Series,
    output_dir: Path, universe: str,
) -> Path:
    """Per-asset summary statistics, written as both a figure and a CSV."""
    rows = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        if len(series) < 2:
            continue
        excess = (series - risk_free.reindex(series.index)).dropna()
        price_series = prices[ticker].dropna()

        rows.append({
            "ticker": ticker,
            "n_obs": len(series),
            "ann_return": series.mean() * TRADING_DAYS_PER_YEAR,
            "ann_vol": series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR),
            "excess_sharpe": (
                excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                if excess.std(ddof=1) > 0 else np.nan
            ),
            "skew": scipy_stats.skew(series, bias=False),
            "excess_kurtosis": scipy_stats.kurtosis(series, fisher=True, bias=False),
            "max_drawdown": (price_series / price_series.cummax() - 1.0).min(),
        })

    summary = pd.DataFrame(rows).set_index("ticker")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "D10_summary_statistics.csv")

    display = summary.copy()
    for column, fmt in (
        ("ann_return", "{:.2%}"), ("ann_vol", "{:.2%}"), ("excess_sharpe", "{:.2f}"),
        ("skew", "{:+.2f}"), ("excess_kurtosis", "{:+.1f}"),
        ("max_drawdown", "{:.1%}"), ("n_obs", "{:.0f}"),
    ):
        display[column] = display[column].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))

    figure, axis = plt.subplots(figsize=(11, 0.42 * len(display) + 1.4))
    axis.axis("off")
    table = axis.table(cellText=display.reset_index().values,
                       colLabels=["ticker", *display.columns],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for column_index in range(display.shape[1] + 1):
        table[0, column_index].set_facecolor("#0072B2")
        table[0, column_index].set_text_props(color="white", weight="bold")

    axis.set_title(f"D10 · {universe} — summary statistics (Sharpe is excess, over r_f)",
                   fontsize=11, loc="left")
    figure.tight_layout()
    return save(figure, output_dir, "D10_summary_statistics")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_for_universe(universe: str) -> int:
    """Build every diagnostic for one universe. Returns the count written."""
    logger.info("=== %s ===", universe)
    artifacts = load_universe(universe)

    prices = artifacts["prices_adj"]
    returns_log = artifacts["returns_log"]
    signals = artifacts["signals"]
    risk_free = artifacts["risk_free"]["r_f"]
    output_dir = DIAGNOSTICS_DIR / universe

    d1_cumulative_growth(prices, output_dir, universe)
    d2_correlation_heatmap(returns_log, output_dir, universe)
    d3_return_distributions(returns_log, output_dir, universe)
    d4_drawdowns(prices, output_dir, universe)
    d5_rolling_volatility(returns_log, output_dir, universe)
    d6_rolling_correlation(returns_log, output_dir, universe)
    d7_signals(signals, output_dir, universe)
    d8_signal_correlation(signals, output_dir, universe)
    d9_vix_regime_shading(prices, signals, output_dir, universe)
    d10_summary_table(prices, returns_log, risk_free, output_dir, universe)
    return 10


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Exploratory data diagnostics (not results).")
    parser.add_argument("--universe", default="all", help="etf10, sector11, or all.")
    args = parser.parse_args(argv)

    universes = ["etf10", "sector11"] if args.universe == "all" else [args.universe]

    total = 0
    for universe in universes:
        total += build_for_universe(universe)

    logger.info("Wrote %d diagnostics into %s", total, DIAGNOSTICS_DIR)
    logger.info("These are for understanding only - never cite them as results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
