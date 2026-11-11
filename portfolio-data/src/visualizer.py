"""the summary stats table and two figures per universe."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # write files, never try to open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .processor import Panels  # noqa: E402

log = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.year = cfg.trading_days_per_year
        self.rf = cfg.risk_free_rate_pct / 100.0
        plt.style.use(f"seaborn-v0_8-{cfg.palette}")

    def summary_stats(self, panels: Panels) -> pd.DataFrame:
        """per asset annual return, volatility, sharpe, shape of the distribution."""
        rows = []
        n_days = max(len(panels.prices), 1)
        for ticker in panels.tickers:
            returns = panels.returns[ticker].dropna()
            prices = panels.prices[ticker].dropna()
            if len(returns) < 2:
                rows.append({"ticker": ticker, "n_obs": len(returns)})
                continue
            annual_return = float(returns.mean() * self.year)
            annual_vol = float(returns.std(ddof=1) * np.sqrt(self.year))
            rows.append({
                "ticker": ticker,
                "ann_return": annual_return,
                "ann_vol": annual_vol,
                "sharpe": (annual_return - self.rf) / annual_vol if annual_vol > 0 else np.nan,
                "skew": float(returns.skew()),
                "excess_kurtosis": float(returns.kurt()),
                "max_drawdown": float((prices / prices.cummax() - 1.0).min()),
                "n_obs": len(returns),
                "pct_missing": 100.0 * (1.0 - panels.prices[ticker].notna().sum() / n_days),
            })
        return pd.DataFrame(rows).set_index("ticker")

    def render(self, panels: Panels, summary: pd.DataFrame) -> None:
        out_dir = self.cfg.plots_dir / panels.name
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save(self._cumulative_growth(panels), out_dir / "cumulative_growth.png")
        self._save(self._summary_table(summary, panels.name), out_dir / "summary_table.png")
        log.info("%s: 2 figures written to %s", panels.name, out_dir)

    def _cumulative_growth(self, panels: Panels) -> plt.Figure:
        """growth of 1.0, linear on top and log underneath.

        each line starts at that asset's own first price, so a fund launched in
        2015 begins in 2015 rather than pretending it existed earlier. the log
        panel is there because over nineteen years the linear one squashes
        everything that is not the best performer.
        """
        prices = panels.prices
        normed = pd.DataFrame(index=prices.index)
        for ticker in panels.tickers:
            first = prices[ticker].first_valid_index()
            if first is not None:
                normed[ticker] = prices[ticker] / prices[ticker].loc[first]

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        for ticker in normed.columns:
            top.plot(normed.index, normed[ticker], lw=1.1, label=ticker)
            bottom.plot(normed.index, normed[ticker], lw=1.1)
        top.set_ylabel("growth of 1.0")
        top.set_title(f"cumulative growth, {panels.name} (each line starts at its own inception)")
        top.axhline(1.0, color="grey", lw=0.8, ls="--")
        top.legend(loc="upper left", fontsize=7, ncol=1 if normed.shape[1] <= 12 else 2)
        bottom.set_yscale("log")
        bottom.set_ylabel("growth of 1.0, log scale")
        bottom.set_xlabel("date")
        bottom.axhline(1.0, color="grey", lw=0.8, ls="--")
        fig.tight_layout()
        return fig

    def _summary_table(self, summary: pd.DataFrame, name: str) -> plt.Figure:
        formats = {
            "ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
            "skew": "{:+.2f}", "excess_kurtosis": "{:+.2f}", "max_drawdown": "{:.2%}",
            "n_obs": "{:.0f}", "pct_missing": "{:.1f}%",
        }
        shown = summary.copy()
        for column, spec in formats.items():
            if column in shown.columns:
                shown[column] = shown[column].map(lambda v, s=spec: "" if pd.isna(v) else s.format(v))

        fig, ax = plt.subplots(figsize=(min(14, 1.4 * shown.shape[1] + 2), 0.4 * len(shown) + 1.5))
        ax.axis("off")
        table = ax.table(cellText=shown.reset_index().values,
                         colLabels=["ticker", *shown.columns], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)
        for column in range(shown.shape[1] + 1):
            table[0, column].set_facecolor("#40466e")
            table[0, column].set_text_props(color="white", weight="bold")
        ax.set_title(f"summary statistics, {name}", pad=10)
        fig.tight_layout()
        return fig

    def _save(self, fig: plt.Figure, path) -> None:
        fig.savefig(path, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close(fig)
