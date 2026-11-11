"""Smoke-test the ledger against real data, and verify its two identities.

    python scripts/run_smoke_backtest.py

This is P2's definition of done: run the engine over the full ETF-10 sample and
check that the two portfolios whose answers we already know come out right.

* **All-cash** must reproduce the compounded risk-free path exactly, and score
  an excess Sharpe of zero. If it does not, the cash column, the risk-free
  series, or the excess convention is wrong - and every Sharpe in the project
  would be wrong with it.
* **Buy-and-hold SPY** with zero costs must equal SPY's own total return. If it
  does not, the drift arithmetic is wrong, and every multi-asset result would
  inherit that error invisibly.

A third case, a costed weekly 1/N rebalance, is run to show the cost machinery
biting on a portfolio that actually trades. It has no closed-form identity to
check against; it is there to confirm the numbers are sane and to preview the
rung-0 baseline.

Output is written to ``results/diagnostics/smoke/`` - a verification artifact,
not a result of record.
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
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.backtest.costs import DEFAULT_COST_PER_UNIT  # noqa: E402
from rrc.backtest.ledger import (  # noqa: E402
    LEDGER_DTYPE,
    run_ledger,
    weekly_execution_schedule,
)
from rrc.backtest.metrics import summarise  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("smoke")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "smoke"
UNIVERSE = "etf10"
CASH_COLUMN = "CASH"


def load_panel() -> tuple[pd.DataFrame, pd.Series, int]:
    """Load the accounting return panel, the risk-free series, and cash's index."""
    returns = pd.read_parquet(PROCESSED_DIR / f"{UNIVERSE}_returns_simple.parquet")
    risk_free = pd.read_parquet(PROCESSED_DIR / f"{UNIVERSE}_risk_free.parquet")["r_f"]

    # The first row has no previous close, so no return exists for it; and the
    # risk-free series is NaN until its own first shifted observation.
    usable = returns.notna().all(axis=1) & risk_free.notna()
    returns, risk_free = returns.loc[usable], risk_free.loc[usable]

    return returns, risk_free, list(returns.columns).index(CASH_COLUMN)


def constant_weight_targets(
    weights: np.ndarray, n_periods: int
) -> torch.Tensor:
    """Repeat one weight vector across every rebalance."""
    return torch.as_tensor(
        np.tile(weights, (n_periods, 1)), dtype=LEDGER_DTYPE
    )


def report(name: str, net_returns: np.ndarray, risk_free: np.ndarray, **extra) -> dict:
    """Log and return the Section 13 summary for one strategy."""
    summary = summarise(net_returns, risk_free, **extra)
    logger.info(
        "  %-24s ann.return %7.2f%% | ann.vol %6.2f%% | excess Sharpe %6.3f | maxDD %7.2f%%",
        name,
        100 * summary["ann_return"],
        100 * summary["ann_vol"],
        summary["excess_sharpe"],
        100 * summary["max_drawdown"],
    )
    return summary


def main() -> int:
    """Run the smoke backtests and assert the two identities."""
    returns, risk_free, cash_index = load_panel()
    n_days, n_assets = returns.shape
    # .copy() because pandas may hand back a read-only view, which torch warns
    # about sharing memory with.
    returns_tensor = torch.as_tensor(returns.to_numpy().copy(), dtype=LEDGER_DTYPE)
    risk_free_array = risk_free.to_numpy()

    logger.info(
        "Panel: %d days x %d assets, %s to %s (cash column at index %d).",
        n_days, n_assets, returns.index.min().date(), returns.index.max().date(), cash_index,
    )

    signal_days, execution_days = weekly_execution_schedule(returns.index)
    logger.info(
        "Weekly schedule: %d rebalances; first signal %s -> executes %s (E7 lag).",
        len(execution_days),
        returns.index[signal_days[0]].date(),
        returns.index[execution_days[0]].date(),
    )

    results: dict[str, np.ndarray] = {}

    # ---- B1: all cash -----------------------------------------------------
    logger.info("--- B1: all-cash identity ---")
    cash_weights = np.zeros(n_assets)
    cash_weights[cash_index] = 1.0
    all_cash = run_ledger(
        constant_weight_targets(cash_weights, len(execution_days)),
        execution_days, returns_tensor, cash_index,
    )
    cash_summary = report("all cash", all_cash.net_returns.numpy(), risk_free_array)
    results["all cash"] = all_cash.net_returns.numpy()

    worst_cash_error = float(np.max(np.abs(all_cash.net_returns.numpy() - risk_free_array)))
    assert worst_cash_error < 1e-15, f"B1 FAILED: cash deviates from r_f by {worst_cash_error:.2e}"
    assert abs(cash_summary["excess_sharpe"]) < 1e-6, "B1 FAILED: cash has nonzero excess Sharpe"
    logger.info("  B1 PASSED: max |cash - r_f| = %.2e, excess Sharpe = %.2e",
                worst_cash_error, cash_summary["excess_sharpe"])

    # ---- B2: buy and hold SPY, zero costs ---------------------------------
    logger.info("--- B2: buy-and-hold SPY identity (zero costs) ---")
    spy_index = list(returns.columns).index("SPY")
    spy_weights = np.zeros(n_assets)
    spy_weights[spy_index] = 1.0

    buy_and_hold = run_ledger(
        torch.as_tensor(spy_weights[None, :], dtype=LEDGER_DTYPE),
        [0], returns_tensor, cash_index, cost_per_unit=0.0,
    )
    report("buy & hold SPY", buy_and_hold.net_returns.numpy(), risk_free_array)
    results["buy & hold SPY"] = buy_and_hold.net_returns.numpy()

    ledger_growth = float(np.prod(1.0 + buy_and_hold.net_returns.numpy()[1:]))
    spy_growth = float(np.prod(1.0 + returns["SPY"].to_numpy()[1:]))
    relative_error = abs(ledger_growth - spy_growth) / spy_growth
    assert relative_error < 1e-10, f"B2 FAILED: relative error {relative_error:.2e}"
    logger.info(
        "  B2 PASSED: ledger growth %.6f vs SPY total return %.6f (rel. error %.2e)",
        ledger_growth, spy_growth, relative_error,
    )

    # ---- Sanity: costed weekly 1/N over the risky assets -------------------
    logger.info("--- sanity: weekly 1/N over risky assets, costs on ---")
    equal_weights = np.full(n_assets, 1.0 / (n_assets - 1))
    equal_weights[cash_index] = 0.0

    equal_weight = run_ledger(
        constant_weight_targets(equal_weights, len(execution_days)),
        execution_days, returns_tensor, cash_index,
        cost_per_unit=DEFAULT_COST_PER_UNIT, return_diagnostics=True,
    )
    equal_summary = report(
        "weekly 1/N (costed)", equal_weight.net_returns.numpy(), risk_free_array,
        turnover=equal_weight.turnover.numpy(),
        held_weights=equal_weight.held_weights.numpy(), cash_index=cash_index,
    )
    results["weekly 1/N (costed)"] = equal_weight.net_returns.numpy()

    total_cost = float(equal_weight.costs.sum())
    logger.info(
        "  total costs %.4f of initial capital over %.1f years "
        "(annualised turnover %.2f, holdings %.1f, effective N %.2f)",
        total_cost, n_days / 252,
        equal_summary["ann_turnover"], equal_summary["holdings_count"],
        equal_summary["effective_n"],
    )

    # ---- Plot -------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 6))
    for label, series in results.items():
        axis.plot(returns.index, np.cumprod(1.0 + series), lw=1.3, label=label)

    axis.set_yscale("log")
    axis.set_title(
        "P2 smoke backtest — ledger identities on the real ETF-10 panel "
        "(log scale, growth of 1.0)", fontsize=11, loc="left",
    )
    axis.set_ylabel("growth of 1.0 (log)")
    axis.set_xlabel("date")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    plot_path = OUTPUT_DIR / "smoke_backtest.png"
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    logger.info("=== ALL LEDGER IDENTITIES HOLD ON REAL DATA ===")
    logger.info("Plot: %s", plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
