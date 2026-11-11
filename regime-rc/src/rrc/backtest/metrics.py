"""Performance metrics - the Section 13 definitions, in one place.

Two flavours live here, and the distinction matters:

* **Reporting metrics** (NumPy) are what go into the registry and the results
  tables. They are computed once per run, on a finished return stream.
* **Differentiable metrics** (torch, suffixed ``_torch``) are what the *loss*
  is built from at P5. Excess Sharpe and CVaR appear in both flavours, so they
  are defined once here and imported by the loss stack, guaranteeing that the
  quantity being optimised is exactly the quantity being reported.

Every metric is defined on **excess** returns (over the same ``r_f`` the cash
asset earns) wherever that is meaningful, which is what makes an all-cash
portfolio score exactly zero rather than something spuriously attractive.
"""

from __future__ import annotations

import numpy as np
import torch

TRADING_DAYS_PER_YEAR = 252

# Guards the Sharpe denominator. A portfolio that de-risks into cash has a
# vanishing standard deviation, and without this the ratio explodes and the
# gradient with it.
SHARPE_EPSILON = 1e-6

# A weight below this is treated as no position when counting holdings.
HOLDING_THRESHOLD = 1e-4


# --------------------------------------------------------------------------- #
# Reporting metrics (NumPy)
# --------------------------------------------------------------------------- #
def annualised_return(net_returns: np.ndarray) -> float:
    """Geometric annualised return of the compounded net path.

    Geometric, not the arithmetic mean times 252: only the compounded figure
    answers "what did an investor actually end up with".
    """
    returns = np.asarray(net_returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size == 0:
        return float("nan")
    total_growth = float(np.prod(1.0 + returns))
    if total_growth <= 0:
        return float("nan")  # the portfolio was wiped out; a rate is meaningless
    return total_growth ** (TRADING_DAYS_PER_YEAR / returns.size) - 1.0


def annualised_volatility(net_returns: np.ndarray) -> float:
    """Annualised standard deviation of daily returns."""
    returns = np.asarray(net_returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size < 2:
        return float("nan")
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def excess_sharpe(net_returns: np.ndarray, risk_free: np.ndarray) -> float:
    """Annualised Sharpe ratio computed on returns in excess of ``r_f``.

    Excess, not raw. With a cash asset earning ``r_f`` and zero variance, a raw
    Sharpe would reward holding cash forever with an essentially infinite score.
    Measuring excess over the same rate cash earns makes an all-cash portfolio
    score 0 - cash becomes a refuge, not an attractor.

    The ``SHARPE_EPSILON`` in the denominator is not cosmetic. An all-cash
    portfolio's excess returns are zero only in exact arithmetic; in float64
    they are round-off noise of order 1e-19. The ratio of two such numbers is
    an arbitrary O(1) value, so without stabilisation a pure cash portfolio
    reports a *large* Sharpe rather than zero. The same epsilon appears in the
    differentiable version, so the quantity reported is the quantity optimised.
    """
    excess = _excess(net_returns, risk_free)
    if excess.size < 2:
        return float("nan")
    deviation = float(np.std(excess, ddof=1))
    return float(
        np.mean(excess) / (deviation + SHARPE_EPSILON) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def sortino(net_returns: np.ndarray, risk_free: np.ndarray) -> float:
    """Annualised Sortino ratio: excess return over *downside* deviation.

    Downside deviation is ``sqrt(mean(min(excess, 0)^2))`` - upside volatility
    is not penalised, which is the point of the measure.

    Carries the same epsilon stabilisation as :func:`excess_sharpe`, and for the
    same reason: an all-cash portfolio has a downside deviation of round-off
    noise, not of zero.
    """
    excess = _excess(net_returns, risk_free)
    if excess.size < 2:
        return float("nan")
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    return float(
        np.mean(excess)
        / (downside_deviation + SHARPE_EPSILON)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def max_drawdown(net_returns: np.ndarray) -> float:
    """Largest peak-to-trough fall of the compounded net path, as a negative number."""
    returns = np.asarray(net_returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size == 0:
        return float("nan")
    wealth_path = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(wealth_path)
    return float(np.min(wealth_path / running_peak - 1.0))


def cvar(net_returns: np.ndarray, risk_free: np.ndarray, level: float = 0.95) -> float:
    """Conditional value at risk of daily excess returns, as a positive loss.

    The mean of the worst ``1 - level`` fraction of days. Reported positive, so
    a larger number always means worse tail risk.

    CVaR rather than variance because the return distributions have heavy tails
    (see diagnostic D3): variance treats a -20% day as merely four times worse
    than a -5% day, while CVaR looks directly at the part of the distribution
    that actually hurts.
    """
    excess = _excess(net_returns, risk_free)
    if excess.size == 0:
        return float("nan")
    tail_size = max(1, int(round((1.0 - level) * excess.size)))
    worst = np.sort(excess)[:tail_size]
    return float(-np.mean(worst))


def realised_holdings_count(
    held_weights: np.ndarray, cash_index: int, threshold: float = HOLDING_THRESHOLD
) -> float:
    """Average number of genuinely held risky positions per day.

    Cash is excluded: holding cash is not a "position" in the selection sense,
    and the cardinality constraint counts risky names only.
    """
    weights = np.asarray(held_weights, dtype=float)
    risky = np.delete(weights, cash_index, axis=1)
    return float(np.mean((risky > threshold).sum(axis=1)))


def effective_number_of_assets(held_weights: np.ndarray, cash_index: int) -> float:
    """Average of ``1 / sum_i w_i^2`` over risky holdings.

    A diversification-equivalent count: five assets at 20% each give 5.0, while
    one at 96% plus four at 1% gives barely above 1, even though both "hold five
    names". It is the honest companion to a raw holdings count.
    """
    weights = np.asarray(held_weights, dtype=float)
    risky = np.delete(weights, cash_index, axis=1)
    row_sums = risky.sum(axis=1, keepdims=True)
    normalised = np.divide(risky, row_sums, out=np.zeros_like(risky), where=row_sums > 0)
    concentration = (normalised**2).sum(axis=1)
    valid = concentration > 0
    if not valid.any():
        return 0.0
    return float(np.mean(1.0 / concentration[valid]))


def summarise(
    net_returns: np.ndarray,
    risk_free: np.ndarray,
    turnover: np.ndarray | None = None,
    held_weights: np.ndarray | None = None,
    cash_index: int | None = None,
) -> dict[str, float]:
    """Compute the full Section 13 metric set for one return stream.

    Returns a dict whose keys match the registry's metric columns exactly, so a
    run can pass it straight through to ``RegistryRow``.
    """
    from rrc.backtest.costs import annualised_turnover

    summary = {
        "ann_return": annualised_return(net_returns),
        "ann_vol": annualised_volatility(net_returns),
        "excess_sharpe": excess_sharpe(net_returns, risk_free),
        "sortino": sortino(net_returns, risk_free),
        "max_drawdown": max_drawdown(net_returns),
        "cvar_95": cvar(net_returns, risk_free, level=0.95),
    }
    if turnover is not None:
        summary["ann_turnover"] = annualised_turnover(
            turnover, n_days=len(np.asarray(net_returns))
        )
    if held_weights is not None and cash_index is not None:
        summary["holdings_count"] = realised_holdings_count(held_weights, cash_index)
        summary["effective_n"] = effective_number_of_assets(held_weights, cash_index)
    return summary


def _excess(net_returns: np.ndarray, risk_free: np.ndarray) -> np.ndarray:
    """Excess returns with non-finite rows dropped from both series together."""
    returns = np.asarray(net_returns, dtype=float)
    rates = np.asarray(risk_free, dtype=float)
    if returns.shape != rates.shape:
        raise ValueError(
            f"net_returns {returns.shape} and risk_free {rates.shape} must align."
        )
    usable = np.isfinite(returns) & np.isfinite(rates)
    return returns[usable] - rates[usable]


# --------------------------------------------------------------------------- #
# Differentiable metrics (torch) - the loss is built from these at P5
# --------------------------------------------------------------------------- #
def excess_sharpe_torch(
    net_returns: torch.Tensor,
    risk_free: torch.Tensor,
    epsilon: float = SHARPE_EPSILON,
) -> torch.Tensor:
    """Differentiable excess Sharpe over the whole span (not annualised).

    Note two deliberate choices:

    * the ``epsilon`` in the denominator, which stops the ratio (and its
      gradient) exploding as a de-risking portfolio's volatility approaches
      zero;
    * **no annualisation factor** - a constant scaling has no effect on the
      argmax, and leaving it out keeps the loss's magnitude interpretable
      against the other loss terms.

    Sharpe is *not separable*: the denominator couples every day in the span, so
    this must be computed over the whole training fold in one go rather than
    accumulated over mini-batches. That is why the project uses full-span
    batching everywhere.
    """
    excess = net_returns - risk_free
    return excess.mean() / (excess.std(unbiased=True) + epsilon)


def cvar_torch(
    net_returns: torch.Tensor,
    risk_free: torch.Tensor,
    level: float = 0.95,
) -> torch.Tensor:
    """Differentiable CVaR of daily excess returns, as a positive loss.

    Implemented as the mean of the worst ``1 - level`` fraction of days. The
    ``topk`` selection is piecewise-linear, so its subgradients are well defined
    and autograd handles it correctly.
    """
    excess = net_returns - risk_free
    tail_size = max(1, int(round((1.0 - level) * excess.numel())))
    worst = torch.topk(excess, tail_size, largest=False).values
    return -worst.mean()
