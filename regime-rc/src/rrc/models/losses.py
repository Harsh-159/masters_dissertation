"""The decision loss - what "good" means to the optimiser.

The core idea, stated once
--------------------------
This project never trains on forecast error. The loss is the **risk-adjusted
quality of the entire realised return stream** that the portfolio would have
produced, after costs. A head is good if the portfolio it runs is good - that
is the whole "decision-focused" thesis, and this file is where it becomes a
number.

The stack (Table 3 of the implementation plan)
----------------------------------------------
::

    L = - excess Sharpe (daily, full span)      the decision signal; always on
        + lambda_cvar * CVaR_95(excess)         tail term; default 1.0
        + lambda_balance * CV^2(importance)     anti-collapse; only bites for K > 1
        + lambda_turn * mean one-way turnover   explicit trading pressure; off by default
        [ + lambda_div * pairwise corr ]        anti-redundancy; off by default

    The turnover term, and why it defaults to zero (review fix E9)
    -------------------------------------------------------------
    Costs already flow through the daily net returns (the ledger charges 5 bp per
    unit traded), so the Sharpe term already *pays* for turnover. ``lambda_turn``
    is therefore 0 by default: adding it counts the same pain twice. It exists as
    a first-class, dynamic knob because a diagnostic may show the gate whipsawing
    - fast regime switches that trade a lot without earning it - and an explicit
    pressure on the *amount* traded (not just its cost) is then the documented
    lever. Turning it on is a one-line config change; it reads the per-rebalance
    one-way turnover the ledger already returns, so nothing new is computed. Any
    nonzero value is a new experiment (a new ``cfg_hash``), never a silent change
    to an existing rung.

Why Sharpe must see the whole span (A3)
---------------------------------------
Mean squared error is separable - a sum of per-day terms - so mini-batches are
fine. Sharpe is **not**: its denominator is the standard deviation across the
span, which couples every day to every other. Compute it on a shuffled
mini-batch and you have silently changed the objective. So the rule, enforced
by how these functions are called: one optimisation step consumes the entire
training fold. This is affordable precisely because the states are cached and
the ledger is closed-form.

Why the epsilon and the excess basis matter has been established at P2: the
epsilon stops a de-risking portfolio's vanishing volatility exploding the
ratio, and measuring excess over the same ``r_f`` cash earns makes the all-cash
portfolio score exactly zero (E1/D5).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rrc.backtest.metrics import cvar_torch, excess_sharpe_torch


@dataclass(frozen=True)
class LossParts:
    """The loss and its components, kept separate for logging.

    Watching the parts individually is a diagnostic: a falling total driven by
    a falling CVaR term while the Sharpe term stagnates tells a different story
    from the reverse, and the loss curves the run script saves plot exactly
    these.
    """

    total: torch.Tensor
    negative_sharpe: torch.Tensor
    cvar_term: torch.Tensor
    balance_term: torch.Tensor
    turnover_term: torch.Tensor
    divergence_term: torch.Tensor
    drawdown_term: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        """Detach to plain floats for history logging."""
        return {
            "loss": float(self.total.detach()),
            "negative_sharpe": float(self.negative_sharpe.detach()),
            "cvar_term": float(self.cvar_term.detach()),
            "balance_term": float(self.balance_term.detach()),
            "turnover_term": float(self.turnover_term.detach()),
            "divergence_term": float(self.divergence_term.detach()),
            "drawdown_term": float(self.drawdown_term.detach()),
        }


def balance_penalty(importance: torch.Tensor) -> torch.Tensor:
    """Squared coefficient of variation of per-expert importance.

    ``importance[k]`` is expert k's total gate mass over the span,
    ``Imp_k = sum_t pi_{t,k}``. If usage is perfectly even the penalty is zero;
    if one expert absorbs everything it grows. This is the required
    anti-collapse pressure for the mixture (Spec 8.4): without it, the
    self-reinforcing routing loop starves all but one expert.

    With a single expert (rung 1) the importance vector has one entry, its
    dispersion is zero, and the term vanishes identically - which is why the
    same loss function serves every rung.
    """
    if importance.numel() <= 1:
        return torch.zeros((), dtype=importance.dtype)

    mean_importance = importance.mean()
    if float(mean_importance.detach()) == 0.0:
        return torch.zeros((), dtype=importance.dtype)

    coefficient_of_variation = importance.std(unbiased=False) / mean_importance
    return coefficient_of_variation**2


def divergence_penalty(expert_weights: torch.Tensor) -> torch.Tensor:
    """Mean pairwise correlation between the experts' weight schedules.

    The anti-redundancy pressure of Specification 8.5, which the plan's Table 3
    says to switch on **only if gate G2 finds near-identical experts**. G2 did
    exactly that on the sector universe (median max pairwise correlation 0.923),
    and the consequence was measured on fold F03: all three experts held ~9.5%
    cash, the gate routed uniformly between three copies of one portfolio, and
    the model rode 2008 fully invested for -29.8%. With no distinct expert to
    route to, a mixture cannot express a defensive regime however well the gate
    detects one.

    ``expert_weights`` is ``(J, K, N)``: each expert's proposed portfolio at each
    signal day. Each expert's schedule is flattened to a single vector and
    correlated against the others; the penalty is the **mean** off-diagonal
    correlation. The plan states a sum, but with K fixed the two differ only by a
    constant absorbed into ``lambda_div``, and the mean is directly comparable to
    the pairwise correlations G2 reports.

    Returns 0 for a single expert, so the same loss serves every rung.
    """
    if expert_weights.ndim != 3:
        raise ValueError(
            f"expert_weights must be (J, K, N), got {tuple(expert_weights.shape)}."
        )
    n_experts = expert_weights.shape[1]
    if n_experts < 2:
        return torch.zeros((), dtype=expert_weights.dtype)

    # (K, J*N): one flat schedule per expert.
    flat = expert_weights.permute(1, 0, 2).reshape(n_experts, -1)
    centred = flat - flat.mean(dim=1, keepdim=True)
    norms = centred.pow(2).sum(dim=1).sqrt()
    correlation = (centred @ centred.T) / (norms[:, None] * norms[None, :] + 1e-12)

    off_diagonal = correlation - torch.diag(torch.diagonal(correlation))
    return off_diagonal.sum() / (n_experts * (n_experts - 1))



def drawdown_penalty(net_returns: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Underwater depth of the compounded net path - a differentiable drawdown.

    Why the loss needs this at all
    ------------------------------
    The tail term is CVaR over the worst 5 percent of **days**. 2008 was not
    remarkable day by day; it was an eight-month *sustained* decline, and daily
    CVaR is close to blind to that shape. Raising ``lambda_cvar`` was measured to
    make sector11 worse, because it buys generic day-to-day caution rather than
    protection against a long slide. Drawdown is defined on the *path*, so it
    penalises exactly the failure mode that costs the money.

    ``mode``
        ``"mean"`` (default) is the average underwater depth over the span. Every
        day spent below the previous peak contributes, so the gradient is dense
        and it directly punishes *staying* down - the 2008 signature.
        ``"max"`` is the textbook maximum drawdown, which matches the reported
        metric but concentrates its whole gradient on a single day, making it a
        poor training signal. The two are strongly related; ``mean`` is the
        trainable proxy and ``max`` remains what the registry reports.

    Returns a non-negative scalar (0 for a path that never falls below its peak).
    """
    if mode not in ("mean", "max"):
        raise ValueError(f"mode must be 'mean' or 'max', got {mode!r}.")

    value = torch.cumprod(1.0 + net_returns, dim=0)
    running_peak = torch.cummax(value, dim=0).values
    underwater = (running_peak - value) / (running_peak + 1e-12)
    return underwater.mean() if mode == "mean" else underwater.max()


def decision_loss(
    net_returns: torch.Tensor,
    risk_free: torch.Tensor,
    lambda_cvar: float = 1.0,
    lambda_balance: float = 0.0,
    importance: torch.Tensor | None = None,
    cvar_level: float = 0.95,
    lambda_turn: float = 0.0,
    turnover: torch.Tensor | None = None,
    lambda_div: float = 0.0,
    expert_weights: torch.Tensor | None = None,
    lambda_drawdown: float = 0.0,
    drawdown_mode: str = "mean",
) -> LossParts:
    """Assemble the full decision loss over one training span.

    Parameters
    ----------
    net_returns
        ``(D,)`` daily net portfolio returns from the ledger - costs already
        inside, which is why turnover is a *separate, optional* pressure rather
        than the only place trading is felt.
    risk_free
        ``(D,)`` daily risk-free returns, aligned.
    lambda_cvar
        Weight on the tail term. The rung-1 grid is {0, 1, 5}; frozen after.
    lambda_balance
        Weight on the anti-collapse term. 0 at rung 1 (meaningless for K = 1);
        0.01 from rung 2.
    importance
        ``(K,)`` per-expert gate mass over the span. ``None`` at rung 1.
    cvar_level
        Tail level for the CVaR term (0.95 fixed by the spec).
    lambda_turn
        Weight on the explicit turnover pressure. 0 by default (E9: costs
        already flow through ``net_returns``). A positive value is a deliberate,
        logged experiment - see the module docstring.
    turnover
        ``(J,)`` per-rebalance one-way turnover from the ledger. Required only
        when ``lambda_turn > 0``; the term is the mean over rebalances, so it is
        on the same "fraction of book per trade" scale regardless of fold length.

    Returns
    -------
    LossParts
        Total plus the individual components, all differentiable.
    """
    if net_returns.shape != risk_free.shape:
        raise ValueError(
            f"net_returns {tuple(net_returns.shape)} and risk_free "
            f"{tuple(risk_free.shape)} must align."
        )

    negative_sharpe = -excess_sharpe_torch(net_returns, risk_free)
    cvar_term = lambda_cvar * cvar_torch(net_returns, risk_free, level=cvar_level)

    if importance is not None and lambda_balance > 0:
        balance_term = lambda_balance * balance_penalty(importance)
    else:
        balance_term = torch.zeros((), dtype=net_returns.dtype)

    if lambda_turn > 0 and turnover is not None and turnover.numel() > 0:
        # Mean per-rebalance one-way turnover: differentiable in the executed
        # weights, so the gradient actively discourages needless trading.
        turnover_term = lambda_turn * turnover.mean()
    else:
        turnover_term = torch.zeros((), dtype=net_returns.dtype)

    if lambda_div > 0 and expert_weights is not None:
        divergence_term = lambda_div * divergence_penalty(expert_weights)
    else:
        divergence_term = torch.zeros((), dtype=net_returns.dtype)

    if lambda_drawdown > 0:
        drawdown_term = lambda_drawdown * drawdown_penalty(net_returns, drawdown_mode)
    else:
        drawdown_term = torch.zeros((), dtype=net_returns.dtype)

    total = (negative_sharpe + cvar_term + balance_term + turnover_term
             + divergence_term + drawdown_term)
    return LossParts(
        total=total,
        negative_sharpe=negative_sharpe,
        cvar_term=cvar_term,
        balance_term=balance_term,
        turnover_term=turnover_term,
        divergence_term=divergence_term,
        drawdown_term=drawdown_term,
    )
