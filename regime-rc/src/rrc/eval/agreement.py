"""Regime-agreement metrics - RQ1 as numbers, not eyeballed heatmaps (A10).

The question and how it is measured
-----------------------------------
"Do decision-defined regimes carve the market differently from statistically
defined ones?" has to be a number, because either answer is only publishable if
it is measured. Four measurements, each answering a distinct sub-question:

* **Adjusted mutual information / adjusted Rand index** between the learned
  gate's hard assignments (argmax pi) and a statistical labelling (filtered
  HMM states, or VIX terciles). Both are chance-corrected: 0 means the two
  labellings agree no more than random relabellings would, 1 means identical
  partitions. Low AMI together with a positive rung-4 performance gap is the
  strongest form of the thesis claim; high AMI is the honest null.
* **Dwell time**: the average number of consecutive signal days a labelling
  stays in one regime - are these slow macro states or fast switches?
* **Transition rate**: the fraction of steps that change label - the same
  question from the other side.
* **Decision divergence** ``D = mean_t || w_t^learned - w_t^stat ||_1``:
  holding the EXPERTS fixed, how differently do the two routings actually
  trade? Two gates can disagree about labels while inducing nearly identical
  portfolios (if the experts are similar), or agree on labels while trading
  differently. This measures what matters - the decisions - directly.

Plus **occupancy-conditioned Sharpe**: the learned strategy's Sharpe computed
separately over the days each regime governed, which is what makes statements
like "expert 2's regime is the crisis regime" quantitative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from rrc.backtest.metrics import TRADING_DAYS_PER_YEAR, SHARPE_EPSILON


def hard_labels(memberships: np.ndarray) -> np.ndarray:
    """Argmax over the membership columns - soft routing made categorical."""
    return np.asarray(memberships).argmax(axis=1)


def label_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    """Chance-corrected agreement between two label series.

    Returns
    -------
    dict
        ``ami`` (adjusted mutual information) and ``ari`` (adjusted Rand
        index). Both are 1 for identical partitions and about 0 for unrelated
        ones, and both are invariant to renaming the labels - essential here,
        because the learned gate's "expert 0" and the HMM's "state 0" have no
        reason to share names even when they mean the same thing.
    """
    first = np.asarray(labels_a)
    second = np.asarray(labels_b)
    if first.shape != second.shape:
        raise ValueError(f"Label series differ in length: {first.shape} vs {second.shape}.")
    return {
        "ami": float(adjusted_mutual_info_score(first, second)),
        "ari": float(adjusted_rand_score(first, second)),
    }


def dwell_and_transition(labels: np.ndarray) -> dict[str, float]:
    """Mean dwell time (in signal days) and per-step transition rate."""
    series = np.asarray(labels)
    if len(series) < 2:
        return {"mean_dwell": float(len(series)), "transition_rate": 0.0}

    changes = series[1:] != series[:-1]
    n_runs = 1 + int(changes.sum())
    return {
        "mean_dwell": float(len(series) / n_runs),
        "transition_rate": float(changes.mean()),
    }


def occupancy_conditioned_sharpe(
    daily_net_returns: pd.Series,
    daily_risk_free: pd.Series,
    signal_labels: np.ndarray,
    signal_positions: np.ndarray,
) -> dict[int, dict[str, float]]:
    """The strategy's Sharpe, split by which regime governed each day.

    A label decided at signal day t governs the days from that signal's
    execution until the next one - so each daily return is attributed to the
    most recent signal's label, and the Sharpe is computed per label over its
    own days.

    Returns
    -------
    dict
        ``label -> {"sharpe": ..., "n_days": ..., "share_of_days": ...}``.
    """
    n_days = len(daily_net_returns)
    governing = np.full(n_days, -1, dtype=int)
    for index, signal_position in enumerate(signal_positions):
        start = signal_position + 1  # the day after the signal (execution day)
        end = signal_positions[index + 1] + 1 if index + 1 < len(signal_positions) else n_days
        governing[start:end] = signal_labels[index]

    excess = (daily_net_returns.to_numpy() - daily_risk_free.to_numpy())
    results: dict[int, dict[str, float]] = {}
    for label in sorted(set(signal_labels.tolist())):
        mask = governing == label
        if mask.sum() < 2:
            continue
        segment = excess[mask]
        sharpe = float(
            segment.mean() / (segment.std(ddof=1) + SHARPE_EPSILON)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        results[int(label)] = {
            "sharpe": sharpe,
            "n_days": int(mask.sum()),
            "share_of_days": float(mask.mean()),
        }
    return results


def decision_divergence(
    learned_memberships: np.ndarray,
    statistical_memberships: np.ndarray,
    expert_weights: np.ndarray,
) -> float:
    """``mean_t || w_t^learned - w_t^stat ||_1`` under identical experts.

    Parameters
    ----------
    learned_memberships, statistical_memberships
        ``(J, K)`` membership matrices from the two gates, over the same
        signal days. The statistical gate may have fewer states than K (a
        degeneracy-reduced HMM); it is then padded with zero columns, which is
        exact: a state that does not exist receives no mass.
    expert_weights
        ``(J, K, N)`` the SAME experts' proposals - so any difference in the
        blended portfolios comes from the routing alone.

    Returns
    -------
    float
        The average L1 gap between the two blended weight vectors. 0 means
        the routings induce identical trades; 2 is the maximum possible
        (completely disjoint portfolios).
    """
    learned = np.asarray(learned_memberships, dtype=float)
    statistical = np.asarray(statistical_memberships, dtype=float)
    experts = np.asarray(expert_weights, dtype=float)

    n_experts = experts.shape[1]
    if statistical.shape[1] < n_experts:
        padding = np.zeros((statistical.shape[0], n_experts - statistical.shape[1]))
        statistical = np.hstack([statistical, padding])
    if learned.shape != statistical.shape:
        raise ValueError(
            f"Membership shapes differ after padding: {learned.shape} vs "
            f"{statistical.shape}."
        )

    blended_learned = np.einsum("jk,jkn->jn", learned, experts)
    blended_statistical = np.einsum("jk,jkn->jn", statistical, experts)
    return float(np.abs(blended_learned - blended_statistical).sum(axis=1).mean())
