"""Exact cardinality: restricting a portfolio to its top-K conviction names.

What rung 5 is, after the NSGA-II front was dropped
----------------------------------------------------
A softmax head cannot express a real book. It spreads a little weight across
every asset, whereas a portfolio that is actually traded holds a handful of
names. Rung 5 is the layer that imposes that constraint: keep the ``K_card``
assets the trained mixture likes most, drop the rest, renormalise.

Why there is no optimiser here any more
---------------------------------------
The plan (R1) specified a per-rebalance-date NSGA-II front: search continuous
weights against a return-versus-CVaR objective under a combinatorial
cardinality constraint. Two measurements retired it:

* **The combinatorics do not bite at this size.** Choosing 5 of 10 assets is
  252 combinations, and once a subset is fixed the remaining problem is convex.
  A stochastic population search is the wrong tool for a problem that small.
* **Its objective fought the trained system.** The front scored candidates on an
  EWMA momentum estimate ``mu_hat`` - a predict-then-optimise step bolted onto
  the end of a decision-trained system, which is the very pattern this project
  argues against. Measured on validation (``run_front_attribution``), simply
  imposing cardinality on the trained blend scored **+0.909 / +0.342** (etf10 /
  sector11) while the NSGA front scored **-0.325 / -0.231** on the same folds,
  same model, same ledger.

So the trained mixture chooses *what* to hold - that is what it was trained on -
and this module only enforces *how many*. The result is that cardinality turns
out to be roughly free (it slightly beat the dense blend on both universes),
which is the honest finding rung 5 exists to report.

Note on costs
-------------
Nothing here changes trading costs. The 5 bp per unit of notional traded (A9)
is charged inside the ledger exactly as before, so every number this layer
produces is net of realistic transaction costs.
"""

from __future__ import annotations

import numpy as np


def repair_scores(raw_scores: np.ndarray, active_mask: np.ndarray, k_card: int) -> np.ndarray:
    """Keep the ``k_card`` largest scores among active assets; renormalise to 1.

    Parameters
    ----------
    raw_scores
        ``(n_risky,)`` non-negative preference scores - in practice the trained
        mixture's own risky weights.
    active_mask
        ``(n_risky,)`` boolean availability at this date.
    k_card
        Maximum number of risky names to hold.

    Returns
    -------
    numpy.ndarray
        ``(n_risky,)`` weights summing to 1, zero on inactive assets, with at
        most ``k_card`` nonzero entries. Cardinality holds by construction, not
        as a penalty something could violate.
    """
    active = np.asarray(active_mask, dtype=bool)
    scores = np.clip(np.asarray(raw_scores, dtype=float), 0.0, None) * active

    n_active = int(active.sum())
    if n_active == 0:
        # No risky asset available; the caller's cash weight absorbs the budget.
        return np.zeros_like(scores)

    keep = min(int(k_card), n_active)
    # Indices of the `keep` largest scores; ties break by index, deterministically.
    top = np.argsort(scores, kind="stable")[::-1][:keep]
    selected = np.zeros_like(scores)
    selected[top] = scores[top]

    total = selected.sum()
    if total <= 0.0:
        # All top scores were zero - fall back to equal weight over the kept set.
        selected[top] = 1.0
        total = float(len(top))
    return selected / total


def full_weight_vector(
    risky_weights: np.ndarray, gross: float, cash_index: int, n_assets: int
) -> np.ndarray:
    """Assemble a universe-native weight vector, inserting cash at ``cash_index``.

    ``risky_weights`` sum to 1 over the risky assets; scaling by ``gross`` and
    putting ``1 - gross`` in cash gives a long-only, fully-invested portfolio in
    the same column order the ledger and every baseline use.
    """
    gross = float(np.clip(gross, 0.0, 1.0))
    full = np.zeros(n_assets, dtype=float)
    risky_positions = [i for i in range(n_assets) if i != cash_index]
    full[risky_positions] = gross * np.asarray(risky_weights, dtype=float)
    full[cash_index] = 1.0 - gross
    return full


def cardinality_schedule(
    blend: np.ndarray, pack, artifacts, k_card: int
) -> np.ndarray:
    """Restrict a trained weight schedule to its top-``k_card`` names per date.

    The mixture's own ranking decides *which* assets survive, and its own gross
    exposure (hence its cash weight) is carried through unchanged - so the only
    thing this changes is the number of names held.

    Parameters
    ----------
    blend
        ``(J, N)`` the trained model's weight schedule, cash column included.
    pack
        The :class:`~rrc.optim.fold_data.SegmentPack` the schedule belongs to.
    artifacts
        The universe's artifacts, for the availability matrix.
    k_card
        Maximum number of risky names.

    Returns
    -------
    numpy.ndarray
        ``(J, N)`` the cardinality-constrained schedule, ready for the ledger.
    """
    n_assets = blend.shape[1]
    cash_index = pack.cash_index
    risky_positions = [i for i in range(n_assets) if i != cash_index]
    availability = artifacts.availability.to_numpy()

    restricted = np.zeros_like(blend)
    for j in range(pack.n_signals):
        # Availability is read at the SIGNAL day (execution day minus one, E7).
        signal_date = pack.dates[pack.execution_positions[j] - 1]
        abs_index = int(artifacts.dates.get_loc(signal_date))
        active = availability[abs_index].astype(bool)

        risky_raw = blend[j][risky_positions]
        restricted[j] = full_weight_vector(
            repair_scores(risky_raw, active, k_card),
            float(risky_raw.sum()),       # keep the blend's own gross exposure
            cash_index, n_assets,
        )
    return restricted
