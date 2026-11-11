"""Tests for the exact-cardinality execution layer (rung 5).

The load-bearing invariant: whatever the trained model proposes, the executed
portfolio must hold at most K_card risky names, be long-only, and be fully
invested. These pin that, plus the availability rule and the cash assembly.

(The per-date NSGA-II front these replace was retired - see
``rrc.optim.cardinality`` for the measurements that retired it.)
"""

from __future__ import annotations

import numpy as np
import pytest

from rrc.optim.cardinality import full_weight_vector, repair_scores


# --------------------------------------------------------------------------- #
# Repair: cardinality, availability, budget
# --------------------------------------------------------------------------- #
def test_repair_keeps_at_most_k_card_assets():
    scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
    weights = repair_scores(scores, np.ones(6, dtype=bool), k_card=3)
    assert int((weights > 0).sum()) == 3
    assert weights.sum() == pytest.approx(1.0)


def test_repair_selects_the_largest_scores():
    scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
    weights = repair_scores(scores, np.ones(6, dtype=bool), k_card=2)
    assert set(np.flatnonzero(weights).tolist()) == {1, 3}


def test_repair_preserves_relative_conviction():
    """Kept weights stay proportional to the model's own scores, just rescaled."""
    scores = np.array([0.6, 0.3, 0.1])
    weights = repair_scores(scores, np.ones(3, dtype=bool), k_card=2)
    assert weights[0] / weights[1] == pytest.approx(0.6 / 0.3)


def test_repair_never_holds_an_inactive_asset():
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    active = np.array([True, False, True, False])
    weights = repair_scores(scores, active, k_card=3)
    assert weights[1] == 0.0 and weights[3] == 0.0
    assert weights.sum() == pytest.approx(1.0)


def test_repair_all_zero_scores_falls_back_to_equal_weight():
    weights = repair_scores(np.zeros(5), np.ones(5, dtype=bool), k_card=3)
    assert int((weights > 0).sum()) == 3
    assert weights.sum() == pytest.approx(1.0)


def test_repair_with_fewer_active_than_k_card():
    active = np.array([True, False, False, True])
    weights = repair_scores(np.array([0.4, 0.9, 0.9, 0.6]), active, k_card=3)
    assert int((weights > 0).sum()) == 2


def test_repair_with_no_active_assets_returns_zeros():
    weights = repair_scores(np.array([0.5, 0.5]), np.zeros(2, dtype=bool), k_card=2)
    assert weights.sum() == pytest.approx(0.0)   # the caller's cash absorbs it


# --------------------------------------------------------------------------- #
# Full weight vector: cash insertion, budget
# --------------------------------------------------------------------------- #
def test_full_weight_vector_inserts_cash_and_sums_to_one():
    full = full_weight_vector(np.array([0.5, 0.5]), gross=0.8, cash_index=2, n_assets=3)
    assert full[2] == pytest.approx(0.2)
    assert full[:2] == pytest.approx([0.4, 0.4])
    assert full.sum() == pytest.approx(1.0)


def test_full_weight_vector_clips_gross_to_unit_interval():
    full = full_weight_vector(np.array([1.0]), gross=1.5, cash_index=1, n_assets=2)
    assert full.sum() == pytest.approx(1.0)
    assert full[1] == pytest.approx(0.0)          # no negative cash


def test_full_weight_vector_places_cash_at_any_index():
    full = full_weight_vector(np.array([1.0, 0.0]), gross=0.5, cash_index=0, n_assets=3)
    assert full[0] == pytest.approx(0.5)          # cash first
    assert full[1] == pytest.approx(0.5)
    assert full.sum() == pytest.approx(1.0)
