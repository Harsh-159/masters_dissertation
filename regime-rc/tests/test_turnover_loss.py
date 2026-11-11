"""Tests for the dynamic lambda_turn penalty in the decision loss.

The contract: off by default (so every pre-P8 result and cfg_hash is unchanged),
additive when on, and differentiable so it can actually push the model to trade
less.
"""

from __future__ import annotations

import pytest
import torch

from rrc.models.losses import decision_loss


def _returns_and_rf(n=252, seed=0):
    generator = torch.Generator().manual_seed(seed)
    net = 0.0004 + 0.01 * torch.randn(n, generator=generator, dtype=torch.float64)
    risk_free = torch.full((n,), 0.0001, dtype=torch.float64)
    return net, risk_free


def test_turnover_term_is_zero_by_default():
    net, risk_free = _returns_and_rf()
    parts = decision_loss(net, risk_free, lambda_cvar=1.0)
    assert float(parts.turnover_term) == 0.0


def test_turnover_term_is_zero_when_lambda_is_zero_even_with_turnover():
    net, risk_free = _returns_and_rf()
    turnover = torch.full((50,), 0.2, dtype=torch.float64)
    parts = decision_loss(net, risk_free, lambda_turn=0.0, turnover=turnover)
    assert float(parts.turnover_term) == 0.0


def test_turnover_term_penalises_the_mean_turnover():
    net, risk_free = _returns_and_rf()
    turnover = torch.full((50,), 0.2, dtype=torch.float64)
    parts = decision_loss(net, risk_free, lambda_cvar=0.0, lambda_turn=0.5, turnover=turnover)
    # 0.5 * mean(0.2) = 0.1
    assert float(parts.turnover_term) == pytest.approx(0.1)


def test_turnover_raises_the_total_loss():
    net, risk_free = _returns_and_rf()
    turnover = torch.full((50,), 0.3, dtype=torch.float64)
    without = decision_loss(net, risk_free, lambda_cvar=1.0)
    with_turn = decision_loss(net, risk_free, lambda_cvar=1.0, lambda_turn=1.0, turnover=turnover)
    assert float(with_turn.total) > float(without.total)
    # Only the turnover term should differ.
    assert float(with_turn.negative_sharpe) == pytest.approx(float(without.negative_sharpe))


def test_turnover_term_is_differentiable_in_the_turnover():
    turnover = torch.full((10,), 0.2, dtype=torch.float64, requires_grad=True)
    net, risk_free = _returns_and_rf()
    parts = decision_loss(net, risk_free, lambda_cvar=0.0, lambda_turn=1.0, turnover=turnover)
    parts.total.backward()
    # d(mean)/d(each) = 1/n, so every entry gets a positive gradient.
    assert torch.allclose(turnover.grad, torch.full((10,), 0.1, dtype=torch.float64))


def test_empty_turnover_is_safe():
    net, risk_free = _returns_and_rf()
    parts = decision_loss(net, risk_free, lambda_turn=1.0, turnover=torch.empty(0, dtype=torch.float64))
    assert float(parts.turnover_term) == 0.0
