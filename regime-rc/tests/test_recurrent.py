"""Tests for the LSTM comparison arm.

The arm exists to test decision D2 (a frozen temporal encoder), so what matters
is that it is a FAIR comparison: same feasibility guarantees as the reservoir
heads, same causality rule on its inputs, and drivable by the same training
loop. These pin exactly that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rrc.models.recurrent import LSTMPortfolioModel


def _mask(n_signals: int, n_assets: int) -> torch.Tensor:
    return torch.zeros(n_signals, n_assets, dtype=torch.float64)


def test_output_is_a_valid_portfolio():
    """Long-only and fully invested, exactly like the reservoir heads."""
    model = LSTMPortfolioModel(n_inputs=4, n_assets=5, hidden_size=8)
    windows = torch.randn(7, 20, 4, dtype=torch.float64)
    weights = model(windows, _mask(7, 5))
    assert weights.shape == (7, 5)
    assert torch.allclose(weights.sum(dim=1), torch.ones(7, dtype=torch.float64))
    assert (weights >= 0).all()


def test_unavailable_assets_get_exactly_zero_weight():
    """The availability mask must bind here as it does everywhere else."""
    model = LSTMPortfolioModel(n_inputs=3, n_assets=4, hidden_size=8)
    mask = _mask(5, 4)
    mask[:, 1] = -1e9                     # asset 1 unavailable
    weights = model(torch.randn(5, 15, 3, dtype=torch.float64), mask)
    assert torch.all(weights[:, 1] == 0.0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5, dtype=torch.float64))


def test_gradients_reach_the_recurrent_encoder():
    """The whole point is that the encoder is TRAINED, so it must get gradient.

    Note the objective: softmax rows sum to 1, so ``weights.sum()`` is the
    constant J and would legitimately carry zero gradient. A randomly weighted
    sum is a genuine function of the outputs.
    """
    torch.manual_seed(0)
    model = LSTMPortfolioModel(n_inputs=3, n_assets=4, hidden_size=8)
    weights = model(torch.randn(6, 10, 3, dtype=torch.float64), _mask(6, 4))
    (weights * torch.randn_like(weights)).sum().backward()
    encoder_grads = [p.grad for p in model.lstm.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in encoder_grads)
    assert any(g.abs().sum() > 0 for g in encoder_grads)


def test_route_pack_presents_a_single_expert():
    """The shared training loop drives every arm through route_pack."""
    model = LSTMPortfolioModel(n_inputs=3, n_assets=4, hidden_size=8)

    class Pack:
        input_windows = torch.randn(9, 12, 3, dtype=torch.float64)
        logit_mask = _mask(9, 4)

    memberships, expert_weights = model.route_pack(Pack())
    assert memberships.shape == (9, 1)
    assert torch.allclose(memberships, torch.ones_like(memberships))
    assert expert_weights.shape == (9, 1, 4)


def test_missing_windows_raise_rather_than_silently_misbehave():
    model = LSTMPortfolioModel(n_inputs=3, n_assets=4, hidden_size=8)

    class Pack:
        input_windows = None
        logit_mask = _mask(4, 4)

    with pytest.raises(ValueError, match="input_windows"):
        model.weights_for_pack(Pack())


def test_window_shape_is_validated():
    model = LSTMPortfolioModel(n_inputs=4, n_assets=5, hidden_size=8)
    with pytest.raises(ValueError, match="windows must be"):
        model(torch.randn(7, 4, dtype=torch.float64), _mask(7, 5))


def test_parameter_count_exceeds_a_linear_head_by_an_order_of_magnitude():
    """The capacity argument the arm is designed to test, made explicit."""
    from rrc.models.experts import LinearPortfolioHead, count_parameters

    lstm = LSTMPortfolioModel(n_inputs=9, n_assets=12, hidden_size=32)
    linear = LinearPortfolioHead(32, 12)
    assert count_parameters(lstm) > 5 * count_parameters(linear)
