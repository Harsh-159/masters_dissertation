"""Tests for the stationary bootstrap.

The bootstrap feeds every confidence interval the dissertation reports, so the
properties checked here are the ones an interval silently depends on: the
resamples really are blocks of the original data, the same seed reproduces the
same answer, and dependence in the data actually widens the interval relative
to naive i.i.d. resampling (the entire reason for using blocks).
"""

from __future__ import annotations

import numpy as np
import pytest

from rrc.eval.stats import (
    BootstrapInterval,
    annualised_excess_sharpe,
    sharpe_confidence_interval,
    stationary_bootstrap_indices,
)


def test_indices_have_the_right_shape_and_range():
    indices = stationary_bootstrap_indices(
        n_observations=300, n_resamples=50, mean_block_length=21, seed=0
    )
    assert indices.shape == (50, 300)
    assert indices.min() >= 0
    assert indices.max() < 300


def test_same_seed_reproduces_the_same_indices():
    first = stationary_bootstrap_indices(200, 10, 21, seed=3)
    second = stationary_bootstrap_indices(200, 10, 21, seed=3)
    np.testing.assert_array_equal(first, second)


def test_resamples_are_made_of_consecutive_runs():
    """Within a block, successive indices step by exactly one (mod n).

    This is what distinguishes a block bootstrap from naive resampling: long
    consecutive runs of the original series survive into the resample.
    """
    indices = stationary_bootstrap_indices(500, 20, mean_block_length=50, seed=1)

    steps = np.diff(indices, axis=1) % 500
    fraction_consecutive = float((steps == 1).mean())
    # With mean block length 50, roughly 98% of steps continue a block.
    assert fraction_consecutive > 0.9


def test_block_length_parameter_controls_run_length():
    short_blocks = stationary_bootstrap_indices(500, 20, mean_block_length=2, seed=2)
    long_blocks = stationary_bootstrap_indices(500, 20, mean_block_length=100, seed=2)

    consecutive_short = float((np.diff(short_blocks, axis=1) % 500 == 1).mean())
    consecutive_long = float((np.diff(long_blocks, axis=1) % 500 == 1).mean())
    assert consecutive_long > consecutive_short


def test_interval_brackets_the_point_estimate():
    generator = np.random.default_rng(0)
    net_returns = generator.normal(0.0006, 0.01, 1500)
    risk_free = np.full(1500, 0.0001)

    interval = sharpe_confidence_interval(
        net_returns, risk_free, n_resamples=500, seed=0
    )
    assert isinstance(interval, BootstrapInterval)
    assert interval.low < interval.point < interval.high
    assert interval.n_resamples == 500


def test_interval_is_reproducible():
    generator = np.random.default_rng(1)
    net_returns = generator.normal(0.0005, 0.01, 800)
    risk_free = np.zeros(800)

    first = sharpe_confidence_interval(net_returns, risk_free, n_resamples=300, seed=7)
    second = sharpe_confidence_interval(net_returns, risk_free, n_resamples=300, seed=7)
    assert (first.low, first.point, first.high) == (second.low, second.point, second.high)


def test_a_strong_sharpe_yields_an_interval_above_zero():
    """A genuinely strong return stream should not have zero inside its CI."""
    generator = np.random.default_rng(2)
    net_returns = generator.normal(0.002, 0.008, 2000)  # unrealistically good
    risk_free = np.zeros(2000)

    interval = sharpe_confidence_interval(net_returns, risk_free, n_resamples=500, seed=0)
    assert interval.low > 0


def test_positive_autocorrelation_widens_the_interval():
    """Autocorrelated returns must produce a wider CI than the same values shuffled.

    This is the whole justification for the block bootstrap. When today's return
    carries over into tomorrow's (positive autocorrelation), the effective
    number of independent observations is smaller than the row count, so the
    uncertainty around the mean - and hence around the Sharpe - is genuinely
    larger. Resampling in blocks preserves that carry-over and reports the wider
    interval; shuffling the identical values destroys it and the interval
    narrows. An i.i.d. bootstrap would report the narrow interval for both,
    which is exactly the flattering mistake this machinery exists to avoid.
    """
    generator = np.random.default_rng(3)
    n_days = 2000
    persistence = 0.6  # strong AR(1) carry-over, so the effect is unambiguous

    innovations = generator.normal(0.0, 0.01, n_days)
    autocorrelated = np.empty(n_days)
    autocorrelated[0] = innovations[0]
    for day in range(1, n_days):
        autocorrelated[day] = persistence * autocorrelated[day - 1] + innovations[day]
    autocorrelated += 0.0005  # a modest positive drift

    shuffled = generator.permutation(autocorrelated)  # same values, no memory
    risk_free = np.zeros(n_days)

    autocorrelated_interval = sharpe_confidence_interval(
        autocorrelated, risk_free, n_resamples=400, seed=0
    )
    shuffled_interval = sharpe_confidence_interval(
        shuffled, risk_free, n_resamples=400, seed=0
    )

    autocorrelated_width = autocorrelated_interval.high - autocorrelated_interval.low
    shuffled_width = shuffled_interval.high - shuffled_interval.low
    assert autocorrelated_width > 1.3 * shuffled_width


def test_annualised_sharpe_matches_the_reporting_metric():
    """One definition project-wide: the bootstrap's Sharpe equals metrics.py's."""
    from rrc.backtest.metrics import excess_sharpe

    generator = np.random.default_rng(4)
    net_returns = generator.normal(0.0005, 0.01, 900)
    risk_free = np.full(900, 0.0001)

    assert annualised_excess_sharpe(net_returns - risk_free) == pytest.approx(
        excess_sharpe(net_returns, risk_free), rel=1e-12
    )


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="Shapes must match"):
        sharpe_confidence_interval(np.zeros(10), np.zeros(9))


def test_degenerate_inputs_are_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        stationary_bootstrap_indices(1, 10, 21, seed=0)
    with pytest.raises(ValueError, match="mean_block_length"):
        stationary_bootstrap_indices(100, 10, 0.5, seed=0)
