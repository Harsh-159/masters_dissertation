"""Tests for the P9 inference layer: paired bootstrap, Ledoit-Wolf, DSR.

These three functions decide what the results chapter is allowed to claim, so
the properties pinned here are the ones a wrong claim would depend on:

* the deflation actually punishes searching, and punishes it by the right
  amount (a hand-computed value is pinned, not just a direction);
* Sharpes go into the deflation in **per-day** units even though the project
  reports annualised ones - the sqrt(252) trap gets its own test;
* the paired bootstrap really does preserve pairing, which is the whole reason
  P4 built an index-returning bootstrap rather than a value-returning one;
* the HAC standard error widens under autocorrelation, which is the only reason
  to prefer the Ledoit-Wolf test over a naive one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from rrc.backtest.metrics import TRADING_DAYS_PER_YEAR
from rrc.eval.stats import (
    annualise_sharpe,
    deannualise_sharpe,
    default_hac_lag,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    ledoit_wolf_sharpe_test,
    paired_sharpe_difference_interval,
)


# --------------------------------------------------------------------------- #
# Fixtures: return streams with known, controlled properties
# --------------------------------------------------------------------------- #
def gaussian_stream(n_days: int, annual_sharpe: float, seed: int) -> np.ndarray:
    """A daily excess-return stream with a target annualised Sharpe."""
    generator = np.random.default_rng(seed)
    daily_volatility = 0.01
    daily_mean = deannualise_sharpe(annual_sharpe) * daily_volatility
    return generator.normal(daily_mean, daily_volatility, n_days)


def autocorrelated_stream(n_days: int, rho: float, seed: int) -> np.ndarray:
    """An AR(1) stream: today's return carries part of yesterday's."""
    generator = np.random.default_rng(seed)
    shocks = generator.normal(0.0, 0.01, n_days)
    values = np.empty(n_days)
    values[0] = shocks[0]
    for t in range(1, n_days):
        values[t] = rho * values[t - 1] + shocks[t]
    return values


# --------------------------------------------------------------------------- #
# Unit conversion - the sqrt(252) trap
# --------------------------------------------------------------------------- #
def test_annualisation_round_trips():
    assert deannualise_sharpe(annualise_sharpe(0.037)) == pytest.approx(0.037)


def test_annualisation_uses_the_project_trading_year():
    assert annualise_sharpe(1.0) == pytest.approx(np.sqrt(TRADING_DAYS_PER_YEAR))


# --------------------------------------------------------------------------- #
# The selection-bias benchmark
# --------------------------------------------------------------------------- #
def test_expected_max_matches_a_hand_computed_value():
    """Pin the registry's own case: N = 305 trials at a dispersion of 0.365.

    Worked by hand from the Gumbel formula:
        Z(1 - 1/305)     = Z(0.996721)  = 2.7168
        Z(1 - 1/(305*e)) = Z(0.9987939) = 3.0353
        0.4228 * 2.7168 + 0.5772 * 3.0353 = 2.9010
        0.365 * 2.9010 = 1.0589
    """
    assert expected_max_sharpe_under_null(0.365, 305) == pytest.approx(1.0589, abs=1e-3)


def test_more_trials_raise_the_bar_but_only_slowly():
    """The trial count enters logarithmically; dispersion does not.

    This asymmetry is the reason the project fixes N by pre-registration and
    spends its methodological effort on the dispersion estimator instead.
    """
    few = expected_max_sharpe_under_null(0.1, 50)
    many = expected_max_sharpe_under_null(0.1, 305)
    assert many > few
    # Six times the searching, but well under half again as high a bar.
    assert many < 1.5 * few

    # Dispersion, by contrast, is exactly proportional.
    assert expected_max_sharpe_under_null(0.2, 305) == pytest.approx(
        2.0 * expected_max_sharpe_under_null(0.1, 305)
    )


def test_zero_dispersion_means_no_selection_bias():
    """If every trial scored identically, picking the best cannot flatter you."""
    assert expected_max_sharpe_under_null(0.0, 305) == pytest.approx(0.0)


def test_expected_max_rejects_a_degenerate_trial_count():
    with pytest.raises(ValueError, match="at least 2 trials"):
        expected_max_sharpe_under_null(0.3, 1)


# --------------------------------------------------------------------------- #
# The deflated Sharpe ratio
# --------------------------------------------------------------------------- #
def test_searching_more_lowers_the_deflated_sharpe():
    """The headline property: the same track record, deflated harder."""
    stream = gaussian_stream(1250, annual_sharpe=0.9, seed=0)
    barely_searched = deflated_sharpe_ratio(stream, trial_dispersion=0.1, n_trials=5)
    heavily_searched = deflated_sharpe_ratio(stream, trial_dispersion=0.1, n_trials=1000)

    assert barely_searched.deflated_probability > heavily_searched.deflated_probability
    assert heavily_searched.benchmark_sharpe > barely_searched.benchmark_sharpe
    # The observed Sharpe itself is untouched by how much you searched.
    assert barely_searched.observed_sharpe == pytest.approx(
        heavily_searched.observed_sharpe
    )


def test_wider_trial_dispersion_lowers_the_deflated_sharpe():
    stream = gaussian_stream(1250, annual_sharpe=0.9, seed=1)
    tight = deflated_sharpe_ratio(stream, trial_dispersion=0.05, n_trials=305)
    scattered = deflated_sharpe_ratio(stream, trial_dispersion=0.40, n_trials=305)
    assert tight.deflated_probability > scattered.deflated_probability


def test_a_longer_track_record_raises_the_deflated_sharpe():
    """Same Sharpe, more evidence for it."""
    short = deflated_sharpe_ratio(
        gaussian_stream(300, 0.9, seed=2), trial_dispersion=0.1, n_trials=50
    )
    long = deflated_sharpe_ratio(
        gaussian_stream(3000, 0.9, seed=2), trial_dispersion=0.1, n_trials=50
    )
    assert long.deflated_probability > short.deflated_probability


def test_zero_dispersion_reduces_to_the_plain_sharpe_t_test():
    """With no selection bias the benchmark vanishes and the DSR is a t-test.

    This is the algebraic anchor: it checks the units, the sqrt(T - 1), and the
    non-normality denominator all at once, against an independently written
    expression.
    """
    stream = gaussian_stream(1250, annual_sharpe=0.8, seed=3)
    result = deflated_sharpe_ratio(stream, trial_dispersion=0.0, n_trials=305)

    daily_sharpe = deannualise_sharpe(result.observed_sharpe)
    skewness = scipy_stats.skew(stream, bias=False)
    kurtosis = scipy_stats.kurtosis(stream, fisher=False, bias=False)
    expected_statistic = (
        daily_sharpe
        * np.sqrt(len(stream) - 1)
        / np.sqrt(1.0 - skewness * daily_sharpe + (kurtosis - 1.0) / 4.0 * daily_sharpe**2)
    )

    assert result.benchmark_sharpe == pytest.approx(0.0)
    assert result.deflated_probability == pytest.approx(
        scipy_stats.norm.cdf(expected_statistic), abs=1e-9
    )


def test_the_sharpe_enters_in_daily_units_not_annualised():
    """Guard the sqrt(252) trap directly.

    An implementation that fed the *annualised* Sharpe into a formula whose T
    counts days would produce a statistic sqrt(252) times too large - and a DSR
    of essentially 1.0 for any stream at all. Here a deliberately weak stream
    must NOT come out significant.
    """
    weak = gaussian_stream(1250, annual_sharpe=0.15, seed=4)
    result = deflated_sharpe_ratio(weak, trial_dispersion=0.3, n_trials=305)

    assert not result.clears_benchmark
    assert result.deflated_probability < 0.5


def test_negative_skew_is_penalised():
    """Two streams, same mean and variance; the lopsided one is less credible.

    A strategy that gains a little most days and occasionally loses badly has
    the same Sharpe as a symmetric one but is a worse bet, and the DSR's
    denominator is what expresses that.
    """
    generator = np.random.default_rng(5)
    n_days = 2000

    symmetric = generator.normal(0.0004, 0.01, n_days)
    # Negatively skewed: flip a right-skewed lognormal-ish series.
    raw = -generator.lognormal(mean=0.0, sigma=0.8, size=n_days)
    skewed = (raw - raw.mean()) / raw.std() * symmetric.std() + symmetric.mean()

    assert scipy_stats.skew(skewed) < -0.5  # the fixture really is lopsided
    symmetric_result = deflated_sharpe_ratio(symmetric, 0.05, 50)
    skewed_result = deflated_sharpe_ratio(skewed, 0.05, 50)

    # Same Sharpe by construction, different credibility.
    assert skewed_result.observed_sharpe == pytest.approx(
        symmetric_result.observed_sharpe, rel=0.02
    )
    assert skewed_result.deflated_probability < symmetric_result.deflated_probability


def test_deflated_sharpe_echoes_its_inputs():
    """A result must carry the population it was deflated against."""
    result = deflated_sharpe_ratio(gaussian_stream(500, 0.5, 6), 0.2, 77)
    assert result.n_trials == 77
    assert result.trial_dispersion == pytest.approx(0.2)
    assert result.n_observations == 500


# --------------------------------------------------------------------------- #
# Paired bootstrap on the difference
# --------------------------------------------------------------------------- #
def test_identical_streams_have_exactly_zero_difference():
    stream = gaussian_stream(800, 0.7, seed=7)
    interval = paired_sharpe_difference_interval(stream, stream, n_resamples=200)
    assert interval.point == pytest.approx(0.0)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(0.0)


def test_pairing_narrows_the_interval_on_co_moving_streams():
    """The reason the bootstrap returns indices instead of values.

    Two rungs trading the same assets share most of their daily variation.
    Resampling them jointly keeps that shared part shared; resampling them
    independently would destroy it and inflate the difference's variance.
    """
    generator = np.random.default_rng(8)
    common = generator.normal(0.0, 0.01, 1000)
    a = common + generator.normal(0.0002, 0.002, 1000)
    b = common + generator.normal(0.0000, 0.002, 1000)

    paired = paired_sharpe_difference_interval(a, b, n_resamples=400, seed=0)

    # Independent resampling: two different index matrices.
    from rrc.eval.stats import annualised_excess_sharpe, stationary_bootstrap_indices

    rows_a = stationary_bootstrap_indices(1000, 400, 21, seed=0)
    rows_b = stationary_bootstrap_indices(1000, 400, 21, seed=99)
    unpaired_differences = np.array(
        [
            annualised_excess_sharpe(a[ra]) - annualised_excess_sharpe(b[rb])
            for ra, rb in zip(rows_a, rows_b)
        ]
    )
    unpaired_width = np.percentile(unpaired_differences, 97.5) - np.percentile(
        unpaired_differences, 2.5
    )
    paired_width = paired.high - paired.low

    assert paired_width < unpaired_width


def test_paired_interval_rejects_misaligned_streams():
    with pytest.raises(ValueError, match="aligned"):
        paired_sharpe_difference_interval(np.zeros(100), np.zeros(90))


def test_paired_interval_is_reproducible():
    a = gaussian_stream(600, 0.8, seed=9)
    b = gaussian_stream(600, 0.3, seed=10)
    first = paired_sharpe_difference_interval(a, b, n_resamples=200, seed=4)
    second = paired_sharpe_difference_interval(a, b, n_resamples=200, seed=4)
    assert first == second


# --------------------------------------------------------------------------- #
# Ledoit-Wolf robust Sharpe-difference test
# --------------------------------------------------------------------------- #
def test_hac_lag_follows_the_plans_rule():
    assert default_hac_lag(1000) == int(np.floor(1.5 * 1000 ** (1 / 3)))
    assert default_hac_lag(1000) == 14


def test_identical_streams_are_not_significantly_different():
    stream = gaussian_stream(1500, 0.6, seed=11)
    result = ledoit_wolf_sharpe_test(stream, stream)
    assert result.difference == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_a_clearly_better_stream_is_detected():
    strong = gaussian_stream(3000, annual_sharpe=1.2, seed=12)
    weak = gaussian_stream(3000, annual_sharpe=0.0, seed=13)
    result = ledoit_wolf_sharpe_test(strong, weak)
    assert result.difference > 0
    assert result.p_value < 0.05


def test_autocorrelation_widens_the_standard_error():
    """The entire point of the HAC estimator.

    An AR(1) stream carries information across days, so it holds less
    independent evidence than its length suggests. A naive standard error
    ignores that; the HAC one must not.
    """
    n_days = 2000
    independent = autocorrelated_stream(n_days, rho=0.0, seed=14)
    persistent = autocorrelated_stream(n_days, rho=0.5, seed=14)
    # One fixed comparator for both, so only the autocorrelation differs.
    comparator = gaussian_stream(n_days, annual_sharpe=0.0, seed=140)

    independent_result = ledoit_wolf_sharpe_test(independent, comparator)
    persistent_result = ledoit_wolf_sharpe_test(persistent, comparator)

    assert persistent_result.standard_error > independent_result.standard_error


def test_lag_zero_recovers_a_plain_sandwich_estimator():
    """At lag 0 no autocorrelation terms are added, so the SE must shrink."""
    a = autocorrelated_stream(1500, rho=0.4, seed=15)
    b = gaussian_stream(1500, 0.2, seed=16)
    no_lags = ledoit_wolf_sharpe_test(a, b, hac_lag=0)
    with_lags = ledoit_wolf_sharpe_test(a, b)
    assert no_lags.hac_lag == 0
    assert no_lags.standard_error < with_lags.standard_error


def test_statistic_is_invariant_to_annualisation():
    """difference and standard_error are both annualised, so their ratio is not."""
    a = gaussian_stream(1200, 0.9, seed=17)
    b = gaussian_stream(1200, 0.2, seed=18)
    result = ledoit_wolf_sharpe_test(a, b)
    assert result.statistic == pytest.approx(result.difference / result.standard_error)


def test_ledoit_wolf_rejects_too_short_a_sample():
    with pytest.raises(ValueError, match="at least 10"):
        ledoit_wolf_sharpe_test(np.zeros(5) + 0.001, np.zeros(5) + 0.002)
