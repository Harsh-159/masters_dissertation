"""Statistical inference on return streams. P4 starts it; P9 extends it.

The first resident is the **stationary bootstrap** (Politis and Romano, 1994),
which is the project's primary inference tool (Section 13: "inference, in order
of trust").

Why an ordinary bootstrap will not do
-------------------------------------
Daily returns are not independent: volatility clusters, so tomorrow's return
size depends on today's. Resampling days one at a time destroys that structure
and produces confidence intervals that are too narrow - precisely the flattering
direction. The stationary bootstrap resamples **blocks** of consecutive days
instead, preserving the local dependence, and draws the block lengths from a
geometric distribution (mean 21 days here, about one trading month) so the
resampled series remains stationary rather than inheriting artefacts of a fixed
block size.

What P4 needs from it
---------------------
A confidence interval on each baseline's annualised excess Sharpe: resample the
daily excess-return stream B = 2000 times, compute the Sharpe of each resample,
and take percentiles. P9 adds the paired-difference machinery (resampling two
strategies' returns jointly to compare rungs); the index generator built here is
already written to support that, because resampling the *indices* once and
applying them to both series is exactly how the pairing is preserved.

What P9 adds
------------
Three tools, in the plan's own order of trust:

1. :func:`paired_sharpe_difference_interval` - the stationary bootstrap applied
   to a *difference* of Sharpes. One index matrix, both series: the pairing
   survives, so day-level co-movement between two rungs is not resampled away.
2. :func:`ledoit_wolf_sharpe_test` - a parametric test on the same difference,
   robust to the two things daily returns always do (autocorrelation and fat
   tails) via a HAC covariance estimator.
3. :func:`deflated_sharpe_ratio` - the correction for having *searched*. This is
   the one that decides how the results chapter is framed, so its two inputs
   (the trial count and the trial dispersion) are documented at length below.

A note on units, because it is the standard way to get the DSR wrong
--------------------------------------------------------------------
The registry stores **annualised** Sharpe ratios. The deflation algebra is
derived for **per-observation** ones, and the observation count ``T`` it uses is
a count of *days*. Feeding an annualised Sharpe into a daily-T formula produces
a plausible-looking number that is wrong by a factor of sqrt(252). Every
function here therefore states its expected units in the signature docstring,
and :func:`annualise_sharpe` / :func:`deannualise_sharpe` exist so the
conversion is always a named call rather than a bare ``/ np.sqrt(252)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from rrc.backtest.metrics import TRADING_DAYS_PER_YEAR, SHARPE_EPSILON

# Defaults fixed by the plan (Section 13 / Appendix B).
DEFAULT_MEAN_BLOCK_LENGTH = 21
DEFAULT_N_RESAMPLES = 2000

# The Euler-Mascheroni constant. It is not a tuning choice: the maximum of many
# independent normal draws converges to a Gumbel distribution, and gamma is that
# limiting distribution's mean. It enters the expected-maximum formula for that
# reason alone.
EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class BootstrapInterval:
    """A bootstrap confidence interval for one statistic.

    Attributes
    ----------
    point
        The statistic computed on the original, unresampled series.
    low, high
        The percentile interval bounds.
    level
        The nominal coverage, e.g. 0.95.
    n_resamples
        How many bootstrap replicates were used.
    """

    point: float
    low: float
    high: float
    level: float
    n_resamples: int


def stationary_bootstrap_indices(
    n_observations: int,
    n_resamples: int,
    mean_block_length: float,
    seed: int,
) -> np.ndarray:
    """Generate index matrices for the stationary bootstrap.

    Each row is one resample: a sequence of positions into the original series,
    built from blocks of consecutive positions. A block starts at a uniformly
    random position; at every step the block continues with probability
    ``1 - 1/mean_block_length`` or a new block starts, which makes block lengths
    geometric with the requested mean. Indices wrap around the end of the series
    (the "circular" convention), so every position is equally likely to appear.

    Returning *indices* rather than resampled values is deliberate: applying the
    same index matrix to two strategies' return series resamples them jointly,
    preserving their day-by-day pairing - which is what the paired comparisons
    at P9 require.

    Parameters
    ----------
    n_observations
        Length of the series being resampled.
    n_resamples
        Number of bootstrap replicates.
    mean_block_length
        Average block length in days.
    seed
        Seed for reproducibility; the same seed always yields the same indices.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_resamples, n_observations)``.
    """
    if n_observations < 2:
        raise ValueError(f"Need at least 2 observations, got {n_observations}.")
    if mean_block_length < 1:
        raise ValueError(f"mean_block_length must be >= 1, got {mean_block_length}.")

    generator = np.random.default_rng(seed)
    restart_probability = 1.0 / mean_block_length
    indices = np.empty((n_resamples, n_observations), dtype=np.int64)

    for row in range(n_resamples):
        # Where do new blocks start? Position 0 always starts one.
        restarts = generator.random(n_observations) < restart_probability
        restarts[0] = True
        block_start_positions = np.flatnonzero(restarts)

        # Length of each block = distance to the next restart.
        block_lengths = np.diff(np.append(block_start_positions, n_observations))

        # Each block copies from a uniformly random anchor in the original series.
        anchors = generator.integers(0, n_observations, size=len(block_start_positions))

        # Expand: for each position, its block's anchor plus its offset within
        # the block, wrapping around the end of the series.
        anchor_per_position = np.repeat(anchors, block_lengths)
        offset_within_block = np.arange(n_observations) - np.repeat(
            block_start_positions, block_lengths
        )
        indices[row] = (anchor_per_position + offset_within_block) % n_observations

    return indices


def annualised_excess_sharpe(excess_returns: np.ndarray) -> float:
    """Annualised Sharpe of a daily excess-return array.

    Same definition (and the same epsilon-stabilised denominator) as the
    reporting metric in :mod:`rrc.backtest.metrics`, restated here on a plain
    array so the bootstrap can apply it to thousands of resamples cheaply.
    """
    values = np.asarray(excess_returns, dtype=float)
    deviation = values.std(ddof=1)
    return float(values.mean() / (deviation + SHARPE_EPSILON) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_confidence_interval(
    net_returns: np.ndarray,
    risk_free: np.ndarray,
    level: float = 0.95,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    mean_block_length: float = DEFAULT_MEAN_BLOCK_LENGTH,
    seed: int = 0,
) -> BootstrapInterval:
    """Stationary-bootstrap confidence interval for the annualised excess Sharpe.

    Parameters
    ----------
    net_returns
        Daily net returns of the strategy.
    risk_free
        Daily risk-free returns, aligned with ``net_returns``.
    level
        Nominal coverage of the interval.
    n_resamples, mean_block_length, seed
        Bootstrap parameters; defaults are the plan's.

    Returns
    -------
    BootstrapInterval
        Point estimate plus the percentile interval.
    """
    returns = np.asarray(net_returns, dtype=float)
    rates = np.asarray(risk_free, dtype=float)
    if returns.shape != rates.shape:
        raise ValueError(f"Shapes must match: {returns.shape} vs {rates.shape}.")

    usable = np.isfinite(returns) & np.isfinite(rates)
    excess = returns[usable] - rates[usable]

    point_estimate = annualised_excess_sharpe(excess)

    index_matrix = stationary_bootstrap_indices(
        n_observations=len(excess),
        n_resamples=n_resamples,
        mean_block_length=mean_block_length,
        seed=seed,
    )
    resampled_sharpes = np.array(
        [annualised_excess_sharpe(excess[row]) for row in index_matrix]
    )

    tail = (1.0 - level) / 2.0
    low, high = np.percentile(resampled_sharpes, [100 * tail, 100 * (1 - tail)])
    return BootstrapInterval(
        point=point_estimate,
        low=float(low),
        high=float(high),
        level=level,
        n_resamples=n_resamples,
    )


# --------------------------------------------------------------------------- #
# Unit conversion (see the module docstring's note on units)
# --------------------------------------------------------------------------- #
def annualise_sharpe(per_period_sharpe: float) -> float:
    """Convert a per-observation Sharpe to an annualised one."""
    return float(per_period_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR))


def deannualise_sharpe(annualised_sharpe: float) -> float:
    """Convert an annualised Sharpe back to per-observation units.

    The deflation algebra needs this: its ``T`` counts trading days, so the
    Sharpe it consumes must be a per-day quantity.
    """
    return float(annualised_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR))


# --------------------------------------------------------------------------- #
# P9: paired comparison of two return streams
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SharpeDifferenceTest:
    """Outcome of a Ledoit-Wolf robust Sharpe-difference test.

    Attributes
    ----------
    difference
        Annualised Sharpe of ``a`` minus annualised Sharpe of ``b``.
    standard_error
        HAC standard error of that difference, annualised to match.
    statistic
        ``difference / standard_error``. Unaffected by annualisation, since both
        numerator and denominator scale by the same sqrt(252).
    p_value
        Two-sided p-value against the null of equal Sharpe ratios.
    hac_lag
        The truncation lag actually used by the HAC estimator.
    n_observations
        Number of paired, finite observations the test ran on.
    """

    difference: float
    standard_error: float
    statistic: float
    p_value: float
    hac_lag: int
    n_observations: int


def paired_sharpe_difference_interval(
    excess_a: np.ndarray,
    excess_b: np.ndarray,
    level: float = 0.95,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    mean_block_length: float = DEFAULT_MEAN_BLOCK_LENGTH,
    seed: int = 0,
) -> BootstrapInterval:
    """Stationary-bootstrap interval for the *difference* of two Sharpes.

    Why pairing matters
    -------------------
    Two rungs of the same ladder trade the same assets on the same days, so
    their returns move together. Bootstrapping them independently would break
    that co-movement and inflate the variance of the difference - the interval
    would come out far too wide, and a real improvement would be reported as
    inconclusive. Drawing **one** index matrix and applying it to both series
    resamples them jointly, so whatever they share on a given day stays shared.

    This is the reason :func:`stationary_bootstrap_indices` returns indices
    rather than values, decided back at P4.

    Parameters
    ----------
    excess_a, excess_b
        Daily **excess** return streams, already aligned day-for-day. Positive
        results mean ``a`` beat ``b``.
    level, n_resamples, mean_block_length, seed
        Bootstrap parameters; defaults are the plan's.

    Returns
    -------
    BootstrapInterval
        ``point`` is the observed annualised Sharpe difference. An interval that
        excludes zero is the bootstrap's version of a significant difference.
    """
    first = np.asarray(excess_a, dtype=float)
    second = np.asarray(excess_b, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"Streams must be aligned: {first.shape} vs {second.shape}.")

    # Drop any day either stream is missing, so the pairing stays exact.
    usable = np.isfinite(first) & np.isfinite(second)
    first, second = first[usable], second[usable]
    if first.size < 2:
        raise ValueError("Need at least 2 paired observations.")

    point_estimate = annualised_excess_sharpe(first) - annualised_excess_sharpe(second)

    index_matrix = stationary_bootstrap_indices(
        n_observations=first.size,
        n_resamples=n_resamples,
        mean_block_length=mean_block_length,
        seed=seed,
    )
    # The same row of indices addresses both series - that is the pairing.
    differences = np.array(
        [
            annualised_excess_sharpe(first[row]) - annualised_excess_sharpe(second[row])
            for row in index_matrix
        ]
    )

    tail = (1.0 - level) / 2.0
    low, high = np.percentile(differences, [100 * tail, 100 * (1 - tail)])
    return BootstrapInterval(
        point=float(point_estimate),
        low=float(low),
        high=float(high),
        level=level,
        n_resamples=n_resamples,
    )


def default_hac_lag(n_observations: int) -> int:
    """The plan's HAC truncation lag, ``floor(1.5 * T**(1/3))``.

    A HAC estimator has to decide how many lags of autocorrelation to carry.
    Too few and the standard error is too small (flattering); too many and it
    becomes noisy. The cube-root rule is the standard compromise and is fixed by
    the plan so it is never a tuned quantity.
    """
    return int(np.floor(1.5 * n_observations ** (1.0 / 3.0)))


def _hac_covariance(deviations: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West HAC covariance of a multivariate series' sample mean.

    ``deviations`` is (T, k), already demeaned. Returns the (k, k) long-run
    covariance ``Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')`` with Bartlett
    weights ``w_j = 1 - j/(lag+1)``, which is what guarantees the result is
    positive semi-definite.

    Ledoit and Wolf's own paper prewhitens and uses a Parzen kernel. We use the
    plainer Bartlett form at the plan's fixed lag: the difference is immaterial
    at T in the thousands, and the simpler estimator is one a reader can check.
    """
    n_observations = deviations.shape[0]
    covariance = deviations.T @ deviations / n_observations
    for j in range(1, lag + 1):
        cross = deviations[j:].T @ deviations[:-j] / n_observations
        weight = 1.0 - j / (lag + 1.0)
        covariance = covariance + weight * (cross + cross.T)
    return covariance


def ledoit_wolf_sharpe_test(
    excess_a: np.ndarray,
    excess_b: np.ndarray,
    hac_lag: int | None = None,
) -> SharpeDifferenceTest:
    """Robust test of ``H0: Sharpe(a) == Sharpe(b)`` (Ledoit and Wolf, 2008).

    The problem it solves
    ---------------------
    A naive test of a Sharpe difference assumes returns are independent and
    normal. Daily returns are neither: they autocorrelate in volatility and have
    fat tails. Both violations push a naive standard error *down*, so the naive
    test declares differences significant that are not. This test rebuilds the
    standard error with a HAC estimator that tolerates both.

    How it works
    ------------
    Write each Sharpe in terms of the first two uncentred moments,
    ``SR = mu / sqrt(gamma - mu**2)`` with ``gamma = E[r**2]``. Then the
    difference of Sharpes is a smooth function ``f`` of the four-vector
    ``v = (mu_a, mu_b, gamma_a, gamma_b)``. The delta method gives its variance
    as ``grad_f' @ Psi @ grad_f / T``, where ``Psi`` is the long-run (HAC)
    covariance of ``v``. Everything below is that sentence, spelled out.

    Parameters
    ----------
    excess_a, excess_b
        Aligned daily **excess** return streams.
    hac_lag
        Truncation lag; defaults to :func:`default_hac_lag`.

    Returns
    -------
    SharpeDifferenceTest
        Annualised difference, its standard error, the test statistic and a
        two-sided p-value.
    """
    first = np.asarray(excess_a, dtype=float)
    second = np.asarray(excess_b, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"Streams must be aligned: {first.shape} vs {second.shape}.")

    usable = np.isfinite(first) & np.isfinite(second)
    first, second = first[usable], second[usable]
    n_observations = first.size
    if n_observations < 10:
        raise ValueError(f"Need at least 10 paired observations, got {n_observations}.")

    mean_a, mean_b = first.mean(), second.mean()
    second_moment_a = (first**2).mean()
    second_moment_b = (second**2).mean()

    variance_a = second_moment_a - mean_a**2
    variance_b = second_moment_b - mean_b**2
    if variance_a <= 0 or variance_b <= 0:
        raise ValueError("A stream has non-positive variance; the Sharpe is undefined.")

    # Per-period Sharpes, and the difference the test is about.
    sharpe_a = mean_a / np.sqrt(variance_a)
    sharpe_b = mean_b / np.sqrt(variance_b)
    difference = sharpe_a - sharpe_b

    # Gradient of f(mu_a, mu_b, gamma_a, gamma_b) = SR_a - SR_b.
    #   d SR / d mu    =  gamma / (gamma - mu**2)**1.5
    #   d SR / d gamma = -mu / (2 * (gamma - mu**2)**1.5)
    scale_a = variance_a**1.5
    scale_b = variance_b**1.5
    gradient = np.array(
        [
            second_moment_a / scale_a,
            -second_moment_b / scale_b,
            -mean_a / (2.0 * scale_a),
            mean_b / (2.0 * scale_b),
        ]
    )

    # The four-dimensional series whose mean is v, demeaned for the HAC step.
    observations = np.column_stack([first, second, first**2, second**2])
    deviations = observations - observations.mean(axis=0)

    lag = default_hac_lag(n_observations) if hac_lag is None else hac_lag
    long_run_covariance = _hac_covariance(deviations, lag)

    variance_of_difference = gradient @ long_run_covariance @ gradient / n_observations
    standard_error = float(np.sqrt(max(variance_of_difference, 0.0)))

    if standard_error > 0:
        statistic = difference / standard_error
    elif difference == 0.0:
        # Comparing a stream against itself: the gradient is antisymmetric across
        # the duplicated moment pairs, so the HAC variance collapses to exactly
        # zero. That is 0/0, and the honest reading is "no difference, and no
        # evidence of one" - not an infinitely significant result.
        statistic = 0.0
    else:
        statistic = np.inf if difference > 0 else -np.inf
    p_value = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(statistic))))

    # Report on the annualised scale the rest of the project uses. The statistic
    # and p-value are scale-free, so only the first two fields change.
    return SharpeDifferenceTest(
        difference=annualise_sharpe(difference),
        standard_error=annualise_sharpe(standard_error),
        statistic=float(statistic),
        p_value=p_value,
        hac_lag=lag,
        n_observations=int(n_observations),
    )


# --------------------------------------------------------------------------- #
# P9: the deflated Sharpe ratio
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeflatedSharpe:
    """Outcome of a deflated-Sharpe computation.

    Attributes
    ----------
    deflated_probability
        The DSR itself: probability the true Sharpe exceeds zero, *given* that
        ``n_trials`` configurations were tried. Above 0.95 is the conventional
        significance line.
    observed_sharpe, benchmark_sharpe
        The strategy's annualised Sharpe, and the annualised expected maximum
        under the null (what the luckiest worthless configuration would score).
    n_trials, trial_dispersion
        The two selection-bias inputs, echoed back so a result always carries
        the population it was deflated against.
    n_observations, skewness, kurtosis
        Track-record length and shape. Longer records, positive skew and thin
        tails all make a given Sharpe more believable.
    """

    deflated_probability: float
    observed_sharpe: float
    benchmark_sharpe: float
    n_trials: int
    trial_dispersion: float
    n_observations: int
    skewness: float
    kurtosis: float

    @property
    def clears_benchmark(self) -> bool:
        """Did the observed Sharpe beat the selection-bias benchmark at all?

        A necessary condition, not a sufficient one: clearing it with a short or
        badly-shaped track record can still leave the DSR below 0.95.
        """
        return self.observed_sharpe > self.benchmark_sharpe


def expected_max_sharpe_under_null(
    trial_dispersion: float,
    n_trials: int,
) -> float:
    """Expected **maximum** Sharpe when every trial has zero true skill.

    This is the bar that selection bias creates. Search ``N`` worthless
    strategies and the best of them still looks good, purely because measuring
    anything ``N`` times and reporting the maximum is a biased procedure. The
    quantity returned here is how good, on average, that best-of-N looks.

    Bailey and Lopez de Prado (2014), from the Gumbel limit of the maximum of
    ``N`` independent normal draws::

        SR_0 = dispersion * [ (1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N*e)) ]

    Both inputs are worth understanding, because they behave very differently:

    * ``n_trials`` enters through two inverse-normal terms and therefore grows
      roughly logarithmically. Going from 50 trials to 305 - six times the
      searching - raises the bar by only about 27%.
    * ``trial_dispersion`` multiplies the whole expression, so it sets the
      answer almost single-handedly. Estimating it on the right population is
      the substantive methodological decision; the trial count is not.

    Parameters
    ----------
    trial_dispersion
        Standard deviation of the Sharpe ratios **across trials** - one Sharpe
        per configuration, then the spread of that list. This is *not* the
        seed-to-seed spread of a single configuration, and it is not the
        volatility of any return stream. Units follow the input: pass
        annualised Sharpes and the benchmark comes back annualised.
    n_trials
        Number of configurations that were given a chance to be selected. For
        this project the counting rule was pre-registered before any experiment
        existed: distinct ``cfg_hash`` values in the registry with
        ``split == "val"``.

    Returns
    -------
    float
        The expected maximum Sharpe under the null, in the same units as
        ``trial_dispersion``.
    """
    if n_trials < 2:
        raise ValueError(f"Need at least 2 trials to have a maximum, got {n_trials}.")
    if trial_dispersion < 0:
        raise ValueError(f"Dispersion must be non-negative, got {trial_dispersion}.")

    upper = scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
    lower = scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(
        trial_dispersion * ((1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * lower)
    )


def deflated_sharpe_ratio(
    excess_returns: np.ndarray,
    trial_dispersion: float,
    n_trials: int,
) -> DeflatedSharpe:
    """The deflated Sharpe ratio of one return stream.

    In one sentence: a t-test whose null hypothesis is *"the best of ``n_trials``
    lucky coins"* rather than the usual *"zero"*::

        DSR = Phi[ (SR - SR_0) * sqrt(T - 1)
                   / sqrt(1 - skew*SR + ((kurtosis - 1)/4) * SR**2) ]

    Reading the pieces:

    * ``SR - SR_0`` is the deflation. Beating zero is not the question once you
      have searched; beating what luck alone produces is.
    * ``sqrt(T - 1)`` rewards a longer track record, exactly as in any t-test.
    * The denominator is the standard error of a Sharpe estimate. Negative skew
      (small gains most days, occasional disasters) *increases* it and so shrinks
      the DSR - the formula declines to be impressed by strategies that look
      like selling insurance. Fat tails do the same. For normal returns the
      denominator is approximately 1 and the whole thing reduces to the plain
      t-test above.

    All internal algebra runs in **per-day** units, because ``T`` counts days;
    the returned dataclass reports Sharpes annualised, matching the registry.

    Parameters
    ----------
    excess_returns
        The strategy's daily excess-return stream - the concatenated
        out-of-sample stream, per the project's locked metric convention.
    trial_dispersion
        Across-trial standard deviation of **annualised** Sharpes. See
        :func:`expected_max_sharpe_under_null` for what this must and must not
        be measured over.
    n_trials
        Pre-registered trial count.

    Returns
    -------
    DeflatedSharpe
        The probability plus every input that produced it.
    """
    values = np.asarray(excess_returns, dtype=float)
    values = values[np.isfinite(values)]
    n_observations = values.size
    if n_observations < 10:
        raise ValueError(f"Need at least 10 observations, got {n_observations}.")

    observed_annual = annualised_excess_sharpe(values)

    # Everything below is per-day, so that sqrt(T - 1) with T in days is right.
    observed = deannualise_sharpe(observed_annual)
    benchmark_annual = expected_max_sharpe_under_null(trial_dispersion, n_trials)
    benchmark = deannualise_sharpe(benchmark_annual)

    skewness = float(scipy_stats.skew(values, bias=False))
    kurtosis = float(scipy_stats.kurtosis(values, fisher=False, bias=False))

    # Standard error of the Sharpe estimate under non-normal returns.
    variance_term = (
        1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed**2
    )
    if variance_term <= 0:
        # Only reachable with pathological higher moments; refuse rather than
        # return a number that looks fine.
        raise ValueError(
            f"Non-positive Sharpe variance term ({variance_term:.4g}); "
            "skewness/kurtosis are degenerate for this stream."
        )

    statistic = (observed - benchmark) * np.sqrt(n_observations - 1) / np.sqrt(variance_term)

    return DeflatedSharpe(
        deflated_probability=float(scipy_stats.norm.cdf(statistic)),
        observed_sharpe=float(observed_annual),
        benchmark_sharpe=float(benchmark_annual),
        n_trials=int(n_trials),
        trial_dispersion=float(trial_dispersion),
        n_observations=int(n_observations),
        skewness=skewness,
        kurtosis=kurtosis,
    )
