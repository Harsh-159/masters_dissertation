"""Tests for the reservoir: construction, dynamics, caching, and health.

The properties that matter most here are the ones that would otherwise fail
silently:

* the SCR's spectral radius must be **exactly** what was asked for, because the
  echo state property depends on it and nothing downstream re-checks;
* the construction must be **deterministic**, since the entire seed budget
  argument (D2) rests on there being no reservoir-draw variance;
* the state must actually **remember** - a reservoir whose state depends only on
  today's input is a pointless nonlinear map;
* the influence of the arbitrary initial state must **fade**, or the washout
  period is not long enough and every result inherits a startup artefact;
* the usable mask must exclude every NaN, or training fails far from the cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rrc.reservoir.cache import (
    CachedStates,
    build_usable_mask,
    cache_path,
    load_states,
    reservoir_config_hash,
    save_states,
)
from rrc.reservoir.diagnostics import (
    canonical_correlations,
    memory_capacity,
    saturation_statistics,
    state_pca_spectrum,
)
from rrc.reservoir.esn import build_esn
from rrc.reservoir.scr import (
    build_cycle_matrix,
    build_scr,
    build_scr_input_weights,
    calibrate_input_scale,
    pi_digits,
    pi_sign_sequence,
    run_reservoir,
)


@pytest.fixture
def small_scr():
    """A small deterministic reservoir, fast enough for every test."""
    return build_scr(
        n_reservoir=50, n_inputs=3, spectral_radius=0.9,
        input_scale=0.5, leak_rate=0.3,
    )


@pytest.fixture
def driving_inputs():
    """A finite, non-constant 3-channel input series."""
    generator = np.random.default_rng(0)
    return generator.normal(0.0, 1.0, size=(600, 3))


# --------------------------------------------------------------------------- #
# The aperiodic sign pattern
# --------------------------------------------------------------------------- #
def test_pi_digits_are_correct():
    """Computed, not transcribed - so verify against the known expansion."""
    assert pi_digits(15) == [1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]


def test_pi_digits_stay_correct_at_length():
    """Guard digits must be sufficient that later digits are not rounding noise."""
    long_expansion = pi_digits(500)
    assert len(long_expansion) == 500
    assert long_expansion[:10] == [1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    assert all(0 <= digit <= 9 for digit in long_expansion)


def test_sign_sequence_is_balanced_and_aperiodic():
    """Roughly half the signs are positive, and the pattern does not repeat.

    A periodic pattern would make groups of units respond identically, silently
    shrinking the effective reservoir.
    """
    signs = pi_sign_sequence(300)

    assert set(np.unique(signs)) <= {-1.0, 1.0}
    assert 0.35 < (signs > 0).mean() < 0.65
    # No short period: the first half must not equal the second half.
    assert not np.array_equal(signs[:150], signs[150:])


def test_sign_sequence_is_reproducible():
    np.testing.assert_array_equal(pi_sign_sequence(100), pi_sign_sequence(100))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_cycle_matrix_has_exactly_one_connection_per_unit():
    recurrent = build_cycle_matrix(6, 0.9)

    assert (np.count_nonzero(recurrent, axis=1) == 1).all()
    assert recurrent[0, 5] == pytest.approx(0.9)  # unit 0 closes the ring
    assert recurrent[3, 2] == pytest.approx(0.9)


@pytest.mark.parametrize("requested_radius", [0.5, 0.9, 0.99])
def test_scr_spectral_radius_is_exact(requested_radius):
    """The ring's eigenvalues all have modulus r, so no rescaling is needed.

    This is a real advantage over the random construction, where the radius is
    only achieved up to the eigenvalue solver's precision.
    """
    weights = build_scr(
        n_reservoir=64, n_inputs=2, spectral_radius=requested_radius,
        input_scale=0.5, leak_rate=0.3,
    )
    assert weights.spectral_radius() == pytest.approx(requested_radius, abs=1e-12)


def test_scr_construction_is_deterministic():
    """No randomness anywhere - the basis of the reduced seed budget (D2)."""
    first = build_scr(64, 3, 0.9, 0.5, 0.3)
    second = build_scr(64, 3, 0.9, 0.5, 0.3)

    np.testing.assert_array_equal(first.recurrent_weights, second.recurrent_weights)
    np.testing.assert_array_equal(first.input_weights, second.input_weights)


def test_scr_input_weights_give_each_unit_one_input():
    """W_in = v * D * C: one connection per unit, magnitude v, sign +/-1."""
    input_weights = build_scr_input_weights(n_reservoir=9, n_inputs=3, input_scale=0.4)

    assert (np.count_nonzero(input_weights, axis=1) == 1).all()
    nonzero = input_weights[input_weights != 0]
    np.testing.assert_allclose(np.abs(nonzero), 0.4)
    # Cyclic assignment: unit i listens to input i mod n_inputs.
    assert input_weights[0, 0] != 0 and input_weights[1, 1] != 0 and input_weights[2, 2] != 0
    assert input_weights[3, 0] != 0


def test_each_input_drives_an_equal_share_of_the_reservoir():
    input_weights = build_scr_input_weights(n_reservoir=300, n_inputs=10, input_scale=1.0)
    units_per_input = np.count_nonzero(input_weights, axis=0)
    assert (units_per_input == 30).all()


def test_spectral_radius_at_or_above_one_is_rejected():
    """Without rho < 1 the echo state property fails and states can diverge."""
    with pytest.raises(ValueError, match="echo state property"):
        build_scr(64, 2, 1.0, 0.5, 0.3)


def test_invalid_leak_rate_is_rejected():
    with pytest.raises(ValueError, match="leak_rate"):
        build_scr(64, 2, 0.9, 0.5, 0.0)


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
def test_states_have_the_expected_shape(small_scr, driving_inputs):
    states, pre_activations = run_reservoir(small_scr, driving_inputs)
    assert states.shape == (600, 50)
    assert pre_activations.shape == (600, 50)


def test_states_stay_bounded(small_scr, driving_inputs):
    """tanh bounds the update, and leaky integration keeps the state in [-1, 1]."""
    states, _ = run_reservoir(small_scr, driving_inputs)
    assert np.abs(states).max() <= 1.0 + 1e-12
    assert np.isfinite(states).all()


def test_state_depends_on_history_not_only_on_the_current_input(small_scr):
    """The defining property: identical inputs today, different pasts, different states.

    If this failed the reservoir would be a memoryless nonlinear map and the
    whole premise - that x_t encodes the fading history - would be false.
    """
    generator = np.random.default_rng(1)
    shared_tail = generator.normal(size=(1, 3))

    first_history = np.vstack([generator.normal(size=(200, 3)), shared_tail])
    second_history = np.vstack([generator.normal(size=(200, 3)), shared_tail])

    first_states, _ = run_reservoir(small_scr, first_history)
    second_states, _ = run_reservoir(small_scr, second_history)

    assert not np.allclose(first_states[-1], second_states[-1])


def test_echo_state_property_makes_the_initial_state_fade(small_scr, driving_inputs):
    """Two different starting states must converge given the same input.

    This is what the washout period is for, and it is worth measuring rather
    than assuming: if the influence of x_0 did not decay, every result would
    carry an artefact of an arbitrary choice.
    """
    generator = np.random.default_rng(2)
    far_apart = generator.uniform(-1, 1, size=small_scr.n_reservoir)

    from_zero, _ = run_reservoir(small_scr, driving_inputs, initial_state=None)
    from_elsewhere, _ = run_reservoir(small_scr, driving_inputs, initial_state=far_apart)

    early_gap = np.abs(from_zero[0] - from_elsewhere[0]).max()
    late_gap = np.abs(from_zero[-1] - from_elsewhere[-1]).max()

    assert late_gap < early_gap
    assert late_gap < 1e-6, "the initial state has not faded; washout is too short"


def test_leak_rate_of_one_removes_the_leaky_integration():
    """alpha = 1 reduces the update to the plain tanh form."""
    weights = build_scr(32, 2, 0.9, 0.5, leak_rate=1.0)
    inputs = np.random.default_rng(3).normal(size=(50, 2))

    states, pre_activations = run_reservoir(weights, inputs)
    np.testing.assert_allclose(states, np.tanh(pre_activations), atol=1e-12)


def test_non_finite_inputs_are_rejected(small_scr, driving_inputs):
    """A NaN would poison every subsequent state, so it must stop the run."""
    poisoned = driving_inputs.copy()
    poisoned[10, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        run_reservoir(small_scr, poisoned)


def test_wrong_input_width_is_rejected(small_scr):
    with pytest.raises(ValueError, match="input channels"):
        run_reservoir(small_scr, np.zeros((100, 5)))


# --------------------------------------------------------------------------- #
# Input-scale calibration
# --------------------------------------------------------------------------- #
def test_calibration_hits_the_saturation_target(driving_inputs):
    """The whole point: reach a chosen pre-tanh spread rather than guessing v."""
    input_scale, achieved_std = calibrate_input_scale(
        inputs=driving_inputs, n_reservoir=50, spectral_radius=0.9,
        leak_rate=0.3, target_pre_activation_std=0.7,
    )
    assert achieved_std == pytest.approx(0.7, rel=0.03)
    assert input_scale > 0


def test_calibration_adapts_to_input_magnitude():
    """Tiny inputs need a large scale; large inputs need a small one.

    Raw daily log returns are ~0.01, which is exactly why an uncalibrated
    reservoir would sit in the linear region of tanh.
    """
    generator = np.random.default_rng(4)
    tiny_inputs = 0.01 * generator.normal(size=(400, 3))
    large_inputs = 10.0 * generator.normal(size=(400, 3))

    scale_for_tiny, _ = calibrate_input_scale(tiny_inputs, 40, 0.9, 0.3)
    scale_for_large, _ = calibrate_input_scale(large_inputs, 40, 0.9, 0.3)

    assert scale_for_tiny > scale_for_large


def test_calibration_rejects_a_dead_input_series():
    """An all-zero input leaves the state at zero, so no scale can fix it.

    Note this is the *zero* case, not the merely constant one: a constant input
    still produces varying pre-activations, because the +/-1 sign pattern gives
    neighbouring units opposite drive. Only an input carrying no signal at all
    collapses the spread to exactly zero, and that is what must be caught.
    """
    with pytest.raises(ValueError, match="zero spread"):
        calibrate_input_scale(np.zeros((200, 2)), 40, 0.9, 0.3)


def test_calibration_survives_a_constant_nonzero_input():
    """A constant input is degenerate for finance but fine for the reservoir."""
    input_scale, achieved_std = calibrate_input_scale(
        np.ones((300, 2)), 40, 0.9, 0.3, target_pre_activation_std=0.7
    )
    assert input_scale > 0
    assert achieved_std == pytest.approx(0.7, rel=0.05)


# --------------------------------------------------------------------------- #
# The random ESN variant
# --------------------------------------------------------------------------- #
def test_esn_reaches_the_requested_spectral_radius():
    """Achieved by rescaling, so exact only to solver precision - unlike the SCR."""
    weights = build_esn(80, 3, spectral_radius=0.9, input_scale=0.5, leak_rate=0.3, seed=0)
    assert weights.spectral_radius() == pytest.approx(0.9, abs=1e-8)


def test_esn_is_sparse_and_seed_dependent():
    first = build_esn(100, 3, 0.9, 0.5, 0.3, seed=0)
    second = build_esn(100, 3, 0.9, 0.5, 0.3, seed=1)

    density = np.count_nonzero(first.recurrent_weights) / first.recurrent_weights.size
    assert 0.05 < density < 0.15
    assert not np.array_equal(first.recurrent_weights, second.recurrent_weights)


def test_esn_is_reproducible_for_a_given_seed():
    first = build_esn(60, 2, 0.9, 0.5, 0.3, seed=7)
    second = build_esn(60, 2, 0.9, 0.5, 0.3, seed=7)
    np.testing.assert_array_equal(first.recurrent_weights, second.recurrent_weights)


def test_both_reservoir_kinds_share_the_same_dynamics(driving_inputs):
    """Any SCR-vs-ESN difference must come from the weights, not a second code path."""
    scr = build_scr(50, 3, 0.9, 0.5, 0.3)
    esn = build_esn(50, 3, 0.9, 0.5, 0.3, seed=0)

    scr_states, _ = run_reservoir(scr, driving_inputs)
    esn_states, _ = run_reservoir(esn, driving_inputs)

    assert scr_states.shape == esn_states.shape
    assert np.isfinite(scr_states).all() and np.isfinite(esn_states).all()


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_memory_capacity_grows_towards_the_edge_of_stability():
    """A larger spectral radius should remember further back.

    This is the shape figure F2 must show. A flat or inverted curve would mean
    the construction or the scaling is wrong.

    Probed with a **small input scale**, which is the standard convention and is
    load-bearing here: memory capacity measures *linear* recoverability, so
    driving the reservoir into the curved part of tanh trades that memory for
    nonlinear computation and flattens the curve. Measured at an operating-size
    input scale this property genuinely does not hold, which is a fact about the
    measurement, not about the reservoir.
    """
    def capacity_at(radius: float) -> float:
        weights = build_scr(60, 1, radius, input_scale=0.05, leak_rate=0.6)
        return memory_capacity(
            weights, max_delay=40, n_train=1500, n_test=800, washout=150
        ).total

    assert capacity_at(0.9) > 1.5 * capacity_at(0.5)


def test_driving_the_reservoir_hard_suppresses_linear_memory():
    """Documents why the MC probe uses a small input scale.

    The same reservoir, measured at an operating-size input scale, shows much
    lower capacity. Pinning this as a test stops anyone (including a future me)
    from "fixing" the probe scale back to 1.0 and then puzzling over a flat F2.
    """
    def capacity_at_scale(input_scale: float) -> float:
        weights = build_scr(60, 1, 0.9, input_scale=input_scale, leak_rate=0.6)
        return memory_capacity(
            weights, max_delay=40, n_train=1500, n_test=800, washout=150
        ).total

    assert capacity_at_scale(0.05) > 2 * capacity_at_scale(1.0)


def test_memory_capacity_is_bounded_by_the_reservoir_size():
    """The theoretical ceiling: you cannot recover more than N_r past inputs."""
    weights = build_scr(40, 1, 0.9, input_scale=1.0, leak_rate=0.6)
    result = memory_capacity(weights, max_delay=30, n_train=1200, n_test=600, washout=150)

    assert 0 < result.total <= 40
    assert len(result.per_delay) == 30
    # Recoverability must decay with delay: the recent past beats the distant past.
    assert result.per_delay[0] > result.per_delay[-1]


def test_memory_capacity_requires_a_single_input_channel(small_scr):
    with pytest.raises(ValueError, match="1-input reservoir"):
        memory_capacity(small_scr)


def test_saturation_statistics_detect_a_saturated_reservoir():
    """Over-scaled inputs pin the units at +/-1 and the diagnostic must say so."""
    linear_region = np.random.default_rng(5).normal(0, 0.1, size=(1000, 20))
    saturated = np.random.default_rng(5).normal(0, 12.0, size=(1000, 20))

    assert saturation_statistics(linear_region)["saturated_fraction"] < 0.01
    assert saturation_statistics(saturated)["saturated_fraction"] > 0.5


def test_pca_spectrum_sums_to_one_and_decreases(small_scr, driving_inputs):
    states, _ = run_reservoir(small_scr, driving_inputs)
    spectrum = state_pca_spectrum(states)

    assert spectrum.sum() == pytest.approx(1.0, abs=1e-9)
    assert (np.diff(spectrum) <= 1e-12).all()


def test_canonical_correlation_is_one_for_a_rotated_copy():
    """A rotation changes the basis but not the information, so CCA must give 1."""
    generator = np.random.default_rng(6)
    states = generator.normal(size=(500, 10))
    rotation, _ = np.linalg.qr(generator.normal(size=(10, 10)))

    correlations = canonical_correlations(states, states @ rotation, n_components=10)
    np.testing.assert_allclose(correlations, 1.0, atol=1e-8)


def test_canonical_correlation_is_low_for_unrelated_states():
    generator = np.random.default_rng(7)
    first = generator.normal(size=(2000, 5))
    second = generator.normal(size=(2000, 5))

    assert canonical_correlations(first, second, n_components=5).mean() < 0.2


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def test_usable_mask_excludes_warmup_and_washout():
    mask = build_usable_mask(n_days=1000, first_computed_row=100, washout=250)

    assert mask[:350].sum() == 0     # warm-up plus washout
    assert mask[350:].all()
    assert mask.sum() == 650


def test_usable_mask_handles_a_washout_longer_than_the_sample():
    mask = build_usable_mask(n_days=100, first_computed_row=50, washout=250)
    assert mask.sum() == 0


def test_cache_round_trip_preserves_everything(tmp_path):
    """States, mask, dates and config must survive save and load unchanged."""
    n_days, n_units = 400, 20
    states = np.full((n_days, n_units), np.nan, dtype=np.float32)
    states[100:] = np.random.default_rng(8).normal(size=(300, n_units)).astype(np.float32)

    reservoir_config = {"universe": "etf10", "n_reservoir": n_units, "spectral_radius": 0.9}
    cached = CachedStates(
        states=states,
        dates=pd.bdate_range("2010-01-01", periods=n_days, name="date"),
        usable=build_usable_mask(n_days, 100, 250),
        reservoir_config=reservoir_config,
        cfg_hash=reservoir_config_hash(reservoir_config),
        kind="scr",
    )

    path = save_states(cached, tmp_path)
    reloaded = load_states(path)

    np.testing.assert_array_equal(reloaded.states, cached.states)
    np.testing.assert_array_equal(reloaded.usable, cached.usable)
    # Dates are canonicalised to nanoseconds by the cache, so compare on that.
    pd.testing.assert_index_equal(reloaded.dates, cached.dates.as_unit("ns"))
    assert reloaded.cfg_hash == cached.cfg_hash
    assert reloaded.kind == "scr"
    assert reloaded.n_usable == cached.n_usable


def test_cache_filename_changes_with_the_configuration(tmp_path):
    """A changed hyperparameter must produce a new file, never a stale hit."""
    first = reservoir_config_hash({"universe": "etf10", "spectral_radius": 0.9})
    second = reservoir_config_hash({"universe": "etf10", "spectral_radius": 0.95})

    assert first != second
    assert cache_path(tmp_path, "etf10", "scr", first) != cache_path(
        tmp_path, "etf10", "scr", second
    )


def test_cache_rejects_nan_inside_the_usable_region(tmp_path):
    """The mask and the data must agree, or training fails far from the cause."""
    states = np.zeros((300, 5), dtype=np.float32)
    states[280, 0] = np.nan  # a NaN inside what the mask calls usable

    cached = CachedStates(
        states=states,
        dates=pd.bdate_range("2010-01-01", periods=300, name="date"),
        usable=build_usable_mask(300, 0, 250),
        reservoir_config={"universe": "etf10"},
        cfg_hash="deadbeef",
        kind="scr",
    )
    with pytest.raises(ValueError, match="usable mask does not match"):
        cached.assert_usable_states_are_finite()


def test_cache_requires_a_universe_in_the_config(tmp_path):
    cached = CachedStates(
        states=np.zeros((10, 2), dtype=np.float32),
        dates=pd.bdate_range("2010-01-01", periods=10, name="date"),
        usable=np.ones(10, dtype=bool),
        reservoir_config={},  # missing "universe"
        cfg_hash="abc",
        kind="scr",
    )
    with pytest.raises(ValueError, match="must contain a 'universe' key"):
        save_states(cached, tmp_path)
