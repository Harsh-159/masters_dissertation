"""Tests for the thesis week: sentiment arms, statistical gates, agreement.

The two tests the plan calls out by name - "the filter unit tests of Section 11
must pass before any rung-4 number is believed" - are here: the forward filter
is causal to the bit, and it provably DIVERGES from hmmlearn's smoothed
posteriors (if the two ever matched, the wrong function would be in use).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from rrc.baselines.hmm_gate import (
    FittedHmmGate,
    build_observations,
    diagonal_variances,
    fit_hmm_gate,
    forward_filter,
    log_gaussian_diag,
    memberships_at,
)
from rrc.baselines.vix_gate import memberships_at as vix_memberships_at
from rrc.baselines.vix_gate import vix_tercile_labels
from rrc.eval.agreement import (
    decision_divergence,
    dwell_and_transition,
    hard_labels,
    label_agreement,
    occupancy_conditioned_sharpe,
)
from rrc.models.experts import count_parameters
from rrc.models.gating import FixedRoutingMixture, MixtureOfExperts, SignalRoutedMixture
from rrc.optim.fold_data import SegmentPack
from rrc.optim.toy import make_toy_fold


# --------------------------------------------------------------------------- #
# Synthetic two-regime observations for the HMM tests
# --------------------------------------------------------------------------- #
def two_regime_observations(n_days: int = 800, seed: int = 0) -> pd.DataFrame:
    """Calm and violent stretches alternating every 100 days - unambiguous."""
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days, name="date")

    violent = (np.arange(n_days) // 100) % 2 == 1
    returns = np.where(
        violent,
        generator.normal(-0.001, 0.03, n_days),
        generator.normal(0.0006, 0.006, n_days),
    )
    observations = pd.DataFrame(index=dates)
    observations["excess_return"] = returns
    observations["realised_vol"] = (
        observations["excess_return"].rolling(21).std(ddof=1)
    )
    return observations


def fitted_toy_gate(observations: pd.DataFrame) -> FittedHmmGate:
    train_window = observations.index[:500]
    return fit_hmm_gate(observations, train_window, n_states=2, random_state=0)


# --------------------------------------------------------------------------- #
# THE two filter tests (Section 11)
# --------------------------------------------------------------------------- #
def test_filter_is_causal_to_the_bit():
    """Altering any observation after t leaves filtered[: t+1] bit-identical."""
    observations = two_regime_observations().dropna()
    gate = fitted_toy_gate(two_regime_observations())

    # Re-run the filter on a copy whose FUTURE is violently corrupted.
    cut = 400
    corrupted = observations.to_numpy().copy()
    corrupted[cut + 1 :] *= -7.0

    from hmmlearn.hmm import GaussianHMM  # fit once more for raw params

    model = GaussianHMM(n_components=2, covariance_type="diag",
                        n_iter=200, random_state=0)
    model.fit(observations.to_numpy()[:480])

    def run(matrix: np.ndarray) -> np.ndarray:
        return forward_filter(
            matrix,
            np.log(model.startprob_ + 1e-300),
            np.log(model.transmat_ + 1e-300),
            model.means_,
            diagonal_variances(model),
        )

    baseline = run(observations.to_numpy())
    perturbed = run(corrupted)

    np.testing.assert_array_equal(baseline[: cut + 1], perturbed[: cut + 1])
    assert not np.allclose(baseline[cut + 1 :], perturbed[cut + 1 :])


def test_filter_diverges_from_hmmlearn_smoothed_posteriors():
    """predict_proba is forward-BACKWARD; matching it would mean look-ahead."""
    observations = two_regime_observations().dropna().to_numpy()

    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=2, covariance_type="diag",
                        n_iter=200, random_state=0)
    model.fit(observations)

    filtered = forward_filter(
        observations,
        np.log(model.startprob_ + 1e-300),
        np.log(model.transmat_ + 1e-300),
        model.means_,
        diagonal_variances(model),
    )
    smoothed = model.predict_proba(observations)

    # Same shape, both simplex rows - but NOT the same numbers.
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-9)
    largest_gap = np.abs(filtered - smoothed).max()
    assert largest_gap > 1e-3, (
        "filtered and smoothed probabilities agree suspiciously exactly - "
        "the wrong function is being called somewhere"
    )


def test_filter_rows_are_probability_distributions():
    observations = two_regime_observations().dropna()
    gate = fitted_toy_gate(two_regime_observations())

    assert gate.filtered.shape[1] == gate.n_states
    np.testing.assert_allclose(gate.filtered.sum(axis=1), 1.0, atol=1e-9)
    assert (gate.filtered >= 0).all()


def test_hmm_finds_the_planted_regimes():
    """On unambiguous calm/violent data the filter must track the truth."""
    observations = two_regime_observations()
    gate = fitted_toy_gate(observations)

    finite = observations.dropna()
    truth = ((np.arange(len(observations)) // 100) % 2 == 1)[-len(finite):]
    inferred = hard_labels(gate.filtered)

    agreement = label_agreement(inferred, truth.astype(int))
    assert agreement["ami"] > 0.5  # far above chance on a planted problem


def test_log_gaussian_diag_matches_scipy():
    from scipy.stats import norm

    observations = np.array([[0.5, -1.0]])
    means = np.array([[0.0, 0.0]])
    variances = np.array([[1.0, 4.0]])

    ours = log_gaussian_diag(observations, means, variances)[0, 0]
    reference = norm.logpdf(0.5, 0, 1) + norm.logpdf(-1.0, 0, 2)
    assert ours == pytest.approx(reference, rel=1e-12)


def test_memberships_at_rejects_warmup_signal_days():
    observations = two_regime_observations()
    gate = fitted_toy_gate(observations)

    with pytest.raises(ValueError, match="warm-up"):
        memberships_at(gate, observations.index, np.array([0]))  # inside warm-up


# --------------------------------------------------------------------------- #
# The VIX gate
# --------------------------------------------------------------------------- #
def make_vix_series(n_days: int = 900, seed: int = 1) -> tuple[pd.Series, pd.DatetimeIndex]:
    generator = np.random.default_rng(seed)
    calendar = pd.bdate_range("2010-01-01", periods=n_days, name="date")
    level = 20.0 + np.cumsum(generator.normal(0, 0.5, n_days))
    return pd.Series(np.clip(level, 9.0, 80.0), index=calendar), calendar


def test_vix_labels_are_causal():
    """Perturbing the future must leave every past label bit-identical."""
    vix, calendar = make_vix_series()
    baseline = vix_tercile_labels(vix, calendar)

    corrupted = vix.copy()
    corrupted.iloc[600:] *= 5.0
    perturbed = vix_tercile_labels(corrupted, calendar)

    pd.testing.assert_series_equal(baseline.iloc[:600], perturbed.iloc[:600])
    assert not baseline.iloc[600:].equals(perturbed.iloc[600:])


def test_vix_labels_use_yesterdays_value():
    """A spike on day t must not move day t's own label (one-day lag)."""
    vix, calendar = make_vix_series()
    baseline = vix_tercile_labels(vix, calendar)

    spiked = vix.copy()
    spiked.iloc[700] = 500.0
    perturbed = vix_tercile_labels(spiked, calendar)

    assert baseline.iloc[700] == perturbed.iloc[700]      # own day unmoved
    assert perturbed.iloc[701] == 2.0                     # visible from t+1


def test_vix_memberships_are_one_hot_and_all_terciles_occur():
    vix, calendar = make_vix_series()
    labels = vix_tercile_labels(vix, calendar)

    signal_positions = np.arange(300, 890, 5)
    memberships = vix_memberships_at(labels, signal_positions)

    np.testing.assert_allclose(memberships.sum(axis=1), 1.0)
    assert set(np.unique(memberships)) == {0.0, 1.0}
    assert (memberships.sum(axis=0) > 0).all()  # every tercile appears


def test_vix_memberships_reject_warmup():
    vix, calendar = make_vix_series()
    labels = vix_tercile_labels(vix, calendar)
    with pytest.raises(ValueError, match="warm-up"):
        vix_memberships_at(labels, np.array([10]))


# --------------------------------------------------------------------------- #
# The rung-3 arms: parameter counts and information routing
# --------------------------------------------------------------------------- #
def signal_pack_from_toy() -> SegmentPack:
    """The planted toy with a synthetic 5-wide signal vector attached."""
    import dataclasses

    toy = make_toy_fold(seed=0)
    generator = np.random.default_rng(7)
    signals = torch.as_tensor(
        generator.normal(size=(toy.pack.n_signals, 5)), dtype=torch.float64
    )
    return dataclasses.replace(toy.pack, signals=signals)


def test_a8_parameter_counts_are_exact():
    """The A8 arithmetic: gate arm adds K*d, expert arm adds K*d*N."""
    n_state, n_assets, n_experts, n_signals = 300, 12, 3, 5
    base = MixtureOfExperts(n_state, n_assets, n_experts)

    to_gate = SignalRoutedMixture(n_state, n_assets, n_experts, n_signals,
                                  signals_to_gate=True, signals_to_experts=False)
    to_experts = SignalRoutedMixture(n_state, n_assets, n_experts, n_signals,
                                     signals_to_gate=False, signals_to_experts=True)
    to_both = SignalRoutedMixture(n_state, n_assets, n_experts, n_signals,
                                  signals_to_gate=True, signals_to_experts=True)

    base_count = count_parameters(base)
    assert count_parameters(to_gate) == base_count + n_signals * n_experts        # K*d
    assert count_parameters(to_experts) == base_count + n_signals * n_experts * n_assets
    assert count_parameters(to_both) == (
        base_count + n_signals * n_experts + n_signals * n_experts * n_assets
    )


def test_signals_reach_exactly_the_component_they_are_routed_to():
    """3a: signals move pi but not the experts' proposals; 3b: the reverse."""
    pack = signal_pack_from_toy()
    perturbed_signals = pack.signals * 3.0 + 1.0
    import dataclasses
    perturbed_pack = dataclasses.replace(pack, signals=perturbed_signals)

    torch.manual_seed(0)
    gate_arm = SignalRoutedMixture(8, 3, 3, 5, signals_to_gate=True,
                                   signals_to_experts=False)
    pi_before, experts_before = gate_arm.route_pack(pack)
    pi_after, experts_after = gate_arm.route_pack(perturbed_pack)
    assert not torch.allclose(pi_before, pi_after)          # routing moved
    torch.testing.assert_close(experts_before, experts_after)  # proposals did not

    torch.manual_seed(0)
    expert_arm = SignalRoutedMixture(8, 3, 3, 5, signals_to_gate=False,
                                     signals_to_experts=True)
    pi_before, experts_before = expert_arm.route_pack(pack)
    pi_after, experts_after = expert_arm.route_pack(perturbed_pack)
    torch.testing.assert_close(pi_before, pi_after)         # routing did not move
    assert not torch.allclose(experts_before, experts_after)   # proposals did


def test_signal_arm_requires_signals_and_a_direction():
    with pytest.raises(ValueError, match="At least one"):
        SignalRoutedMixture(8, 3, 3, 5, signals_to_gate=False,
                            signals_to_experts=False)

    toy = make_toy_fold(seed=0)  # its pack carries no signals
    arm = SignalRoutedMixture(8, 3, 3, 5, signals_to_gate=True,
                              signals_to_experts=False)
    with pytest.raises(ValueError, match="market signals"):
        arm.route_pack(toy.pack)


# --------------------------------------------------------------------------- #
# The rung-4 fixed-routing mixture
# --------------------------------------------------------------------------- #
def test_fixed_routing_uses_the_given_memberships_exactly():
    import dataclasses

    toy = make_toy_fold(seed=0)
    fixed = torch.zeros(toy.pack.n_signals, 2, dtype=torch.float64)
    fixed[: toy.pack.n_signals // 2, 0] = 1.0
    fixed[toy.pack.n_signals // 2 :, 1] = 1.0
    pack = dataclasses.replace(toy.pack, fixed_memberships=fixed)

    model = FixedRoutingMixture(n_state_features=8, n_assets=3, n_experts=2)
    memberships, expert_weights = model.route_pack(pack)

    torch.testing.assert_close(memberships, fixed)
    # With one-hot routing the blend IS the routed expert's proposal.
    blended = model.weights_for_pack(pack)
    torch.testing.assert_close(blended[0], expert_weights[0, 0])
    torch.testing.assert_close(blended[-1], expert_weights[-1, 1])


def test_fixed_routing_has_no_gate_parameters_and_experts_learn():
    model = FixedRoutingMixture(n_state_features=8, n_assets=3, n_experts=2)
    assert count_parameters(model) == 2 * (8 * 3 + 3)  # experts only, no gate

    import dataclasses
    toy = make_toy_fold(seed=0)
    fixed = torch.full((toy.pack.n_signals, 2), 0.5, dtype=torch.float64)
    pack = dataclasses.replace(toy.pack, fixed_memberships=fixed)

    model.weights_for_pack(pack).sum().backward()
    for expert in model.experts:
        assert expert.linear.weight.grad is not None
        assert float(expert.linear.weight.grad.abs().sum()) > 0


def test_fixed_routing_demands_memberships_on_the_pack():
    toy = make_toy_fold(seed=0)
    model = FixedRoutingMixture(8, 3, 2)
    with pytest.raises(ValueError, match="fixed_memberships"):
        model.route_pack(toy.pack)


# --------------------------------------------------------------------------- #
# The agreement metrics
# --------------------------------------------------------------------------- #
def test_agreement_is_one_for_identical_partitions_even_renamed():
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    renamed = np.array([2, 2, 0, 0, 1, 1, 2, 0])  # same partition, new names

    agreement = label_agreement(labels, renamed)
    assert agreement["ami"] == pytest.approx(1.0)
    assert agreement["ari"] == pytest.approx(1.0)


def test_agreement_is_near_zero_for_unrelated_labels():
    generator = np.random.default_rng(0)
    first = generator.integers(0, 3, 3000)
    second = generator.integers(0, 3, 3000)

    agreement = label_agreement(first, second)
    assert abs(agreement["ami"]) < 0.05
    assert abs(agreement["ari"]) < 0.05


def test_dwell_and_transition_arithmetic():
    labels = np.array([0, 0, 0, 1, 1, 2])  # runs of 3, 2, 1 -> mean dwell 2
    statistics = dwell_and_transition(labels)
    assert statistics["mean_dwell"] == pytest.approx(2.0)
    assert statistics["transition_rate"] == pytest.approx(2 / 5)


def test_decision_divergence_is_zero_for_identical_routings():
    generator = np.random.default_rng(1)
    memberships = generator.dirichlet(np.ones(3), size=20)
    experts = generator.dirichlet(np.ones(4), size=(20, 3))

    assert decision_divergence(memberships, memberships, experts) == pytest.approx(0.0)


def test_decision_divergence_pads_a_reduced_hmm():
    """A 2-state gate against 3 experts: the missing state gets zero mass."""
    learned = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    statistical_2state = np.array([[1.0, 0.0], [1.0, 0.0]])
    experts = np.zeros((2, 3, 2))
    experts[:, 0] = [1.0, 0.0]
    experts[:, 1] = [0.5, 0.5]
    experts[:, 2] = [0.0, 1.0]

    divergence = decision_divergence(learned, statistical_2state, experts)
    # Day 0: identical (both route to expert 0). Day 1: [0,1] vs [1,0] -> L1 = 2.
    assert divergence == pytest.approx(1.0)


def test_occupancy_conditioned_sharpe_attributes_days_to_regimes():
    n_days = 60
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    # Regime 0 governs profitable days, regime 1 governs losing days.
    signal_positions = np.array([9, 29, 49])
    signal_labels = np.array([0, 1, 0])

    returns = np.zeros(n_days)
    returns[10:30] = 0.01    # governed by label 0
    returns[30:50] = -0.01   # governed by label 1
    returns[50:] = 0.01      # governed by label 0
    net = pd.Series(returns, index=dates)
    risk_free = pd.Series(0.0, index=dates)

    by_regime = occupancy_conditioned_sharpe(net, risk_free, signal_labels,
                                             signal_positions)
    assert by_regime[0]["sharpe"] > 0
    assert by_regime[1]["sharpe"] < 0
    assert by_regime[0]["n_days"] == 30
    assert by_regime[1]["n_days"] == 20
