"""Causality tests C1-C6: the look-ahead firewall.

These are the tests the whole dissertation leans on. A single leak invalidates
every result downstream, and leaks are invisible in output - a leaking backtest
looks *better*, not broken. So each property is stated as an executable
assertion rather than a convention to be remembered.

===  ==================================================================
C1   max-timestamp: a feature at t depends on no raw datum after t
C2   shuffle-future: destroying post-cut structure changes nothing at/before
     the cut (the strategy-level collapse assertion lands with P5)
C3   cash exclusion: cash is absent from reservoir input and covariance,
     present in accounting, and scores exactly zero excess return
C4   no-backfill: every series starts at its inception; NaN before, never 0
C5   truncation invariance: deleting future rows changes nothing in the past
C6   z-window causality: the trailing z-score at t uses only data up to t
===  ==================================================================

C7 (no same-bar fill) requires the execution ledger and lands with P2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rrc.data.availability import build_availability_matrix
from rrc.data.calendar import align_panel, build_trading_calendar, normalise_index
from rrc.data.fred import annual_yield_to_daily_return
from rrc.data.loaders import log_returns, rolling_zscore
from rrc.data.signals import align_to_calendar_forward_only, shift_then_zscore
from tests.conftest import permute_after

Z_WINDOW = 252
Z_MIN_PERIODS = 126


# ===========================================================================
# C1 - max-timestamp assert
# ===========================================================================
def test_c1_future_raw_value_cannot_change_a_past_feature(synthetic_signal):
    """Perturbing the raw input at t+1 must leave every feature at <= t identical.

    This is the operational form of "the newest raw datum a feature depends on
    has timestamp <= t". Rather than tracking dependency lags symbolically, we
    change the future and check the past did not move.
    """
    baseline = shift_then_zscore(synthetic_signal, 1, Z_WINDOW, Z_MIN_PERIODS)

    perturbation_date = synthetic_signal.index[300]
    perturbed_input = synthetic_signal.copy()
    perturbed_input.loc[perturbation_date] *= 10.0  # a violent change

    perturbed = shift_then_zscore(perturbed_input, 1, Z_WINDOW, Z_MIN_PERIODS)

    up_to_perturbation = baseline.index <= perturbation_date
    pd.testing.assert_series_equal(
        baseline[up_to_perturbation], perturbed[up_to_perturbation]
    )


def test_c1_publication_lag_actually_delays_the_signal(synthetic_signal):
    """A one-day lag means the feature at t reflects the raw value at t-1."""
    lagged_only = synthetic_signal.shift(1)
    assert lagged_only.iloc[5] == synthetic_signal.iloc[4]
    # And the value at t itself is *not* used at t.
    assert lagged_only.iloc[5] != synthetic_signal.iloc[5]


def test_c1_sentiment_lag_is_at_least_a_business_week(synthetic_signal):
    """Sentiment carries a >= 5 business-day lag, so t-1..t-4 are invisible at t."""
    sentiment_like = shift_then_zscore(synthetic_signal, 5, Z_WINDOW, Z_MIN_PERIODS)

    perturbation_index = 300
    perturbed_input = synthetic_signal.copy()
    perturbed_input.iloc[perturbation_index] *= 10.0
    perturbed = shift_then_zscore(perturbed_input, 5, Z_WINDOW, Z_MIN_PERIODS)

    # The perturbation may only surface 5 rows later, never sooner.
    first_difference = np.nonzero(
        ~np.isclose(sentiment_like.to_numpy(), perturbed.to_numpy(), equal_nan=True)
    )[0]
    assert len(first_difference) > 0, "perturbation had no effect at all - check the fixture"
    assert first_difference[0] >= perturbation_index + 5


# ===========================================================================
# C2 - shuffle-future collapse (data-level at P1; strategy-level at P5)
# ===========================================================================
def test_c2_permutation_leaves_the_past_bit_identical(synthetic_signal):
    """The C2 harness must not disturb anything at or before the cut."""
    cut = synthetic_signal.index[200]
    permuted = permute_after(synthetic_signal, cut, seed=0)

    before = synthetic_signal.index <= cut
    pd.testing.assert_series_equal(synthetic_signal[before], permuted[before])


def test_c2_permutation_actually_destroys_future_structure(synthetic_signal):
    """It must genuinely shuffle the future, or the P5 test would be vacuous."""
    cut = synthetic_signal.index[200]
    permuted = permute_after(synthetic_signal, cut, seed=0)

    after = synthetic_signal.index > cut
    original_after = synthetic_signal[after]
    permuted_after = permuted[after]

    # Same multiset of values ...
    np.testing.assert_allclose(
        np.sort(original_after.to_numpy()), np.sort(permuted_after.to_numpy())
    )
    # ... in a different order.
    assert not np.allclose(original_after.to_numpy(), permuted_after.to_numpy())


def test_c2_features_before_the_cut_are_unchanged_after_permutation(synthetic_signal):
    """The property that matters: post-cut shuffling cannot move a pre-cut feature."""
    cut = synthetic_signal.index[250]
    baseline = shift_then_zscore(synthetic_signal, 1, Z_WINDOW, Z_MIN_PERIODS)
    permuted = shift_then_zscore(
        permute_after(synthetic_signal, cut, seed=7), 1, Z_WINDOW, Z_MIN_PERIODS
    )

    before = baseline.index <= cut
    pd.testing.assert_series_equal(baseline[before], permuted[before])


# ===========================================================================
# C3 - cash exclusion
# ===========================================================================
def test_c3_cash_is_absent_from_the_reservoir_input(etf_reservoir_input):
    """Cash must never drive the reservoir: it carries no market dynamics."""
    assert "CASH" not in etf_reservoir_input.columns


def test_c3_cash_is_absent_from_the_log_return_panel(etf_returns_log):
    """The log-return panel is the covariance/reservoir basis; cash is excluded.

    A zero-variance asset makes a covariance matrix singular, which silently
    breaks the Markowitz and Ledoit-Wolf baselines.
    """
    assert "CASH" not in etf_returns_log.columns


def test_c3_cash_is_present_in_the_accounting_panel(etf_returns_simple):
    """Accounting must include cash: it is where risk is parked."""
    assert "CASH" in etf_returns_simple.columns


def test_c3_cash_return_equals_the_risk_free_rate(etf_returns_simple, etf_risk_free):
    """Cash earns exactly r_f, which is what makes all-cash score zero excess."""
    cash = etf_returns_simple["CASH"]
    both_present = cash.notna() & etf_risk_free.notna()
    np.testing.assert_allclose(
        cash[both_present].to_numpy(), etf_risk_free[both_present].to_numpy()
    )


def test_c3_all_cash_portfolio_earns_exactly_zero_excess(etf_returns_simple, etf_risk_free):
    """The load-bearing property: cash is a refuge, not an attractor.

    If cash earned a positive *excess* return with near-zero variance, gradient
    descent would discover that holding cash forever has an enormous Sharpe and
    would learn nothing else. Because excess is measured against the same r_f
    that cash earns, the all-cash excess return is identically zero.
    """
    excess_cash_return = etf_returns_simple["CASH"] - etf_risk_free
    observed = excess_cash_return.dropna()

    assert len(observed) > 1000, "not enough overlap to make this test meaningful"
    np.testing.assert_allclose(observed.to_numpy(), 0.0, atol=1e-15)


# ===========================================================================
# C4 - no back-fill
# ===========================================================================
def test_c4_late_inception_assets_have_nan_not_zero_before_launch(
    sector_prices_filled, sector_availability
):
    """XLRE and XLC must be NaN before their launch, never 0.

    A zero price would flow into returns as a catastrophic move; a NaN correctly
    says "this did not exist".
    """
    for ticker, expected_inception in (("XLRE", "2015-10-08"), ("XLC", "2018-06-19")):
        column = sector_prices_filled[ticker]
        first_real = column.first_valid_index()

        assert str(first_real.date()) == expected_inception
        before_inception = column.loc[: first_real].iloc[:-1]
        assert before_inception.isna().all(), f"{ticker} has values before inception"
        assert not (before_inception == 0).any(), f"{ticker} has zeros before inception"


def test_c4_availability_matches_inception(sector_availability):
    """A[t, i] is 0 before inception and 1 from the first real observation."""
    xlc = sector_availability["XLC"]
    inception = pd.Timestamp("2018-06-19")

    assert xlc.loc[:inception].iloc[:-1].eq(0).all()
    assert xlc.loc[inception] == 1
    assert xlc.loc[inception:].eq(1).all()


def test_c4_no_series_is_backfilled(synthetic_prices):
    """Alignment forward-fills only; leading NaNs survive the process."""
    series = {name: normalise_index(synthetic_prices[name].dropna())
              for name in synthetic_prices.columns}
    calendar = build_trading_calendar(
        series, synthetic_prices.index[0], synthetic_prices.index[-1]
    )
    filled, observed = align_panel(series, calendar, ["OLD", "YOUNG"], ffill_limit=5)

    first_real = observed["YOUNG"].first_valid_index()
    assert filled["YOUNG"].loc[:first_real].iloc[:-1].isna().all()


# ===========================================================================
# C5 - truncation invariance
# ===========================================================================
def test_c5_truncating_the_future_does_not_change_the_past(synthetic_signal):
    """Rebuild on data up to t only; every value at or before t must match.

    This is the strongest single statement of causality: it says the dataset
    could have been built in real time, day by day, and would have produced
    exactly the values it now contains.
    """
    cut = synthetic_signal.index[320]

    full_build = shift_then_zscore(synthetic_signal, 1, Z_WINDOW, Z_MIN_PERIODS)
    truncated_build = shift_then_zscore(
        synthetic_signal.loc[:cut], 1, Z_WINDOW, Z_MIN_PERIODS
    )

    pd.testing.assert_series_equal(full_build.loc[:cut], truncated_build)


def test_c5_truncation_invariance_holds_for_reservoir_input(synthetic_prices):
    """The z-scored return panel is causal under truncation too."""
    returns = log_returns(synthetic_prices[["OLD"]])
    cut = returns.index[250]

    full_build = rolling_zscore(returns, window=60, min_periods=30)
    truncated_build = rolling_zscore(returns.loc[:cut], window=60, min_periods=30)

    pd.testing.assert_frame_equal(full_build.loc[:cut], truncated_build)


def test_c5_availability_is_truncation_invariant(synthetic_prices):
    """Availability computed on a truncated panel matches the full build."""
    cut = synthetic_prices.index[200]

    full = build_availability_matrix(synthetic_prices)
    truncated = build_availability_matrix(synthetic_prices.loc[:cut])

    pd.testing.assert_frame_equal(full.loc[:cut], truncated)


# ===========================================================================
# C6 - z-window causality
# ===========================================================================
def test_c6_zscore_at_t_recomputes_from_data_up_to_t_only(synthetic_signal):
    """Recompute the trailing z-score by hand at a chosen date and compare.

    The hand computation uses a plain slice of the lagged series ending at t,
    which is by construction blind to anything later. Matching the pipeline's
    value proves the pipeline is equally blind.
    """
    pipeline = shift_then_zscore(synthetic_signal, 1, Z_WINDOW, Z_MIN_PERIODS)

    target_position = 350
    target_date = synthetic_signal.index[target_position]

    lagged = synthetic_signal.shift(1)
    window_values = lagged.iloc[target_position - Z_WINDOW + 1 : target_position + 1]

    expected = (
        lagged.iloc[target_position] - window_values.mean()
    ) / window_values.std(ddof=0)

    assert pipeline.loc[target_date] == pytest.approx(expected)


def test_c6_zscore_emits_nothing_before_min_periods(synthetic_signal):
    """No z-score is produced until half a window of history exists."""
    standardised = shift_then_zscore(synthetic_signal, 1, Z_WINDOW, Z_MIN_PERIODS)

    # One row is consumed by the shift, so the first usable value is at
    # index shift + min_periods - 1.
    assert standardised.iloc[: Z_MIN_PERIODS].isna().all()
    assert standardised.iloc[Z_MIN_PERIODS:].notna().any()


def test_c6_forward_only_alignment_never_looks_ahead():
    """Aligning a slow series to a calendar carries values forward, not back."""
    sparse = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2020-01-06", "2020-01-10"], name="date"),
    )
    calendar = pd.bdate_range("2020-01-01", "2020-01-14", name="date")

    aligned = align_to_calendar_forward_only(sparse, calendar)

    # Before the first observation: NaN, never back-filled with 1.0.
    assert aligned.loc["2020-01-01"] != aligned.loc["2020-01-01"]  # NaN check
    assert aligned.loc["2020-01-02"] != aligned.loc["2020-01-02"]
    # Between observations: the last published value is carried forward.
    assert aligned.loc["2020-01-06"] == 1.0
    assert aligned.loc["2020-01-08"] == 1.0
    assert aligned.loc["2020-01-10"] == 2.0
    assert aligned.loc["2020-01-13"] == 2.0


# ===========================================================================
# Confirmatory checks on the real built artifacts
# ===========================================================================
def test_real_signals_are_causal_under_truncation(sector_signals):
    """Every built signal column is finite-or-NaN and starts late, not at row 0.

    A signal that had values from the very first row would mean the trailing
    z-score had been seeded with fewer observations than min_periods.
    """
    for column in sector_signals.columns:
        series = sector_signals[column]
        if series.notna().sum() == 0:
            continue  # an unavailable source; reported in the manifest
        first_valid_position = series.reset_index(drop=True).first_valid_index()
        assert first_valid_position >= Z_MIN_PERIODS, (
            f"signal {column!r} produced a value at row {first_valid_position}, "
            f"before {Z_MIN_PERIODS} observations existed"
        )
        assert np.isfinite(series.dropna().to_numpy()).all()


def test_real_reservoir_input_excludes_late_entrants(sector_reservoir_input):
    """XLRE and XLC are tradeable but must never drive the reservoir (R4)."""
    assert "XLRE" not in sector_reservoir_input.columns
    assert "XLC" not in sector_reservoir_input.columns
    assert len(sector_reservoir_input.columns) == 9


def test_risk_free_conversion_is_compounding_consistent():
    """(1+y/100)^(1/252)-1 compounds back to the annual yield, unlike y/100/252."""
    annual_percent = pd.Series([5.0])
    daily = annual_yield_to_daily_return(annual_percent)

    compounded = (1.0 + daily.iloc[0]) ** 252 - 1.0
    assert compounded == pytest.approx(0.05, abs=1e-12)

    naive_daily = 5.0 / 100.0 / 252.0
    assert daily.iloc[0] < naive_daily  # compounding makes the daily rate smaller


# --------------------------------------------------------------------------- #
# C8 - late entrants cannot influence a pre-inception backtest (end to end)
# --------------------------------------------------------------------------- #
def test_c8_late_entrants_hold_exactly_zero_before_inception():
    """A TRAINED model holds exactly 0.0 in XLRE and XLC before they existed.

    The existing C3/C4/R4 tests check the ingredients in isolation: returns are
    NaN before launch, the availability matrix flips on the right day, and the
    late entrants never drive the reservoir. None of them checks the thing that
    actually matters for a result - that a model trained and evaluated through
    the real pipeline puts *exactly* zero weight on an asset that did not exist.

    "Exactly" is the point. The masked softmax adds -1e9 to unavailable logits,
    which underflows to a true 0.0 rather than a small positive number. A weight
    of 1e-12 in XLC in 2008 would be economically meaningless but would still
    prove the mask had failed, and it would contaminate the holdings count and
    the turnover series. So the assertion is on exact equality, not a tolerance.

    This is what makes "adding two sectors later cannot change earlier results"
    an executable claim rather than an argument.
    """
    from rrc.eval.slicing import folds_for
    from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts
    from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture
    from rrc.reservoir.cache import load_states

    project_root = Path(__file__).resolve().parents[1]
    processed = project_root / "data" / "processed"
    states_dir = project_root / "data" / "states"
    if not (processed / "sector11_returns_simple.parquet").exists():
        pytest.skip("sector11 artifacts not built; run scripts/run_data.py first.")
    caches = sorted(states_dir.glob("sector11_scr_*.npz"))
    if not caches:
        pytest.skip("No sector11 state cache; run scripts/run_reservoir.py first.")

    artifacts = load_universe_artifacts(processed, "sector11")
    states = load_states(caches[0])

    # F03 validates on 2008 - seven years before XLRE and ten before XLC.
    fold = next(f for f in folds_for("sector11") if f.name == "F03")
    packs = build_fold_packs(artifacts, states, fold)

    result = train_mixture(
        packs["train"], packs["val"], MixtureTrainConfig(n_experts=1, max_epochs=3, seed=0)
    )
    # The weights the trained model actually proposes on the 2008 segment.
    proposed = result.model.weights_for_pack(packs["val"]).detach().numpy()

    for ticker, inception in (("XLRE", "2015-10-08"), ("XLC", "2018-06-19")):
        index = artifacts.columns.index(ticker)
        column = proposed[:, index]
        assert (column == 0.0).all(), (
            f"{ticker} (inception {inception}) received non-zero weight during "
            f"the 2008 validation year; max was {column.max():.3e}."
        )

    # And the portfolio is still fully invested without them - the mask removes
    # the assets, it does not quietly leak budget.
    np.testing.assert_allclose(proposed.sum(axis=1), 1.0, atol=1e-10)
