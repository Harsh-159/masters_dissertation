"""Unit tests for the data-layer mechanics.

The causality properties live in ``test_causality.py``; this file checks that
the individual pieces do what their docstrings say - calendar alignment, return
arithmetic, the cash column, the availability matrix, and the defensive parsers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rrc.data.availability import (
    UNAVAILABLE_LOGIT_PENALTY,
    active_assets_on,
    assert_no_prices_before_inception,
    availability_summary,
    build_availability_matrix,
    logit_mask,
)
from rrc.data.calendar import (
    align_panel,
    align_to_calendar,
    build_trading_calendar,
    normalise_index,
)
from rrc.data.fred import _parse_fred_csv, annual_yield_to_daily_return
from rrc.data.loaders import (
    _extract_column,
    append_cash_column,
    log_returns,
    rolling_zscore,
    simple_returns,
)
from rrc.data.signals import SIGNAL_NAMES, build_signal_matrix


# --------------------------------------------------------------------------- #
# calendar.py
# --------------------------------------------------------------------------- #
def test_normalise_index_strips_timezone_and_time_of_day():
    index = pd.to_datetime(["2020-01-02 16:00", "2020-01-03 16:00"]).tz_localize("America/New_York")
    series = pd.Series([1.0, 2.0], index=index)

    cleaned = normalise_index(series)

    assert cleaned.index.tz is None
    assert list(cleaned.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


def test_normalise_index_drops_duplicates_keeping_last():
    """Revised data supersedes the original print for the same date."""
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-02", "2020-01-02"]))
    assert normalise_index(series).iloc[0] == 2.0


def test_normalise_index_sorts():
    series = pd.Series([2.0, 1.0], index=pd.to_datetime(["2020-01-03", "2020-01-02"]))
    assert list(normalise_index(series).to_numpy()) == [1.0, 2.0]


def test_trading_calendar_is_the_union_of_observed_days():
    """A day on which only one asset traded is still a trading day."""
    first = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    second = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-03", "2020-01-06"]))

    calendar = build_trading_calendar(
        {"A": first, "B": second}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31")
    )

    assert list(calendar) == [
        pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")
    ]


def test_trading_calendar_respects_the_window():
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2019-01-02", "2020-01-03"]))
    calendar = build_trading_calendar(
        {"A": series}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
    )
    assert list(calendar) == [pd.Timestamp("2020-01-03")]


def test_trading_calendar_raises_when_the_window_is_empty():
    series = pd.Series([1.0], index=pd.to_datetime(["2019-01-02"]))
    with pytest.raises(ValueError, match="No trading days"):
        build_trading_calendar(
            {"A": series}, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
        )


def test_align_to_calendar_bridges_short_gaps_only():
    """A gap within the limit is carried forward; a longer one stays NaN."""
    calendar = pd.bdate_range("2020-01-01", periods=12, name="date")
    observed = pd.Series(
        [10.0, 11.0], index=[calendar[0], calendar[9]]
    )  # 8 missing days between

    bridged = align_to_calendar(observed, calendar, ffill_limit=2)

    assert bridged.iloc[1] == 10.0            # within the limit
    assert bridged.iloc[2] == 10.0            # still within the limit
    assert np.isnan(bridged.iloc[3])          # limit exhausted
    assert bridged.iloc[9] == 11.0            # the next real observation


def test_align_panel_keeps_a_column_for_a_ticker_with_no_data():
    """A missing asset appears as an all-NaN column, never silently absent."""
    calendar = pd.bdate_range("2020-01-01", periods=5, name="date")
    series = {"A": pd.Series([1.0] * 5, index=calendar)}

    filled, observed = align_panel(series, calendar, ["A", "MISSING"], ffill_limit=5)

    assert list(filled.columns) == ["A", "MISSING"]
    assert filled["MISSING"].isna().all()
    assert observed["MISSING"].isna().all()


# --------------------------------------------------------------------------- #
# loaders.py
# --------------------------------------------------------------------------- #
def test_log_returns_on_a_known_series():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    returns = log_returns(prices)["A"]

    assert np.isnan(returns.iloc[0])  # first observation is NaN, never 0
    assert returns.iloc[1] == pytest.approx(np.log(1.1))
    assert returns.iloc[2] == pytest.approx(np.log(1.1))


def test_simple_returns_on_a_known_series():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    returns = simple_returns(prices)["A"]

    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(0.10)


def test_log_and_simple_returns_differ_and_are_related():
    """log(1+r_simple) == r_log; conflating them is a real and subtle error."""
    prices = pd.DataFrame({"A": [100.0, 130.0]})
    simple = simple_returns(prices)["A"].iloc[1]
    logarithmic = log_returns(prices)["A"].iloc[1]

    assert simple != pytest.approx(logarithmic)
    assert np.log(1 + simple) == pytest.approx(logarithmic)


def test_append_cash_column_adds_the_risk_free_return():
    index = pd.bdate_range("2020-01-01", periods=3, name="date")
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=index)
    risk_free = pd.Series([0.0001, 0.0001, 0.0002], index=index)

    with_cash = append_cash_column(returns, risk_free)

    assert list(with_cash.columns) == ["A", "CASH"]
    np.testing.assert_allclose(with_cash["CASH"].to_numpy(), risk_free.to_numpy())


def test_append_cash_column_refuses_to_overwrite():
    index = pd.bdate_range("2020-01-01", periods=2, name="date")
    returns = pd.DataFrame({"CASH": [0.0, 0.0]}, index=index)
    with pytest.raises(ValueError, match="already exists"):
        append_cash_column(returns, pd.Series([0.0, 0.0], index=index))


def test_rolling_zscore_standardises_within_the_window():
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    standardised = rolling_zscore(values, window=3, min_periods=3)

    # At row 2 the window is [1,2,3]: mean 2, population sd sqrt(2/3).
    assert standardised["A"].iloc[2] == pytest.approx((3.0 - 2.0) / np.sqrt(2 / 3))
    assert standardised["A"].iloc[:2].isna().all()


def test_extract_column_handles_multiindex_and_flat_frames():
    """yfinance returns both shapes depending on version; both must work."""
    index = pd.date_range("2020-01-01", periods=3)

    flat = pd.DataFrame({"Adj Close": [1.0, 2.0, 3.0]}, index=index)
    assert _extract_column(flat, "SPY", ("Adj Close",)).tolist() == [1.0, 2.0, 3.0]

    nested = pd.DataFrame({("Adj Close", "SPY"): [1.0, 2.0, 3.0]}, index=index)
    nested.columns = pd.MultiIndex.from_tuples(nested.columns)
    assert _extract_column(nested, "SPY", ("Adj Close",)).tolist() == [1.0, 2.0, 3.0]

    assert _extract_column(flat, "SPY", ("Nonexistent",)) is None


# --------------------------------------------------------------------------- #
# availability.py
# --------------------------------------------------------------------------- #
def test_availability_marks_the_asset_live_between_first_and_last_observation(
    synthetic_prices,
):
    availability = build_availability_matrix(synthetic_prices)

    assert availability["OLD"].eq(1).all()
    assert availability["YOUNG"].iloc[:100].eq(0).all()
    assert availability["YOUNG"].iloc[100:].eq(1).all()


def test_logit_mask_zeroes_unavailable_assets(synthetic_prices):
    """The mask must be large enough that exp() underflows to exactly zero."""
    availability = build_availability_matrix(synthetic_prices)
    mask = logit_mask(availability)

    assert mask["YOUNG"].iloc[0] == UNAVAILABLE_LOGIT_PENALTY
    assert mask["YOUNG"].iloc[-1] == 0.0
    assert np.exp(UNAVAILABLE_LOGIT_PENALTY) == 0.0


def test_active_assets_reflects_the_date(synthetic_prices):
    """1/N must mean one over the assets that existed, not over the full list."""
    availability = build_availability_matrix(synthetic_prices)

    early = active_assets_on(availability, synthetic_prices.index[0])
    late = active_assets_on(availability, synthetic_prices.index[-1])

    assert early == ["OLD"]
    assert late == ["OLD", "YOUNG"]


def test_availability_summary_reports_coverage(synthetic_prices):
    summary = availability_summary(build_availability_matrix(synthetic_prices))

    assert summary.loc["OLD", "fraction_of_window"] == 1.0
    assert 0.0 < summary.loc["YOUNG", "fraction_of_window"] < 1.0
    assert summary.loc["YOUNG", "days_available"] == 200


def test_assert_no_prices_before_inception_catches_a_violation(synthetic_prices):
    """The executable form of 'never fabricate pre-inception prices'."""
    availability = build_availability_matrix(synthetic_prices)

    corrupted = synthetic_prices.copy()
    corrupted.iloc[0, corrupted.columns.get_loc("YOUNG")] = 42.0  # a price before launch

    with pytest.raises(AssertionError, match="unavailable"):
        assert_no_prices_before_inception(corrupted, availability)


def test_assert_no_prices_before_inception_passes_on_clean_data(synthetic_prices):
    availability = build_availability_matrix(synthetic_prices)
    assert_no_prices_before_inception(synthetic_prices, availability)  # must not raise


# --------------------------------------------------------------------------- #
# fred.py - defensive parsing
# --------------------------------------------------------------------------- #
def test_parse_fred_csv_handles_the_legacy_date_header():
    csv_text = "DATE,DGS3MO\n2020-01-02,1.54\n2020-01-03,1.52\n"
    series = _parse_fred_csv(csv_text, "DGS3MO")

    assert list(series.to_numpy()) == [1.54, 1.52]
    assert series.index[0] == pd.Timestamp("2020-01-02")


def test_parse_fred_csv_handles_the_current_date_header():
    """FRED renamed DATE to observation_date; the parser takes it positionally."""
    csv_text = "observation_date,DGS3MO\n2020-01-02,1.54\n"
    assert _parse_fred_csv(csv_text, "DGS3MO").iloc[0] == 1.54


def test_parse_fred_csv_drops_missing_markers():
    """FRED writes '.' for holidays; those become dropped rows, never zeros."""
    csv_text = "DATE,DGS3MO\n2020-01-02,1.54\n2020-01-03,.\n2020-01-06,1.55\n"
    series = _parse_fred_csv(csv_text, "DGS3MO")

    assert len(series) == 2
    assert 0.0 not in series.to_numpy()


def test_parse_fred_csv_raises_on_an_unusable_response():
    with pytest.raises(ValueError):
        _parse_fred_csv("just one column\n1\n", "DGS3MO")


def test_annual_yield_to_daily_return_is_monotonic():
    yields = pd.Series([0.0, 2.0, 5.0])
    daily = annual_yield_to_daily_return(yields)

    assert daily.iloc[0] == 0.0
    assert daily.iloc[1] < daily.iloc[2]


# --------------------------------------------------------------------------- #
# signals.py
# --------------------------------------------------------------------------- #
def test_signal_matrix_has_the_fixed_column_order():
    calendar = pd.bdate_range("2020-01-01", periods=400, name="date")
    values = pd.Series(np.linspace(1, 5, len(calendar)), index=calendar)

    matrix = build_signal_matrix(
        calendar=calendar,
        raw_series={name: values for name in SIGNAL_NAMES},
        shift_days=1, sentiment_shift_days=5, z_window=252, z_min_periods=126,
    )

    assert list(matrix.columns) == list(SIGNAL_NAMES)
    assert len(matrix) == len(calendar)


def test_missing_signal_becomes_an_all_nan_column_not_a_missing_one():
    """An unavailable source must stay visible, so its absence is auditable."""
    calendar = pd.bdate_range("2020-01-01", periods=300, name="date")
    values = pd.Series(np.linspace(1, 5, len(calendar)), index=calendar)

    matrix = build_signal_matrix(
        calendar=calendar,
        raw_series={"vix": values},  # the other four are absent
        shift_days=1, sentiment_shift_days=5, z_window=252, z_min_periods=126,
    )

    assert list(matrix.columns) == list(SIGNAL_NAMES)
    assert matrix["sentiment"].isna().all()
