"""the cleaning rules. these are what stand between us and a look-ahead."""

from __future__ import annotations

import numpy as np
import pytest

from src.processor import DataProcessor
from tests.conftest import make_series


def test_short_gaps_bridge_long_gaps_do_not(make_cfg):
    cfg = make_cfg(start="2020-01-01", end="2020-02-01", ffill_limit=2)
    series = make_series(
        ["2020-01-01", "2020-01-02", "2020-01-08", "2020-01-09"], [10.0, 11.0, 12.0, 13.0], "AAA"
    )
    other = make_series(
        ["2020-01-01", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"],
        [5.0, 5.1, 5.2, 5.25, 5.3],
        "BBB",
    )
    panels = DataProcessor(cfg).process("t", {"AAA": series, "BBB": other}, ["AAA", "BBB"])

    # AAA is missing the 3rd, 6th and 7th. the limit of two carries it two rows,
    # so the 3rd and 6th get filled and the 7th does not
    assert panels.prices.loc["2020-01-03", "AAA"] == 11.0
    assert panels.prices.loc["2020-01-06", "AAA"] == 11.0
    assert np.isnan(panels.prices.loc["2020-01-07", "AAA"])


def test_log_returns_match_hand_calculation(make_cfg):
    cfg = make_cfg(start="2021-01-01", end="2021-01-31")
    series = make_series(["2021-01-04", "2021-01-05", "2021-01-06"], [100.0, 110.0, 99.0], "AAA")
    panels = DataProcessor(cfg).process("t", {"AAA": series}, ["AAA"])
    returns = panels.returns["AAA"]

    # computed as log(P_t) - log(P_t-1), so it matches log(P_t / P_t-1) to floating
    # point rather than exactly
    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(np.log(110.0 / 100.0))
    assert returns.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


def test_bridged_day_has_zero_return(make_cfg):
    cfg = make_cfg(start="2021-01-01", end="2021-01-31")
    series = make_series(["2021-01-04", "2021-01-06"], [100.0, 105.0], "AAA")
    other = make_series(["2021-01-04", "2021-01-05", "2021-01-06"], [1.0, 1.0, 1.0], "BBB")
    panels = DataProcessor(cfg).process("t", {"AAA": series, "BBB": other}, ["AAA", "BBB"])

    # the 5th carries the 4th's price forward, so nothing moved that day
    assert panels.returns.loc["2021-01-05", "AAA"] == 0.0


def test_late_asset_stays_empty_before_it_existed(make_cfg):
    cfg = make_cfg(start="2020-01-01", end="2020-03-31")
    early = make_series(["2020-01-02", "2020-02-03", "2020-03-02"], [10.0, 11.0, 12.0], "OLD")
    late = make_series(["2020-03-02"], [50.0], "NEW")
    panels = DataProcessor(cfg).process("t", {"OLD": early, "NEW": late}, ["OLD", "NEW"])

    assert panels.prices["NEW"].loc[:"2020-02-03"].isna().all()
    assert panels.prices.loc["2020-03-02", "NEW"] == 50.0


def test_dates_outside_the_window_are_dropped(make_cfg):
    cfg = make_cfg(start="2020-02-01", end="2020-02-28")
    series = make_series(["2020-01-15", "2020-02-10", "2020-03-15"], [1.0, 2.0, 3.0], "AAA")
    panels = DataProcessor(cfg).process("t", {"AAA": series}, ["AAA"])

    assert len(panels.prices) == 1
    assert panels.prices.index[0].strftime("%Y-%m-%d") == "2020-02-10"
