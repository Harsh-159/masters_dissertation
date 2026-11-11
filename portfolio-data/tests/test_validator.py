"""hard checks must stop the run, soft ones must not."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.validator import DataError, validate
from tests.conftest import make_panels, make_universe


def clean_prices(n: int = 60) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    rng = np.random.default_rng(0)
    walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=(n, 2)), axis=0))
    return pd.DataFrame(walk, index=index, columns=["AAA", "BBB"])


def test_clean_panel_passes(make_cfg):
    panels = make_panels(clean_prices())
    validate(make_cfg(), panels, make_universe(("AAA", "BBB")))


def test_extreme_return_only_warns(make_cfg, caplog):
    prices = clean_prices()
    prices.iloc[30, 0] = prices.iloc[29, 0] * 3.0  # a 200% day
    panels = make_panels(prices)

    validate(make_cfg(), panels, make_universe(("AAA", "BBB")))
    assert "extreme return" in caplog.text


def test_soft_checks_are_skipped_without_a_validate_block(make_cfg, caplog):
    prices = clean_prices()
    prices.iloc[30, 0] = prices.iloc[29, 0] * 3.0
    panels = make_panels(prices)

    caplog.set_level(logging.INFO)
    validate(make_cfg(soft_checks=False), panels, make_universe(("AAA", "BBB")))
    assert "extreme return" not in caplog.text
    assert "skipping the soft warnings" in caplog.text


def test_hard_checks_still_run_without_a_validate_block(make_cfg):
    prices = clean_prices()
    prices.iloc[10, 0] = -1.0
    with pytest.raises(DataError, match="at or below zero"):
        validate(make_cfg(soft_checks=False), make_panels(prices), make_universe(("AAA", "BBB")))


def test_negative_price_stops_the_run(make_cfg):
    prices = clean_prices()
    prices.iloc[10, 0] = -1.0
    with pytest.raises(DataError, match="at or below zero"):
        validate(make_cfg(), make_panels(prices), make_universe(("AAA", "BBB")))


def test_duplicate_date_stops_the_run(make_cfg):
    prices = clean_prices()
    prices.index = [prices.index[0]] + list(prices.index[1:])
    prices = pd.concat([prices.iloc[:1], prices])
    with pytest.raises(DataError, match="duplicate dates"):
        validate(make_cfg(), make_panels(prices), make_universe(("AAA", "BBB")))


def test_price_before_inception_stops_the_run(make_cfg):
    prices = clean_prices()
    raw = prices.copy()
    raw.iloc[:20, 1] = np.nan  # BBB really starts on day 20
    with pytest.raises(DataError, match="before it existed"):
        validate(make_cfg(), make_panels(prices, raw_aligned=raw), make_universe(("AAA", "BBB")))
