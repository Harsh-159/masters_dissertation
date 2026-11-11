"""shared fixtures. every test runs offline, yahoo is monkeypatched."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Universe, load_config
from src.processor import Panels

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture
def make_cfg():
    """the real config with a short window and no sleeping between downloads.

    the soft check thresholds are set here rather than read, so these tests pass
    whether or not the validate block is enabled in config.yaml.
    """

    def build(start="2018-01-01", end="2018-12-31", ffill_limit=5, raw_dir=None, soft_checks=True):
        cfg = load_config(CONFIG_PATH)
        cfg = dataclasses.replace(
            cfg,
            start=dt.date.fromisoformat(start),
            end=dt.date.fromisoformat(end),
            ffill_limit=ffill_limit,
            min_history_days=1,
            polite_delay_seconds=0.0,
            retry_backoff_seconds=0.0,
            retry_attempts=2,
            outlier_sigma=6.0 if soft_checks else None,
            extreme_return_abs=0.40 if soft_checks else None,
            stale_run_length=5 if soft_checks else None,
        )
        if raw_dir is not None:
            cfg = dataclasses.replace(cfg, raw_dir=Path(raw_dir))
        return cfg

    return build


def make_series(dates: list[str], values: list[float], name: str) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.Series(values, index=index, name=name)


def make_panels(prices: pd.DataFrame, raw_aligned=None, name="test") -> Panels:
    if raw_aligned is None:
        raw_aligned = prices.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(prices).diff()
    return Panels(name=name, prices=prices, returns=returns, raw_aligned=raw_aligned)


def make_universe(tickers: tuple[str, ...]) -> Universe:
    return Universe(name="test", description="", tickers=tickers)
