"""downloading and caching, with yahoo replaced by a fake."""

from __future__ import annotations

import pandas as pd

from src import fetcher as fetcher_module
from src.fetcher import DataFetcher


def yahoo_frame(ticker: str, n: int = 5) -> pd.DataFrame:
    """what yfinance returns now, with (field, ticker) column pairs."""
    index = pd.bdate_range("2020-01-01", periods=n, name="Date")
    columns = pd.MultiIndex.from_product([["Adj Close", "Close"], [ticker]])
    values = [[100.0 + i, 100.0 + i] for i in range(n)]
    return pd.DataFrame(values, index=index, columns=columns)


def test_second_call_comes_from_the_cache(make_cfg, tmp_path, monkeypatch):
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return yahoo_frame(ticker)

    monkeypatch.setattr(fetcher_module.yf, "download", fake_download)
    cfg = make_cfg(raw_dir=tmp_path)

    first = DataFetcher(cfg).fetch_one("AAA")
    second = DataFetcher(cfg).fetch_one("AAA")

    assert len(calls) == 1
    pd.testing.assert_series_equal(first, second)


def test_falls_back_to_auto_adjusted_close(make_cfg, tmp_path, monkeypatch):
    def fake_download(ticker, **kwargs):
        frame = yahoo_frame(ticker)
        if not kwargs.get("auto_adjust"):
            return frame.drop(columns="Adj Close")  # no Adj Close on the first try
        return frame

    monkeypatch.setattr(fetcher_module.yf, "download", fake_download)
    series = DataFetcher(make_cfg(raw_dir=tmp_path)).fetch_one("AAA")

    assert series is not None and len(series) == 5


def test_dead_ticker_is_remembered(make_cfg, tmp_path, monkeypatch):
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return pd.DataFrame()

    monkeypatch.setattr(fetcher_module.yf, "download", fake_download)
    cfg = make_cfg(raw_dir=tmp_path)

    assert DataFetcher(cfg).fetch_one("DEAD") is None
    attempts_first_run = len(calls)

    assert DataFetcher(cfg).fetch_one("DEAD") is None
    assert len(calls) == attempts_first_run  # no network on the second run


def test_force_refresh_downloads_again(make_cfg, tmp_path, monkeypatch):
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        return yahoo_frame(ticker)

    monkeypatch.setattr(fetcher_module.yf, "download", fake_download)
    cfg = make_cfg(raw_dir=tmp_path)

    DataFetcher(cfg).fetch_one("AAA")
    DataFetcher(cfg).fetch_one("AAA", force_refresh=True)

    assert len(calls) == 2
