"""Shared fixtures and helpers for the data-layer tests.

Two principles here:

* **No test touches the network.** Tests either build tiny synthetic series
  whose correct answer can be verified by hand, or read the parquet artifacts
  produced by ``scripts/run_data.py``. Tests that need built artifacts skip
  cleanly when they are absent, so a fresh clone still has a green suite.
* **Synthetic first.** A causality property is best demonstrated on data small
  enough to reason about completely; the real artifacts are then checked as a
  second, confirmatory layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------------------------------- #
# Access to the built artifacts
# --------------------------------------------------------------------------- #
def load_artifact_or_skip(universe: str, name: str) -> pd.DataFrame:
    """Load a built parquet artifact, or skip the test if it does not exist.

    Parameters
    ----------
    universe
        ``'etf10'`` or ``'sector11'``.
    name
        Artifact suffix, e.g. ``'returns_simple'``.
    """
    path = PROCESSED_DIR / f"{universe}_{name}.parquet"
    if not path.is_file():
        pytest.skip(f"Artifact {path.name} not built; run scripts/run_data.py first.")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def etf_prices_observed() -> pd.DataFrame:
    return load_artifact_or_skip("etf10", "prices_observed")


@pytest.fixture(scope="session")
def etf_returns_simple() -> pd.DataFrame:
    return load_artifact_or_skip("etf10", "returns_simple")


@pytest.fixture(scope="session")
def etf_returns_log() -> pd.DataFrame:
    return load_artifact_or_skip("etf10", "returns_log")


@pytest.fixture(scope="session")
def etf_reservoir_input() -> pd.DataFrame:
    return load_artifact_or_skip("etf10", "reservoir_input")


@pytest.fixture(scope="session")
def etf_risk_free() -> pd.Series:
    return load_artifact_or_skip("etf10", "risk_free")["r_f"]


@pytest.fixture(scope="session")
def sector_prices_filled() -> pd.DataFrame:
    return load_artifact_or_skip("sector11", "prices_adj")


@pytest.fixture(scope="session")
def sector_prices_observed() -> pd.DataFrame:
    return load_artifact_or_skip("sector11", "prices_observed")


@pytest.fixture(scope="session")
def sector_availability() -> pd.DataFrame:
    return load_artifact_or_skip("sector11", "availability")


@pytest.fixture(scope="session")
def sector_reservoir_input() -> pd.DataFrame:
    return load_artifact_or_skip("sector11", "reservoir_input")


@pytest.fixture(scope="session")
def sector_signals() -> pd.DataFrame:
    return load_artifact_or_skip("sector11", "signals")


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_signal() -> pd.Series:
    """A deterministic, non-constant daily series on a business-day index.

    Deliberately not random: every test that uses it can be reasoned about, and
    a failure points at the logic rather than at an unlucky draw.
    """
    index = pd.bdate_range("2020-01-01", periods=400, name="date")
    values = np.linspace(1.0, 5.0, len(index)) + np.sin(np.arange(len(index)) / 7.0)
    return pd.Series(values, index=index, name="signal")


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """A two-asset price panel where the second asset starts late.

    ``OLD`` exists throughout; ``YOUNG`` begins on row 100, which is the
    inception pattern that XLRE and XLC exhibit in the real sector universe.
    """
    index = pd.bdate_range("2020-01-01", periods=300, name="date")
    old = pd.Series(100.0 * np.exp(np.cumsum(np.full(len(index), 0.0003))), index=index)

    young = pd.Series(np.nan, index=index)
    young.iloc[100:] = 50.0 * np.exp(np.cumsum(np.full(len(index) - 100, 0.0005)))

    return pd.DataFrame({"OLD": old, "YOUNG": young})


def permute_after(series: pd.Series, cut_date: pd.Timestamp, seed: int = 0) -> pd.Series:
    """Randomly permute a series' values strictly after ``cut_date``.

    This is the mechanism behind causality test C2 (shuffle-future collapse).
    Destroying the temporal structure after a cut date must not change anything
    a model computed at or before the cut. At P1 this function is verified to do
    exactly what it claims; at P5 it is used for the full test, where a strategy
    is retrained on the permuted data and its post-cut performance must collapse
    to noise. Any surviving skill would be evidence of look-ahead.

    Parameters
    ----------
    series
        The series to permute.
    cut_date
        Values at or before this date are left untouched.
    seed
        Seed for the permutation, so the test is reproducible.

    Returns
    -------
    pandas.Series
        A copy with post-cut values shuffled among themselves.
    """
    permuted = series.copy()
    after_cut = permuted.index > cut_date

    values_after = permuted.loc[after_cut].to_numpy()
    generator = np.random.default_rng(seed)
    permuted.loc[after_cut] = generator.permutation(values_after)

    return permuted
