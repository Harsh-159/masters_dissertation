"""Caching of reservoir states - the decision that makes the ladder affordable.

The idea (locked decision D3)
-----------------------------
The reservoir is **frozen**. Its states therefore depend only on the input series
and the reservoir hyperparameters - never on anything that training changes. So
they can be computed **once** and reused by every subsequent experiment.

This is not a minor optimisation, it is what makes the project's schedule
possible. The ablation ladder involves roughly a thousand training runs across
arms, folds and seeds. If each had to re-drive the reservoir over 7,000 trading
days first, the runs would be minutes long instead of seconds. With cached
states, a training run touches only small readout heads on a matrix that is
already in memory.

What is stored, and why the mask matters
----------------------------------------
Each cache entry holds the state path aligned to the **full trading calendar**,
plus a boolean ``usable`` mask. Two distinct things make a row unusable:

1. **Warm-up of the inputs.** The reservoir input is z-scored over a trailing
   window, so it is NaN until that window fills. The reservoir cannot start
   before the first fully finite input row.
2. **Washout.** The state starts at ``x_0 = 0``, an arbitrary condition. The
   echo state property guarantees its influence fades, but not instantly, so the
   first 250 trading days after the start are discarded.

Unusable rows are stored as NaN rather than zero, deliberately: a zero would
flow silently through a matrix multiply and produce a plausible-looking number,
whereas a NaN makes any accidental use immediately obvious.

Keying
------
The filename carries the universe, the reservoir kind, and a hash of the
reservoir configuration. Change any hyperparameter and you get a different file
rather than a silently stale one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rrc.eval.registry import config_hash

logger = logging.getLogger(__name__)

# States are stored at float32: half the size of float64, and far below the
# precision at which reservoir dynamics are meaningful. The *ledger* uses
# float64 because it must satisfy an identity to 1e-10; states have no such
# requirement.
STATE_DTYPE = np.float32


@dataclass(frozen=True)
class CachedStates:
    """A reservoir state path plus everything needed to interpret it.

    Attributes
    ----------
    states
        ``(n_days, n_reservoir)`` float32, aligned to ``dates``. NaN on rows the
        reservoir could not produce.
    dates
        The trading calendar the states are aligned to.
    usable
        ``(n_days,)`` boolean. True only where the state is both computed and
        past the washout period. **Every loss and every metric must respect
        this mask.**
    reservoir_config
        The hyperparameters the states were built from.
    cfg_hash
        Hash of ``reservoir_config``; also appears in the filename.
    kind
        ``"scr"`` or ``"esn"``.
    """

    states: np.ndarray
    dates: pd.DatetimeIndex
    usable: np.ndarray
    reservoir_config: dict
    cfg_hash: str
    kind: str

    @property
    def n_usable(self) -> int:
        """Number of days available to training and evaluation."""
        return int(self.usable.sum())

    def usable_states(self) -> np.ndarray:
        """The state rows that may be used, with the masked rows removed."""
        return self.states[self.usable]

    def usable_dates(self) -> pd.DatetimeIndex:
        """The dates matching :meth:`usable_states`."""
        return self.dates[self.usable]

    def assert_usable_states_are_finite(self) -> None:
        """Fail loudly if any usable state is non-finite.

        Called after every build and every load. A NaN inside the usable region
        would mean the mask is wrong, and would propagate into training as an
        immediate, confusing failure far from its cause.
        """
        if not np.isfinite(self.usable_states()).all():
            n_bad = int((~np.isfinite(self.usable_states())).sum())
            raise ValueError(
                f"{n_bad} non-finite value(s) inside the usable region of the "
                f"cached states - the usable mask does not match the data."
            )


def reservoir_config_hash(reservoir_config: dict) -> str:
    """Hash a reservoir configuration, reusing the project's canonical hashing.

    The same function that identifies an experiment identifies a state cache, so
    "same hash" means the same thing everywhere in the project.
    """
    return config_hash(reservoir_config)


def cache_path(cache_dir: Path | str, universe: str, kind: str, cfg_hash: str) -> Path:
    """Filesystem path for one cache entry.

    The truncated hash keeps the filename readable while remaining collision-free
    in practice for the low hundreds of configurations this project will build.
    """
    return Path(cache_dir) / f"{universe}_{kind}_{cfg_hash[:12]}.npz"


def build_usable_mask(
    n_days: int, first_computed_row: int, washout: int
) -> np.ndarray:
    """Mark which rows survive input warm-up and washout.

    Parameters
    ----------
    n_days
        Length of the full trading calendar.
    first_computed_row
        Index of the first row the reservoir was driven from.
    washout
        Number of trading days to discard after the start, to let the influence
        of ``x_0 = 0`` decay.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``n_days``.
    """
    if washout < 0:
        raise ValueError(f"washout must be non-negative, got {washout}.")

    usable = np.zeros(n_days, dtype=bool)
    first_usable_row = first_computed_row + washout
    if first_usable_row < n_days:
        usable[first_usable_row:] = True
    else:
        logger.warning(
            "Washout of %d days consumes the entire sample (%d rows from row %d); "
            "no usable states remain.", washout, n_days, first_computed_row,
        )
    return usable


def save_states(
    cached: CachedStates, cache_dir: Path | str
) -> Path:
    """Write a state cache to disk.

    Stored as a compressed ``.npz`` holding the states, the mask, the dates as
    integer nanoseconds, and the configuration as JSON text. At roughly 8 MB for
    a 7,000-day, 300-unit reservoir the whole file loads in well under a second,
    so there is nothing to gain from memory-mapping it.

    Dates are canonicalised to **nanosecond** resolution on the way in and out.
    pandas now carries a per-index time unit (``ns``/``us``/``ms``) that varies
    with how the index was built - parquet round-trips give milliseconds,
    ``bdate_range`` gives microseconds - and letting that leak into the cache
    would make an index compare unequal to itself across a save/load cycle even
    though every timestamp matched.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = cache_path(directory, _universe_of(cached), cached.kind, cached.cfg_hash)

    np.savez_compressed(
        path,
        states=cached.states.astype(STATE_DTYPE),
        usable=cached.usable,
        dates=cached.dates.as_unit("ns").asi8,  # int64 nanoseconds; unambiguous
        metadata=json.dumps(
            {
                "reservoir_config": cached.reservoir_config,
                "cfg_hash": cached.cfg_hash,
                "kind": cached.kind,
            },
            default=str,
        ),
    )

    logger.info(
        "Cached %s states: %d x %d (%d usable) -> %s",
        cached.kind, *cached.states.shape, cached.n_usable, path.name,
    )
    return path


def load_states(path: Path | str) -> CachedStates:
    """Read a state cache from disk and validate it."""
    archive = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))

    cached = CachedStates(
        states=archive["states"],
        dates=pd.DatetimeIndex(archive["dates"].astype("datetime64[ns]"), name="date"),
        usable=archive["usable"],
        reservoir_config=metadata["reservoir_config"],
        cfg_hash=metadata["cfg_hash"],
        kind=metadata["kind"],
    )
    cached.assert_usable_states_are_finite()
    return cached


def load_or_none(
    cache_dir: Path | str, universe: str, kind: str, cfg_hash: str
) -> CachedStates | None:
    """Load a cache entry if it exists, otherwise return ``None``."""
    path = cache_path(cache_dir, universe, kind, cfg_hash)
    if not path.is_file():
        return None
    logger.info("Loading cached states from %s", path.name)
    return load_states(path)


def _universe_of(cached: CachedStates) -> str:
    """Extract the universe name from the cached configuration."""
    universe = cached.reservoir_config.get("universe")
    if not universe:
        raise ValueError(
            "reservoir_config must contain a 'universe' key so the cache file "
            "can be named unambiguously."
        )
    return str(universe)
