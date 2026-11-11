"""The variant registry: an append-only ledger of every experiment ever run.

Why this exists, and why it exists *first*
------------------------------------------
The dissertation reports a **deflated Sharpe ratio** (Bailey and Lopez de Prado,
2014), which corrects an observed Sharpe for the fact that it is the best of
many things tried. That correction needs one number: how many distinct
configurations were tried. If that number is reconstructed from memory at the
end of a project, it is an underestimate, and the deflation is theatre.

So the counting rule is **pre-registered here, before the first experiment
exists**:

    N_trials = the number of distinct cfg_hash values appearing in the registry
               with split == "val", measured at the moment the DSR is computed.

Every training run appends exactly one row per (fold, split) it evaluates,
before it writes any other artifact. The registry is append-only: there is
deliberately no update or delete function in this module. A result that turned
out to be wrong is corrected by appending a new row and explaining it in the
``note`` column, never by editing history.

The columns
-----------
Identity (always required) pins down *what was run*: which code (``git_sha``),
which configuration (``cfg_hash``), which universe, rung, arm, seed, fold, and
split, and how many trainable parameters the model had. Metrics (optional) are
the Section 13 quantities. ``note`` is free text.

``git_sha`` records the commit at ``HEAD`` when the row was written. It pins the
**base commit**, not necessarily the exact bytes: development runs happen on
working trees that carry uncommitted edits, and that weaker guarantee is stated
openly in the README rather than being marked per row (see :func:`git_sha`).
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from rrc.utils.io import canonical_json, ensure_dir, utc_timestamp

# regime-rc/src/rrc/eval/registry.py -> parents[3] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "results" / "registry.csv"

# The only legal values for the `split` column. `val` is load-bearing: it is
# the split the deflated-Sharpe trial count is defined over.
VALID_SPLITS = ("train", "val", "test", "holdout")

# Identity columns: what was run. All are required on every row.
IDENTITY_COLUMNS = (
    "run_id",
    "timestamp",
    "git_sha",
    "cfg_hash",
    "universe",
    "rung",
    "arm",
    "seed",
    "fold",
    "split",
    "n_params",
)

# Metric columns: the Section 13 quantities. All optional, because a row is
# also written for a run that crashed or was abandoned - those still count as
# trials, which is precisely why they must not be silently omitted.
METRIC_COLUMNS = (
    "ann_return",        # annualised geometric return of the net compounded path
    "ann_vol",           # sqrt(252) * std of daily net returns
    "excess_sharpe",     # sqrt(252) * mean/std of daily EXCESS returns (over r_f)
    "sortino",           # excess mean / downside deviation, annualised
    "max_drawdown",      # max peak-to-trough on the compounded net path
    "cvar_95",           # daily CVaR at 95%, as a positive loss
    "ann_turnover",      # annualised one-way turnover (52 x mean per-rebalance)
    "holdings_count",    # realised count of weights > 1e-4, cash excluded
    "effective_n",       # 1 / sum(w_i^2), the diversification-equivalent count
    "mean_gate_entropy", # mean entropy of pi_t; a saturation diagnostic (A2)
    "importance_shares", # JSON list of per-expert importance shares, e.g. "[0.31,0.35,0.34]"
)

NOTE_COLUMN = ("note",)

# Columns that must always be read back as text, never inferred.
#
# This matters more than it looks. Rung labels are "0", "1", "2", "3a", "3b",
# "3c", "4", "5", "6". While only numeric rungs exist, pandas infers int64 for
# the column; the moment the first "3a" row is appended (P7) it silently becomes
# an object column. Any filter written as `frame["rung"] == "1"` would therefore
# work early in the project and quietly match nothing later. Pinning the dtype
# from the schema removes that whole class of bug.
STRING_COLUMNS = (
    "run_id",
    "timestamp",
    "git_sha",
    "cfg_hash",
    "universe",
    "rung",
    "arm",
    "fold",
    "split",
    "importance_shares",  # JSON text, not a number
    "note",
)

# Metric columns that really are numeric and should read back as floats.
NUMERIC_METRIC_COLUMNS = tuple(
    column for column in METRIC_COLUMNS if column != "importance_shares"
)

# The full header, in file order. Any change to this tuple is a schema change
# and will be rejected against an existing registry (see `_assert_header_matches`).
REGISTRY_COLUMNS = IDENTITY_COLUMNS + METRIC_COLUMNS + NOTE_COLUMN


class RegistrySchemaError(ValueError):
    """Raised when a row or an existing registry file violates the schema."""


# --------------------------------------------------------------------------- #
# Experiment identity: config hashing, git SHA, run ids
# --------------------------------------------------------------------------- #
def config_hash(config: dict) -> str:
    """Return the SHA-256 of a configuration's canonical JSON form.

    This is the experiment's identity. Two configs that differ only in key
    order, whitespace, or YAML formatting hash to the same value; two configs
    that differ in any actual setting hash differently. That property is what
    makes "distinct cfg_hash" a meaningful count of distinct experiments.

    Parameters
    ----------
    config
        The experiment configuration, typically loaded from a YAML file in
        ``configs/``.

    Returns
    -------
    str
        A 64-character lowercase hex digest.

    Examples
    --------
    >>> config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    True
    """
    canonical_text = canonical_json(config)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def git_sha(short: bool = True, repo_root: Path | None = None) -> str:
    """Return the current git commit SHA.

    What this does and does not pin down
    ------------------------------------
    The recorded SHA is the commit at ``HEAD`` when the row was written. It
    identifies the **base commit**, not necessarily the exact bytes that ran: a
    run started while the working tree carried uncommitted edits records the
    commit those edits sat on top of.

    An earlier version appended a ``-dirty`` suffix in that case. It was removed
    because in practice every development run is dirty - you write code, run it,
    look, tweak, run again - so the flag marked essentially every row and
    conveyed nothing that distinguished one row from another. The write-up
    states the weaker guarantee plainly instead (see README, "The variant
    registry"), which is more useful than a marker that is always on.

    Parameters
    ----------
    short
        If ``True``, return the abbreviated SHA.
    repo_root
        Repository to inspect; defaults to the project root.

    Returns
    -------
    str
        For example ``'a1b2c3d'``, or ``'unknown'`` when the directory is not a
        git repository.
    """
    root = repo_root or PROJECT_ROOT
    rev_parse = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]

    try:
        return subprocess.run(
            rev_parse, cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def new_run_id() -> str:
    """Generate a unique, sortable run identifier.

    The timestamp prefix makes the ids sort chronologically in the registry and
    on disk; the random suffix guarantees uniqueness when several runs start
    within the same second (which happens under parallel execution).

    Returns
    -------
    str
        For example ``'20260718T143012-a1b2c3'``.
    """
    stamp = utc_timestamp().replace("-", "").replace(":", "").replace("+00:00", "")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------- #
# One row of the ledger
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegistryRow:
    """A single registry record: one (run, universe, rung, arm, seed, fold, split).

    Identity fields are required. Metric fields default to ``None`` so that a
    row can be written for a run before or without metrics - a crashed run is
    still a trial and must still be counted.

    Parameters
    ----------
    universe
        Which asset universe, e.g. ``'etf10'``, ``'sector11'``, ``'djia'``.
    rung
        Ablation-ladder rung as a string, because rungs ``'3a'``/``'3b'``/``'3c'``
        are not integers. Rung ``'0'`` is the baselines.
    arm
        Label for the variant within a rung, e.g. ``'scr'``, ``'hmm_gate'``,
        ``'sentiment_to_gating'``.
    seed
        Head-initialisation seed (or reservoir seed for the ESN robustness runs).
    fold
        Walk-forward fold label, e.g. ``'F01'``, or ``'holdout'``.
    split
        One of ``train`` / ``val`` / ``test`` / ``holdout``.
    n_params
        Number of trainable parameters, used for the equal-capacity argument
        that makes the 3a-versus-3b contrast interpretable (A8).
    note
        Free text. Use it to record anything a reader would need in order to
        interpret or distrust the row.
    """

    # --- identity (required) ---
    universe: str
    rung: str
    arm: str
    seed: int
    fold: str
    split: str
    n_params: int
    cfg_hash: str
    run_id: str = ""       # filled by `new_run_id()` if left blank
    timestamp: str = ""    # filled by `utc_timestamp()` if left blank
    git_sha: str = ""      # filled by `git_sha()` if left blank

    # --- metrics (optional) ---
    ann_return: float | None = None
    ann_vol: float | None = None
    excess_sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    cvar_95: float | None = None
    ann_turnover: float | None = None
    holdings_count: float | None = None
    effective_n: float | None = None
    mean_gate_entropy: float | None = None
    importance_shares: str | None = None

    # --- free text ---
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the row at construction time, so bad rows never reach disk."""
        if self.split not in VALID_SPLITS:
            raise RegistrySchemaError(
                f"split must be one of {VALID_SPLITS}, got {self.split!r}."
            )
        for field_name in ("universe", "rung", "arm", "fold", "cfg_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise RegistrySchemaError(
                    f"{field_name} must be a non-empty string, got {value!r}."
                )
        for field_name in ("seed", "n_params"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise RegistrySchemaError(
                    f"{field_name} must be an int, got {type(value).__name__}."
                )
            if value < 0:
                raise RegistrySchemaError(f"{field_name} must be non-negative, got {value}.")

        # Frozen dataclass: use object.__setattr__ to fill the auto fields.
        if not self.run_id:
            object.__setattr__(self, "run_id", new_run_id())
        if not self.timestamp:
            object.__setattr__(self, "timestamp", utc_timestamp())
        if not self.git_sha:
            object.__setattr__(self, "git_sha", git_sha())

    def as_csv_dict(self) -> dict[str, Any]:
        """Return the row as a dict keyed by the registry's column names.

        ``None`` metrics become empty cells, which pandas reads back as ``NaN``.
        """
        raw = asdict(self)
        return {column: ("" if raw[column] is None else raw[column]) for column in REGISTRY_COLUMNS}


# Guard against the dataclass and the header drifting apart during development.
_DATACLASS_FIELDS = {f.name for f in fields(RegistryRow)}
if _DATACLASS_FIELDS != set(REGISTRY_COLUMNS):
    raise RegistrySchemaError(
        "RegistryRow fields and REGISTRY_COLUMNS disagree: "
        f"{_DATACLASS_FIELDS ^ set(REGISTRY_COLUMNS)}"
    )


# --------------------------------------------------------------------------- #
# Append-only writing
# --------------------------------------------------------------------------- #
def _assert_header_matches(registry_path: Path) -> None:
    """Fail loudly if an existing registry has a different header.

    Silently appending rows under a changed schema would corrupt the ledger:
    columns would shift and old rows would be misread. If this raises, the
    correct response is to start a new registry file for the new schema and
    keep the old one, not to edit the old file.
    """
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        existing_header = next(csv.reader(handle), None)

    if existing_header is None:
        return  # empty file; the header will be written on first append
    if tuple(existing_header) != REGISTRY_COLUMNS:
        missing = set(REGISTRY_COLUMNS) - set(existing_header)
        extra = set(existing_header) - set(REGISTRY_COLUMNS)
        raise RegistrySchemaError(
            f"Registry at {registry_path} has a different schema.\n"
            f"  columns missing from the file: {sorted(missing)}\n"
            f"  columns not in the code schema: {sorted(extra)}\n"
            "Start a new registry file rather than editing this one."
        )


def append_row(row: RegistryRow, registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> Path:
    """Append one row to the registry, creating the file with a header if needed.

    The append is guarded by an exclusive file lock, because the per-date
    hyperparameter search and the walk-forward evaluation are run in parallel and
    several processes may append at once; without the lock, two interleaved
    writes can produce a corrupted line.

    Parameters
    ----------
    row
        The record to append. Already validated by :class:`RegistryRow`.
    registry_path
        Destination CSV. Defaults to ``results/registry.csv``.

    Returns
    -------
    pathlib.Path
        The registry path written to.
    """
    path = Path(registry_path)
    ensure_dir(path.parent)

    if path.exists() and path.stat().st_size > 0:
        _assert_header_matches(path)
        write_header = False
    else:
        write_header = True

    with path.open("a", encoding="utf-8", newline="") as handle:
        with _exclusive_lock(handle):
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row.as_csv_dict())
            handle.flush()
    return path


def _exclusive_lock(handle) -> Any:
    """Context manager giving an exclusive advisory lock on an open file.

    Falls back to a no-op on platforms without :mod:`fcntl` (Windows), where
    this project is not run; the fallback keeps the module importable rather
    than failing at import time.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not exercised on POSIX
        from contextlib import nullcontext

        return nullcontext()

    from contextlib import contextmanager

    @contextmanager
    def _locked() -> Iterator[None]:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return _locked()


# --------------------------------------------------------------------------- #
# Reading and the pre-registered trial count
# --------------------------------------------------------------------------- #
def read_registry(registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> pd.DataFrame:
    """Read the whole registry into a DataFrame.

    Parameters
    ----------
    registry_path
        Registry CSV to read.

    Returns
    -------
    pandas.DataFrame
        All rows. Identity columns are always text, numeric metrics are floats
        with empty cells as ``NaN``, and absent text cells are empty strings.
        Dtypes come from the schema, never from pandas' inference over whichever
        rows happen to be present. An empty, correctly-typed frame is returned
        if the file does not exist.
    """
    path = Path(registry_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(REGISTRY_COLUMNS))

    declared_dtypes: dict[str, str] = {column: "string" for column in STRING_COLUMNS}
    declared_dtypes["seed"] = "Int64"
    declared_dtypes["n_params"] = "Int64"

    frame = pd.read_csv(path, dtype=declared_dtypes)

    # Empty text cells read as <NA>; an absent note is better represented as "".
    for column in STRING_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").fillna("")

    for column in NUMERIC_METRIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def count_dsr_trials(registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> int:
    """Return the deflated-Sharpe trial count, under the pre-registered rule.

    The rule, fixed before any experiment existed:

        N_trials = the number of **distinct cfg_hash values** appearing in the
        registry with **split == "val"**.

    Validation rows are the right population because model selection happens on
    validation only: the trial count should measure how many configurations
    were *given a chance to be selected*, not how many were finally reported.
    Counting distinct hashes (rather than rows) means re-running the same
    configuration across folds or seeds does not inflate the count, while
    genuinely different configurations always do.

    Parameters
    ----------
    registry_path
        Registry CSV to read.

    Returns
    -------
    int
        The number of distinct validation configurations tried.
    """
    frame = read_registry(registry_path)
    if frame.empty:
        return 0
    validation_rows = frame[frame["split"] == "val"]
    return int(validation_rows["cfg_hash"].nunique())


def summarise_registry(registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> dict:
    """Return a small human-readable summary of the registry's contents.

    Intended for end-of-run logging and for the sanity check that the results
    chapter is quoting the registry it thinks it is.

    Parameters
    ----------
    registry_path
        Registry CSV to read.

    Returns
    -------
    dict
        Row counts by split, distinct configurations, and the DSR trial count.
    """
    frame = read_registry(registry_path)
    if frame.empty:
        return {"n_rows": 0, "n_configs": 0, "dsr_trials": 0, "rows_by_split": {}}

    return {
        "n_rows": int(len(frame)),
        "n_configs": int(frame["cfg_hash"].nunique()),
        "dsr_trials": count_dsr_trials(registry_path),
        "rows_by_split": frame["split"].value_counts().to_dict(),
        "universes": sorted(frame["universe"].unique().tolist()),
        "rungs": sorted(frame["rung"].astype(str).unique().tolist()),
    }
