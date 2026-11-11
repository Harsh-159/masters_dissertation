"""Tests for the variant registry.

These cover the three properties the audit trail depends on:

1. ``cfg_hash`` is a stable identity for a configuration (order- and
   formatting-invariant, but sensitive to every actual setting);
2. a row survives a write/read round trip unchanged, and the file is
   genuinely append-only with a single header;
3. the deflated-Sharpe trial count implements the pre-registered rule exactly
   - distinct configurations on the validation split, counted once each
   regardless of how many folds or seeds they were run over.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rrc.eval.registry import (
    REGISTRY_COLUMNS,
    RegistryRow,
    RegistrySchemaError,
    append_row,
    config_hash,
    count_dsr_trials,
    git_sha,
    new_run_id,
    read_registry,
    summarise_registry,
)


def make_row(**overrides) -> RegistryRow:
    """Build a valid row, overriding any field.

    ``git_sha`` is supplied explicitly so the tests neither shell out to git nor
    depend on the state of the working tree.
    """
    defaults = dict(
        universe="etf10",
        rung="1",
        arm="scr",
        seed=0,
        fold="F01",
        split="val",
        n_params=3612,
        cfg_hash=config_hash({"model": "single_head"}),
        git_sha="abc1234",
    )
    defaults.update(overrides)
    return RegistryRow(**defaults)


# --------------------------------------------------------------------------- #
# 1. Configuration hashing
# --------------------------------------------------------------------------- #
def test_config_hash_ignores_key_order():
    """Logically identical configs hash identically regardless of key order."""
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_ignores_nesting_order():
    """Order-invariance holds recursively, not just at the top level."""
    first = {"model": {"k": 3, "rho": 0.9}, "seed": 0}
    second = {"seed": 0, "model": {"rho": 0.9, "k": 3}}
    assert config_hash(first) == config_hash(second)


def test_config_hash_changes_when_any_value_changes():
    """A different setting is a different experiment, and must hash differently."""
    baseline = {"model": {"k": 3, "rho": 0.9}}
    changed = {"model": {"k": 3, "rho": 0.95}}
    assert config_hash(baseline) != config_hash(changed)


def test_config_hash_handles_dates_and_is_stable():
    """Dates are common in configs; they must serialise deterministically."""
    config = {"start": dt.date(2006, 2, 1), "end": dt.date(2026, 6, 30)}
    assert config_hash(config) == config_hash(config)
    assert len(config_hash(config)) == 64  # sha256 hex digest


def test_config_hash_is_a_hex_digest():
    """Sanity: the hash is a 64-character lowercase hex string."""
    digest = config_hash({"a": 1})
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


# --------------------------------------------------------------------------- #
# 2. Row validation
# --------------------------------------------------------------------------- #
def test_invalid_split_is_rejected():
    """`split` is constrained because the DSR rule is defined over it."""
    with pytest.raises(RegistrySchemaError, match="split must be one of"):
        make_row(split="validation")  # near-miss spelling of "val"


def test_empty_identity_field_is_rejected():
    """An empty universe would produce an unattributable row."""
    with pytest.raises(RegistrySchemaError, match="universe must be"):
        make_row(universe="")


def test_non_integer_seed_is_rejected():
    """Seeds are integers; a float seed usually means a bug upstream."""
    with pytest.raises(RegistrySchemaError, match="seed must be an int"):
        make_row(seed=0.5)


def test_boolean_seed_is_rejected():
    """`True` is an int in Python; reject it explicitly rather than record seed=1."""
    with pytest.raises(RegistrySchemaError, match="seed must be an int"):
        make_row(seed=True)


def test_auto_filled_fields_are_populated():
    """run_id, timestamp and git_sha are filled in when not supplied."""
    row = make_row()
    assert row.run_id
    assert row.timestamp.endswith("+00:00")  # UTC
    assert row.git_sha == "abc1234"


def test_run_ids_are_unique():
    """Parallel runs starting in the same second must not collide."""
    ids = {new_run_id() for _ in range(200)}
    assert len(ids) == 200


# --------------------------------------------------------------------------- #
# 3. Append-only round trip  (P0's explicit definition of done)
# --------------------------------------------------------------------------- #
def test_append_and_read_round_trip(tmp_path):
    """A written row reads back with identical values."""
    registry = tmp_path / "registry.csv"
    row = make_row(excess_sharpe=0.71, ann_vol=0.134, note="smoke")

    append_row(row, registry)
    frame = read_registry(registry)

    assert len(frame) == 1
    record = frame.iloc[0]
    assert record["universe"] == "etf10"
    assert record["rung"] == "1"
    assert record["arm"] == "scr"
    assert int(record["seed"]) == 0
    assert record["split"] == "val"
    assert int(record["n_params"]) == 3612
    assert record["cfg_hash"] == row.cfg_hash
    assert record["excess_sharpe"] == pytest.approx(0.71)
    assert record["ann_vol"] == pytest.approx(0.134)
    assert record["note"] == "smoke"


def test_missing_metrics_round_trip_as_nan(tmp_path):
    """A crashed run still gets a row; its metrics read back as NaN, not 0."""
    registry = tmp_path / "registry.csv"
    append_row(make_row(note="crashed before evaluation"), registry)

    frame = read_registry(registry)
    assert frame["excess_sharpe"].isna().all()
    assert frame["max_drawdown"].isna().all()


def test_header_written_once_and_rows_accumulate(tmp_path):
    """Appending is append-only: one header, N data lines."""
    registry = tmp_path / "registry.csv"
    for index in range(3):
        append_row(make_row(seed=index), registry)

    lines = registry.read_text().strip().splitlines()
    assert len(lines) == 4  # 1 header + 3 rows
    assert lines[0].split(",") == list(REGISTRY_COLUMNS)
    assert len(read_registry(registry)) == 3


def test_rung_stays_a_string_even_when_all_rungs_look_numeric(tmp_path):
    """Regression: dtypes come from the schema, not from pandas inference.

    Rung labels are strings ("0", "1", "3a", ...). If the dtype were inferred,
    a registry containing only numeric rungs would read back as int64 and every
    `rung == "1"` filter would match nothing once rung "3a" appeared in P7.
    """
    registry = tmp_path / "registry.csv"
    append_row(make_row(rung="1"), registry)
    append_row(make_row(rung="2"), registry)

    frame = read_registry(registry)
    assert frame["rung"].tolist() == ["1", "2"]
    assert (frame["rung"] == "1").sum() == 1


def test_numeric_and_lettered_rungs_coexist(tmp_path):
    """The 3a/3b/3c arms must filter identically to the numeric rungs."""
    registry = tmp_path / "registry.csv"
    for rung in ("1", "2", "3a", "3b", "4"):
        append_row(make_row(rung=rung), registry)

    frame = read_registry(registry)
    assert frame["rung"].tolist() == ["1", "2", "3a", "3b", "4"]
    assert (frame["rung"] == "3a").sum() == 1
    assert (frame["rung"] == "4").sum() == 1


def test_seed_reads_back_as_integer(tmp_path):
    """Seeds stay integers so they can be grouped and sorted numerically."""
    registry = tmp_path / "registry.csv"
    append_row(make_row(seed=3), registry)

    frame = read_registry(registry)
    assert int(frame["seed"].iloc[0]) == 3
    assert frame["seed"].dtype == "Int64"


def test_absent_note_reads_back_as_empty_string(tmp_path):
    """An absent note is "", not NaN, so string operations on it are safe."""
    registry = tmp_path / "registry.csv"
    append_row(make_row(), registry)

    frame = read_registry(registry)
    assert frame["note"].iloc[0] == ""


def test_schema_drift_is_rejected(tmp_path):
    """A registry with a different header must not be silently appended to."""
    registry = tmp_path / "registry.csv"
    registry.write_text("run_id,timestamp,something_else\n")

    with pytest.raises(RegistrySchemaError, match="different schema"):
        append_row(make_row(), registry)


def test_reading_a_missing_registry_returns_empty_frame(tmp_path):
    """Reading before the first run yields an empty, correctly-shaped frame."""
    frame = read_registry(tmp_path / "does_not_exist.csv")
    assert frame.empty
    assert list(frame.columns) == list(REGISTRY_COLUMNS)


# --------------------------------------------------------------------------- #
# 4. The pre-registered deflated-Sharpe trial count
# --------------------------------------------------------------------------- #
def test_trial_count_is_zero_before_any_experiment(tmp_path):
    assert count_dsr_trials(tmp_path / "registry.csv") == 0


def test_trial_count_counts_distinct_validation_configs(tmp_path):
    """Two different configs on val => two trials."""
    registry = tmp_path / "registry.csv"
    append_row(make_row(cfg_hash=config_hash({"rho": 0.9}), split="val"), registry)
    append_row(make_row(cfg_hash=config_hash({"rho": 0.95}), split="val"), registry)

    assert count_dsr_trials(registry) == 2


def test_same_config_across_folds_and_seeds_counts_once(tmp_path):
    """Re-running one configuration must not inflate the trial count."""
    registry = tmp_path / "registry.csv"
    same_config = config_hash({"rho": 0.9})
    for fold in ("F01", "F02", "F03"):
        for seed in (0, 1, 2):
            append_row(
                make_row(cfg_hash=same_config, fold=fold, seed=seed, split="val"),
                registry,
            )

    assert len(read_registry(registry)) == 9
    assert count_dsr_trials(registry) == 1


def test_non_validation_rows_do_not_count_as_trials(tmp_path):
    """Selection happens on validation, so only validation rows are trials."""
    registry = tmp_path / "registry.csv"
    append_row(make_row(cfg_hash=config_hash({"a": 1}), split="train"), registry)
    append_row(make_row(cfg_hash=config_hash({"a": 2}), split="test"), registry)
    append_row(make_row(cfg_hash=config_hash({"a": 3}), split="holdout"), registry)

    assert len(read_registry(registry)) == 3
    assert count_dsr_trials(registry) == 0


def test_summarise_registry_reports_counts(tmp_path):
    """The summary is what a run prints at the end and what the write-up quotes."""
    registry = tmp_path / "registry.csv"
    append_row(make_row(cfg_hash=config_hash({"a": 1}), split="val"), registry)
    append_row(make_row(cfg_hash=config_hash({"a": 2}), split="val"), registry)
    append_row(make_row(cfg_hash=config_hash({"a": 1}), split="test"), registry)

    summary = summarise_registry(registry)
    assert summary["n_rows"] == 3
    assert summary["n_configs"] == 2
    assert summary["dsr_trials"] == 2
    assert summary["rows_by_split"]["val"] == 2
    assert summary["universes"] == ["etf10"]


# --------------------------------------------------------------------------- #
# 5. Code identity
# --------------------------------------------------------------------------- #
def test_git_sha_returns_a_string():
    """git_sha never raises; it degrades to 'unknown' outside a repository."""
    sha = git_sha()
    assert isinstance(sha, str)
    assert sha
