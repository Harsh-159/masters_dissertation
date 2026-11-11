"""Tests for the serialisation and file-writing primitives.

The canonical-JSON tests matter because ``cfg_hash`` is built on top of them: if
canonicalisation is not total, two identical experiments can hash differently
and the deflated-Sharpe trial count silently inflates.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from rrc.eval.registry import config_hash
from rrc.utils.io import (
    atomic_write_text,
    canonical_json,
    ensure_dir,
    load_yaml,
    utc_timestamp,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------- #
# Canonical JSON
# --------------------------------------------------------------------------- #
def test_canonical_json_sorts_keys():
    """Key order in the source is irrelevant to the output text."""
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_is_compact():
    """Whitespace is fixed, so formatting differences cannot change the hash."""
    text = canonical_json({"a": 1, "b": [1, 2]})
    assert " " not in text
    assert text == '{"a":1,"b":[1,2]}'


def test_canonical_json_round_trips_as_valid_json():
    """Whatever we hash must still be readable JSON, for debugging."""
    text = canonical_json({"model": {"k": 3}, "seed": 0})
    assert json.loads(text) == {"model": {"k": 3}, "seed": 0}


def test_canonical_json_serialises_dates_deterministically():
    """Dates appear in every experiment config (study windows and fold edges)."""
    config = {"start": dt.date(2006, 2, 1)}
    assert canonical_json(config) == '{"start":"2006-02-01"}'


def test_canonical_json_serialises_paths_and_sets():
    """Paths and sets appear occasionally; both must be stable."""
    assert canonical_json({"p": Path("/tmp/x")}) == '{"p":"/tmp/x"}'
    # A set has no inherent order, so it is sorted before serialising.
    assert canonical_json({"s": {"b", "a"}}) == canonical_json({"s": {"a", "b"}})


# --------------------------------------------------------------------------- #
# YAML loading, and the end-to-end config -> hash path
# --------------------------------------------------------------------------- #
def test_load_yaml_reads_a_mapping():
    config = load_yaml(CONFIGS_DIR / "smoke.yaml")
    assert config["experiment"]["name"] == "smoke"
    assert config["data"]["universe"] == "etf10"
    assert config["run"]["seed"] == 0


def test_smoke_config_hashes_stably():
    """The real config file hashes to the same value on repeated loads."""
    first = config_hash(load_yaml(CONFIGS_DIR / "smoke.yaml"))
    second = config_hash(load_yaml(CONFIGS_DIR / "smoke.yaml"))
    assert first == second
    assert len(first) == 64


def test_load_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml(CONFIGS_DIR / "no_such_config.yaml")


def test_load_yaml_rejects_non_mapping(tmp_path):
    """A config must be a mapping; a bare list would break every consumer."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml(bad_config)


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
def test_atomic_write_creates_file_with_content(tmp_path):
    target = tmp_path / "nested" / "out.txt"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_overwrites_cleanly(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text() == "second"


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    """A stray .tmp file would later look like an artifact; there must be none."""
    target = tmp_path / "out.txt"
    atomic_write_text(target, "content")
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_ensure_dir_is_idempotent(tmp_path):
    directory = tmp_path / "a" / "b"
    assert ensure_dir(directory).is_dir()
    assert ensure_dir(directory).is_dir()  # calling twice must not raise


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def test_utc_timestamp_is_utc_and_second_precision():
    stamp = utc_timestamp()
    assert stamp.endswith("+00:00")
    assert "." not in stamp  # microseconds stripped, so rows stay comparable
