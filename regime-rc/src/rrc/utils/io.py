"""Serialisation and file-writing primitives shared across the project.

Two concerns live here, both of which exist to protect the audit trail:

**Canonical JSON.** A configuration must hash to the same value every time it
describes the same experiment, regardless of the order keys happen to appear in
a YAML file or a Python dict. ``canonical_json`` fixes a single, total ordering
and a single formatting convention, so ``cfg_hash`` (see
:mod:`rrc.eval.registry`) is a stable identity for an experiment rather than an
accident of authoring.

**Atomic writes.** A run that dies halfway through writing a results file must
not leave a half-written file that later looks like a valid result.
``atomic_write_text`` writes to a temporary file in the same directory and then
renames it, which is atomic on POSIX filesystems: readers see either the old
file or the complete new one, never a partial one.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _json_default(value: Any) -> str:
    """Convert non-JSON-native values to a stable string.

    Configurations legitimately contain dates, paths, and occasionally NumPy
    scalars. Rather than let :func:`json.dumps` raise, we convert them to a
    deterministic string form. Determinism is the only requirement here: the
    same object must always produce the same text, because this feeds a hash.

    Parameters
    ----------
    value
        An object that :mod:`json` cannot serialise natively.

    Returns
    -------
    str
        A stable textual representation.
    """
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        # Sets have no inherent order; sort so the text is deterministic.
        return json.dumps(sorted(str(item) for item in value))
    # NumPy scalars and anything else exotic: fall back to repr-like text.
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def canonical_json(config: Any) -> str:
    """Render an object as canonical JSON text.

    "Canonical" here means three things, all chosen so that logically identical
    configurations produce byte-identical text:

    1. keys are sorted, so authoring order is irrelevant;
    2. separators are tight and fixed, so whitespace is irrelevant;
    3. non-JSON types are converted deterministically by :func:`_json_default`.

    Parameters
    ----------
    config
        Any JSON-like structure, typically a nested dict loaded from YAML.

    Returns
    -------
    str
        Canonical JSON text suitable for hashing.

    Examples
    --------
    >>> canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    True
    """
    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dictionary.

    Parameters
    ----------
    path
        Path to a ``.yaml`` file.

    Returns
    -------
    dict
        The parsed mapping.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not parse to a mapping (a config must be a mapping,
        because every downstream consumer indexes it by key).
    """
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    parsed = yaml.safe_load(yaml_path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Config at {yaml_path} must be a mapping at the top level, "
            f"got {type(parsed).__name__}."
        )
    return parsed


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path
        Directory to create.

    Returns
    -------
    pathlib.Path
        The directory path, now guaranteed to exist.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Write text to a file atomically.

    The text is written to a temporary file in the *same directory* as the
    target (same directory matters: ``os.replace`` is only atomic within a
    filesystem) and then renamed over the target. A crash mid-write therefore
    leaves either the previous file or nothing - never a truncated file that a
    later reader might mistake for a complete result.

    Parameters
    ----------
    path
        Destination file path.
    text
        Content to write.
    encoding
        Text encoding.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    destination = Path(path)
    ensure_dir(destination.parent)

    # delete=False because we rename the file ourselves; the finally block
    # cleans up only if the rename never happened.
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())  # force to disk before the rename
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string, seconds precision.

    All timestamps in the registry are UTC so that rows remain sortable and
    comparable regardless of where or when the machine's clock is set.

    Returns
    -------
    str
        For example ``'2026-07-18T14:30:12+00:00'``.
    """
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
