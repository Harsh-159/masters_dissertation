"""Print a summary of the variant registry, including the DSR trial count.

Run this at any point to see how many distinct configurations have been given a
chance to be selected. The number only grows, and it is the number the deflated
Sharpe ratio will be computed against - seeing it regularly is a useful check on
how much search the project has actually done.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrc.eval.registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    read_registry,
    summarise_registry,
)


def main() -> int:
    """Print the registry summary. Returns a process exit code."""
    registry_path = DEFAULT_REGISTRY_PATH
    frame = read_registry(registry_path)

    print(f"Registry: {registry_path}")
    if frame.empty:
        print("  (empty - no experiments have been run yet)")
        print("  deflated-Sharpe trial count: 0")
        return 0

    summary = summarise_registry(registry_path)
    print(f"  rows              : {summary['n_rows']}")
    print(f"  distinct configs  : {summary['n_configs']}")
    print(f"  universes         : {', '.join(summary['universes'])}")
    print(f"  rungs             : {', '.join(summary['rungs'])}")
    print("  rows by split     :")
    for split_name, count in sorted(summary["rows_by_split"].items()):
        print(f"      {split_name:<8} {count}")
    print()
    print(f"  deflated-Sharpe trial count (distinct cfg_hash on split='val'): "
          f"{summary['dsr_trials']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
