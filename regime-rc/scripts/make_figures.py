"""Regenerate every figure of record from the frozen registry.

Design rule this script enforces
--------------------------------
No figure in the dissertation is produced by hand, and none is produced in a
notebook. Every figure is a function registered in ``FIGURE_BUILDERS`` below,
which reads the registry (and the artifacts the registry points at) and writes a
file into ``results/figures/``. Running ``make figures`` therefore rebuilds the
entire figure set from the recorded results, which is what makes the claim
"every figure regenerates from the frozen registry" literally true.

At P0 the dictionary is empty: there are no results yet. Later phases add their
figures here as they are built, so this file becomes the index of figures F1-F12
named in Section 13 of the implementation plan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Allow `python scripts/make_figures.py` to work without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rrc.eval.registry import DEFAULT_REGISTRY_PATH, read_registry  # noqa: E402
from rrc.utils.io import ensure_dir  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

# Maps a figure identifier (F1-F12 in the plan) to the callable that builds it.
# Each builder takes the registry DataFrame and an output directory, and
# returns the path it wrote.
#
#   Planned, filled in by the phase that produces the underlying result:
#     F1  data coverage / availability map          (P1)
#     F2  memory-capacity curve                     (P3)
#     F3  pre-tanh saturation histogram             (P3)
#     F4  ladder bar chart with bootstrap CIs       (P9)
#     F5  log equity curves per rung vs baselines   (P9)
#     F6  regime occupancy timeline                 (P6)
#     F7  expert-weight heatmaps + correlations     (P6)
#     F8  regime agreement (AMI bars, confusions)   (P7)
#     F9  selected-front-quantile trace vs VIX      (P8)
#     F10 example per-date fronts                   (P8)
#     F11 drawdown traces                           (P9)
#     F12 holdings-count / turnover distributions   (P9)
def requires_registry(builder: Callable) -> Callable:
    """Mark a figure builder as needing recorded experiment results.

    Most figures plot results and cannot be drawn before any experiment has run.
    A few - F1 coverage, F2/F3 reservoir health - describe the *data* or the
    *model's dynamics* and are drawable as soon as their phase completes. Marking
    the distinction explicitly means `make figures` can render what is currently
    renderable instead of refusing to draw anything until the very end.
    """
    builder.requires_registry = True
    return builder


def build_f1_coverage(registry, figures_dir: Path) -> Path:
    """F1 - data coverage and availability map.

    Shows, per universe and asset, the span over which the asset was actually
    available to trade. This is the figure that makes the late entrants visible:
    XLRE (Oct 2015) and XLC (Jun 2018) appear as short bars against the nine
    original SPDRs running from 1999, which is precisely the condition the
    availability matrix exists to handle.

    Reads the data artifacts rather than the registry, because coverage is a
    property of the dataset, not of any experiment.
    """
    import matplotlib.pyplot as plt

    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    universes = ["etf10", "sector11"]
    available = [u for u in universes if (processed_dir / f"{u}_availability.parquet").is_file()]
    if not available:
        raise FileNotFoundError("No availability artifacts; run scripts/run_data.py first.")

    figure, axes = plt.subplots(
        len(available), 1, figsize=(12, 3.2 * len(available)), squeeze=False
    )

    for row, universe in enumerate(available):
        axis = axes[row][0]
        availability = pd.read_parquet(processed_dir / f"{universe}_availability.parquet")

        for position, ticker in enumerate(availability.columns):
            live_days = availability.index[availability[ticker] == 1]
            if len(live_days) == 0:
                continue
            axis.barh(
                y=position,
                width=(live_days.max() - live_days.min()).days,
                left=live_days.min(),
                height=0.6,
                color="#3b6ea5",
            )

        axis.set_yticks(range(len(availability.columns)))
        axis.set_yticklabels(availability.columns, fontsize=8)
        axis.invert_yaxis()
        axis.set_title(
            f"{universe} - asset availability "
            f"({len(availability)} trading days, "
            f"{availability.index.min().date()} to {availability.index.max().date()})",
            fontsize=10,
        )
        axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    output_path = figures_dir / "F1_data_coverage.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


FIGURE_BUILDERS: dict[str, Callable] = {
    "F1": build_f1_coverage,
}


def main() -> int:
    """Rebuild all registered figures. Returns a process exit code."""
    ensure_dir(FIGURES_DIR)
    registry = read_registry(DEFAULT_REGISTRY_PATH)

    if not FIGURE_BUILDERS:
        print("No figure builders are registered yet.")
        return 0

    built, skipped = 0, []
    for figure_id, builder in FIGURE_BUILDERS.items():
        needs_results = getattr(builder, "requires_registry", False)
        if needs_results and registry.empty:
            skipped.append(figure_id)
            continue
        written = builder(registry, FIGURES_DIR)
        print(f"  {figure_id}: {written.name}")
        built += 1

    if skipped:
        print(
            f"  skipped {len(skipped)} result-dependent figure(s) "
            f"({', '.join(skipped)}) - the registry is still empty."
        )
    print(f"Rebuilt {built} figure(s) into {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
