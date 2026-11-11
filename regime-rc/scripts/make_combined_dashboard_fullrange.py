"""The combined 50/50 dashboard over the FULL out-of-sample range (2015-2024).

    # first build (regenerates the streams; ~30-60 min, background it):
    python scripts/make_combined_dashboard_fullrange.py --repopulate
    # later replots (reuses the cached full-range streams; seconds):
    python scripts/make_combined_dashboard_fullrange.py

Why this script exists
----------------------
`make_combined_dashboard.py` blends etf10 and sector11 on the *matched
2015-2017* window, because:

  * the etf10 model streams cached on disk cover only folds F01-F03
    (test 2015-2017), even though PREVIEW_FOLDS already lists all ten; and
  * the matched sector sleeve is trained only on folds F09-F11 (test 2015-2017).

This script widens both sides to the full common out-of-sample era:

  * etf10   -> folds F01-F10  (test 2015-2024), regenerated in place;
  * sector11 matched -> folds F09-F18 (test 2015-2024).

Everything else - the daily-rebalanced 50/50 blend, the per-panel metrics, the
approximate turnover, the quarantine (diagnostics dir, no registry writes, no
F-number) - is exactly the tested machinery of the 2015-2017 dashboard, reused
unchanged. It is still an INTERIM VISUAL, superseded by P9's figures of record.

Cost note
---------
`--repopulate` retrains 7 arms x 10 folds for each universe at seed 0 on cached
states (~140 trainings). Run it once in the background; the streams are cached
afterwards, so subsequent runs (no flag) just reblend and replot in seconds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# The machinery we reuse verbatim. We only widen two fold lists (below) before
# calling the populate functions, so the identical training/blending/plotting
# path produces the full-range picture.
import make_combined_ladder as combined_ladder  # noqa: E402
import make_combined_dashboard as combined_dashboard  # noqa: E402
from make_ladder_preview import (  # noqa: E402
    PREVIEW_FOLDS,
    copy_rung0_streams,
    load_streams,
    populate_model_streams,
)
from make_combined_ladder import blend_streams, load_matched_streams  # noqa: E402
from make_ladder_dashboard import plot_dashboard, stream_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("combined-dashboard-full")

# etf10 folds whose test years span the full out-of-sample era 2015-2024.
FULL_ETF_FOLDS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10"]

# sector11 folds whose TEST years are 2015..2024 - the same calendar years the
# etf10 folds above test - so the daily 50/50 blend is well-defined across the
# whole window. (sector11 F09 tests 2015, ..., F18 tests 2024.)
FULL_MATCHED_SECTOR_FOLDS = [
    "F09", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18"
]


def _apply_full_range_overrides() -> None:
    """Point the reused machinery at the full-range fold sets.

    The populate functions read these module globals at call time, so patching
    them here - before any populate call - is all that is needed. We assert
    PREVIEW_FOLDS already covers F01-F10 (it does in the current script) rather
    than mutating it, to avoid surprising the plain `make ladder` preview.
    """
    missing = [f for f in FULL_ETF_FOLDS if f not in PREVIEW_FOLDS]
    if missing:
        raise RuntimeError(
            f"make_ladder_preview.PREVIEW_FOLDS is missing {missing}; the etf10 "
            "streams cannot cover the full range. Widen PREVIEW_FOLDS first."
        )
    # The matched sector sleeve: widen F09-F11 -> F09-F18.
    combined_ladder.MATCHED_SECTOR_FOLDS = FULL_MATCHED_SECTOR_FOLDS
    # The turnover panel reads etf10 registry turnover over these folds.
    combined_dashboard.ETF_FOLDS = FULL_ETF_FOLDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combined 50/50 dashboard over the full 2015-2024 window."
    )
    parser.add_argument(
        "--repopulate", action="store_true",
        help="Regenerate the model streams over the full fold sets. Required on "
             "the first run (the cached streams are the narrow 2015-2017 ones); "
             "omit on later runs to reuse the full-range cache.",
    )
    args = parser.parse_args(argv)

    _apply_full_range_overrides()

    # Ensure both ingredient stores exist and are full-range.
    copy_rung0_streams("etf10")
    copy_rung0_streams("sector11")
    populate_model_streams("etf10", repopulate=args.repopulate)
    combined_ladder.populate_matched_sector_streams(repopulate=args.repopulate)

    # Daily-rebalanced 50/50 blend on the shared calendar (now 2015-2024).
    blended = blend_streams(load_streams("etf10"), load_matched_streams())
    if not blended:
        raise RuntimeError(
            "No blended streams were produced. If this is the first full-range "
            "run, pass --repopulate so the streams are regenerated over all folds."
        )

    span_start = min(s.index.min() for s in blended.values()).date()
    span_end = max(s.index.max() for s in blended.values()).date()
    logger.info("blended window: %s -> %s (%d methods)", span_start, span_end, len(blended))

    # Both universes share the same DGS3MO-derived risk-free series.
    risk_free = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "etf10_risk_free.parquet"
    )["r_f"]

    metrics = stream_metrics(blended, risk_free)
    turnover = combined_dashboard.blended_turnover(list(blended))

    path = plot_dashboard(
        label="combined 50/50 (etf10 + sector11 sleeves, daily-rebalanced) - full range",
        streams=blended,
        metrics=metrics,
        turnover=turnover,
        output_stem="ladder_dashboard_combined_50_50_fullrange",
        turnover_note=(
            "turnover = 0.5 x etf10 sleeve (registry F01-F03 where available) + "
            "0.5 x sector11 sleeve (matched-stream sidecars F09-F18)"
        ),
    )
    logger.info("%d blended lines (%s -> %s) -> %s",
                len(blended), span_start, span_end, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
