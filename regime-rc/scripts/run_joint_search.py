"""Final joint search over K and the reservoir, then a held-out check.

    python scripts/run_joint_search.py --universe sector11
    python scripts/run_joint_search.py --universe sector11 --dry-run

Why a JOINT search, when both axes have already been searched
--------------------------------------------------------------
Because they were searched one at a time, and each pass held the other fixed at
a value the other pass later changed:

* the K search ran at the DEFAULT reservoir and selected K = 1;
* the reservoir grid ran at K = 3, was found contaminated, and was re-run at
  K = 1 - but only after K had been fixed.

So no measurement in this project has ever varied K and the reservoir together.
If they interact, both earlier answers are conditional on a value the other
search moved. That is the one genuine gap left on this axis, and it is what this
script closes.

Deliberately lean, for two reasons
----------------------------------
Every configuration here is a distinct ``cfg_hash`` on validation and therefore
a deflated-Sharpe trial. Searching wider makes the deflation benchmark higher
for every result in the dissertation, so breadth has a real price.

And the evidence says breadth would not pay. Held-out verification of three
reservoir configurations (``run_verify_frozen.py``) found the spread ACROSS
configurations (0.064) to be smaller than the seed spread WITHIN one (0.085).
So the grid fixes N_r, the leak rate and the input scale at their measured
optima and varies only the two axes with an established effect:

* **K**, which showed a real and monotone effect on sector11 (0.670 / 0.639 at
  K = 1 / 3 in the twelve-fold round);
* **spectral radius**, the one reservoir parameter with a large measured swing
  at K = 1 (0.459 at rho = 0.70 against 0.615 at rho = 0.95).

Twelve cells, not two hundred.

The held-out check is not optional
----------------------------------
The winner is re-scored on folds F13-F18, which no search in this project has
ever touched. Three configurations have already been selected on validation in
this project and then failed to transfer. A winner here is adopted only if it
survives that check; otherwise the pre-specified defaults stand and the search
is reported as negative. That rule is stated here, before the run, so the
outcome cannot be reinterpreted afterwards.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.optim.hyper_search import evaluate_candidate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("joint")

# Held at their measured optima; see the module docstring for why they are not
# searched again.
FIXED = {
    "washout": 250,
    "bias_scale": 0.0,
    "calibration_fraction": 0.6,
    "max_iterations": 10,
    "tolerance": 0.02,
}
ANCHOR = {
    "n_reservoir": 300,
    "leak_rate": 0.30,
    "target_pre_activation_std": 0.70,
    "gate_temperature": 1.0,
}

K_VALUES = [1, 2, 3]
SPECTRAL_RADII = [0.85, 0.90, 0.95, 0.99]

# Which folds each universe's searches actually looked at, and which they never
# did. These are facts about the run history, not choices:
#
#   sector11 - the grid, the K search and the reservoir re-run all used F01-F08
#              (configs/grid.yaml, configs/grid_k1.yaml). F13-F18 are untouched.
#   etf10    - the ONLY registered search is the 50-draw random search, which
#              used F01-F05 (configs/hyper.yaml). F06-F10 are untouched.
#              Caveat: the cross-universe K diagnostic reported in
#              SEARCH_FINDINGS.md ran on 8 etf10 folds but left no registry
#              rows, so F06-F08 may have been seen once in a diagnostic that
#              never fed a selection. F09-F10 are unambiguously clean.
FOLD_SETS = {
    "sector11": {
        "selection": ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"],
        "held_out": ["F13", "F14", "F15", "F16", "F17", "F18"],
    },
    "etf10": {
        "selection": ["F01", "F02", "F03", "F04", "F05"],
        "held_out": ["F06", "F07", "F08", "F09", "F10"],
    },
}


def run_cell(universe, k, rho, folds, seeds, processed_dir, cache_dir):
    """Score one (K, spectral radius) cell on the concatenated validation stream."""
    params = {**ANCHOR, "spectral_radius": rho}
    return evaluate_candidate(
        universe=universe,
        params=params,
        fixed=FIXED,
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        fold_names=folds,
        seeds=seeds,
        n_experts=k,
        lambda_cvar=0.0,       # every regulariser selects to zero; see configs
        lambda_balance=0.01 if k > 1 else 0.0,   # inert at K = 1
        lambda_turn=0.0,
        lambda_div=0.0,
        max_epochs=200,
        patience=15,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Joint K x reservoir search.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir, cache_dir = data_root / "data" / "processed", data_root / "data" / "states"
    registry_path = data_root / "results" / "registry.csv"
    seeds = [int(v) for v in args.seeds.split(",")]

    selection_folds = FOLD_SETS[args.universe]["selection"]
    held_out_folds = FOLD_SETS[args.universe]["held_out"]
    logger.info("=== joint K x spectral-radius grid: %d cells, %d folds x %d seeds ===",
                len(K_VALUES) * len(SPECTRAL_RADII), len(selection_folds), len(seeds))

    rows = []
    for k, rho in itertools.product(K_VALUES, SPECTRAL_RADII):
        candidate = run_cell(args.universe, k, rho, selection_folds, seeds,
                             processed_dir, cache_dir)
        rows.append({"K": k, "spectral_radius": rho, "sharpe": candidate.val_sharpe,
                     "cvar": candidate.val_cvar})
        logger.info("  K=%d rho=%.2f  ->  %+.3f", k, rho, candidate.val_sharpe)
        if not args.dry_run:
            identity = {"universe": args.universe, "rung": "joint-search",
                        "n_experts": k, **{**ANCHOR, "spectral_radius": rho}}
            append_row(
                RegistryRow(
                    universe=args.universe, rung="joint", arm=f"K{k}_rho{rho:.2f}",
                    seed=seeds[0], fold="joint", split="val", n_params=0,
                    cfg_hash=config_hash(identity), excess_sharpe=candidate.val_sharpe,
                    note=f"joint K x rho grid; K={k} rho={rho:.2f}",
                ),
                registry_path,
            )

    frame = pd.DataFrame(rows)
    table = frame.pivot(index="K", columns="spectral_radius", values="sharpe")
    logger.info("")
    logger.info("=== selection folds %s: concatenated validation Sharpe ===", "-".join([selection_folds[0], selection_folds[-1]]))
    for line in table.round(3).to_string().splitlines():
        logger.info("  %s", line)

    best = frame.loc[frame["sharpe"].idxmax()]
    logger.info("")
    logger.info("Best on selection folds: K=%d rho=%.2f at %+.3f",
                int(best["K"]), best["spectral_radius"], best["sharpe"])

    # --- the held-out check, pre-registered above ---------------------------
    logger.info("")
    logger.info("=== held-out %s (never used in any selection) ===", "-".join([held_out_folds[0], held_out_folds[-1]]))
    held = []
    contenders = [(int(best["K"]), float(best["spectral_radius"]), "search winner"),
                  (1, 0.90, "defaults (K=1)"),
                  (3, 0.90, "defaults (K=3)")]
    for k, rho, label in contenders:
        candidate = run_cell(args.universe, k, rho, held_out_folds, seeds,
                             processed_dir, cache_dir)
        held.append({"K": k, "spectral_radius": rho, "label": label,
                     "sharpe": candidate.val_sharpe})
        logger.info("  %-16s K=%d rho=%.2f  ->  %+.3f", label, k, rho, candidate.val_sharpe)

    held_frame = pd.DataFrame(held)
    winner_held = held_frame.iloc[0]["sharpe"]
    default_held = held_frame[held_frame.label == "defaults (K=1)"].iloc[0]["sharpe"]
    logger.info("")
    logger.info("=== verdict ===")
    logger.info("  search winner minus K=1 defaults, held out: %+.3f", winner_held - default_held)
    if winner_held <= default_held:
        logger.info("  -> the winner does NOT transfer. Keep the pre-specified defaults")
        logger.info("     and report the search as a negative result.")
    else:
        logger.info("  -> the winner transfers. Adopt it, and say so with this number.")

    out_dir = data_root / "results" / "runs" / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"joint_selection_{args.universe}.csv", index=False)
    held_frame.to_csv(out_dir / f"joint_heldout_{args.universe}.csv", index=False)
    logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
