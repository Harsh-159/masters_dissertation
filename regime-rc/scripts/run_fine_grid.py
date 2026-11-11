"""A dense local grid around each universe's best-capacity reservoir.

    python scripts/run_fine_grid.py --universe sector11
    python scripts/run_fine_grid.py --universe etf10

The capacity curve searched a coarse 9-cell reservoir grid (N_r in {150,300,600}
x rho in {0.85,0.90,0.95}) inside each K and found the best K per universe:
sector11 K=2 at (300, 0.85), etf10 K=5 at (600, 0.95). This script asks whether a
FINER search around those winners finds anything better - the coarse grid could
have stepped over a local optimum.

Design
------
A full factorial, densest on the two axes with measured effect in this project
(N_r, rho) and 3-point fine sweeps around the optimum on the three the project
found flat or secondary (leak alpha, input std, gate temperature tau_g). The
grids are centred on each universe's capacity-curve winner.

Two numbers per cell, exactly as the capacity curve reports them:
* ``val``  - concatenated Sharpe on the VAL split of the selection folds
  (sector11 F01-F08, etf10 F01-F05), 2 seeds. This is the fitness the grid
  maximises.
* ``test`` - for the top cells only, the concatenated Sharpe on the TEST split of
  ALL folds, 3 seeds. Nothing in selection saw these. If the val winner does not
  top the test table, that is the transfer-failure finding (L2/F4) reproduced on
  a finer grid - report it, do not chase it.

Honesty / scope
---------------
* States are built into a SCRATCH states dir (``--states-dir``), NOT the project
  ``data/states/``, so the ~1,500 candidate reservoirs here do not clutter the
  real cache or its default-by-construction resolution.
* Writes ``results/runs/grid_fine/`` only. It appends NO registry rows and
  touches NO holdout data. NOTE for the write-up: a grid this fine is ~1,500
  configurations "given a chance to be selected"; under the pre-registered rule
  that would raise the DSR bar if any winner here were quoted as selected. It is
  reported as a robustness probe of the FROZEN defaults, not as a new selection.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.metrics import excess_sharpe  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.gating import MixtureOfExperts  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.hyper_search import build_states_for_config  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture  # noqa: E402
from run_joint_search import FOLD_SETS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("fine-grid")

FIXED = {"washout": 250, "bias_scale": 0.0, "calibration_fraction": 0.6,
         "max_iterations": 10, "tolerance": 0.02}

# Centred on each universe's capacity-curve winner. N_r and rho dense (the axes
# with effect); alpha, std, tau_g are 3-point fine sweeps around the optimum.
GRID = {
    "sector11": dict(k=2,
                     n_reservoir=[250, 270, 290, 310, 330, 350],
                     spectral_radius=[0.80, 0.83, 0.86, 0.89, 0.92],
                     leak_rate=[0.20, 0.30, 0.40],
                     target_pre_activation_std=[0.60, 0.70, 0.80],
                     gate_temperature=[0.80, 1.00, 1.20]),
    "etf10": dict(k=5,
                  n_reservoir=[500, 550, 600, 650, 700],
                  spectral_radius=[0.90, 0.93, 0.95, 0.97, 0.99],
                  leak_rate=[0.20, 0.30, 0.40],
                  target_pre_activation_std=[0.60, 0.70, 0.80],
                  gate_temperature=[0.80, 1.00, 1.20]),
}
# The capacity-curve winner, reproduced here so the fine grid brackets it.
REFERENCE = {"sector11": (300, 0.85, 0.505), "etf10": (600, 0.95, 0.411)}


def train_config(k: int, tau: float, seed: int) -> MixtureTrainConfig:
    return MixtureTrainConfig(
        n_experts=k, lambda_cvar=0.0, lambda_balance=0.01 if k > 1 else 0.0,
        lambda_turn=0.0, lambda_div=0.0, gate_temperature=tau, seed=seed,
    )


def score(packs_by_fold, folds, k, tau, seeds, split):
    """Concatenated Sharpe on one split: mean over seeds (+ sd)."""
    per_seed = []
    for seed in seeds:
        streams, rfs = [], []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            n_state = packs["train"].states.shape[1]
            n_assets = packs["train"].logit_mask.shape[1]
            result = train_mixture(
                packs["train"], packs["val"], train_config(k, tau, seed),
                model=lambda: MixtureOfExperts(n_state, n_assets, k, gate_temperature=tau),
            )
            ev = evaluate_on_pack(result.model, packs[split])
            streams.append(ev["net_returns"].to_numpy())
            rfs.append(packs[split].risk_free.numpy())
        per_seed.append(excess_sharpe(np.concatenate(streams), np.concatenate(rfs)))
    sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    return float(np.mean(per_seed)), sd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense local reservoir grid at the best K.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--val-seeds", default="0,1")
    parser.add_argument("--test-seeds", default="0,1,2")
    parser.add_argument("--top-n", type=int, default=12, help="Cells re-scored on test.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    parser.add_argument("--states-dir", default="", help="Scratch dir for candidate reservoirs.")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir = data_root / "data" / "processed"
    states_dir = Path(args.states_dir).expanduser().resolve() if args.states_dir \
        else data_root / "results" / "runs" / "grid_fine" / "_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    val_seeds = [int(s) for s in args.val_seeds.split(",")]
    test_seeds = [int(s) for s in args.test_seeds.split(",")]

    spec = GRID[args.universe]
    k = spec["k"]
    artifacts = load_universe_artifacts(processed_dir, args.universe)
    all_folds = {f.name: f for f in folds_for(args.universe)}
    sel_folds = [all_folds[n] for n in FOLD_SETS[args.universe]["selection"]]
    every_fold = list(all_folds.values())

    # A state config is (N_r, rho, alpha, std); tau_g does NOT rebuild states.
    state_axes = list(itertools.product(
        spec["n_reservoir"], spec["spectral_radius"],
        spec["leak_rate"], spec["target_pre_activation_std"]))
    n_cells = len(state_axes) * len(spec["gate_temperature"])
    ref_nr, ref_rho, ref_val = REFERENCE[args.universe]
    logger.info("=== fine grid: %s | K=%d ===", args.universe, k)
    logger.info("  %d state configs x %d tau = %d cells | val %d folds x %d seeds",
                len(state_axes), len(spec["gate_temperature"]), n_cells,
                len(sel_folds), len(val_seeds))
    logger.info("  bracketing capacity winner (N_r %d, rho %.2f, val ref %.3f)",
                ref_nr, ref_rho, ref_val)

    rows = []
    t0 = time.time()
    for i, (nr, rho, alpha, std) in enumerate(state_axes, 1):
        params = {"leak_rate": alpha, "target_pre_activation_std": std,
                  "gate_temperature": 1.0, "n_reservoir": nr, "spectral_radius": rho}
        states = build_states_for_config(args.universe, params, FIXED, processed_dir, states_dir)
        sel_packs = {f.name: build_fold_packs(artifacts, states, f) for f in sel_folds}
        for tau in spec["gate_temperature"]:
            val, val_sd = score(sel_packs, sel_folds, k, tau, val_seeds, "val")
            rows.append(dict(universe=args.universe, k=k, n_reservoir=nr,
                             spectral_radius=rho, leak_rate=alpha,
                             target_pre_activation_std=std, gate_temperature=tau,
                             val_sharpe=val, val_sd=val_sd))
        done = i * len(spec["gate_temperature"])
        if i % 10 == 0 or i == len(state_axes):
            rate = (time.time() - t0) / done
            eta = rate * (n_cells - done) / 60
            logger.info("  %4d/%d cells | %.2fs/cell | ETA %.0f min",
                        done, n_cells, rate, eta)

    grid = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False).reset_index(drop=True)
    out_dir = data_root / "results" / "runs" / "grid_fine"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / f"grid_fine_{args.universe}.csv", index=False)

    logger.info("")
    logger.info("=== top %d by validation Sharpe ===", args.top_n)
    logger.info("  N_r  rho   alpha std  tau  |   val   (sd)")
    top = grid.head(args.top_n).copy()
    for _, r in top.iterrows():
        logger.info("  %4d %.2f  %.2f  %.2f %.2f |  %.3f  (%.3f)",
                    int(r.n_reservoir), r.spectral_radius, r.leak_rate,
                    r.target_pre_activation_std, r.gate_temperature, r.val_sharpe, r.val_sd)

    # Honest number: re-score the top cells on the full walk-forward TEST split.
    logger.info("")
    logger.info("=== the same top cells on TEST (all folds, %d seeds) - the honest number ===",
                len(test_seeds))
    logger.info("  N_r  rho   alpha std  tau  |   val  -> test  (sd)")
    test_scores = []
    for _, r in top.iterrows():
        params = {"leak_rate": r.leak_rate, "target_pre_activation_std": r.target_pre_activation_std,
                  "gate_temperature": 1.0, "n_reservoir": int(r.n_reservoir),
                  "spectral_radius": r.spectral_radius}
        states = build_states_for_config(args.universe, params, FIXED, processed_dir, states_dir)
        packs = {f.name: build_fold_packs(artifacts, states, f) for f in every_fold}
        test, test_sd = score(packs, every_fold, k, r.gate_temperature, test_seeds, "test")
        test_scores.append(dict(**r, test_sharpe=test, test_sd=test_sd))
        logger.info("  %4d %.2f  %.2f  %.2f %.2f |  %.3f -> %.3f  (%.3f)",
                    int(r.n_reservoir), r.spectral_radius, r.leak_rate,
                    r.target_pre_activation_std, r.gate_temperature, r.val_sharpe, test, test_sd)
    pd.DataFrame(test_scores).to_csv(out_dir / f"grid_fine_top_test_{args.universe}.csv", index=False)

    best_val = grid.iloc[0]
    best_test = max(test_scores, key=lambda d: d["test_sharpe"])
    logger.info("")
    logger.info("  best on VALIDATION : N_r %d rho %.2f a %.2f std %.2f tau %.2f  val %.3f",
                int(best_val.n_reservoir), best_val.spectral_radius, best_val.leak_rate,
                best_val.target_pre_activation_std, best_val.gate_temperature, best_val.val_sharpe)
    logger.info("  best on TEST       : N_r %d rho %.2f a %.2f std %.2f tau %.2f  test %.3f "
                "(capacity ref %.3f)",
                int(best_test["n_reservoir"]), best_test["spectral_radius"], best_test["leak_rate"],
                best_test["target_pre_activation_std"], best_test["gate_temperature"],
                best_test["test_sharpe"], ref_val)
    logger.info("  wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
