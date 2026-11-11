"""How does out-of-sample performance vary with the number of experts K?

    python scripts/run_capacity_curve.py --universe sector11
    python scripts/run_capacity_curve.py --universe etf10 --k-values 1,2,3

The question, and why the obvious version of it is wrong
--------------------------------------------------------
"Does the mixture help?" is usually answered by running K = 1, 2, 3 ... at one
fixed set of reservoir hyperparameters and comparing. That answer is worthless,
because the hyperparameters were tuned at ONE value of K. Every other K is then
being judged at settings chosen for its competitor, and the earlier joint search
in this project showed the two genuinely interact: on sector11 the spectral
radius swung performance by 0.062 at K = 1 and by 0.006 at K = 3.

So this script searches the reservoir **inside each K**. Every K gets its own
best configuration, found on its own, and only then are they compared. If K = 1
still wins, it wins against opponents that were each given a fair tuning budget.

Two numbers per K, and they answer different questions
-------------------------------------------------------
``validation``
    The concatenated Sharpe on the selection folds - the quantity the inner
    search maximised. It says how well capacity K can be made to FIT.

``test``
    The concatenated Sharpe over the FULL walk-forward (all 18 sector11 folds,
    all 10 etf10 folds), at the configuration the inner search chose, on the
    test split. Nothing in the selection ever saw these segments at this
    configuration. This is the honest number.

Reporting both is the point. The gap between them IS the overfitting, and if it
widens with K then extra experts are buying fit rather than generalisation -
which is the capacity story the rest of the project has been circling.

Scope: the plain mixture only
-----------------------------
Only ``MixtureOfExperts`` - K experts and a learned gate, both reading the frozen
reservoir state and nothing else. No market signals into the gate or the experts
(rungs 3a/3b/3c), and no statistical routing (HMM, VIX terciles). Those arms
change the INFORMATION available, and mixing an information change into a
capacity sweep would make the curve uninterpretable. This measures capacity, one
variable, over a fixed information set.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.metrics import excess_sharpe  # noqa: E402
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.gating import MixtureOfExperts  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.hyper_search import build_states_for_config  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture  # noqa: E402
from run_joint_search import FOLD_SETS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("capacity")

FIXED = {"washout": 250, "bias_scale": 0.0, "calibration_fraction": 0.6,
         "max_iterations": 10, "tolerance": 0.02}

# The inner search grid. Deliberately restricted to the two reservoir axes with
# any measured effect in this project - width and spectral radius. The leak rate
# and input scale were re-measured at their defaults and never transferred, so
# searching them again inside every K would multiply the cost for noise.
INNER_GRID = {
    "n_reservoir": [150, 300, 600],
    "spectral_radius": [0.85, 0.90, 0.95],
}
ANCHOR = {"leak_rate": 0.30, "target_pre_activation_std": 0.70, "gate_temperature": 1.0}


def train_config(k: int, seed: int) -> MixtureTrainConfig:
    """One training configuration. lambda_balance is inert at K = 1."""
    return MixtureTrainConfig(
        n_experts=k,
        lambda_cvar=0.0,
        lambda_balance=0.01 if k > 1 else 0.0,
        lambda_turn=0.0,
        lambda_div=0.0,
        seed=seed,
    )


def score_split(packs_by_fold, folds, k, seeds, split):
    """Concatenated Sharpe on one split: mean and seed spread.

    Per seed the fold segments are stitched into one continuous daily stream and
    scored once - the project's locked aggregation, so these numbers sit on the
    same scale as every other Sharpe in the dissertation.
    """
    per_seed = []
    for seed in seeds:
        streams, risk_frees = [], []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            n_state = packs["train"].states.shape[1]
            n_assets = packs["train"].logit_mask.shape[1]
            result = train_mixture(
                packs["train"], packs["val"], train_config(k, seed),
                model=lambda: MixtureOfExperts(n_state, n_assets, k),
            )
            evaluation = evaluate_on_pack(result.model, packs[split])
            streams.append(evaluation["net_returns"].to_numpy())
            risk_frees.append(packs[split].risk_free.numpy())
        per_seed.append(excess_sharpe(np.concatenate(streams), np.concatenate(risk_frees)))

    spread = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    return float(np.mean(per_seed)), spread


def parameter_count(n_state: int, n_assets: int, k: int) -> int:
    """Trainable parameters: K experts plus the gate."""
    return k * (n_state * n_assets + n_assets) + (n_state * k + k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Out-of-sample performance versus K.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--k-values", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--search-seeds", default="0,1", help="Screening seeds for the inner search.")
    parser.add_argument("--test-seeds", default="0,1,2", help="Seeds for the final walk-forward.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir = data_root / "data" / "processed"
    states_dir = data_root / "data" / "states"
    registry_path = data_root / "results" / "registry.csv"

    k_values = [int(v) for v in args.k_values.split(",")]
    search_seeds = [int(v) for v in args.search_seeds.split(",")]
    test_seeds = [int(v) for v in args.test_seeds.split(",")]

    artifacts = load_universe_artifacts(processed_dir, args.universe)
    all_folds = {f.name: f for f in folds_for(args.universe)}
    selection_folds = [all_folds[n] for n in FOLD_SETS[args.universe]["selection"]]
    every_fold = list(all_folds.values())

    logger.info("=== capacity curve: %s ===", args.universe)
    logger.info("  inner search : %d cells per K, %d folds x %d seeds (validation)",
                len(INNER_GRID["n_reservoir"]) * len(INNER_GRID["spectral_radius"]),
                len(selection_folds), len(search_seeds))
    logger.info("  final score  : %d folds x %d seeds (TEST)",
                len(every_fold), len(test_seeds))

    # Reservoir states depend only on (N_r, rho), not on K, so build each once
    # and reuse across every K. This is most of the saving that makes the sweep
    # affordable.
    states_cache = {}
    for n_reservoir, rho in itertools.product(
        INNER_GRID["n_reservoir"], INNER_GRID["spectral_radius"]
    ):
        params = {**ANCHOR, "n_reservoir": n_reservoir, "spectral_radius": rho}
        states_cache[(n_reservoir, rho)] = build_states_for_config(
            args.universe, params, FIXED, processed_dir, states_dir
        )
    logger.info("  built/loaded %d reservoirs", len(states_cache))

    rows = []
    for k in k_values:
        logger.info("")
        logger.info("--- K = %d ---", k)

        # ---- inner search: best reservoir FOR THIS K, on selection folds ----
        best = None
        for (n_reservoir, rho), states in states_cache.items():
            packs_by_fold = {
                f.name: build_fold_packs(artifacts, states, f) for f in selection_folds
            }
            val_sharpe, val_sd = score_split(
                packs_by_fold, selection_folds, k, search_seeds, "val"
            )
            logger.info("    N_r=%4d rho=%.2f  val %+.3f (sd %.3f)",
                        n_reservoir, rho, val_sharpe, val_sd)
            if best is None or val_sharpe > best["val_sharpe"]:
                best = {"n_reservoir": n_reservoir, "spectral_radius": rho,
                        "val_sharpe": val_sharpe, "val_sd": val_sd}
            if not args.dry_run:
                identity = {"universe": args.universe, "rung": "capacity",
                            "n_experts": k, "n_reservoir": n_reservoir,
                            "spectral_radius": rho, **ANCHOR}
                append_row(
                    RegistryRow(
                        universe=args.universe, rung="capacity", arm=f"K{k}",
                        seed=search_seeds[0], fold="capacity-search", split="val",
                        n_params=0, cfg_hash=config_hash(identity),
                        excess_sharpe=val_sharpe,
                        note=f"capacity curve inner search; K={k} "
                             f"N_r={n_reservoir} rho={rho:.2f}",
                    ),
                    registry_path,
                )

        # ---- outer score: that winner, on the FULL walk-forward, TEST split --
        states = states_cache[(best["n_reservoir"], best["spectral_radius"])]
        packs_by_fold = {f.name: build_fold_packs(artifacts, states, f) for f in every_fold}
        test_sharpe, test_sd = score_split(
            packs_by_fold, every_fold, k, test_seeds, "test"
        )

        sample = packs_by_fold[every_fold[0].name]["train"]
        n_params = parameter_count(sample.states.shape[1], sample.logit_mask.shape[1], k)

        logger.info("  best for K=%d: N_r=%d rho=%.2f  |  val %+.3f  ->  TEST %+.3f (sd %.3f)"
                    "  |  %d params",
                    k, best["n_reservoir"], best["spectral_radius"],
                    best["val_sharpe"], test_sharpe, test_sd, n_params)

        rows.append({"K": k, **{key: best[key] for key in
                                ("n_reservoir", "spectral_radius", "val_sharpe", "val_sd")},
                     "test_sharpe": test_sharpe, "test_sd": test_sd,
                     "n_params": n_params,
                     "gap": best["val_sharpe"] - test_sharpe})
        if not args.dry_run:
            identity = {"universe": args.universe, "rung": "capacity-test",
                        "n_experts": k, "n_reservoir": best["n_reservoir"],
                        "spectral_radius": best["spectral_radius"], **ANCHOR}
            append_row(
                RegistryRow(
                    universe=args.universe, rung="capacity", arm=f"K{k}",
                    seed=test_seeds[0], fold="capacity-test", split="test",
                    n_params=n_params, cfg_hash=config_hash(identity),
                    excess_sharpe=test_sharpe,
                    note=f"capacity curve, full walk-forward; K={k} "
                         f"N_r={best['n_reservoir']} rho={best['spectral_radius']:.2f}",
                ),
                registry_path,
            )

    frame = pd.DataFrame(rows)
    logger.info("")
    logger.info("=== capacity curve, %s ===", args.universe)
    for line in frame.round(3).to_string(index=False).splitlines():
        logger.info("  %s", line)

    out_dir = data_root / "results" / "runs" / "capacity"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"capacity_{args.universe}.csv", index=False)
    logger.info("Wrote %s", out_dir / f"capacity_{args.universe}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
