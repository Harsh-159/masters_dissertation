"""Verify a frozen configuration on folds that were never used to select it.

    python scripts/run_verify_frozen.py --universe sector11 --dry-run
    python scripts/run_verify_frozen.py --universe sector11 --seeds 0,1,2,3,4

Why this script exists
----------------------
Every configuration in this project was chosen by looking at validation Sharpe
on folds F01-F12. A number chosen that way is not evidence that the choice was
right - it is the thing that was maximised, and maximising a noisy quantity over
many candidates is exactly how selection bias is manufactured.

sector11's later folds F13-F18 (validation years 2018-2023) were never touched
by any search, grid, or K sweep. They are therefore a genuine held-out check on
the *selection process itself*, distinct from the 2025-2026 holdout, which stays
firewalled for the single scripted P11 run.

The check matters here more than usual: two previous "winners" in this project
collapsed at precisely this step. The first random search froze a configuration
at validation Sharpe 0.817 that scored 0.298 when re-run across seeds. The first
reservoir grid was run at K = 3 and its winner did not survive the move to K = 1.
A configuration that has not passed this test has not earned a `config-freeze`
tag.

What it compares
----------------
Three configurations, so the answer distinguishes "the correction helped" from
"nothing here matters":

``old_frozen``
    What ``configs/frozen/sector11.yaml`` contained before it was rewritten -
    the random-search winner, searched at K = 3.
``corrected``
    The K = 1 grid winner now in that file.
``defaults``
    ``configs/reservoir.yaml`` untouched. If the corrected config cannot beat
    plain defaults out of sample, the honest write-up says the reservoir
    hyperparameters do not matter on this universe - which is itself a finding,
    and one the response-surface analysis already points at.

All three run at K = 1 with every regulariser at zero, so the only thing varying
is the reservoir. Scored the project's standard way: each seed's fold validation
segments stitched into one continuous daily stream, one excess Sharpe, then mean
and spread across seeds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.backtest.metrics import excess_sharpe  # noqa: E402
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.hyper_search import build_states_for_config  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("verify")

FIXED = {
    "washout": 250,
    "bias_scale": 0.0,
    "calibration_fraction": 0.6,
    "max_iterations": 10,
    "tolerance": 0.02,
}

# The reservoirs under test, PER UNIVERSE. These must be the configurations that
# universe's own searches actually proposed - testing sector11's candidates on
# etf10 data would compare three arbitrary points and answer nothing.
#
# The plan's Appendix B defaults are the common reference in both, because they
# are the only configuration chosen without looking at any data and therefore
# the only one carrying no selection bias.
DEFAULTS = {
    "n_reservoir": 300,
    "spectral_radius": 0.90,
    "leak_rate": 0.30,
    "target_pre_activation_std": 0.70,
    "gate_temperature": 1.0,
}
CONFIGURATIONS = {
    "sector11": {
        # The random-search winner that configs/frozen/sector11.yaml used to hold.
        "random_search": {
            "n_reservoir": 479,
            "spectral_radius": 0.7197482002031762,
            "leak_rate": 0.9568609690062003,
            "target_pre_activation_std": 0.8998749764251764,
            "gate_temperature": 1.0,
        },
        # The K = 1 reservoir grid winner (results/runs/grid/grid_sector11.csv).
        "grid_winner": {**DEFAULTS, "spectral_radius": 0.95},
        "defaults": DEFAULTS,
    },
    "etf10": {
        # The random-search winner still sitting in configs/frozen/etf10.yaml at
        # val_sharpe 1.0616 - never once checked on folds it was not selected on.
        "random_search": {
            "n_reservoir": 415,
            "spectral_radius": 0.8256546735919826,
            "leak_rate": 0.31015664366957935,
            "target_pre_activation_std": 1.1431824847917436,
            "gate_temperature": 1.9157021404131362,
        },
        "defaults": DEFAULTS,
    },
}

# K per universe. sector11 selected K = 1 over three rounds of the K search;
# etf10 never made a selection (every K tied), so the plan's default stands.
# Verifying etf10's reservoir at K = 1 would test a configuration nothing chose.
EXPERTS = {"sector11": 1, "etf10": 3}

# Folds every search in this project looked at, and folds none of them did.
# See run_joint_search.FOLD_SETS for why each split is what it is.
FOLD_SETS = {
    "sector11": (["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"],
                 ["F13", "F14", "F15", "F16", "F17", "F18"]),
    "etf10": (["F01", "F02", "F03", "F04", "F05"],
              ["F06", "F07", "F08", "F09", "F10"]),
}


def score(
    universe: str,
    params: dict,
    artifacts,
    folds: list,
    seeds: list[int],
    states_dir: Path,
    processed_dir: Path,
    n_experts: int,
) -> tuple[float, float, list[float]]:
    """Concatenated validation Sharpe for one reservoir config: mean, sd, per-seed.

    Identical aggregation to ``run_grid_search.score_config`` - deliberately, so
    a number here is comparable with a number there without any adjustment.
    """
    states = build_states_for_config(universe, params, FIXED, processed_dir, states_dir)
    packs_by_fold = {f.name: build_fold_packs(artifacts, states, f) for f in folds}

    per_seed = []
    for seed in seeds:
        streams, risk_frees = [], []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            result = train_mixture(
                packs["train"],
                packs["val"],
                MixtureTrainConfig(
                    n_experts=n_experts,  # sector11 selected 1; etf10 defaults to 3
                    lambda_cvar=0.0,      # every regulariser selects to zero
                    lambda_balance=0.01 if n_experts > 1 else 0.0,
                    lambda_turn=0.0,
                    lambda_div=0.0,
                    gate_temperature=float(params["gate_temperature"]),
                    seed=seed,
                ),
            )
            evaluation = evaluate_on_pack(result.model, packs["val"])
            streams.append(evaluation["net_returns"].to_numpy())
            risk_frees.append(packs["val"].risk_free.numpy())
        per_seed.append(excess_sharpe(np.concatenate(streams), np.concatenate(risk_frees)))

    spread = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    return float(np.mean(per_seed)), spread, [float(value) for value in per_seed]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out verification of a frozen config.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT),
        help="Checkout holding data/ and results/ (defaults to this script's repo).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write no registry rows. Use while orienting; the run of record should register.",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir, states_dir = data_root / "data" / "processed", data_root / "data" / "states"
    registry_path = data_root / "results" / "registry.csv"
    seeds = [int(value) for value in args.seeds.split(",")]

    configurations = CONFIGURATIONS[args.universe]
    n_experts = EXPERTS[args.universe]
    selection_folds, held_out_folds = FOLD_SETS[args.universe]
    held_out_label = f"held-out {held_out_folds[0]}-{held_out_folds[-1]}"
    artifacts = load_universe_artifacts(processed_dir, args.universe)
    all_folds = {fold.name: fold for fold in folds_for(args.universe)}

    rows = []
    for fold_set_name, fold_names in [
        (f"selection {selection_folds[0]}-{selection_folds[-1]}", selection_folds),
        (f"held-out {held_out_folds[0]}-{held_out_folds[-1]}", held_out_folds),
    ]:
        folds = [all_folds[name] for name in fold_names]
        logger.info("")
        logger.info("=== %s (%d folds x %d seeds) ===", fold_set_name, len(folds), len(seeds))
        for label, params in configurations.items():
            mean, spread, per_seed = score(
                args.universe, params, artifacts, folds, seeds, states_dir,
                processed_dir, n_experts,
            )
            logger.info(
                "  %-13s N_r=%4d rho=%.3f alpha=%.3f std=%.2f  ->  %+.3f (sd %.3f)  seeds %s",
                label, params["n_reservoir"], params["spectral_radius"],
                params["leak_rate"], params["target_pre_activation_std"],
                mean, spread, " ".join(f"{v:+.2f}" for v in per_seed),
            )
            rows.append(
                {"fold_set": fold_set_name, "config": label, "sharpe": mean,
                 "sharpe_sd": spread, **params}
            )
            if not args.dry_run:
                identity = {"universe": args.universe, "rung": "verify",
                            "fold_set": fold_set_name, **params}
                append_row(
                    RegistryRow(
                        universe=args.universe, rung="verify", arm=label,
                        seed=seeds[0], fold=fold_set_name.split()[0], split="val",
                        n_params=0, cfg_hash=config_hash(identity),
                        excess_sharpe=mean,
                        note=f"held-out verification; {fold_set_name}; sd {spread:.3f}",
                    ),
                    registry_path,
                )

    frame = pd.DataFrame(rows)
    out_dir = data_root / "results" / "runs" / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"verify_frozen_{args.universe}.csv", index=False)

    # The question the script exists to answer, stated explicitly.
    logger.info("")
    logger.info("=== verdict ===")
    held = frame[frame.fold_set == held_out_label].set_index("config")
    default = held.loc["defaults", "sharpe"]
    spread = held["sharpe_sd"].max()
    best_searched = max(
        (name for name in held.index if name != "defaults"),
        key=lambda name: held.loc[name, "sharpe"],
    )
    corrected = held.loc[best_searched, "sharpe"]
    for name in held.index:
        logger.info("  %-14s held out: %+.3f", name, held.loc[name, "sharpe"])
    logger.info("  best searched (%s) minus defaults : %+.3f",
                best_searched, corrected - default)
    if corrected - default < spread:
        logger.info("  -> the correction is WITHIN seed noise of plain defaults out of sample.")
        logger.info("     Report the reservoir search as finding nothing on this universe.")
    logger.info("")
    logger.info("Wrote %s", out_dir / f"verify_frozen_{args.universe}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
