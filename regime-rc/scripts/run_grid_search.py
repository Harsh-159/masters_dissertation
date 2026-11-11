"""Coarse-to-fine grid search over the reservoir hyperparameters.

    python scripts/run_grid_search.py --universe sector11

Why a grid now, when the earlier pass was random
-------------------------------------------------
Random search and grid search answer different questions, and the project used
each for the one it suits.

The **random** pass (50 draws over five dimensions) was a *screening* exercise:
nobody knew which hyperparameters mattered, and a grid would have spent equal
budget on every dimension before finding out. Its answer, from the binned
response surface:

===================  ==================  ===============================
 hyperparameter       quartile spread     verdict (seed noise ~0.03-0.08)
===================  ==================  ===============================
 N_reservoir          0.186 -> 0.281      the only clear effect, ~2x noise
 input target_std     0.197 -> 0.255      marginal, higher is better
 spectral radius      0.051               flat across its range
 gate temperature     0.020               flat
 leak rate            0.020               flat
===================  ==================  ===============================

With the effective dimensionality now known to be one, possibly two, the curse
of dimensionality that rules grids out no longer applies, and a grid is the
better tool: even coverage of the dimension that matters, and a plottable
marginal curve rather than a scatter of unevenly-spaced draws.

The division of budget follows the measurement: **many levels of N_r, few of
input target_std, fewest of the three flat parameters**. The flat three are not
dropped - they are checked one-at-a-time at the winning point, so "no
interaction" is verified rather than assumed.

Note that "flat" means flat *within the admissible range*. Every searched
spectral radius satisfies the echo state property and every leak rate gives a
working integrator; outside those ranges these parameters matter enormously.
The grid explores the same pre-vetted box.

Coarse to fine, and where to stop
---------------------------------
Stage 1 grids N_r x target_std coarsely. Stage 2 varies each flat parameter
alone at the stage-1 winner. Stage 3 refines N_r around it.

Refinement stops at the **noise floor**: seed-to-seed spread is about 0.03-0.06
and the whole N_r effect is about 0.095, so resolving N_r more finely than the
stage-3 spacing would be fitting seed variance, not signal. One refinement round
is what the data supports; the script does not offer more.

Every configuration is a distinct cfg_hash on validation and therefore a
deflated-Sharpe trial, so the grid is deliberately lean.
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.metrics import excess_sharpe  # noqa: E402
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.hyper_search import build_states_for_config  # noqa: E402
from rrc.optim.train_loop import (  # noqa: E402
    MixtureTrainConfig,
    evaluate_on_pack,
    train_mixture,
)
from rrc.utils.io import load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("grid")


def score_config(
    universe: str, params: dict, config: dict, artifacts, folds, seeds: list[int]
) -> tuple[float, float]:
    """Concatenated validation Sharpe for one reservoir configuration.

    Per seed the fold validation segments are stitched into one continuous daily
    stream and scored once - the same aggregation the baselines and the
    literature use. Averaging per-fold Sharpes was measured to differ from this
    by 0.48 on sector11 and is not used anywhere.

    Returns the mean over seeds and the seed spread, because the spread is what
    says whether a difference between two configurations is real.
    """
    states = build_states_for_config(
        universe, params, config["fixed"],
        PROJECT_ROOT / config["paths"]["processed_dir"],
        PROJECT_ROOT / config["paths"]["states_dir"],
    )
    packs_by_fold = {f.name: build_fold_packs(artifacts, states, f) for f in folds}
    loss = config["loss"]

    per_seed = []
    for seed in seeds:
        streams, risk_frees = [], []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            result = train_mixture(
                packs["train"], packs["val"],
                MixtureTrainConfig(
                    n_experts=int(config["mixture"]["n_experts"]),
                    lambda_cvar=float(loss["lambda_cvar"]),
                    lambda_balance=float(loss["lambda_balance"]),
                    lambda_turn=float(loss["lambda_turn"]),
                    lambda_div=float(loss["lambda_div"]),
                    gate_temperature=float(params["gate_temperature"]),
                    seed=seed,
                ),
            )
            evaluation = evaluate_on_pack(result.model, packs["val"])
            streams.append(evaluation["net_returns"].to_numpy())
            risk_frees.append(packs["val"].risk_free.numpy())
        per_seed.append(
            excess_sharpe(np.concatenate(streams), np.concatenate(risk_frees))
        )

    spread = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    return float(np.mean(per_seed)), spread


def evaluate_points(
    universe, points, config, artifacts, folds, seeds, registry_path, dry_run, stage
) -> pd.DataFrame:
    """Score a list of parameter dicts; log and register each one."""
    rows = []
    for index, params in enumerate(points, start=1):
        sharpe, spread = score_config(universe, params, config, artifacts, folds, seeds)
        rows.append({**params, "sharpe": sharpe, "sharpe_sd": spread, "stage": stage})
        logger.info(
            "  [%s %2d/%2d] N_r=%4d std=%.2f rho=%.2f a=%.2f tau=%.2f "
            "-> Sharpe %+.3f (sd %.3f)",
            stage, index, len(points), params["n_reservoir"],
            params["target_pre_activation_std"], params["spectral_radius"],
            params["leak_rate"], params["gate_temperature"], sharpe, spread,
        )
        if not dry_run:
            identity = {"universe": universe, "rung": "hyper-grid",
                        "stage": stage, **params}
            append_row(
                RegistryRow(
                    universe=universe, rung="hyper", arm="grid",
                    seed=seeds[0], fold="grid", split="val",
                    n_params=0, cfg_hash=config_hash(identity),
                    excess_sharpe=sharpe,
                    note=(f"grid {stage}; N_r={params['n_reservoir']} "
                          f"std={params['target_pre_activation_std']:.2f} "
                          f"rho={params['spectral_radius']:.2f} "
                          f"alpha={params['leak_rate']:.2f} "
                          f"tau={params['gate_temperature']:.2f}"),
                ),
                registry_path,
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coarse-to-fine reservoir grid search.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "grid.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Score everything but write no registry rows.")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    grid = config["grid"]
    defaults = config["defaults"]
    seeds = [int(s) for s in config["evaluation"]["seeds"]]
    fold_names = list(config["evaluation"]["fold_names"])

    artifacts = load_universe_artifacts(
        PROJECT_ROOT / config["paths"]["processed_dir"], args.universe
    )
    folds = [f for f in folds_for(args.universe) if f.name in fold_names]
    registry_path = PROJECT_ROOT / "results" / "registry.csv"

    logger.info("=== %s grid search: folds %s, seeds %s ===",
                args.universe, ",".join(fold_names), seeds)

    def point(**overrides) -> dict:
        return {**defaults, **overrides}

    # --- Stage 1: the two dimensions that matter -------------------------
    logger.info("--- stage 1: N_reservoir x input target_std (others at default) ---")
    stage1_points = [
        point(n_reservoir=int(n), target_pre_activation_std=float(std))
        for n in grid["n_reservoir"]
        for std in grid["target_pre_activation_std"]
    ]
    stage1 = evaluate_points(args.universe, stage1_points, config, artifacts, folds,
                             seeds, registry_path, args.dry_run, "coarse")
    best = stage1.loc[stage1["sharpe"].idxmax()].to_dict()
    logger.info("  stage-1 best: N_r=%d std=%.2f -> %+.3f",
                int(best["n_reservoir"]), best["target_pre_activation_std"],
                best["sharpe"])

    # --- Stage 2: do the flat parameters interact with that winner? -------
    logger.info("--- stage 2: one-at-a-time check on the flat three ---")
    stage2_points = []
    # Whatever the grid lists beyond the two stage-1 dimensions. At K = 1 the
    # gate temperature is a mathematical no-op (softmax over a single logit is
    # 1.0 whatever the temperature), so the config simply omits it rather than
    # spending evaluations on a parameter that cannot change the answer.
    stage2_names = [n for n in grid
                    if n not in ('n_reservoir', 'target_pre_activation_std')]
    for name in stage2_names:
        for value in grid[name]:
            if float(value) == float(defaults[name]):
                continue                      # already scored in stage 1
            stage2_points.append(point(
                n_reservoir=int(best["n_reservoir"]),
                target_pre_activation_std=float(best["target_pre_activation_std"]),
                **{name: float(value)},
            ))
    stage2 = evaluate_points(args.universe, stage2_points, config, artifacts, folds,
                             seeds, registry_path, args.dry_run, "interact")

    # --- Stage 3: refine N_r around the winner, once ----------------------
    logger.info("--- stage 3: refine N_reservoir (one round; noise floor) ---")
    ladder = sorted(int(n) for n in grid["n_reservoir"])
    chosen = int(best["n_reservoir"])
    position = ladder.index(chosen)
    low = ladder[position - 1] if position > 0 else max(50, chosen // 2)
    high = ladder[position + 1] if position < len(ladder) - 1 else chosen * 2
    fine_values = sorted({int(round(v)) for v in np.linspace(low, high, 5)} - {chosen})
    stage3_points = [
        point(n_reservoir=n,
              target_pre_activation_std=float(best["target_pre_activation_std"]))
        for n in fine_values
    ]
    stage3 = evaluate_points(args.universe, stage3_points, config, artifacts, folds,
                             seeds, registry_path, args.dry_run, "fine")

    # --- report -----------------------------------------------------------
    table = pd.concat([stage1, stage2, stage3], ignore_index=True)
    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / f"grid_{args.universe}.csv"
    table.round(4).to_csv(csv_path, index=False)

    overall = table.loc[table["sharpe"].idxmax()]
    logger.info("=" * 72)
    logger.info("BEST: N_r=%d std=%.2f rho=%.2f alpha=%.2f tau=%.2f -> %+.3f (sd %.3f)",
                int(overall["n_reservoir"]), overall["target_pre_activation_std"],
                overall["spectral_radius"], overall["leak_rate"],
                overall["gate_temperature"], overall["sharpe"], overall["sharpe_sd"])
    logger.info("  %d configurations scored -> %s", len(table), csv_path)
    logger.info("  Each is a deflated-Sharpe trial; compare gains against the "
                "seed spread before believing them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
