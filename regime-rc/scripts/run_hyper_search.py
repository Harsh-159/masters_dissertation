"""P8 outer loop: random search over reservoir hyperparameters, then freeze one.

    python scripts/run_hyper_search.py --universe etf10
    python scripts/run_hyper_search.py --universe etf10
    python scripts/run_hyper_search.py --n-random 10   # a quick smaller search

What it does
------------
1. Draws ``n_random`` reservoir configurations (random search - the R1 fallback,
   the better choice here; see hyper_search's docstring).
2. Scores each by training the mixture on a small fold subset and reading
   validation Sharpe and CVaR.
3. Writes ONE ``split == "val"`` registry row per configuration, so the
   deflated-Sharpe trial count honestly reflects every config tried.
4. Freezes the single winner (max validation Sharpe on the non-dominated front)
   to ``configs/frozen/<universe>.yaml``.

This script does not tag ``config-freeze`` - do that by hand once the frozen
configs are reviewed, so the freeze is a deliberate act.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.optim.hyper_search import non_dominated, random_search, select_frozen  # noqa: E402
from rrc.utils.io import atomic_write_text, load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("hyper")


def candidate_identity(universe: str, candidate, evaluation: dict) -> dict:
    """The registry identity dict for one search candidate (its cfg_hash source)."""
    return {
        "universe": universe,
        "rung": "hyper",
        "arm": "random_search",
        "reservoir_cfg_hash": candidate.reservoir_cfg_hash,
        "gate_temperature": round(candidate.params["gate_temperature"], 6),
        "loss": {k: evaluation[k] for k in ("lambda_cvar", "lambda_balance", "lambda_turn")},
    }


def search_universe(universe: str, config: dict, registry_path: Path) -> dict:
    """Run the search for one universe; write registry rows and the frozen config."""
    logger.info("=== %s : random hyperparameter search ===", universe)
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    cache_dir = PROJECT_ROOT / config["paths"]["states_dir"]
    evaluation = config["evaluation"]

    candidates = random_search(
        universe=universe,
        ranges=config["ranges"],
        fixed=config["fixed"],
        processed_dir=processed_dir,
        cache_dir=cache_dir,
        fold_names=list(evaluation["fold_names"]),
        n_random=int(config["search"]["n_random"]),
        search_seed=int(config["search"]["search_seed"]),
        eval_seeds=[int(s) for s in config["search"]["eval_seeds"]],
        n_experts=int(evaluation["n_experts"]),
        lambda_cvar=float(evaluation["lambda_cvar"]),
        lambda_balance=float(evaluation["lambda_balance"]),
        lambda_turn=float(evaluation["lambda_turn"]),
        max_epochs=int(evaluation["max_epochs"]),
        patience=int(evaluation["patience"]),
        lambda_div=float(evaluation.get("lambda_div", 0.0)),
    )

    # One validation row per distinct configuration tried - the DSR population.
    for candidate in candidates:
        append_row(
            RegistryRow(
                universe=universe, rung="hyper", arm="random_search",
                seed=int(config["search"]["eval_seeds"][0]), fold="search", split="val",
                n_params=0, cfg_hash=config_hash(candidate_identity(universe, candidate, evaluation)),
                excess_sharpe=candidate.val_sharpe, cvar_95=candidate.val_cvar,
                note=(
                    f"random-search candidate; rho={candidate.params['spectral_radius']:.3f} "
                    f"alpha={candidate.params['leak_rate']:.3f} "
                    f"target_std={candidate.params['target_pre_activation_std']:.2f} "
                    f"tau_g={candidate.params['gate_temperature']:.2f} "
                    f"N_r={candidate.params['n_reservoir']}"
                ),
            ),
            registry_path,
        )

    frozen = select_frozen(candidates)
    front = non_dominated(candidates)
    logger.info(
        "  frozen: rho=%.3f alpha=%.3f target_std=%.2f tau_g=%.2f N_r=%d "
        "(val Sharpe %.3f, CVaR %.4f; %d on the non-dominated front)",
        frozen.params["spectral_radius"], frozen.params["leak_rate"],
        frozen.params["target_pre_activation_std"], frozen.params["gate_temperature"],
        frozen.params["n_reservoir"], frozen.val_sharpe, frozen.val_cvar, len(front),
    )

    # Persist the frozen reservoir configuration for P9 to load.
    frozen_dir = PROJECT_ROOT / config["paths"]["frozen_dir"]
    frozen_config = {
        "universe": universe,
        "reservoir": {
            "n_reservoir": frozen.params["n_reservoir"],
            "spectral_radius": frozen.params["spectral_radius"],
            "leak_rate": frozen.params["leak_rate"],
            "bias_scale": config["fixed"]["bias_scale"],
            "washout": config["fixed"]["washout"],
        },
        "input_scaling": {
            "target_pre_activation_std": frozen.params["target_pre_activation_std"],
            "calibration_fraction": config["fixed"]["calibration_fraction"],
            "max_iterations": config["fixed"]["max_iterations"],
            "tolerance": config["fixed"]["tolerance"],
        },
        "gate_temperature": frozen.params["gate_temperature"],
        "reservoir_cfg_hash": frozen.reservoir_cfg_hash,
        "val_sharpe": frozen.val_sharpe,
        "val_cvar": frozen.val_cvar,
    }
    atomic_write_text(frozen_dir / f"{universe}.yaml", yaml.safe_dump(frozen_config, sort_keys=False))
    logger.info("  wrote %s", frozen_dir / f"{universe}.yaml")

    return {
        "n_candidates": len(candidates),
        "n_non_dominated": len(front),
        "frozen": frozen_config,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Outer reservoir-hyperparameter search.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "hyper.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars "
             "(etf10 from 2006, sector11 from 1999), so they are run and "
             "reported independently and never aggregated.",
    )
    parser.add_argument("--n-random", type=int, default=None, help="Override n_random.")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    if args.n_random is not None:
        config["search"]["n_random"] = args.n_random

    registry_path = PROJECT_ROOT / "results" / "registry.csv"

    report = {"universes": {
        args.universe: search_universe(args.universe, config, registry_path)
    }}

    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "hyper_report.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("=== hyperparameter search complete; frozen configs in %s ===",
                config["paths"]["frozen_dir"])
    logger.info("Review the frozen configs, then `git tag config-freeze` by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
