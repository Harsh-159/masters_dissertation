"""Rung 5: exact cardinality on the trained mixture's book.

    python scripts/run_rung5.py --universe etf10
    python scripts/run_rung5.py --universe etf10 --dry-run    # write no registry rows

What this rung is
-----------------
A softmax head spreads a sliver of weight across every asset; a book that is
actually traded holds a handful of names. Rung 5 imposes that: each rebalance
date, keep the ``K_card`` assets the trained mixture likes most, renormalise,
and run the result through the SAME ledger and the SAME 5 bp cost function as
every other rung. Nothing is retrained - this is an execution layer, not a new
model.

Why there is no NSGA-II front here any more
--------------------------------------------
The plan (R1) specified a per-date NSGA-II front over a return-versus-CVaR
objective. It was built, measured, and dropped, for two reasons recorded in
``rrc.optim.cardinality``:

* choosing 5 of 10 assets is 252 combinations - too small to warrant a
  population search, and convex once the subset is fixed;
* its ``mu_hat`` objective (an EWMA momentum forecast) overrode the trained
  system's own view, which is exactly the predict-then-optimise pattern this
  project argues against. On identical folds it scored -0.325 / -0.231 against
  +0.909 / +0.342 for plain cardinality.

The comparison that retired it regenerates from ``run_front_attribution.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.ledger import run_ledger  # noqa: E402
from rrc.backtest.metrics import summarise  # noqa: E402
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.optim.cardinality import cardinality_schedule  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("rung5")


def evaluate_schedule(executed: np.ndarray, pack) -> dict:
    """Run an executed weight schedule through the ledger and summarise it.

    The ledger charges the project's standard 5 bp per unit of notional traded
    (A9), so every metric here is net of realistic transaction costs.
    """
    result = run_ledger(
        targets=torch.as_tensor(executed, dtype=torch.float64),
        exec_days=pack.execution_positions,
        simple_returns=pack.returns,
        cash_index=pack.cash_index,
        return_diagnostics=True,
    )
    return summarise(
        result.net_returns.numpy(),
        pack.risk_free.numpy(),
        turnover=result.turnover.numpy(),
        held_weights=result.held_weights.numpy(),
        cash_index=pack.cash_index,
    )


def run_universe(universe: str, config: dict, registry_path: Path, dry_run: bool = False) -> None:
    """Train the mixture, restrict it to K_card names, and evaluate."""
    logger.info("=== %s : rung 5 (exact cardinality) ===", universe)
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    states_dir = PROJECT_ROOT / config["paths"]["states_dir"]

    artifacts = load_universe_artifacts(processed_dir, universe)
    cached_states = find_cached_states(states_dir, universe)

    loss_cfg = config["loss"]
    k_card = int(config["k_card"])
    folds = [f for f in folds_for(universe) if f.name in config["folds"]]
    seeds = [int(s) for s in config["seeds"]]

    identity = {
        "universe": universe, "rung": "5", "arm": config["arm"],
        "k_card": k_card,
        "loss": {k: loss_cfg[k] for k in sorted(loss_cfg)},
        "reservoir_cfg_hash": cached_states.cfg_hash,
    }
    arm_hash = config_hash(identity)

    for fold in folds:
        packs = build_fold_packs(artifacts, cached_states, fold)
        for seed in seeds:
            trained = train_mixture(
                packs["train"], packs["val"],
                MixtureTrainConfig(
                    n_experts=3,
                    lambda_cvar=float(loss_cfg["lambda_cvar"]),
                    lambda_balance=float(loss_cfg["lambda_balance"]),
                    lambda_turn=float(loss_cfg["lambda_turn"]),
                    seed=seed,
                ),
            )

            sharpe_by_split = {}
            for split in ("val", "test"):
                pack = packs[split]
                with torch.no_grad():
                    blend = trained.model.weights_for_pack(pack).numpy()
                executed = cardinality_schedule(blend, pack, artifacts, k_card)
                metrics = evaluate_schedule(executed, pack)
                sharpe_by_split[split] = metrics["excess_sharpe"]

                if not dry_run:
                    append_row(
                        RegistryRow(
                            universe=universe, rung="5", arm=config["arm"],
                            seed=seed, fold=fold.name, split=split,
                            n_params=0, cfg_hash=arm_hash,
                            note=(
                                f"exact cardinality K_card={k_card}; "
                                f"lambda_turn={loss_cfg['lambda_turn']}; "
                                "no front (NSGA-II retired, see cardinality.py)"
                            ),
                            **metrics,
                        ),
                        registry_path,
                    )
            logger.info(
                "  %s seed %d: val Sharpe %+.3f, test Sharpe %+.3f",
                fold.name, seed, sharpe_by_split["val"], sharpe_by_split["test"],
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rung 5 - exact cardinality.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rung5.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars "
             "(etf10 from 2006, sector11 from 1999), so they are run and "
             "reported independently and never aggregated.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the pipeline but write nothing to the registry (for smoke tests).",
    )
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    registry_path = PROJECT_ROOT / "results" / "registry.csv"
    run_universe(args.universe, config, registry_path, dry_run=args.dry_run)

    logger.info("=== rung 5 complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
