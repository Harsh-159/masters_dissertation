"""What does exact cardinality cost? The rung-5 attribution diagnostic.

    python scripts/run_front_attribution.py --universe etf10
    python scripts/run_front_attribution.py --universe sector11

The question
------------
Rung 5 restricts the trained mixture's book to its top-``K_card`` conviction
names. Two schedules measure what that constraint costs, on ONE trained model,
ONE segment, through the SAME ledger and the SAME 5 bp cost function:

===  ==========================================  ==============================
 A   the mixture's own blend                     rung 2 behaviour, ~10 names
 B   that blend, restricted to the top-K_card    rung 5 - cardinality only
===  ==========================================  ==============================

``A -> B`` is the price of realistic execution. Measured on validation it came
out slightly **positive** on both universes (+0.909 vs +0.856 on etf10, +0.342
vs +0.332 on sector11): concentrating into the model's own top names costs
nothing and may help a little.

Historical note
---------------
A third schedule used to sit here - the plan's per-date NSGA-II front (R1),
which re-picked assets by an EWMA momentum estimate. On these same folds it
scored -0.325 / -0.231, roughly 1.2 Sharpe below plain cardinality, and it was
retired; ``rrc.optim.cardinality`` records why. This script is what measured it.

Quarantine
----------
A diagnostic, not a result of record: schedule A is rung 2's arm and schedule B
is rung 5's, so neither needs a new registry row here. Output is a CSV under
``results/runs/rung5/`` that regenerates from this one command.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.optim.cardinality import cardinality_schedule  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402
from run_rung5 import evaluate_schedule  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("attribution")


def attribute_universe(universe: str, config: dict, args) -> pd.DataFrame:
    """Run both schedules over the configured folds; return tidy rows."""
    logger.info("=== %s : cardinality attribution ===", universe)
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    states_dir = PROJECT_ROOT / config["paths"]["states_dir"]

    artifacts = load_universe_artifacts(processed_dir, universe)
    cached_states = find_cached_states(states_dir, universe)

    loss_cfg = config["loss"]
    k_card = int(config["k_card"])
    folds = [f for f in folds_for(universe) if f.name in args.folds.split(",")]
    rows: list[dict] = []

    for fold in folds:
        packs = build_fold_packs(artifacts, cached_states, fold)
        segment = packs[args.split]

        for seed in [int(s) for s in args.seeds.split(",")]:
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
            with torch.no_grad():
                blend = trained.model.weights_for_pack(segment).numpy()

            schedules = {
                "A_mixture_blend": blend,
                "B_cardinality": cardinality_schedule(blend, segment, artifacts, k_card),
            }

            # The evaluated calendar year, so the timeframe is never ambiguous.
            year = fold.val_year if args.split == "val" else fold.test_year
            for label, schedule in schedules.items():
                metrics = evaluate_schedule(schedule, segment)
                rows.append({
                    "universe": universe, "fold": fold.name, "split": args.split,
                    "year": year, "seed": seed, "schedule": label,
                    "excess_sharpe": metrics["excess_sharpe"],
                    "ann_turnover": metrics["ann_turnover"],
                    "holdings_count": metrics["holdings_count"],
                    "max_drawdown": metrics["max_drawdown"],
                    "ann_vol": metrics["ann_vol"],
                })
                logger.info(
                    "  %s %s (%d) seed %d  %-18s Sharpe %+7.3f  turn %6.2f  hold %5.2f",
                    universe, fold.name, year, seed, label,
                    metrics["excess_sharpe"], metrics["ann_turnover"],
                    metrics["holdings_count"],
                )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rung-5 cardinality attribution (A/B).")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rung5.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars.",
    )
    parser.add_argument("--folds", default="F01,F02,F03")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Which segment to attribute on. Defaults to val: this "
                             "is a diagnostic and should not consume test.")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    table = attribute_universe(args.universe, config, args)

    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / f"cardinality_attribution_{args.universe}.csv"
    table.round(6).to_csv(csv_path, index=False)

    logger.info("=============== ATTRIBUTION (mean over folds x seeds) ===============")
    years = sorted(table["year"].unique())
    logger.info("  %s  (%s years %s)", args.universe, args.split,
                ", ".join(str(y) for y in years))
    means = table.groupby("schedule")["excess_sharpe"].mean()
    for label in ("A_mixture_blend", "B_cardinality"):
        if label in means:
            logger.info("      %-18s %+.3f", label, means[label])
    if {"A_mixture_blend", "B_cardinality"} <= set(means.index):
        logger.info("      -> cardinality costs %+.3f",
                    means["B_cardinality"] - means["A_mixture_blend"])
    logger.info("  csv -> %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
