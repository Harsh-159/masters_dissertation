"""Hyperparameter search for the LSTM arm, on the protocol the reservoir got.

    python scripts/run_lstm_search.py --universe sector11
    python scripts/run_lstm_search.py --universe etf10 --seeds 0,1

Why this exists
---------------
Decision D2 claims a FROZEN temporal encoder beats a TRAINED one. As the ladder
stood, that claim rested on an unfair comparison:

    reservoir / model-side search trials .... 238
    LSTM search trials ......................   0

The LSTM ran at one hand-picked configuration (hidden 32, lookback 60) while the
frozen-encoder side had 238 configurations evaluated. "The frozen encoder wins"
is not a claim that survives that asymmetry, and an examiner will say so.

This gives the LSTM a search on the SAME protocol the reservoir got: same folds,
same seeds, the same concatenated validation metric, and the same held-out check
with the same pre-registered adoption rule.

What is searched, and why these axes
------------------------------------
Two stages, mirroring ``run_grid_search``'s coarse-to-fine structure.

**Stage 1** grids the two axes that carry the actual criticism:

* ``hidden_size`` - capacity. The original 32 was chosen by judgement, on the
  argument that a fold holds only a few hundred weekly decisions. That argument
  may be right, but it was never tested.
* ``lookback`` - the memory horizon. This is the sharper one. The reservoir's
  measured memory capacity at operating settings is 19.53, with 69% of it inside
  60 days - so a 60-day window is not obviously a handicap. But it is a hard
  truncation where the reservoir has a decaying tail (recoverability 0.112 at
  60 days, 0.078 at 90, 0.052 at 120), and 250 days is the common default in
  financial sequence models. Searching it settles the question with a number.

**Stage 2** varies ``num_layers``, ``dropout`` and the learning rate ONE AT A
TIME at the stage-1 winner - the same "check the flat parameters rather than
assume them" discipline the reservoir grid used.

The adoption rule, stated before the run
----------------------------------------
The stage-2 winner is re-scored on folds no search has touched (F13-F18 on
sector11, F06-F10 on etf10). It is adopted only if it beats the current default
configuration there. Five reservoir-side selections have now been made on
validation and checked this way, and none transferred. If the LSTM's does not
either, the finding generalises from "reservoir hyperparameters do not transfer"
to "hyperparameter selection does not transfer for either encoder at this data
size" - which is a stronger and more interesting result than the search finding
a winner would be.

Cost note: 16 stage-1 cells at 2 seeds is deliberately a SCREEN, matching the
project's own round-1 convention. Resolution is bought back with more seeds at
the winner, not with a finer mesh.
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
from rrc.models.recurrent import LSTMPortfolioModel, attach_input_windows  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture  # noqa: E402
from run_joint_search import FOLD_SETS  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("lstm-search")

# The configuration the ladder has been running at - the thing to beat.
BASELINE = {
    "hidden_size": 32,
    "lookback": 60,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.003,
}

STAGE1_HIDDEN = [16, 32, 64, 128]
STAGE1_LOOKBACK = [20, 60, 120, 250]

# Stage 2: one-at-a-time variations at the stage-1 winner.
STAGE2_VARIATIONS = [
    ("num_layers", 2),
    ("dropout", 0.2),
    ("learning_rate", 0.001),
    ("learning_rate", 0.01),
]

MAX_EPOCHS = 150
PATIENCE = 15


def score(universe, params, artifacts, states, reservoir_panel, folds, seeds):
    """Concatenated validation Sharpe for one LSTM configuration.

    Identical aggregation to every other search in this project: per seed, the
    fold validation segments are stitched into one continuous daily stream and
    scored once; the mean and spread across seeds are returned.
    """
    per_seed = []
    for seed in seeds:
        streams, risk_frees = [], []
        for fold in folds:
            packs = build_fold_packs(artifacts, states, fold)
            windowed = attach_input_windows(
                packs, reservoir_panel, artifacts, int(params["lookback"])
            )
            result = train_mixture(
                windowed["train"], windowed["val"],
                MixtureTrainConfig(
                    n_experts=1,
                    lambda_cvar=0.0, lambda_balance=0.0,
                    lambda_turn=0.0, lambda_div=0.0,
                    learning_rate=float(params["learning_rate"]),
                    max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=seed,
                ),
                model=lambda: LSTMPortfolioModel(
                    n_inputs=reservoir_panel.shape[1],
                    n_assets=packs["train"].logit_mask.shape[1],
                    hidden_size=int(params["hidden_size"]),
                    num_layers=int(params["num_layers"]),
                    dropout=float(params["dropout"]),
                ),
            )
            evaluation = evaluate_on_pack(result.model, windowed["val"])
            streams.append(evaluation["net_returns"].to_numpy())
            risk_frees.append(windowed["val"].risk_free.numpy())
        per_seed.append(excess_sharpe(np.concatenate(streams), np.concatenate(risk_frees)))

    spread = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    return float(np.mean(per_seed)), spread


def register(universe, params, sharpe, stage, registry_path, dry_run):
    if dry_run:
        return
    identity = {"universe": universe, "rung": "lstm-search", "stage": stage, **params}
    append_row(
        RegistryRow(
            universe=universe, rung="lstm", arm="lstm_search", seed=0,
            fold=stage, split="val", n_params=0, cfg_hash=config_hash(identity),
            excess_sharpe=sharpe,
            note=(f"lstm {stage}; hidden={params['hidden_size']} "
                  f"lookback={params['lookback']} layers={params['num_layers']} "
                  f"dropout={params['dropout']} lr={params['learning_rate']}"),
        ),
        registry_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSTM hyperparameter search.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--seeds", default="0,1", help="Screening seeds for stage 1.")
    parser.add_argument("--verify-seeds", default="0,1,2", help="Seeds for the held-out check.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir = data_root / "data" / "processed"
    registry_path = data_root / "results" / "registry.csv"
    seeds = [int(v) for v in args.seeds.split(",")]
    verify_seeds = [int(v) for v in args.verify_seeds.split(",")]

    artifacts = load_universe_artifacts(processed_dir, args.universe)
    states = find_cached_states(data_root / "data" / "states", args.universe)
    reservoir_panel = pd.read_parquet(
        processed_dir / f"{args.universe}_reservoir_input.parquet"
    )
    all_folds = {f.name: f for f in folds_for(args.universe)}
    selection = [all_folds[n] for n in FOLD_SETS[args.universe]["selection"]]
    held_out = [all_folds[n] for n in FOLD_SETS[args.universe]["held_out"]]

    def run(params, folds, seed_list):
        return score(args.universe, params, artifacts, states,
                     reservoir_panel, folds, seed_list)

    # ---- stage 1: capacity x memory horizon -------------------------------
    logger.info("=== stage 1: hidden_size x lookback, %d cells, %d folds x %d seeds ===",
                len(STAGE1_HIDDEN) * len(STAGE1_LOOKBACK), len(selection), len(seeds))
    rows = []
    for hidden, lookback in itertools.product(STAGE1_HIDDEN, STAGE1_LOOKBACK):
        params = {**BASELINE, "hidden_size": hidden, "lookback": lookback}
        sharpe, spread = run(params, selection, seeds)
        rows.append({"hidden_size": hidden, "lookback": lookback,
                     "sharpe": sharpe, "sharpe_sd": spread, "stage": "coarse"})
        logger.info("  hidden=%3d lookback=%3d  ->  %+.3f (sd %.3f)",
                    hidden, lookback, sharpe, spread)
        register(args.universe, params, sharpe, "coarse", registry_path, args.dry_run)

    frame = pd.DataFrame(rows)
    table = frame.pivot(index="hidden_size", columns="lookback", values="sharpe")
    logger.info("")
    logger.info("=== stage 1 grid (concatenated validation Sharpe) ===")
    for line in table.round(3).to_string().splitlines():
        logger.info("  %s", line)

    best = frame.loc[frame["sharpe"].idxmax()]
    winner = {**BASELINE, "hidden_size": int(best["hidden_size"]),
              "lookback": int(best["lookback"])}
    logger.info("")
    logger.info("Stage-1 winner: hidden=%d lookback=%d at %+.3f",
                winner["hidden_size"], winner["lookback"], best["sharpe"])

    # ---- stage 2: one-at-a-time at the winner -----------------------------
    logger.info("")
    logger.info("=== stage 2: layers / dropout / learning rate at the winner ===")
    stage2 = [{**winner, "label": "stage-1 winner", "sharpe": float(best["sharpe"])}]
    for key, value in STAGE2_VARIATIONS:
        params = {**winner, key: value}
        sharpe, spread = run(params, selection, seeds)
        stage2.append({**params, "label": f"{key}={value}", "sharpe": sharpe})
        logger.info("  %-18s ->  %+.3f (sd %.3f)", f"{key}={value}", sharpe, spread)
        register(args.universe, params, sharpe, "refine", registry_path, args.dry_run)

    stage2_frame = pd.DataFrame(stage2)
    champion_row = stage2_frame.loc[stage2_frame["sharpe"].idxmax()]
    champion = {k: champion_row[k] for k in BASELINE}
    logger.info("")
    logger.info("Search champion: %s at %+.3f", champion_row["label"], champion_row["sharpe"])

    # ---- held-out check, per the rule stated in the docstring --------------
    logger.info("")
    logger.info("=== held-out %s-%s (never used in any selection), %d seeds ===",
                held_out[0].name, held_out[-1].name, len(verify_seeds))
    champion_held, champion_sd = run(champion, held_out, verify_seeds)
    baseline_held, baseline_sd = run(BASELINE, held_out, verify_seeds)
    logger.info("  search champion  hidden=%d lookback=%d layers=%d dropout=%.1f lr=%.4f"
                "  ->  %+.3f (sd %.3f)",
                int(champion["hidden_size"]), int(champion["lookback"]),
                int(champion["num_layers"]), float(champion["dropout"]),
                float(champion["learning_rate"]), champion_held, champion_sd)
    logger.info("  ladder baseline  hidden=32  lookback=60  layers=1 dropout=0.0 lr=0.0030"
                "  ->  %+.3f (sd %.3f)", baseline_held, baseline_sd)

    logger.info("")
    logger.info("=== verdict ===")
    delta = champion_held - baseline_held
    noise = max(champion_sd, baseline_sd)
    logger.info("  champion minus ladder baseline, held out: %+.3f (seed spread up to %.3f)",
                delta, noise)
    if delta <= noise:
        logger.info("  -> the LSTM search does NOT transfer either. The D2 comparison")
        logger.info("     stands, and the transfer failure now covers BOTH encoders.")
    else:
        logger.info("  -> the LSTM search DOES transfer. Re-run the ladder at this")
        logger.info("     configuration before quoting any D2 conclusion.")

    out_dir = data_root / "results" / "runs" / "lstm_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"stage1_{args.universe}.csv", index=False)
    stage2_frame.to_csv(out_dir / f"stage2_{args.universe}.csv", index=False)
    pd.DataFrame([
        {"config": "search champion", **champion, "held_out": champion_held, "sd": champion_sd},
        {"config": "ladder baseline", **BASELINE, "held_out": baseline_held, "sd": baseline_sd},
    ]).to_csv(out_dir / f"heldout_{args.universe}.csv", index=False)
    logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
