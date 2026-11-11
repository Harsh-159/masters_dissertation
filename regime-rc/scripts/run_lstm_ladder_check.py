"""The ladder's LSTM arm at its SEARCHED configuration, on the test split.

    python scripts/run_lstm_ladder_check.py --universe sector11

Why this exists
---------------
`run_lstm_search.py` searched the LSTM and found hidden 128 / lookback 20 on
both universes, but every number it reports is a *validation* Sharpe - its
`score()` stitches fold validation segments, and so does its held-out check.
The ladder (`run_walkforward.py`) reports concatenated *test* Sharpe, at the
pre-specified hidden 32 / lookback 60. The two are therefore not comparable,
and the searched configuration has never been scored on the metric the ladder
uses.

This script closes that gap and nothing else. It runs the LSTM block of
`run_walkforward.learned_arms` verbatim - same folds, same seeds, same
`train_mixture`, same `score_schedule`, same concatenation as `summarise` -
at two configurations:

* `ladder baseline`  hidden 32, lookback 60, lr 0.003  (reproduces the ladder
  row, and is the control: if it does not come back at the published value,
  this script is wrong, not the ladder)
* `search champion`  hidden 128, lookback 20, and the per-universe learning
  rate that won stage 2 (0.01 on sector11, 0.003 on etf10)

Reported as a sensitivity check on the LSTM row, exactly as HANDOFF section 8
proposed. It selects nothing: the ladder keeps the pre-specified arm.

No registry rows (headline walk-forward results write none), no holdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.costs import annualised_turnover  # noqa: E402
from rrc.backtest.metrics import excess_sharpe, max_drawdown  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.recurrent import LSTMPortfolioModel, attach_input_windows  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402

from run_comparison import score_schedule  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("lstm-ladder")

MAX_EPOCHS, PATIENCE = 150, 15          # identical in the ladder and the search

CONFIGS = {
    "sector11": [
        ("ladder baseline", dict(hidden_size=32, lookback=60, learning_rate=0.003)),
        ("search champion", dict(hidden_size=128, lookback=20, learning_rate=0.01)),
    ],
    "etf10": [
        ("ladder baseline", dict(hidden_size=32, lookback=60, learning_rate=0.003)),
        ("search champion", dict(hidden_size=128, lookback=20, learning_rate=0.003)),
    ],
}


def score_one(packs, artifacts, reservoir_panel, params, seed):
    """One (fold, seed) LSTM fit, scored on TEST - the ladder's code path."""
    windowed = attach_input_windows(
        packs, reservoir_panel, artifacts, int(params["lookback"])
    )
    result = train_mixture(
        windowed["train"], windowed["val"],
        MixtureTrainConfig(
            n_experts=1, lambda_cvar=0.0, lambda_balance=0.0,
            lambda_turn=0.0, lambda_div=0.0,
            learning_rate=float(params["learning_rate"]),
            max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=seed,
        ),
        model=lambda: LSTMPortfolioModel(
            n_inputs=reservoir_panel.shape[1],
            n_assets=packs["train"].logit_mask.shape[1],
            hidden_size=int(params["hidden_size"]),
        ),
    )
    with torch.no_grad():
        weights = result.model.weights_for_pack(windowed["test"])
    return score_schedule(weights.numpy(), windowed["test"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "comparison.yaml"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    seeds = [int(v) for v in args.seeds.split(",")]
    data_root = Path(args.data_root).expanduser().resolve()
    config = load_yaml(Path(args.config))
    processed = data_root / config["paths"]["processed_dir"]

    artifacts = load_universe_artifacts(processed, args.universe)
    states = find_cached_states(data_root / config["paths"]["states_dir"], args.universe)
    reservoir_panel = pd.read_parquet(processed / f"{args.universe}_reservoir_input.parquet")
    folds = folds_for(args.universe)

    logger.info("=== %s: %d folds x %d seeds x %d configs ===",
                args.universe, len(folds), len(seeds), len(CONFIGS[args.universe]))

    rows = []
    for label, params in CONFIGS[args.universe]:
        logger.info("--- %s: hidden %d, lookback %d, lr %g ---", label,
                    params["hidden_size"], params["lookback"], params["learning_rate"])
        per_fold = {seed: [] for seed in seeds}
        for fold in folds:
            packs = build_fold_packs(artifacts, states, fold)
            for seed in seeds:
                pieces = score_one(packs, artifacts, reservoir_panel, params, seed)
                per_fold[seed].append((fold.test_year, pieces))
            logger.info("  %s (test %d) done", fold.name, fold.test_year)

        # concatenation exactly as run_walkforward.summarise does it
        sharpes, turnovers, drawdowns = [], [], []
        for seed in seeds:
            ordered = sorted(per_fold[seed], key=lambda t: t[0])
            returns = np.concatenate([p["net_returns"] for _, p in ordered])
            risk_free = np.concatenate([p["risk_free"] for _, p in ordered])
            turnover = np.concatenate([p["turnover"] for _, p in ordered])
            sharpes.append(excess_sharpe(returns, risk_free))
            turnovers.append(annualised_turnover(turnover, n_days=len(returns)))
            drawdowns.append(max_drawdown(returns))
        rows.append({
            "universe": args.universe, "config": label, **params,
            "sharpe": float(np.mean(sharpes)),
            "sharpe_sd": float(np.std(sharpes, ddof=1)),
            "ann_turnover": float(np.mean(turnovers)),
            "max_drawdown": float(np.mean(drawdowns)),
            "per_seed": ";".join(f"{s:.4f}" for s in sharpes),
        })
        logger.info("  %s -> concatenated TEST Sharpe %.3f (sd %.3f)",
                    label, rows[-1]["sharpe"], rows[-1]["sharpe_sd"])

    out_dir = data_root / "results" / "runs" / "lstm_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / f"lstm_ladder_{args.universe}.csv", index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
