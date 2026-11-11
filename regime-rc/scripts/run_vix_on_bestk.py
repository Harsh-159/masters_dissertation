"""Does adding VIX to the gate help the best-capacity config on each universe?

    python scripts/run_vix_on_bestk.py --universe sector11 --k 2 --n-reservoir 300 --spectral-radius 0.85
    python scripts/run_vix_on_bestk.py --universe etf10    --k 5 --n-reservoir 600 --spectral-radius 0.95

The capacity curve (run_capacity_curve.py) found the best out-of-sample K per
universe - K = 2 on sector11 (test 0.505) and K = 5 on etf10 (test 0.411) - each
at its own searched reservoir. This script holds that exact (K, N_r, rho) fixed
and asks one thing: does letting the GATE also read the five market signals
(VIX + macro; the 3a routing, experts unchanged) move the concatenated test
Sharpe?

Both arms are scored in the SAME run, on the SAME folds/seeds/reservoir, using
run_capacity_curve's exact aggregation (per seed, stitch every fold's test
segment into one continuous daily stream, one excess Sharpe; then mean +- sd
across seeds). Reproducing the no-VIX number here is the control: if it lands on
the published reference, the VIX delta beside it is trustworthy.

Honest scope
------------
* This is NOT a re-search. The reservoir is frozen at the capacity-curve winner
  for the no-VIX gate; a fully fair test would re-search it for the signal gate
  too. This answers the narrower question actually asked - what VIX does to the
  best-K config as it stands.
* Writes results/runs/vix_bestk/ only; it appends NO registry rows and touches
  NO holdout data (all measurement is on the test split of the walk-forward).
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
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.gating import MixtureOfExperts, SignalRoutedMixture  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.hyper_search import build_states_for_config  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("vix-bestk")

FIXED = {"washout": 250, "bias_scale": 0.0, "calibration_fraction": 0.6,
         "max_iterations": 10, "tolerance": 0.02}
ANCHOR = {"leak_rate": 0.30, "target_pre_activation_std": 0.70, "gate_temperature": 1.0}


def train_config(k: int, seed: int) -> MixtureTrainConfig:
    """Identical to run_capacity_curve.train_config - lambda_balance inert at K=1."""
    return MixtureTrainConfig(
        n_experts=k, lambda_cvar=0.0,
        lambda_balance=0.01 if k > 1 else 0.0,
        lambda_turn=0.0, lambda_div=0.0, seed=seed,
    )


def factory_for(arm: str, n_state: int, n_assets: int, n_signals: int, k: int):
    if arm == "state":            # the capacity-curve gate: state only
        return lambda: MixtureOfExperts(n_state, n_assets, k)
    return lambda: SignalRoutedMixture(   # the 3a gate: [state ; signals] -> gate
        n_state, n_assets, k, n_signals,
        gate_temperature=1.0, signals_to_gate=True, signals_to_experts=False,
    )


def score(arm, packs_by_fold, folds, k, seeds):
    """Concatenated test Sharpe: mean and seed spread (capacity-curve aggregation)."""
    per_seed = []
    for seed in seeds:
        streams, risk_frees = [], []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            n_state = packs["train"].states.shape[1]
            n_assets = packs["train"].logit_mask.shape[1]
            n_signals = packs["train"].signals.shape[1]
            result = train_mixture(
                packs["train"], packs["val"], train_config(k, seed),
                model=factory_for(arm, n_state, n_assets, n_signals, k),
            )
            evaluation = evaluate_on_pack(result.model, packs["test"])
            streams.append(evaluation["net_returns"].to_numpy())
            risk_frees.append(packs["test"].risk_free.numpy())
        per_seed.append(excess_sharpe(np.concatenate(streams), np.concatenate(risk_frees)))
    return per_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VIX-in-gate on the best-K config.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n-reservoir", type=int, required=True)
    parser.add_argument("--spectral-radius", type=float, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir = data_root / "data" / "processed"
    states_dir = data_root / "data" / "states"
    seeds = [int(s) for s in args.seeds.split(",")]

    artifacts = load_universe_artifacts(processed_dir, args.universe)
    folds = list(folds_for(args.universe))

    params = {**ANCHOR, "n_reservoir": args.n_reservoir, "spectral_radius": args.spectral_radius}
    states = build_states_for_config(args.universe, params, FIXED, processed_dir, states_dir)
    packs_by_fold = {f.name: build_fold_packs(artifacts, states, f) for f in folds}

    logger.info("=== %s | K=%d | N_r=%d rho=%.2f | %d folds x %d seeds (TEST) ===",
                args.universe, args.k, args.n_reservoir, args.spectral_radius,
                len(folds), len(seeds))

    results = {}
    for arm, label in (("state", "gate = state only  (capacity-curve control)"),
                       ("signal", "gate = state + VIX/signals  (3a)")):
        per_seed = score(arm, packs_by_fold, folds, args.k, seeds)
        results[arm] = per_seed
        mean5 = float(np.mean(per_seed))
        sd5 = float(np.std(per_seed, ddof=1))
        mean3 = float(np.mean(per_seed[:3]))
        logger.info("  %-46s test Sharpe %.3f +- %.3f  (%d seeds) | %.3f (first 3 seeds)",
                    label, mean5, sd5, len(seeds), mean3)

    delta = float(np.mean(results["signal"]) - np.mean(results["state"]))
    paired_sd = float(np.std(np.array(results["signal"]) - np.array(results["state"]), ddof=1))
    logger.info("  --------------------------------------------------------------")
    logger.info("  VIX effect (signal - state): %+0.3f   paired sd across seeds %.3f", delta, paired_sd)

    out_dir = data_root / "results" / "runs" / "vix_bestk"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"seed": seeds, "state": results["state"], "signal": results["signal"]})
    frame["delta"] = frame["signal"] - frame["state"]
    path = out_dir / f"vix_bestk_{args.universe}_K{args.k}.csv"
    frame.to_csv(path, index=False)
    logger.info("  wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
