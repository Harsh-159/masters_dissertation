"""The single holdout touch: frozen config (primary) + baselines (+ best-found secondary).

    python scripts/run_holdout.py --universe sector11
    python scripts/run_holdout.py --universe etf10

This evaluates on the firewalled holdout 2025-01-01 -> 2026-06-30 (~373 trading
days), which every other script removes at load time. It is meant to be run ONCE.

What runs, and why
------------------
* PRIMARY - the FROZEN configuration (configs/frozen/*.yaml): the pre-specified
  Appendix-B defaults, sector11 K=1 / etf10 K=3, on the default frozen
  reservoir. This is the only credible holdout number: it tests a configuration
  committed BEFORE the holdout was seen. Plus the four rung-0 baselines on the
  identical window and the identical ledger.
* SECONDARY (labelled) - the "best-found" search winner per universe
  (sector11 K=2 @ N_r 300, rho 0.85; etf10 K=5 @ N_r 600, rho 0.95). Reported
  as an explicit, pre-declared sensitivity ONLY. It carries selection bias by
  construction and must never be quoted as the headline - this session already
  showed these winners do not transfer out of sample.

How the holdout is reached SAFELY
---------------------------------
The frozen reservoir's state cache already covers 1999/2006 -> 2026-06-30 (the
reservoir is a frozen, causal feature map; its states were computed over the full
series and its input scale was calibrated on the first 60% of the sample, which
ends ~2015 - so nothing here saw the holdout during fitting). We therefore load
artifacts WITHOUT truncation and build one extra "next fold":

    train : universe start -> 2023-12-31 (embargoed, usable-intersected)
    val   : 2024                         (early stopping only)
    HOLD  : 2025-01-01 -> 2026-06-30     (evaluated; never seen in training)

Hard assertions below prove train and val end strictly before the holdout, and
that the holdout window sits exactly inside [HOLDOUT_START, HOLDOUT_END]. If any
fails, nothing is scored.

Scope / honesty
---------------
Writes results/runs/holdout/ only. Appends NO registry rows and makes NO git
commit - the official audit-trail write (split="holdout") belongs to the frozen-
and-committed run. The numbers are nonetheless the real holdout numbers: once
seen, the holdout is spent.
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
from rrc.eval.slicing import (  # noqa: E402
    HOLDOUT_END, HOLDOUT_START, Fold, assert_no_holdout, segment_dates, train_dates,
)
from rrc.models.gating import MixtureOfExperts  # noqa: E402
from rrc.optim.fold_data import (  # noqa: E402
    CASH_COLUMN, UniverseArtifacts, build_segment_pack,
)
from rrc.optim.hyper_search import build_states_for_config  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402
from run_comparison import score_baselines, score_schedule  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("holdout")

# Frozen (primary) K per universe, from configs/frozen/*.yaml.
FROZEN_K = {"sector11": 1, "etf10": 3}
# The "best-found" search winner per universe (capacity curve), reported as a
# labelled secondary only. (N_r, rho, K).
BEST_FOUND = {"sector11": dict(k=2, n_reservoir=300, spectral_radius=0.85),
              "etf10": dict(k=5, n_reservoir=600, spectral_radius=0.95)}
ANCHOR = {"leak_rate": 0.30, "target_pre_activation_std": 0.70, "gate_temperature": 1.0}
FIXED = {"washout": 250, "bias_scale": 0.0, "calibration_fraction": 0.6,
         "max_iterations": 10, "tolerance": 0.02}
SEEDS = [0, 1, 2, 3, 4]


def load_full_artifacts(processed_dir: Path, universe: str) -> UniverseArtifacts:
    """load_universe_artifacts WITHOUT the holdout truncation - the only place in
    the project that deliberately keeps 2025+ rows. Every other check (the
    return-hole guarantee) is preserved verbatim."""
    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(processed_dir / f"{universe}_{name}.parquet")

    returns_simple = read("returns_simple")
    availability = read("availability")
    risk_free = read("risk_free")["r_f"]
    signals = read("signals")

    risky_columns = [c for c in returns_simple.columns if c != CASH_COLUMN]
    available = availability[risky_columns].to_numpy() == 1
    available_yesterday = np.vstack([available[:1] & False, available[:-1]])
    finite = np.isfinite(returns_simple[risky_columns].to_numpy())
    holes = available & available_yesterday & ~finite
    if holes.any():
        raise ValueError(f"{universe}: {int(holes.sum())} return hole(s) inside an "
                         "asset's active life; the data layer's guarantee is broken.")

    return UniverseArtifacts(
        universe=universe, returns_simple=returns_simple,
        ledger_panel=returns_simple.fillna(0.0), availability=availability[risky_columns],
        risk_free=risk_free, signals=signals, dates=returns_simple.index,
        columns=list(returns_simple.columns),
        cash_index=list(returns_simple.columns).index(CASH_COLUMN),
    )


def build_holdout_packs(artifacts, cached_states, universe):
    """train (-> 2023) / val (2024) / HOLDOUT (2025-01-01 -> 2026-06-30) packs.

    The whole point is the safety proof: train and val must end strictly before
    the holdout, and the holdout window must sit exactly inside the firewall.
    """
    full_dates = artifacts.dates
    usable_dates = full_dates[cached_states.usable[: len(full_dates)]]
    train_start = pd.Timestamp("1999-01-01") if universe == "sector11" else pd.Timestamp("2006-02-01")

    # "The fold after the last one": val_year 2024, its test window is the holdout.
    holdout_fold = Fold(name="HOLDOUT", train_start=train_start, val_year=2024, test_year=2025)
    train_window = train_dates(full_dates, holdout_fold).intersection(usable_dates)
    val_window = segment_dates(full_dates, 2024)
    holdout_window = full_dates[(full_dates >= HOLDOUT_START) & (full_dates <= HOLDOUT_END)]

    # --- hard safety proofs; if any fails, nothing is scored ---------------
    assert_no_holdout(train_window, "holdout-run TRAIN")   # train must not reach 2025
    assert_no_holdout(val_window, "holdout-run VAL")       # val (2024) must be < 2025
    if not (holdout_window.min() >= HOLDOUT_START and holdout_window.max() <= HOLDOUT_END):
        raise RuntimeError("holdout window escaped the firewall bounds.")
    if train_window.max() >= HOLDOUT_START or val_window.max() >= HOLDOUT_START:
        raise RuntimeError("train/val crossed the holdout boundary.")
    if val_window.max() >= holdout_window.min():
        raise RuntimeError("val overlaps the holdout.")

    logger.info("  train : %s -> %s  (%d days)", train_window.min().date(),
                train_window.max().date(), len(train_window))
    logger.info("  val   : %s -> %s  (%d days, early stopping only)",
                val_window.min().date(), val_window.max().date(), len(val_window))
    logger.info("  HOLD  : %s -> %s  (%d days)  <-- evaluated once",
                holdout_window.min().date(), holdout_window.max().date(), len(holdout_window))

    return {
        "train": build_segment_pack("train", train_window, artifacts, cached_states),
        "val": build_segment_pack("val", val_window, artifacts, cached_states),
        "test": build_segment_pack("test", holdout_window, artifacts, cached_states),
    }


def metrics_from_pieces(pieces: dict) -> dict:
    r = np.asarray(pieces["net_returns"])
    return {"sharpe": excess_sharpe(r, np.asarray(pieces["risk_free"])),
            "max_drawdown": max_drawdown(r),
            "ann_turnover": annualised_turnover(np.asarray(pieces["turnover"]), n_days=len(r))}


def score_learned(packs, k, seeds):
    """Train the frozen/best-found mixture on train (early-stop on val), score on holdout."""
    n_state = packs["train"].states.shape[1]
    n_assets = packs["train"].logit_mask.shape[1]
    per_seed = []
    for seed in seeds:
        result = train_mixture(
            packs["train"], packs["val"],
            MixtureTrainConfig(n_experts=k, lambda_cvar=0.0,
                               lambda_balance=0.01 if k > 1 else 0.0,
                               lambda_turn=0.0, lambda_div=0.0, seed=seed),
            model=lambda: MixtureOfExperts(n_state, n_assets, k),
        )
        with torch.no_grad():
            weights = result.model.weights_for_pack(packs["test"])
        per_seed.append(metrics_from_pieces(score_schedule(weights.numpy(), packs["test"])))
    return per_seed


def aggregate(name, kind, per_seed):
    sh = [d["sharpe"] for d in per_seed]
    return {"arm": name, "kind": kind, "sharpe": float(np.mean(sh)),
            "sharpe_sd": float(np.std(sh, ddof=1)) if len(sh) > 1 else 0.0,
            "max_drawdown": float(np.mean([d["max_drawdown"] for d in per_seed])),
            "ann_turnover": float(np.mean([d["ann_turnover"] for d in per_seed])),
            "n_seeds": len(per_seed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The single holdout touch.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    parser.add_argument("--states-dir", default="", help="Scratch dir for the best-found reservoir.")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir = data_root / "data" / "processed"
    states_dir = data_root / "data" / "states"
    scratch_states = Path(args.states_dir).expanduser().resolve() if args.states_dir else states_dir
    raw_dir = data_root / "data" / "raw"
    config = load_yaml(data_root / "configs" / "comparison.yaml")

    logger.info("=== HOLDOUT TOUCH: %s | window %s -> %s ===",
                args.universe, HOLDOUT_START.date(), HOLDOUT_END.date())
    artifacts = load_full_artifacts(processed_dir, args.universe)

    rows = []
    streams = {}

    # ---- PRIMARY: frozen config on the default frozen reservoir + baselines ----
    logger.info("")
    logger.info("PRIMARY (frozen defaults, K=%d):", FROZEN_K[args.universe])
    frozen_states = find_cached_states(states_dir, args.universe)
    packs = build_holdout_packs(artifacts, frozen_states, args.universe)

    baselines = score_baselines(packs["test"], artifacts, config, raw_dir)
    for name, pieces in baselines.items():
        rows.append(aggregate(name, "baseline", [metrics_from_pieces(pieces)]))
        streams[name] = pieces["net_returns"]

    frozen_seeds = score_learned(packs, FROZEN_K[args.universe], SEEDS)
    frozen_name = f"FROZEN K={FROZEN_K[args.universe]} (primary)"
    rows.append(aggregate(frozen_name, "frozen", frozen_seeds))

    # ---- SECONDARY (labelled): the best-found search winner --------------------
    bf = BEST_FOUND[args.universe]
    logger.info("")
    logger.info("SECONDARY (best-found, LABELLED sensitivity): K=%d, N_r=%d, rho=%.2f",
                bf["k"], bf["n_reservoir"], bf["spectral_radius"])
    bf_params = {**ANCHOR, "n_reservoir": bf["n_reservoir"], "spectral_radius": bf["spectral_radius"]}
    bf_states = build_states_for_config(args.universe, bf_params, FIXED, processed_dir, scratch_states)
    bf_packs = build_holdout_packs(artifacts, bf_states, args.universe)
    bf_seeds = score_learned(bf_packs, bf["k"], SEEDS)
    bf_name = f"best-found K={bf['k']} @Nr{bf['n_reservoir']},rho{bf['spectral_radius']} (secondary)"
    rows.append(aggregate(bf_name, "best_found", bf_seeds))

    # ---- report ----------------------------------------------------------------
    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    logger.info("")
    logger.info("=== HOLDOUT RESULTS: %s (2025-01-01 -> 2026-06-30, ~%d days) ===",
                args.universe, len(packs["test"].dates))
    logger.info("  %-52s  Sharpe   (sd)   maxDD   turnover  kind", "arm")
    for _, r in table.iterrows():
        logger.info("  %-52s  %+.3f  (%.3f)  %+.1f%%   %5.1f    %s",
                    r.arm, r.sharpe, r.sharpe_sd, 100 * r.max_drawdown,
                    r.ann_turnover, r.kind)

    out_dir = data_root / "results" / "runs" / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"holdout_{args.universe}.csv", index=False)
    logger.info("")
    logger.info("  wrote %s (NO registry rows, NO commit)", out_dir / f"holdout_{args.universe}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
