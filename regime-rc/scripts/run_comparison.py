"""The ladder comparison: every arm, one universe, one protocol, one metric.

    python scripts/run_comparison.py --universe sector11 --folds F01,F02,F03,F04,F05

What it compares
----------------
Seven arms, all scored identically:

=========================  ================================================
 r0 1/N                     equal weight over the active risky assets
 r0 buy-and-hold SPY        the market line
 r0 inverse-vol             risk parity on trailing 63-day volatility
 r0 Ledoit-Wolf MV          rolling mean-variance, monthly refit
 LSTM                       a TRAINED temporal encoder (decision D2's challenger)
 ours-single (rung 1)       one decision-trained head on the frozen reservoir
 ours-group  (rung 2)       K experts blended by a learned gate
=========================  ================================================

The metric, and why it is the concatenated one
-----------------------------------------------
Each fold's out-of-sample test year is a separate model, but the *procedure*
runs continuously: you deploy for a year, retrain, and deploy again without
resetting your wealth. So the arms are scored on the **stitched daily return
stream** across the fold test years, with a single excess Sharpe computed the
standard way - identical to how every baseline and every published figure is
computed.

Averaging per-year Sharpes is deliberately NOT the headline. It implicitly
rebases wealth each January and weights a calm year equally with a crisis year;
on sector11 folds F01-F03 the two differ by 0.48 (+0.43 versus -0.05), and the
concatenated figure is the honest one. Per-year Sharpes are reported beside it
as a consistency breakdown, never as the result.

Turnover is annualised over the same concatenated span by calendar time
(total one-way turnover x 252 / days), so it is comparable across arms that
trade at different cadences.
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
from rrc.backtest.ledger import run_ledger  # noqa: E402
from rrc.backtest.metrics import excess_sharpe, max_drawdown  # noqa: E402
from rrc.baselines.markowitz import mean_variance_schedule  # noqa: E402
from rrc.data.loaders import PriceLoader, adjusted_prices  # noqa: E402
from rrc.baselines.naive import (  # noqa: E402
    inverse_volatility_schedule,
    one_over_n_schedule,
)
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.recurrent import LSTMPortfolioModel, attach_input_windows  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import (  # noqa: E402
    MixtureTrainConfig,
    TrainConfig,
    train_mixture,
    train_single_head,
)
from rrc.utils.io import load_yaml  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("comparison")

CASH = "CASH"


# --------------------------------------------------------------------------- #
# Scoring: one ledger run per (arm, fold), stitched afterwards
# --------------------------------------------------------------------------- #
def score_schedule(targets: np.ndarray | torch.Tensor, pack) -> dict:
    """Run a weight schedule through the shared ledger; return the raw pieces.

    Returns the daily net returns, the risk-free series and the per-rebalance
    turnover rather than summary statistics, because the summaries have to be
    computed on the CONCATENATED stream, not per fold and averaged.
    """
    result = run_ledger(
        targets=torch.as_tensor(np.asarray(targets), dtype=torch.float64),
        exec_days=pack.execution_positions,
        simple_returns=pack.returns,
        cash_index=pack.cash_index,
    )
    return {
        "net_returns": result.net_returns.detach().numpy(),
        "risk_free": pack.risk_free.numpy(),
        "turnover": result.turnover.detach().numpy(),
    }


def signal_positions_for(pack, artifacts) -> np.ndarray:
    """Absolute calendar positions of a pack's signal days (execution day - 1)."""
    return np.array(
        [
            int(artifacts.dates.get_loc(pack.dates[position - 1]))
            for position in pack.execution_positions
        ]
    )


# --------------------------------------------------------------------------- #
# The baseline arms (rung 0), rebuilt on exactly these folds
# --------------------------------------------------------------------------- #
def baseline_schedules(pack, artifacts, config: dict) -> dict[str, np.ndarray]:
    """Build all four rung-0 schedules for one segment.

    Rebuilt here rather than read from earlier runs so that every arm in this
    table covers the identical fold set - the baselines were originally run over
    all 18 sector11 folds, and comparing those to a 5-fold model would be
    comparing different decades.
    """
    n_assets = pack.logit_mask.shape[1]
    cash_index = pack.cash_index
    signal_absolute = signal_positions_for(pack, artifacts)
    execution = pack.execution_positions
    arms_config = config["arms"]

    risky_columns = [c for c in artifacts.columns if c != CASH]
    risky_returns = artifacts.returns_simple[risky_columns]

    schedules: dict[str, np.ndarray] = {}

    schedules["r0 1/N"] = one_over_n_schedule(
        artifacts.availability, signal_absolute, execution, n_assets, cash_index
    )[0]

    schedules["r0 inverse-vol"] = inverse_volatility_schedule(
        risky_returns, artifacts.availability, signal_absolute, execution,
        n_assets, cash_index,
        window=int(arms_config["inverse_vol"]["window"]),
        min_observations=int(arms_config["inverse_vol"]["min_observations"]),
    )[0]

    schedules["r0 Ledoit-Wolf MV"] = mean_variance_schedule(
        risky_returns, artifacts.risk_free, artifacts.availability, artifacts.dates,
        signal_absolute, execution, n_assets, cash_index,
        lookback=int(arms_config["mean_variance"]["lookback"]),
        min_observations=int(arms_config["mean_variance"]["min_observations"]),
        risk_aversion=float(arms_config["mean_variance"]["risk_aversion"]),
    )[0]

    # Buy-and-hold SPY is handled separately (see spy_benchmark_stream): SPY is
    # the market line for BOTH universes, and in sector11 it is not a tradeable
    # column, so it needs its own two-asset panel rather than a weight vector
    # over this universe.
    return schedules



def spy_benchmark_stream(pack, artifacts, raw_dir: Path) -> dict:
    """Buy-and-hold SPY - the market line, for BOTH universes.

    SPY is not a tradeable column in sector11, so it cannot be expressed as a
    weight vector over that universe. Instead it gets its own two-asset
    ``[SPY, CASH]`` panel and a single trade at the segment's first execution
    day, exactly as ``run_baselines`` does - the same ledger, the same 5 bp
    entry cost, so it stays comparable with every other arm here. Its history
    comes from the P1 raw cache, never a download.
    """
    loader = PriceLoader(raw_dir)
    histories, outcomes = loader.load_many(["SPY"])
    if not outcomes[0].succeeded:
        raise RuntimeError("SPY history unavailable from the raw cache.")

    spy_returns = adjusted_prices(histories)["SPY"].pct_change(fill_method=None)
    panel = pd.DataFrame(index=pack.dates)
    panel["SPY"] = spy_returns.reindex(pack.dates)
    panel["CASH"] = pd.Series(pack.risk_free.numpy(), index=pack.dates)
    if panel["SPY"].isna().any():
        raise ValueError("SPY returns have gaps inside the evaluated segment.")

    targets = np.array([[1.0, 0.0]])          # all SPY, traded once, then held
    result = run_ledger(
        targets=torch.as_tensor(targets, dtype=torch.float64),
        exec_days=pack.execution_positions[:1],
        simple_returns=torch.as_tensor(panel.to_numpy(), dtype=torch.float64),
        cash_index=1,
    )
    return {
        "net_returns": result.net_returns.detach().numpy(),
        "risk_free": pack.risk_free.numpy(),
        "turnover": result.turnover.detach().numpy(),
    }


def score_baselines(pack, artifacts, config: dict, raw_dir: Path) -> dict[str, dict]:
    """Score every baseline arm on one segment."""
    scored = {
        name: score_schedule(targets, pack)
        for name, targets in baseline_schedules(pack, artifacts, config).items()
    }
    scored["r0 buy-and-hold SPY"] = spy_benchmark_stream(pack, artifacts, raw_dir)
    return scored


# --------------------------------------------------------------------------- #
# The model arms
# --------------------------------------------------------------------------- #
def score_models(packs, artifacts, reservoir_panel, config: dict, seed: int) -> dict[str, dict]:
    """Train and score the three learned arms on one fold."""
    loss_cfg = config["loss"]
    scored: dict[str, dict] = {}

    # --- ours-single: rung 1, one decision-trained head -------------------
    single = train_single_head(
        packs["train"], packs["val"],
        TrainConfig(lambda_cvar=float(loss_cfg["lambda_cvar"]),
                    lambda_turn=float(loss_cfg["lambda_turn"]), seed=seed),
    )
    with torch.no_grad():
        weights = single.head(packs["test"].states, packs["test"].logit_mask)
    scored["ours-single (rung 1)"] = score_schedule(weights.numpy(), packs["test"])

    # --- ours-group: rung 2, K experts under a learned gate ---------------
    group = train_mixture(
        packs["train"], packs["val"],
        MixtureTrainConfig(
            n_experts=int(config["mixture"]["n_experts"]),
            lambda_cvar=float(loss_cfg["lambda_cvar"]),
            lambda_balance=float(loss_cfg["lambda_balance"]),
            lambda_turn=float(loss_cfg["lambda_turn"]),
            lambda_div=float(loss_cfg["lambda_div"]),
            seed=seed,
        ),
    )
    with torch.no_grad():
        weights = group.model.weights_for_pack(packs["test"])
    scored["ours-group (rung 2)"] = score_schedule(weights.numpy(), packs["test"])

    # --- LSTM: the trained-encoder challenger (D2) -------------------------
    lstm_cfg = config["lstm"]
    windowed = attach_input_windows(
        packs, reservoir_panel, artifacts, int(lstm_cfg["lookback"])
    )
    factory = lambda: LSTMPortfolioModel(  # noqa: E731 - built after seeding
        n_inputs=reservoir_panel.shape[1],
        n_assets=packs["train"].logit_mask.shape[1],
        hidden_size=int(lstm_cfg["hidden_size"]),
    )
    lstm = train_mixture(
        windowed["train"], windowed["val"],
        MixtureTrainConfig(
            n_experts=1,
            lambda_cvar=float(loss_cfg["lambda_cvar"]),
            lambda_balance=0.0,          # meaningless for a single expert
            lambda_turn=float(loss_cfg["lambda_turn"]),
            learning_rate=float(lstm_cfg["learning_rate"]),
            max_epochs=int(lstm_cfg["max_epochs"]),
            patience=int(lstm_cfg["patience"]),
            seed=seed,
        ),
        model=factory,
    )
    with torch.no_grad():
        weights = lstm.model.weights_for_pack(windowed["test"])
    scored["LSTM (trained encoder)"] = score_schedule(weights.numpy(), windowed["test"])

    return scored


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(universe: str, fold_names: list[str], seeds: list[int], config: dict) -> pd.DataFrame:
    """Score every arm over every fold and seed; return tidy per-fold rows."""
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    states_dir = PROJECT_ROOT / config["paths"]["states_dir"]

    artifacts = load_universe_artifacts(processed_dir, universe)
    cached_states = find_cached_states(states_dir, universe)
    reservoir_panel = pd.read_parquet(
        processed_dir / f"{universe}_reservoir_input.parquet"
    )
    folds = [f for f in folds_for(universe) if f.name in fold_names]

    rows: list[dict] = []
    for fold in folds:
        packs = build_fold_packs(artifacts, cached_states, fold)
        test = packs["test"]
        logger.info("%s %s: test year %d", universe, fold.name, fold.test_year)

        # Baselines are deterministic - scored once, recorded under every seed
        # so the concatenation logic below is uniform.
        baselines = score_baselines(
            test, artifacts, config, PROJECT_ROOT / "data" / "raw"
        )

        for seed in seeds:
            scored = dict(baselines)
            scored.update(score_models(packs, artifacts, reservoir_panel, config, seed))
            for arm, pieces in scored.items():
                rows.append({
                    "universe": universe, "fold": fold.name, "year": fold.test_year,
                    "seed": seed, "arm": arm, **pieces,
                })
            logger.info("  seed %d done", seed)

    return pd.DataFrame(rows)


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Concatenate each arm's per-fold streams, then compute one set of metrics.

    Per seed: stitch the fold test years into one continuous daily stream and
    score it. Across seeds: report the mean and the spread. This is the correct
    order - concatenate first, summarise second.
    """
    records = []
    for arm, arm_rows in rows.groupby("arm", sort=False):
        per_seed_sharpe, per_seed_turnover, per_seed_dd = [], [], []
        for seed, seed_rows in arm_rows.groupby("seed"):
            seed_rows = seed_rows.sort_values("year")
            returns = np.concatenate(seed_rows["net_returns"].tolist())
            risk_free = np.concatenate(seed_rows["risk_free"].tolist())
            turnover = np.concatenate(seed_rows["turnover"].tolist())
            per_seed_sharpe.append(excess_sharpe(returns, risk_free))
            per_seed_turnover.append(annualised_turnover(turnover, n_days=len(returns)))
            per_seed_dd.append(max_drawdown(returns))

        # Per-year Sharpes, as a consistency breakdown only.
        yearly = {
            int(year): float(np.mean([
                excess_sharpe(r["net_returns"], r["risk_free"])
                for _, r in year_rows.iterrows()
            ]))
            for year, year_rows in arm_rows.groupby("year")
        }
        records.append({
            "arm": arm,
            "sharpe": float(np.mean(per_seed_sharpe)),
            "sharpe_sd": float(np.std(per_seed_sharpe, ddof=1)) if len(per_seed_sharpe) > 1 else 0.0,
            "ann_turnover": float(np.mean(per_seed_turnover)),
            "max_drawdown": float(np.mean(per_seed_dd)),
            **{f"y{year}": value for year, value in sorted(yearly.items())},
        })
    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ladder comparison across all arms.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "comparison.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars.",
    )
    parser.add_argument("--folds", default="F01,F02,F03,F04,F05")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    fold_names = args.folds.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = run(args.universe, fold_names, seeds, config)
    table = summarise(rows)

    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / f"comparison_{args.universe}.csv"
    table.round(4).to_csv(csv_path, index=False)

    years = sorted(rows["year"].unique())
    logger.info("=" * 78)
    logger.info("%s - concatenated out-of-sample test years %s (seeds %s)",
                args.universe, ", ".join(str(y) for y in years),
                ", ".join(str(s) for s in seeds))
    logger.info("=" * 78)
    logger.info("%-28s %9s %9s %10s %9s", "arm", "Sharpe", "+/-sd", "turnover", "maxDD")
    for _, row in table.iterrows():
        logger.info("%-28s %9.3f %9.3f %10.2f %9.3f",
                    row["arm"], row["sharpe"], row["sharpe_sd"],
                    row["ann_turnover"], row["max_drawdown"])
    logger.info("")
    logger.info("per-year Sharpe (consistency breakdown, NOT the headline):")
    year_columns = [c for c in table.columns if c.startswith("y")]
    logger.info("%-28s " + " ".join(f"{c[1:]:>8}" for c in year_columns), "arm")
    for _, row in table.iterrows():
        logger.info("%-28s " % row["arm"] + " ".join(f"{row[c]:>8.2f}" for c in year_columns))
    logger.info("  csv -> %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
