"""P9: every arm, every sector11 fold, one protocol, one metric.

    python scripts/run_walkforward.py --universe sector11
    python scripts/run_walkforward.py --universe sector11 --folds F01,F02 --seeds 0  # smoke

What this adds over run_comparison.py
--------------------------------------
``run_comparison.py`` scores seven arms on a chosen fold subset. This runs the
FULL ladder over ALL folds and, crucially, saves each arm's stitched daily
stream so the figures and the inference layer regenerate without retraining.

Arms, and why each is here
--------------------------
======================  ====================================================
arm                     what it answers
======================  ====================================================
r0 1/N                  is anything better than naive diversification?
r0 buy-and-hold SPY     ... than simply owning the market?
r0 inverse-vol          ... than trivial risk scaling?
r0 Ledoit-Wolf MV       ... than classical mean-variance?
LSTM (trained encoder)  does a TRAINED encoder beat the frozen one? (D2)
ours K=1 (rung 1)       does decision-focused training add value at all?
ours K=3 (rung 2)       does the mixture help? (G2's question)
3a sent-gate (K=3)      does sentiment help by routing?          (RQ3)
3b sent-experts (K=3)   ... or by informing the allocation?      (RQ3)
3c sent-both (K=3)      the capacity ceiling for the 3a/3b pair
ours K=1 + sentiment    the single head, given the signals
r4 HMM gate             THE thesis question: decision-defined vs
r4 VIX gate             statistically-defined regimes, equal conditions
======================  ====================================================

That is also the presentation order (``LADDER_ORDER``): baselines, then the
challenger, then our ladder one rung at a time. Tables and figure legends both
follow it, so a reader meets the arms in the order the argument builds rather
than in whatever order they happened to score.

Note on "ours K=1 + sentiment". At K = 1 the gate is mathematically inert - a
softmax over one logit is 1.0 whatever it reads - so sentiment-into-the-gate
(3a) is a NO-OP there. The only way sentiment can enter a K = 1 system is
through the expert, so that arm is 3b's routing at K = 1. On sector11, where
K = 1 was selected, this means the selection forecloses the routing hypothesis
that rung 3a exists to test - worth stating plainly in the write-up.

Everything reuses the existing scorers
--------------------------------------
Baselines come from ``run_comparison`` and model construction from
``run_rung3``; nothing is reimplemented here. That is deliberate - R6's
"equal conditions" requirement is satisfied by literal code reuse, not by two
implementations that are believed to match.
"""

from __future__ import annotations

import argparse
import json
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
from rrc.models.gating import MixtureOfExperts, SignalRoutedMixture  # noqa: E402
from rrc.models.recurrent import LSTMPortfolioModel, attach_input_windows  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import atomic_write_text, load_yaml  # noqa: E402

# Reused wholesale - see the module docstring.
from run_comparison import score_baselines, score_schedule  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402
from run_rung3 import (  # noqa: E402
    attach_fixed_memberships, model_factory, statistical_memberships,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("walkforward")

# The two expert counts every universe is run at. They are NAMED NEUTRALLY on
# purpose. An earlier version called K = 1 "ours-optimal" on both universes,
# which silently imported sector11's K selection into etf10 - and no etf10
# evidence supports it:
#
#   sector11 - K = 1 was selected, in three rounds of the K search
#              (0.670 / 0.639 at K = 1 / 3 on twelve folds).
#   etf10    - NO K selection was ever registered. The cross-universe diagnostic
#              had every K tied (K=1 1.062, K=2 1.044, K=3 1.064, K=5 1.066
#              against seed spreads of 0.03-0.13), so the plan's Appendix B
#              default of K = 3 stands.
#
# Both are always run, so the ladder carries the selected configuration AND the
# alternative on either universe; which one is "ours" is stated in the write-up
# per universe rather than baked into an arm name.
K_SINGLE = 1
K_MIXTURE = 3

# Canonical presentation order: the humility line first, then the trained-encoder
# challenger, then our ladder one rung at a time. Results tables and figure
# legends both use this, so a reader meets the arms in the order the argument
# builds rather than in whatever order they happened to score.
LADDER_ORDER = [
    "r0 1/N",
    "r0 buy-and-hold SPY",
    "r0 inverse-vol",
    "r0 Ledoit-Wolf MV",
    "LSTM (trained encoder)",
    "ours K=1 (rung 1)",
    "ours K=3 (rung 2)",
    "3a sent-gate (K=3)",
    "3b sent-experts (K=3)",
    "3c sent-both (K=3)",
    "ours K=1 + sentiment",
    "r4 HMM gate",
    "r4 VIX gate",
]


def in_ladder_order(arms):
    """Sort arm names into LADDER_ORDER, leaving unknown names at the end."""
    rank = {name: index for index, name in enumerate(LADDER_ORDER)}
    return sorted(arms, key=lambda a: (rank.get(a, len(rank)), a))


def learned_arms(packs, artifacts, reservoir_panel, config, seed, fold) -> dict[str, dict]:
    """Train and score every learned arm on one fold, at one seed."""
    loss = config["loss"]
    train_pack, val_pack, test_pack = packs["train"], packs["val"], packs["test"]
    n_state = train_pack.states.shape[1]
    n_assets = train_pack.logit_mask.shape[1]
    n_signals = train_pack.signals.shape[1]
    scored: dict[str, dict] = {}

    def train_and_score(name, factory, k, packs_override=None):
        """One training run through the SHARED loop, then the shared ledger."""
        use = packs_override or packs
        result = train_mixture(
            use["train"], use["val"],
            MixtureTrainConfig(
                n_experts=k,
                lambda_cvar=float(loss["lambda_cvar"]),
                lambda_balance=float(loss["lambda_balance"]) if k > 1 else 0.0,
                lambda_turn=float(loss["lambda_turn"]),
                lambda_div=float(loss["lambda_div"]),
                seed=seed,
            ),
            model=factory,
        )
        with torch.no_grad():
            weights = result.model.weights_for_pack(use["test"])
        scored[name] = score_schedule(weights.numpy(), use["test"])

    # --- the selected configuration, and the K = 3 mixture it beat ----------
    train_and_score("ours K=1 (rung 1)",
                    lambda: MixtureOfExperts(n_state, n_assets, K_SINGLE), K_SINGLE)
    train_and_score("ours K=3 (rung 2)",
                    lambda: MixtureOfExperts(n_state, n_assets, K_MIXTURE), K_MIXTURE)

    # --- the three sentiment arms, at K = 3 where routing is meaningful -----
    for name, to_gate, to_experts in (
        ("3a sent-gate (K=3)", True, False),
        ("3b sent-experts (K=3)", False, True),
        ("3c sent-both (K=3)", True, True),
    ):
        train_and_score(
            name,
            lambda tg=to_gate, te=to_experts: SignalRoutedMixture(
                n_state, n_assets, K_MIXTURE, n_signals,
                signals_to_gate=tg, signals_to_experts=te,
            ),
            K_MIXTURE,
        )

    # --- the selected config WITH sentiment. Only 3b's route exists at K = 1.
    train_and_score(
        "ours K=1 + sentiment",
        lambda: SignalRoutedMixture(
            n_state, n_assets, K_SINGLE, n_signals,
            signals_to_gate=False, signals_to_experts=True,
        ),
        K_SINGLE,
    )

    # --- rung 4: the statistical gates, at equal conditions ------------------
    for arm, label in (("hmm_gate", "r4 HMM gate"), ("vix_gate", "r4 VIX gate")):
        try:
            per_split, n_states, _ = statistical_memberships(
                arm, artifacts, fold, packs, config
            )
        except Exception as error:                     # noqa: BLE001
            logger.warning("  %s unavailable on this fold (%s); skipped", label, error)
            continue
        routed = attach_fixed_memberships(packs, per_split)
        train_and_score(
            label,
            model_factory("4", arm, routed["train"], config, n_states),
            n_states,
            packs_override=routed,
        )

    # --- the LSTM challenger (D2) -------------------------------------------
    lstm_cfg = config["lstm"]
    windowed = attach_input_windows(
        packs, reservoir_panel, artifacts, int(lstm_cfg["lookback"])
    )
    result = train_mixture(
        windowed["train"], windowed["val"],
        MixtureTrainConfig(
            n_experts=1, lambda_cvar=float(loss["lambda_cvar"]), lambda_balance=0.0,
            lambda_turn=float(loss["lambda_turn"]),
            learning_rate=float(lstm_cfg["learning_rate"]),
            max_epochs=int(lstm_cfg["max_epochs"]),
            patience=int(lstm_cfg["patience"]), seed=seed,
        ),
        model=lambda: LSTMPortfolioModel(
            n_inputs=reservoir_panel.shape[1], n_assets=n_assets,
            hidden_size=int(lstm_cfg["hidden_size"]),
        ),
    )
    with torch.no_grad():
        weights = result.model.weights_for_pack(windowed["test"])
    scored["LSTM (trained encoder)"] = score_schedule(weights.numpy(), windowed["test"])

    return scored


def run(universe, fold_names, seeds, config, data_root) -> pd.DataFrame:
    """Score every arm over every fold and seed; return tidy per-fold rows."""
    processed_dir = data_root / config["paths"]["processed_dir"]
    states_dir = data_root / config["paths"]["states_dir"]

    artifacts = load_universe_artifacts(processed_dir, universe)
    cached_states = find_cached_states(states_dir, universe)
    reservoir_panel = pd.read_parquet(processed_dir / f"{universe}_reservoir_input.parquet")
    folds = [f for f in folds_for(universe) if f.name in fold_names]

    rows: list[dict] = []
    for fold in folds:
        packs = build_fold_packs(artifacts, cached_states, fold)
        logger.info("%s %s: test year %d", universe, fold.name, fold.test_year)

        # Baselines are deterministic: scored once, recorded under every seed so
        # the concatenation below stays uniform.
        baselines = score_baselines(packs["test"], artifacts, config, data_root / "data" / "raw")

        for seed in seeds:
            scored = dict(baselines)
            scored.update(learned_arms(packs, artifacts, reservoir_panel, config, seed, fold))
            for arm, pieces in scored.items():
                rows.append({"fold": fold.name, "year": fold.test_year,
                             "seed": seed, "arm": arm, **pieces})
            logger.info("  seed %d: %d arms", seed, len(scored))

    return pd.DataFrame(rows)


def safe_stem(arm: str) -> str:
    """Path-safe filename stem for an arm. Lossy - see arms.json."""
    return arm.replace(" ", "_").replace("/", "over").replace("(", "").replace(")", "")


def stitch_streams(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per arm, the seed-averaged daily net-return and risk-free streams.

    Seeds are averaged at the RETURN level, not the Sharpe level: averaging
    Sharpes across seeds would be the same aggregation error the project already
    rejected across folds.
    """
    streams = {}
    for arm, arm_rows in rows.groupby("arm", sort=False):
        per_seed = []
        for _, seed_rows in arm_rows.groupby("seed"):
            ordered = seed_rows.sort_values("year")
            per_seed.append(np.concatenate(ordered["net_returns"].tolist()))
        first = arm_rows[arm_rows.seed == arm_rows.seed.min()].sort_values("year")
        streams[arm] = pd.DataFrame({
            "net_return": np.mean(per_seed, axis=0),
            "risk_free": np.concatenate(first["risk_free"].tolist()),
            "year": np.concatenate([
                np.full(len(r["net_returns"]), r["year"]) for _, r in first.iterrows()
            ]),
        })
    return streams


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: concatenated metrics, plus per-seed spread."""
    records = []
    for arm, arm_rows in rows.groupby("arm", sort=False):
        sharpes, turnovers, drawdowns = [], [], []
        for _, seed_rows in arm_rows.groupby("seed"):
            ordered = seed_rows.sort_values("year")
            returns = np.concatenate(ordered["net_returns"].tolist())
            risk_free = np.concatenate(ordered["risk_free"].tolist())
            turnover = np.concatenate(ordered["turnover"].tolist())
            sharpes.append(excess_sharpe(returns, risk_free))
            turnovers.append(annualised_turnover(turnover, n_days=len(returns)))
            drawdowns.append(max_drawdown(returns))
        records.append({
            "arm": arm,
            "sharpe": float(np.mean(sharpes)),
            "sharpe_sd": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
            "ann_turnover": float(np.mean(turnovers)),
            "max_drawdown": float(np.mean(drawdowns)),
        })
    frame = pd.DataFrame(records).set_index("arm")
    return frame.reindex(in_ladder_order(frame.index)).reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full walk-forward across all arms.")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "comparison.yaml"))
    parser.add_argument("--folds", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    config = load_yaml(Path(args.config))
    seeds = [int(v) for v in args.seeds.split(",")]
    fold_names = (
        args.folds.split(",") if args.folds
        else [f.name for f in folds_for(args.universe)]
    )

    logger.info("=== walk-forward: %d folds x %d seeds ===", len(fold_names), len(seeds))
    rows = run(args.universe, fold_names, seeds, config, data_root)

    out_dir = data_root / "results" / "runs" / "walkforward"
    stream_dir = out_dir / args.universe
    stream_dir.mkdir(parents=True, exist_ok=True)

    # Stream filenames have to be path-safe, which loses characters ("(", ")",
    # "/") and therefore cannot round-trip back to the arm name. Rather than
    # guess the name back from the filename - which silently mismatched the
    # money table against the summary table - the mapping is written out
    # explicitly and read back by the figure script.
    streams = stitch_streams(rows)
    manifest = {}
    for arm, frame in streams.items():
        stem = safe_stem(arm)
        manifest[stem] = arm
        frame.to_parquet(stream_dir / f"{stem}.parquet")
    atomic_write_text(stream_dir / "arms.json", json.dumps(manifest, indent=2))

    table = summarise(rows)
    table.to_csv(out_dir / f"summary_{args.universe}.csv", index=False)

    logger.info("")
    logger.info("=== concatenated out-of-sample results, %s ===", args.universe)
    for row in table.itertuples():
        logger.info("  %-24s Sharpe %+.3f (sd %.3f)  turnover %5.2f  maxDD %+.3f",
                    row.arm, row.sharpe, row.sharpe_sd, row.ann_turnover, row.max_drawdown)
    logger.info("")
    logger.info("Wrote %s and %d streams to %s", out_dir, len(streams), stream_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
