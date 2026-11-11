"""Quick re-check of the two loss penalties at the FROZEN config, post-anneal-fix.

    python scripts/run_penalty_sweep.py --universe etf10

The lambda_cvar numbers on record were measured at P8, before the gate-noise
anneal fix and before the configuration settled. This re-checks both penalties
at the frozen configuration as it stands now, so the write-up can say
"re-verified" rather than "measured earlier".

etf10 is used because its frozen config is K=3, where BOTH penalties act;
sector11 is K=1, where lambda_balance is identically zero. Each penalty is swept
one-at-a-time around the frozen point (cvar=0, balance=0.01), at the frozen
default reservoir, scored on the concatenated walk-forward TEST split. No
registry rows, no holdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.costs import annualised_turnover  # noqa: E402
from rrc.backtest.metrics import excess_sharpe, max_drawdown  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.gating import MixtureOfExperts  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("penalty")

K_FROZEN = {"etf10": 3, "sector11": 1}


def score(packs_by_fold, folds, k, lam_cvar, lam_balance, seeds):
    """Concatenated test Sharpe / maxDD at one (cvar, balance)."""
    sh, dd = [], []
    for seed in seeds:
        streams, rfs = [], []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            n_state = packs["train"].states.shape[1]
            n_assets = packs["train"].logit_mask.shape[1]
            result = train_mixture(
                packs["train"], packs["val"],
                MixtureTrainConfig(n_experts=k, lambda_cvar=lam_cvar,
                                   lambda_balance=lam_balance if k > 1 else 0.0,
                                   lambda_turn=0.0, lambda_div=0.0, seed=seed),
                model=lambda: MixtureOfExperts(n_state, n_assets, k),
            )
            ev = evaluate_on_pack(result.model, packs["test"])
            streams.append(ev["net_returns"].to_numpy())
            rfs.append(packs["test"].risk_free.numpy())
        r = np.concatenate(streams)
        sh.append(excess_sharpe(r, np.concatenate(rfs)))
        dd.append(max_drawdown(r))
    return float(np.mean(sh)), float(np.std(sh, ddof=1)), float(np.mean(dd))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-check the two loss penalties at the frozen config.")
    parser.add_argument("--universe", default="etf10", choices=["etf10", "sector11"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    seeds = [int(s) for s in args.seeds.split(",")]
    k = K_FROZEN[args.universe]

    artifacts = load_universe_artifacts(data_root / "data" / "processed", args.universe)
    states = find_cached_states(data_root / "data" / "states", args.universe)
    folds = list(folds_for(args.universe))
    packs_by_fold = {f.name: build_fold_packs(artifacts, states, f) for f in folds}

    # (cvar, balance) points: frozen is (0, 0.01). Sweep each one-at-a-time.
    configs = [
        (0.0, 0.01, "FROZEN (cvar 0, bal 0.01)"),
        (0.5, 0.01, "cvar 0.5"),
        (1.0, 0.01, "cvar 1.0"),
        (5.0, 0.01, "cvar 5.0"),
        (0.0, 0.00, "bal 0.00"),
        (0.0, 0.05, "bal 0.05"),
        (0.0, 0.10, "bal 0.10"),
    ]

    logger.info("=== %s | K=%d | frozen reservoir | %d folds x %d seeds (TEST) ===",
                args.universe, k, len(folds), len(seeds))
    logger.info("  %-26s  Sharpe  (sd)    maxDD", "config")
    for lc, lb, label in configs:
        s, sd, d = score(packs_by_fold, folds, k, lc, lb, seeds)
        logger.info("  %-26s  %+.3f (%.3f)  %+.1f%%", label, s, sd, 100 * d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
