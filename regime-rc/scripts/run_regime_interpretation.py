"""What ARE the learned regimes? Level 1: what market state do they track?

    python scripts/run_regime_interpretation.py --universe sector11

The question this answers
-------------------------
Gate G3 established that decision-defined regimes carve the market DIFFERENTLY
from statistically-defined ones: adjusted mutual information between the learned
gate's labels and an HMM's is about zero on every fold and universe. Two
labellings of the same weeks that agree no more than chance.

That says they are different. It does not say what either one IS.

The analogy: two forecasters both say "there are three kinds of day", and their
diaries never agree. AMI measures the disagreement. It cannot tell you that one
is tracking temperature and the other humidity. This script asks that question.

Method
------
Every weekly decision already carries a regime label from each scheme (the
argmax of its membership vector). Separately, we describe what the market was
actually doing that week using standard, causal state variables. Then, for each
scheme, we ask two things:

1. **Profile** - within each regime, what is the average level of each state
   variable? Standardised, so "+0.8" reads as "0.8 standard deviations above
   the sample average whenever this scheme says regime k".

2. **Separation (eta squared)** - what fraction of a state variable's total
   variation is explained by the regime labelling? This is the one-way ANOVA
   effect size, and it is the headline: a scheme that splits cleanly on
   volatility will show a high eta squared there and near zero elsewhere.

The prior, stated before running
--------------------------------
A Gaussian HMM fitted to returns is, mechanically, a variance-state detector, so
it should separate strongly on realised volatility and little else. The learned
gate is the open question. Three outcomes, all informative:

* it separates on a DIFFERENT variable -> that is the mechanism behind AMI ~ 0.
  The two schemes are not disagreeing about one axis; they read different axes.
* it separates on the SAME variable but disagrees anyway -> the difference is
  in threshold or timing, not in what is being measured.
* it separates on NOTHING we test -> the decision objective found structure that
  no standard market descriptor captures. A more interesting negative than
  AMI ~ 0 alone, and a pointer to the Level-2 frequency analysis.

Why K = 3 and not the frozen K = 1
-----------------------------------
At K = 1 the gate is mathematically inert - a softmax over one logit is 1.0
whatever it reads - so there are NO regimes to interpret. The frozen sector11
configuration is K = 1, which means this analysis necessarily runs on the K = 3
mixture. That is a real limitation and it is stated in the results: we are
interpreting the regime structure of a configuration that was not selected.
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

from rrc.eval.agreement import dwell_and_transition, hard_labels, label_agreement  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.gating import MixtureOfExperts  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402
from run_rung3 import statistical_memberships  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("regimes")

N_EXPERTS = 3          # the gate is inert at K = 1; see the module docstring
CASH = "CASH"


# --------------------------------------------------------------------------- #
# Market state at each signal day
# --------------------------------------------------------------------------- #
def market_state(artifacts, signal_absolute: np.ndarray) -> pd.DataFrame:
    """Describe what the market was doing at each signal day.

    Every column is TRAILING - computed from data up to and including the signal
    day - so a reader cannot object that the description peeks. These variables
    are descriptive only; nothing here feeds a trading decision.

    The five signal columns (vix, sentiment, term_slope, short_rate,
    credit_spread) are reused from the P1 signal matrix, so they carry the
    publication lag and trailing z-score the rest of the project uses.
    """
    risky = [c for c in artifacts.columns if c != CASH]
    returns = artifacts.returns_simple[risky]

    equal_weight = returns.mean(axis=1)

    frame = pd.DataFrame(index=artifacts.dates)
    # How choppy has the market been?
    frame["realised_vol_21d"] = equal_weight.rolling(21, min_periods=10).std()
    # How far apart did the assets move from each other today? A market where
    # everything falls together is a different regime from one where sectors
    # rotate, even at identical volatility.
    frame["dispersion"] = returns.std(axis=1).rolling(21, min_periods=10).mean()
    # Trend strength over a quarter.
    frame["momentum_63d"] = equal_weight.rolling(63, min_periods=30).mean()
    # Average pairwise correlation - the "everything moves together" measure.
    #
    # Computed by the implied-correlation identity rather than by building a
    # rolling correlation matrix per day. For an equal-weight basket,
    #     n^2 * Var(mean) = sum(var_i) + rho_bar * [ (sum sd_i)^2 - sum(var_i) ]
    # so rho_bar falls out in closed form. This is the same construction the
    # CBOE implied-correlation indices use, it is a single vectorised pass
    # instead of thousands of matrix builds, and it degrades gracefully when an
    # asset is missing (late entrants) rather than producing NaN for the day.
    n_assets_live = returns.notna().sum(axis=1)
    variances = returns.rolling(63, min_periods=30).var()
    deviations = returns.rolling(63, min_periods=30).std()
    sum_var = variances.sum(axis=1, skipna=True)
    sum_sd_squared = deviations.sum(axis=1, skipna=True) ** 2
    basket_var = equal_weight.rolling(63, min_periods=30).var()
    numerator = n_assets_live**2 * basket_var - sum_var
    denominator = sum_sd_squared - sum_var
    frame["avg_correlation"] = (numerator / denominator).where(denominator > 0)
    # Recent direction, separate from trend strength.
    frame["return_5d"] = equal_weight.rolling(5, min_periods=3).mean()

    for column in artifacts.signals.columns:
        frame[column] = artifacts.signals[column]

    return frame.iloc[signal_absolute].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The two summaries
# --------------------------------------------------------------------------- #
def regime_profile(state: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Standardised mean of each state variable within each regime.

    Standardising first means a cell reads directly as "this many standard
    deviations from the sample average, whenever this scheme says regime k".
    """
    standardised = (state - state.mean()) / state.std(ddof=1)
    return standardised.groupby(labels).mean().T


def separation(state: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Eta squared: the fraction of each variable explained by the labelling.

    One-way ANOVA effect size - between-group variance over total variance.
    0 means the regimes tell you nothing about this variable; 1 means the
    labelling splits it perfectly. This is the headline number: it says WHAT
    each scheme is tracking, in a form comparable across schemes.
    """
    scores = {}
    for column in state.columns:
        values = state[column].to_numpy()
        usable = np.isfinite(values)
        if usable.sum() < 30:
            scores[column] = np.nan
            continue
        y, g = values[usable], labels[usable]
        grand = y.mean()
        between = sum(
            (g == k).sum() * (y[g == k].mean() - grand) ** 2 for k in np.unique(g)
        )
        total = ((y - grand) ** 2).sum()
        scores[column] = float(between / total) if total > 0 else np.nan
    return pd.Series(scores)


# --------------------------------------------------------------------------- #
# Collecting labels across folds
# --------------------------------------------------------------------------- #
def collect(universe, fold_names, seed, config, data_root):
    """Per-fold test-segment labels from each scheme, plus the market state."""
    processed_dir = data_root / "data" / "processed"
    artifacts = load_universe_artifacts(processed_dir, universe)
    states_cache = find_cached_states(data_root / "data" / "states", universe)
    folds = [f for f in folds_for(universe) if f.name in fold_names]

    rows = {"learned": [], "hmm": [], "vix": []}
    state_blocks = []

    for fold in folds:
        packs = build_fold_packs(artifacts, states_cache, fold)
        test = packs["test"]
        window_start = int(artifacts.dates.get_loc(test.dates[0]))
        signal_absolute = test.execution_positions - 1 + window_start

        # --- the learned gate -------------------------------------------
        n_state = packs["train"].states.shape[1]
        n_assets = packs["train"].logit_mask.shape[1]
        trained = train_mixture(
            packs["train"], packs["val"],
            MixtureTrainConfig(n_experts=N_EXPERTS, lambda_cvar=0.0,
                               lambda_balance=0.01, lambda_turn=0.0,
                               lambda_div=0.0, seed=seed),
            model=lambda: MixtureOfExperts(n_state, n_assets, N_EXPERTS),
        )
        with torch.no_grad():
            memberships, _ = trained.model.route_pack(test, noise_scale=0.0)
        rows["learned"].append(hard_labels(memberships.numpy()))

        # --- the statistical gates, on the identical signal days ---------
        for arm, key in (("hmm_gate", "hmm"), ("vix_gate", "vix")):
            per_split, _, _ = statistical_memberships(arm, artifacts, fold, packs, config)
            rows[key].append(hard_labels(per_split["test"]))

        state_blocks.append(market_state(artifacts, signal_absolute))
        logger.info("  %s: %d decisions", fold.name, len(signal_absolute))

    labels = {k: np.concatenate(v) for k, v in rows.items()}
    return pd.concat(state_blocks, ignore_index=True), labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="What market state do the regimes track?")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--folds", default="")
    parser.add_argument("--seed", type=int, default=0, help="Seed-0 convention.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "comparison.yaml"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    config = load_yaml(Path(args.config))
    fold_names = (args.folds.split(",") if args.folds
                  else [f.name for f in folds_for(args.universe)])

    logger.info("=== %s: collecting regime labels over %d folds ===",
                args.universe, len(fold_names))
    state, labels = collect(args.universe, fold_names, args.seed, config, data_root)

    logger.info("")
    logger.info("=== how the three schemes label the same weeks ===")
    for name, series in labels.items():
        stats = dwell_and_transition(series)
        occupancy = np.bincount(series) / len(series)
        logger.info("  %-8s %d regimes | mean dwell %5.1f wks | switches %4.1f%% of weeks"
                    " | occupancy %s", name, len(np.unique(series)),
                    stats["mean_dwell"], 100 * stats["transition_rate"],
                    " ".join(f"{o:.2f}" for o in occupancy))

    logger.info("")
    logger.info("=== agreement between schemes (0 = chance, 1 = identical) ===")
    for a, b in (("learned", "hmm"), ("learned", "vix"), ("hmm", "vix")):
        agreement = label_agreement(labels[a], labels[b])
        logger.info("  %-8s vs %-8s  AMI %+.3f   ARI %+.3f",
                    a, b, agreement["ami"], agreement["ari"])

    logger.info("")
    logger.info("=== SEPARATION: fraction of each variable explained (eta squared) ===")
    sep = pd.DataFrame({name: separation(state, series) for name, series in labels.items()})
    for line in (sep * 100).round(1).to_string().splitlines():
        logger.info("  %s", line)
    logger.info("  (values are percent; higher = this scheme tracks this variable)")

    logger.info("")
    for name, series in labels.items():
        logger.info("=== PROFILE: %s - standardised mean per regime ===", name)
        for line in regime_profile(state, series).round(2).to_string().splitlines():
            logger.info("  %s", line)
        logger.info("")

    out_dir = data_root / "results" / "runs" / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)
    sep.to_csv(out_dir / f"separation_{args.universe}.csv")
    for name, series in labels.items():
        regime_profile(state, series).to_csv(
            out_dir / f"profile_{name}_{args.universe}.csv"
        )
    state.assign(**{f"label_{k}": v for k, v in labels.items()}).to_parquet(
        out_dir / f"state_and_labels_{args.universe}.parquet"
    )
    logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
