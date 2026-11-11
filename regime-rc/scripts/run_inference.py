"""P9 inference: deflated Sharpe, paired bootstrap CIs, Ledoit-Wolf tests.

    python scripts/run_inference.py --universe sector11
    python scripts/run_inference.py --universe etf10 --data-root ~/Desktop/AI/dissertation/regime-rc

This script writes no registry rows. It reads what the ladder already produced -
the per-arm daily streams under ``results/runs/streams/`` and the trial record in
``results/registry.csv`` - and turns those numbers into evidence.

The deflation, and the one judgement call it needs
--------------------------------------------------
The deflated Sharpe corrects a result for the fact that it was *selected*. Search
N configurations, report the best, and the winner flatters you even if every
configuration is worthless. Two inputs set the size of that correction:

* **the trial count N**, which this project pre-registered before any experiment
  existed: distinct ``cfg_hash`` values in the registry with ``split == "val"``.
  It is currently 305. **This script never varies it.** Shrinking N is the one
  move that would destroy the pre-registration, and the pre-registration is
  worth more than any Sharpe number it could rescue.

* **the trial dispersion sigma**, the spread of Sharpes *across* those trials.
  The pre-registered rule fixed N and said nothing about sigma, so this is a
  genuine open methodological choice - and it dominates the answer, because the
  benchmark is proportional to sigma but only logarithmic in N.

Because sigma is a choice, this script computes **both** defensible estimators
and reports both:

``pooled``
    Every val trial in the registry, both universes, one Sharpe each. The most
    conservative possible reading and the one a sceptical reader would compute
    first. On the current registry it gives sigma ~ 0.365.

``coherent``
    The same N, but sigma estimated only over trials that were genuinely
    competing for the same selection decision. Three exclusions, each justified
    by a convention this project locked in *before* this calculation existed:

    1. one universe at a time - HANDOFF section 3: "the two universes are NEVER
       run or aggregated together", because they sit on different calendars;
    2. drop the all-cash rung-5 rows - HANDOFF section 9 already instructs
       filtering these with ``ann_turnover > 0``; they are a documented scale
       bug, not strategies that lost;
    3. search rungs only - the ladder arms are reported *results*, not search
       candidates, and the pre-fix rows use the mean-of-folds aggregation that
       HANDOFF section 5.3 shows is invalid and differs by up to 0.48.

Reporting the pooled number first, then arguing for the coherent one, is the
honest order. A reader who computes the sceptical number themselves and finds it
already in the table will trust the rest of the chapter.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.eval.stats import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    ledoit_wolf_sharpe_test,
    paired_sharpe_difference_interval,
)
from rrc.utils.io import atomic_write_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("inference")

# The ladder, in rung order. Adjacent pairs are what the plan asks to be tested
# against each other ("keep a rung only if it beats the rung below").
LADDER_ORDER = [
    # walk-forward names (P9 source of record)
    "r0_1overN", "r0_buy-and-hold_SPY", "r0_inverse-vol", "r0_Ledoit-Wolf_MV",
    "ours-optimal_K=1", "ours-K3_rung_2",
    "3a_sent-gate", "3b_sent-experts", "3c_sent-both",
    "optimal+sentiment_K=1", "r4_HMM_gate", "r4_VIX_gate",
    "LSTM_trained_encoder",
    # older rung3-era names, kept so the legacy streams still compare
    "r0_one_over_n", "r0_buy_hold_spy", "r0_inverse_vol", "r0_mean_variance",
    "r1_single_head", "r2_moe_k3",
    "r3a_sent_gate", "r3b_sent_experts", "r3c_sent_both",
    "r4_hmm_gate", "r4_vix_gate",
]

# Rungs whose val rows are search candidates rather than reported ladder arms.
SEARCH_RUNGS = {"hyper", "k-search", "grid"}

# Arms that were specified in advance and never searched over. Deflating these
# against the model search count would be a category error: the DSR corrects a
# result for the selection that produced it, and nothing selected buy-and-hold
# SPY from 305 candidates - it was written into the plan at P4. They are
# therefore scored with the *undeflated* benchmark of zero, which reduces the
# DSR to the probabilistic Sharpe ratio (Bailey and Lopez de Prado's own
# special case) and keeps them on the same scale as the searched arms.
PRE_SPECIFIED_ARMS = {
    "r0_1overN", "r0_buy-and-hold_SPY", "r0_inverse-vol", "r0_Ledoit-Wolf_MV",
    "r0_one_over_n", "r0_buy_hold_spy", "r0_inverse_vol", "r0_mean_variance",
}


@dataclass(frozen=True)
class TrialPopulation:
    """The (N, sigma) pair a deflation is computed against."""

    name: str
    n_trials: int
    dispersion: float
    benchmark_sharpe: float
    n_trials_in_dispersion: int
    description: str


# --------------------------------------------------------------------------- #
# Building the trial populations
# --------------------------------------------------------------------------- #
def pre_registered_trial_count(registry: pd.DataFrame) -> int:
    """The pre-registered N: distinct cfg_hash among validation rows.

    Global across both universes and every rung, exactly as written down at P0
    before any experiment ran. Never filtered, never split.
    """
    return int(registry.loc[registry["split"] == "val", "cfg_hash"].nunique())


def trial_sharpes(rows: pd.DataFrame) -> pd.Series:
    """One Sharpe per trial: average a cfg_hash's folds and seeds into a point.

    A configuration run over five folds and five seeds is *one* candidate that
    was given one chance to be selected, not twenty-five. Averaging first is
    what keeps re-running a configuration from inflating the dispersion.
    """
    sharpes = pd.to_numeric(rows["excess_sharpe"], errors="coerce")
    return rows.assign(sharpe=sharpes).groupby("cfg_hash")["sharpe"].mean().dropna()


def build_populations(registry: pd.DataFrame, universe: str) -> list[TrialPopulation]:
    """The two sigma estimators, both at the pre-registered trial count."""
    n_trials = pre_registered_trial_count(registry)
    validation = registry[registry["split"] == "val"]

    pooled_sharpes = trial_sharpes(validation)
    pooled_dispersion = float(pooled_sharpes.std(ddof=1))

    # The coherent population: this universe, search rungs, live rung-5 rows only.
    turnover = pd.to_numeric(validation.get("ann_turnover"), errors="coerce")
    coherent_rows = validation[
        (validation["universe"] == universe)
        & (validation["rung"].isin(SEARCH_RUNGS))
        & (turnover.isna() | (turnover > 0))
    ]
    coherent_sharpes = trial_sharpes(coherent_rows)
    if len(coherent_sharpes) < 2:
        raise SystemExit(
            f"Only {len(coherent_sharpes)} coherent trials for {universe}; "
            "cannot estimate a dispersion. Run the searches first."
        )
    coherent_dispersion = float(coherent_sharpes.std(ddof=1))

    return [
        TrialPopulation(
            name="pooled",
            n_trials=n_trials,
            dispersion=pooled_dispersion,
            benchmark_sharpe=expected_max_sharpe_under_null(pooled_dispersion, n_trials),
            n_trials_in_dispersion=len(pooled_sharpes),
            description=(
                "every validation trial in the registry, both universes, all rungs - "
                "the most sceptical reading"
            ),
        ),
        TrialPopulation(
            name="coherent",
            n_trials=n_trials,
            dispersion=coherent_dispersion,
            benchmark_sharpe=expected_max_sharpe_under_null(coherent_dispersion, n_trials),
            n_trials_in_dispersion=len(coherent_sharpes),
            description=(
                f"{universe} search trials only, all-cash rung-5 rows excluded, "
                "one metric convention - trials that genuinely competed"
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Loading the streams
# --------------------------------------------------------------------------- #
def load_excess_streams(data_root: Path, universe: str, source: str) -> dict[str, pd.Series]:
    """Read each arm's daily returns and convert them to excess returns.

    Excess over the same r_f the cash column earns, which is what makes an
    all-cash book score exactly zero (E1) and what every other metric in this
    project is defined on.

    Two stream layouts exist and both are supported:

    ``walkforward``
        Written by ``run_walkforward.py``. Carries its own ``risk_free`` column
        because the stream is a concatenation of fold segments and the matching
        r_f days have already been selected. This is the P9 source of record.
    ``rung3``
        The older per-rung streams, date-indexed with only ``net_return``, so
        r_f is joined here from the processed panel.
    """
    if source == "walkforward":
        stream_dir = data_root / "results" / "runs" / "walkforward" / universe
    else:
        stream_dir = data_root / "results" / "runs" / "streams" / universe
    if not stream_dir.exists():
        raise SystemExit(f"No streams at {stream_dir}. Run the producing script first.")

    risk_free = None
    streams: dict[str, pd.Series] = {}
    for path in sorted(stream_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "risk_free" in frame.columns:
            excess = frame["net_return"] - frame["risk_free"]
        else:
            if risk_free is None:
                risk_free = pd.read_parquet(
                    data_root / "data" / "processed" / f"{universe}_risk_free.parquet"
                )["r_f"]
            excess = frame["net_return"] - risk_free.reindex(frame.index)
        streams[path.stem] = excess.dropna()
    return streams


def align_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Restrict two streams to the days both cover, preserving order."""
    shared = a.index.intersection(b.index)
    return a.loc[shared].to_numpy(), b.loc[shared].to_numpy()


# --------------------------------------------------------------------------- #
# The three computations
# --------------------------------------------------------------------------- #
def deflate_every_arm(
    streams: dict[str, pd.Series], populations: list[TrialPopulation]
) -> pd.DataFrame:
    """Deflated Sharpe for each arm, under each dispersion estimator.

    Pre-specified baselines are scored at zero dispersion - see
    :data:`PRE_SPECIFIED_ARMS` for why deflating them at all would be wrong.
    """
    records = []
    for arm, stream in streams.items():
        searched = arm not in PRE_SPECIFIED_ARMS
        for population in populations:
            result = deflated_sharpe_ratio(
                stream.to_numpy(),
                trial_dispersion=population.dispersion if searched else 0.0,
                n_trials=population.n_trials,
            )
            records.append(
                {
                    "arm": arm,
                    "searched": searched,
                    "estimator": population.name,
                    "sharpe": result.observed_sharpe,
                    "benchmark": result.benchmark_sharpe,
                    "dsr": result.deflated_probability,
                    "clears": result.clears_benchmark,
                    "n_days": result.n_observations,
                    "skew": result.skewness,
                    "kurtosis": result.kurtosis,
                }
            )
    return pd.DataFrame(records)


def compare_adjacent_rungs(
    streams: dict[str, pd.Series], n_resamples: int, seed: int
) -> pd.DataFrame:
    """Paired bootstrap CI and Ledoit-Wolf test for each adjacent ladder pair.

    "Adjacent" follows the plan's rule that a rung is kept only if it beats the
    rung below it. Both tests run on the same pair of aligned streams, so where
    they disagree that disagreement is about the inference, not the data.
    """
    present = [arm for arm in LADDER_ORDER if arm in streams]
    records = []
    for lower, upper in zip(present, present[1:]):
        a, b = align_pair(streams[upper], streams[lower])
        if a.size < 30:
            logger.warning("  %s vs %s: only %d shared days, skipped", upper, lower, a.size)
            continue

        interval = paired_sharpe_difference_interval(
            a, b, n_resamples=n_resamples, seed=seed
        )
        test = ledoit_wolf_sharpe_test(a, b)
        records.append(
            {
                "arm": upper,
                "versus": lower,
                "difference": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "ci_excludes_zero": (interval.low > 0) or (interval.high < 0),
                "lw_statistic": test.statistic,
                "lw_p_value": test.p_value,
                "hac_lag": test.hac_lag,
                "n_days": test.n_observations,
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(
    universe: str,
    populations: list[TrialPopulation],
    deflation: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    logger.info("")
    logger.info("=== %s: the trial populations ===", universe)
    for population in populations:
        logger.info(
            "  %-9s N=%d (pre-registered)  sigma=%.3f over %d trials  ->  benchmark=%.3f",
            population.name,
            population.n_trials,
            population.dispersion,
            population.n_trials_in_dispersion,
            population.benchmark_sharpe,
        )
        logger.info("            %s", population.description)

    logger.info("")
    logger.info("=== deflated Sharpe by arm ===")
    wide = deflation.pivot(index="arm", columns="estimator", values=["sharpe", "benchmark", "dsr"])
    logger.info("  (baselines marked [fixed] were never searched, so they are not deflated)")
    for arm in deflation["arm"].drop_duplicates():
        sharpe = wide.loc[arm, ("sharpe", "pooled")]
        tag = "       " if arm not in PRE_SPECIFIED_ARMS else "[fixed]"
        line = f"  {arm:20s} {tag} Sharpe {sharpe:+.3f} |"
        for estimator in ("pooled", "coherent"):
            dsr = wide.loc[arm, ("dsr", estimator)]
            bar = wide.loc[arm, ("benchmark", estimator)]
            verdict = "PASS" if dsr > 0.95 else "----"
            line += f"  {estimator}: bar {bar:.3f} DSR {dsr:.3f} {verdict} |"
        logger.info(line)

    if comparisons.empty:
        return
    logger.info("")
    logger.info("=== adjacent-rung differences (paired bootstrap + Ledoit-Wolf) ===")
    for row in comparisons.itertuples():
        flag = "*" if row.ci_excludes_zero or row.lw_p_value < 0.05 else " "
        logger.info(
            "  %s%-16s vs %-16s  d=%+.3f  CI [%+.3f, %+.3f]  LW p=%.3f (lag %d)",
            flag, row.arm, row.versus, row.difference, row.ci_low, row.ci_high,
            row.lw_p_value, row.hac_lag,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT),
        help="Checkout holding results/ and data/ (defaults to this script's repo).",
    )
    parser.add_argument(
        "--source", default="walkforward", choices=["walkforward", "rung3"],
        help="Which stream set to score. walkforward is the P9 source of record.",
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="Seed-0 convention.")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    registry = pd.read_csv(data_root / "results" / "registry.csv", dtype=str)

    populations = build_populations(registry, args.universe)
    streams = load_excess_streams(data_root, args.universe, args.source)
    logger.info("Loaded %d arms for %s", len(streams), args.universe)

    deflation = deflate_every_arm(streams, populations)
    comparisons = compare_adjacent_rungs(streams, args.n_resamples, args.seed)
    report(args.universe, populations, deflation, comparisons)

    out_dir = data_root / "results" / "runs" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    deflation.to_csv(out_dir / f"deflated_sharpe_{args.universe}.csv", index=False)
    comparisons.to_csv(out_dir / f"adjacent_rungs_{args.universe}.csv", index=False)
    atomic_write_text(
        out_dir / f"trial_populations_{args.universe}.json",
        json.dumps([asdict(p) for p in populations], indent=2),
    )
    logger.info("")
    logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
