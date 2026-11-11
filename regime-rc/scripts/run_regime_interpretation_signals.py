"""What do the learned regimes track when the gate ALSO sees the market signals?

    python scripts/run_regime_interpretation_signals.py --universe sector11

Companion to ``run_regime_interpretation.py``. That script interpreted a gate
that reads ONLY the reservoir state. Here the gate additionally reads the five
market signals - it is the 3a routing, ``[state ; signals] -> gate`` - so it is
handed direct access to VIX and the other macro signals: the even, slow,
volatility-carrying inputs an odd, ~1-week-memory reservoir state structurally
cannot construct on its own.

Everything else is held identical to the state-only run: the same folds, the
same K = 3, the same training loss, the SAME HMM and VIX comparators, and the
same eta-squared / AMI / profile summaries. The EXPERTS still read state only
(``signals_to_experts=False``), so the only thing that changes between the two
learned arms is what the GATE can see. That isolates one question:

    does the gate track direction because it CHOSE to (a finding about the
    decision objective), or because it could not see risk (a finding about the
    architecture)?

The pre-registered prediction, stated before running, in the spirit of the
project's other pre-registrations: five signal dimensions against 300 state
dimensions, so unless training lifts the signal weights a great deal, recent
direction should stay the gate's leading axis and realised-volatility eta
squared should rise off the floor but fall well short of the HMM's ~44 %.

Why both arms in one run
------------------------
Training the state-only gate here as well - rather than reading the old CSV -
means the baseline and the new arm share this run's exact folds, seeds and code
path, so the side-by-side difference cannot be an artefact of anything drifting
between two sessions. It also reproduces the published state-only numbers as a
sanity check that nothing regressed.

Output
------
Writes to ``results/runs/regimes_signals/``. The state-only run's directory
``results/runs/regimes/`` is NEVER written here, so the two sit side by side.

Caveats carried over from the state-only run
--------------------------------------------
* K = 3, while the frozen sector11 configuration is K = 1 (gate inert). We are
  interpreting the regime structure of a configuration that was not selected -
  stated, not hidden.
* eta squared and AMI are permutation-invariant, so they are averaged across
  seeds. A regime PROFILE is not (the index k is arbitrary per seed), so
  profiles are reported at seed 0 only and labelled as such.
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
from rrc.models.gating import MixtureOfExperts, SignalRoutedMixture  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402

# Reuse the state-descriptors and summaries VERBATIM from the state-only script,
# so the two runs cannot diverge on how the market is described or how eta
# squared is computed - only on what the gate is allowed to see.
from run_regime_interpretation import market_state, regime_profile, separation  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402
from run_rung3 import statistical_memberships  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("regimes-signals")

N_EXPERTS = 3          # the gate is inert at K = 1; see the module docstring
CASH = "CASH"

# The two learned gates compared here. Both keep the experts on state only.
LEARNED_ARMS = ("learned_state", "learned_signal")


def _build_learned(arm: str, n_state: int, n_assets: int, n_signals: int):
    """Model factory for a learned arm, matching run_rung3's construction."""
    if arm == "learned_state":
        return lambda: MixtureOfExperts(n_state, n_assets, N_EXPERTS)
    # learned_signal == rung 3a: signals join the gate, experts stay on state.
    return lambda: SignalRoutedMixture(
        n_state, n_assets, N_EXPERTS, n_signals,
        gate_temperature=1.0,
        signals_to_gate=True, signals_to_experts=False,
    )


def collect(universe, fold_names, seeds, config, data_root):
    """Per-fold test-segment labels from every scheme, plus the market state.

    Returns
    -------
    state : DataFrame
        The seed-independent market description at every concatenated signal day.
    learned : dict[arm][seed] -> np.ndarray
        Hard labels for each learned arm at each head seed.
    stat : dict[name] -> np.ndarray
        Hard labels for the HMM and VIX gates (seed-independent: the HMM's
        random_state is fixed and the VIX terciles are deterministic).
    """
    processed_dir = data_root / "data" / "processed"
    artifacts = load_universe_artifacts(processed_dir, universe)
    states_cache = find_cached_states(data_root / "data" / "states", universe)
    folds = [f for f in folds_for(universe) if f.name in fold_names]

    learned = {arm: {s: [] for s in seeds} for arm in LEARNED_ARMS}
    stat = {"hmm": [], "vix": []}
    state_blocks = []

    for fold in folds:
        packs = build_fold_packs(artifacts, states_cache, fold)
        test = packs["test"]
        window_start = int(artifacts.dates.get_loc(test.dates[0]))
        signal_absolute = test.execution_positions - 1 + window_start

        n_state = packs["train"].states.shape[1]
        n_assets = packs["train"].logit_mask.shape[1]
        n_signals = packs["train"].signals.shape[1]

        for arm in LEARNED_ARMS:
            factory = _build_learned(arm, n_state, n_assets, n_signals)
            for seed in seeds:
                trained = train_mixture(
                    packs["train"], packs["val"],
                    MixtureTrainConfig(n_experts=N_EXPERTS, lambda_cvar=0.0,
                                       lambda_balance=0.01, lambda_turn=0.0,
                                       lambda_div=0.0, seed=seed),
                    model=factory,
                )
                with torch.no_grad():
                    memberships, _ = trained.model.route_pack(test, noise_scale=0.0)
                learned[arm][seed].append(hard_labels(memberships.numpy()))

        # The statistical gates, on the identical signal days - unchanged.
        for arm, key in (("hmm_gate", "hmm"), ("vix_gate", "vix")):
            per_split, _, _ = statistical_memberships(arm, artifacts, fold, packs, config)
            stat[key].append(hard_labels(per_split["test"]))

        state_blocks.append(market_state(artifacts, signal_absolute))
        logger.info("  %s: %d decisions", fold.name, len(signal_absolute))

    learned = {arm: {s: np.concatenate(v) for s, v in per_seed.items()}
               for arm, per_seed in learned.items()}
    stat = {k: np.concatenate(v) for k, v in stat.items()}
    return pd.concat(state_blocks, ignore_index=True), learned, stat


def _mean_separation(state, per_seed):
    """eta squared averaged across seeds (permutation-invariant, so valid)."""
    frames = [separation(state, labels) for labels in per_seed.values()]
    return pd.concat(frames, axis=1).mean(axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="What market state does the gate track once it can see the signals?")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--folds", default="")
    parser.add_argument("--seeds", default="0,1,2",
                        help="Head seeds; eta squared / AMI are averaged over them.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "comparison.yaml"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    config = load_yaml(Path(args.config))
    seeds = [int(s) for s in args.seeds.split(",")]
    fold_names = (args.folds.split(",") if args.folds
                  else [f.name for f in folds_for(args.universe)])

    logger.info("=== %s: gate WITH signals vs gate WITHOUT, over %d folds, seeds %s ===",
                args.universe, len(fold_names), seeds)
    state, learned, stat = collect(args.universe, fold_names, seeds, config, data_root)

    seed0 = seeds[0]
    # One label series per scheme at seed 0, for the descriptive tables below.
    labels_seed0 = {arm: learned[arm][seed0] for arm in LEARNED_ARMS} | stat

    logger.info("")
    logger.info("=== how each scheme labels the same weeks (seed %d) ===", seed0)
    for name, series in labels_seed0.items():
        stats = dwell_and_transition(series)
        occupancy = np.bincount(series) / len(series)
        logger.info("  %-14s %d regimes | mean dwell %5.1f wks | switches %4.1f%% of weeks"
                    " | occupancy %s", name, len(np.unique(series)),
                    stats["mean_dwell"], 100 * stats["transition_rate"],
                    " ".join(f"{o:.2f}" for o in occupancy))

    logger.info("")
    logger.info("=== agreement between schemes (0 = chance, 1 = identical) ===")
    # Learned-vs-learned and learned-vs-statistical are averaged over seeds;
    # for a learned-vs-learned pair the SAME seed is used on both sides.
    def ami_mean(a_series_by_seed, b_series_by_seed_or_fixed, fixed=False):
        vals = []
        for s in seeds:
            b = b_series_by_seed_or_fixed if fixed else b_series_by_seed_or_fixed[s]
            vals.append(label_agreement(a_series_by_seed[s], b)["ami"])
        return float(np.mean(vals))

    pairs_learned_stat = [("learned_state", "hmm"), ("learned_state", "vix"),
                          ("learned_signal", "hmm"), ("learned_signal", "vix")]
    for a, b in pairs_learned_stat:
        logger.info("  %-14s vs %-8s  AMI %+.3f  (mean over %d seeds)",
                    a, b, ami_mean(learned[a], stat[b], fixed=True), len(seeds))
    logger.info("  %-14s vs %-8s  AMI %+.3f  (how much signals change the carving)",
                "learned_state", "learned_signal",
                ami_mean(learned["learned_state"], learned["learned_signal"]))
    logger.info("  %-14s vs %-8s  AMI %+.3f  (the positive control)",
                "hmm", "vix", label_agreement(stat["hmm"], stat["vix"])["ami"])

    logger.info("")
    logger.info("=== SEPARATION: fraction of each variable explained, eta squared %% ===")
    sep = pd.DataFrame({
        "learned_state":  _mean_separation(state, learned["learned_state"]),
        "learned_signal": _mean_separation(state, learned["learned_signal"]),
        "hmm":  separation(state, stat["hmm"]),
        "vix":  separation(state, stat["vix"]),
    })
    sep["signal_minus_state"] = sep["learned_signal"] - sep["learned_state"]
    for line in (sep * 100).round(1).to_string().splitlines():
        logger.info("  %s", line)
    logger.info("  (learned_* averaged over %d seeds; hmm/vix seed-independent)", len(seeds))

    logger.info("")
    for name, series in labels_seed0.items():
        logger.info("=== PROFILE: %s - standardised mean per regime (seed %d) ===", name, seed0)
        for line in regime_profile(state, series).round(2).to_string().splitlines():
            logger.info("  %s", line)
        logger.info("")

    out_dir = data_root / "results" / "runs" / "regimes_signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    sep.to_csv(out_dir / f"separation_{args.universe}.csv")
    for name, series in labels_seed0.items():
        regime_profile(state, series).to_csv(out_dir / f"profile_{name}_{args.universe}.csv")
    state.assign(**{f"label_{k}": v for k, v in labels_seed0.items()}).to_parquet(
        out_dir / f"state_and_labels_{args.universe}.parquet"
    )
    logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
