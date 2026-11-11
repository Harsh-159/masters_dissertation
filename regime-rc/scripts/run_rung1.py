"""Rung 1: train the single decision-trained head and judge gate G1.

    python scripts/run_rung1.py --universe etf10
    python scripts/run_rung1.py --universe etf10 --skip-c2   # trainings only

What gate G1 asks (the plan's P5 exit criteria)
-----------------------------------------------
1. **Gradients are right**: finite differences match autograd through the whole
   head -> softmax -> ledger -> loss chain, on the synthetic toy.
2. **The mechanism can learn at all**: the head overfits the planted toy to an
   absurd in-sample Sharpe. If it cannot crack a toy, nothing downstream means
   anything.
3. **Real training is stable**: fold F01 trains with no NaNs,
   a loss that broadly falls, and a validation Sharpe that is not degenerate.
4. **No leakage**: the C2 shuffle-future test - shuffle every day after the
   training cut, rebuild the reservoir states from the shuffled data exactly as
   the real pipeline would, and the trained head's performance on that shuffled
   world must collapse to noise. Skill that survives shuffling is leakage, with
   certainty.

Everything is recorded: one registry row per (universe, lambda, seed, split),
loss curves per universe, and a machine-readable G1 report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.data.loaders import rolling_zscore  # noqa: E402
from rrc.eval.causality import collapse_verdict, permute_rows_after  # noqa: E402
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for, truncate_before_holdout  # noqa: E402
from rrc.models.experts import LinearPortfolioHead, count_parameters  # noqa: E402
from rrc.models.losses import decision_loss  # noqa: E402
from rrc.optim.fold_data import (  # noqa: E402
    UniverseArtifacts,
    build_fold_packs,
    load_universe_artifacts,
)
from rrc.optim.toy import finite_difference_check, make_toy_fold  # noqa: E402
from rrc.optim.train_loop import (  # noqa: E402
    TrainConfig,
    TrainResult,
    conditioning_edge,
    evaluate_on_pack,
    portfolio_returns,
    train_single_head,
)
from rrc.reservoir.cache import CachedStates, build_usable_mask, load_states  # noqa: E402
from rrc.reservoir.scr import build_scr, run_reservoir  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("rung1")

SEED_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


# --------------------------------------------------------------------------- #
# Step 1 - the toy protocol
# --------------------------------------------------------------------------- #
def run_toy_protocol(config: dict) -> dict:
    """Finite-difference check and toy overfit; the day-8 plumbing test."""
    logger.info("=== toy protocol ===")
    toy = make_toy_fold(seed=0)
    torch.manual_seed(0)
    head = LinearPortfolioHead(toy.pack.states.shape[1], toy.pack.logit_mask.shape[1])

    def loss_now() -> torch.Tensor:
        returns = portfolio_returns(head, toy.pack)
        return decision_loss(returns, toy.pack.risk_free, lambda_cvar=1.0).total

    loss_now().backward()
    worst_error, largest_gradient = finite_difference_check(
        loss_now, head.linear.weight, n_samples=15, step=1e-6, seed=0
    )
    tolerance = float(config["toy"]["finite_difference_tolerance"])
    gradients_ok = worst_error < tolerance
    logger.info(
        "  finite differences: worst |autograd - numeric| = %.2e "
        "(tolerance %.0e, largest gradient %.2e) -> %s",
        worst_error, tolerance, largest_gradient, "PASS" if gradients_ok else "FAIL",
    )

    overfit_config = TrainConfig(
        max_epochs=int(config["toy"]["overfit_epochs"]),
        patience=int(config["toy"]["overfit_epochs"]),  # never stop early
        lambda_cvar=0.0,
        seed=0,
    )
    overfit = train_single_head(toy.pack, toy.pack, overfit_config)
    reached = float(overfit.history["train_sharpe_ann"].iloc[-1])
    threshold = float(config["toy"]["overfit_sharpe_threshold"])
    overfit_ok = reached > threshold
    logger.info(
        "  toy overfit: in-sample annualised Sharpe %.1f (threshold %.1f) -> %s",
        reached, threshold, "PASS" if overfit_ok else "FAIL",
    )

    return {
        "finite_difference_error": worst_error,
        "finite_difference_pass": bool(gradients_ok),
        "overfit_sharpe": reached,
        "overfit_pass": bool(overfit_ok),
    }


# --------------------------------------------------------------------------- #
# Step 2 - real trainings on fold F01
# --------------------------------------------------------------------------- #
def find_cached_states(states_dir: Path, universe: str) -> CachedStates:
    """Locate the SCR state cache for a universe's DEFAULT reservoir config.

    Before P8 there was exactly one SCR cache per universe. The P8
    hyperparameter search writes an extra cache per candidate reservoir config,
    so "exactly one" no longer holds. When several exist, the right one is the
    cache whose reservoir configuration matches ``configs/reservoir.yaml`` - the
    default the ladder rungs are built on. Matching is on the reservoir
    parameters only (not the calibrated ``input_scale``), which the continuous
    random search never reproduces, so the default is uniquely identifiable.
    """
    candidates = sorted(states_dir.glob(f"{universe}_scr_*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No SCR state cache for {universe}; run scripts/run_reservoir.py."
        )
    if len(candidates) == 1:
        return load_states(candidates[0])

    # Several caches can share the reservoir parameters and still differ - the
    # grid search varies the input-scale TARGET, which changes the calibrated
    # input_scale and hence the config hash, while leaving N_r, rho, alpha,
    # washout and bias untouched. Matching on those keys alone is therefore
    # ambiguous once a search has run.
    #
    # So the default cache is identified by CONSTRUCTION rather than by search:
    # rebuild it from configs/reservoir.yaml through the same calibrate-and-cache
    # path every other caller uses. It is a cache hit whenever the default states
    # already exist, and it can never resolve to a search candidate's states.
    from rrc.optim.hyper_search import build_states_for_config  # local: avoids a cycle

    config = load_yaml(PROJECT_ROOT / "configs" / "reservoir.yaml")
    reservoir, scaling = config["reservoir"], config["input_scaling"]
    return build_states_for_config(
        universe,
        {
            "n_reservoir": reservoir["n_reservoir"],
            "spectral_radius": reservoir["spectral_radius"],
            "leak_rate": reservoir["leak_rate"],
            "target_pre_activation_std": scaling["target_pre_activation_std"],
        },
        {
            "washout": reservoir["washout"],
            "bias_scale": reservoir["bias_scale"],
            "calibration_fraction": scaling["calibration_fraction"],
            "max_iterations": scaling["max_iterations"],
            "tolerance": scaling["tolerance"],
        },
        PROJECT_ROOT / config["paths"]["processed_dir"],
        Path(states_dir),
    )


def train_universe(
    universe: str, config: dict, registry_path: Path
) -> tuple[dict, UniverseArtifacts, CachedStates, dict[float, dict[int, TrainResult]]]:
    """Train the lambda grid x seeds on fold F01; write registry rows."""
    logger.info("=== %s : fold F01 trainings ===", universe)
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    states_dir = PROJECT_ROOT / config["paths"]["states_dir"]

    artifacts = load_universe_artifacts(processed_dir, universe)
    cached_states = find_cached_states(states_dir, universe)
    fold = folds_for(universe)[0]  # F01
    packs = build_fold_packs(artifacts, cached_states, fold)
    logger.info(
        "  train %s..%s (%d signals) | val %d | test %d",
        packs["train"].dates.min().date(), packs["train"].dates.max().date(),
        packs["train"].n_signals, fold.val_year, fold.test_year,
    )

    lambda_grid = [float(v) for v in config["loss"]["lambda_cvar_grid"]]
    seeds = [int(s) for s in config["seeds"]]

    results: dict[float, dict[int, TrainResult]] = {}
    stability_notes = []

    for lambda_cvar in lambda_grid:
        results[lambda_cvar] = {}
        identity = {
            "universe": universe,
            "rung": "1",
            "arm": config["arm"],
            "lambda_cvar": lambda_cvar,
            "train": {k: config["train"][k] for k in sorted(config["train"])},
            "reservoir_cfg_hash": cached_states.cfg_hash,
        }
        arm_hash = config_hash(identity)

        for seed in seeds:
            train_config = TrainConfig(
                learning_rate=float(config["train"]["learning_rate"]),
                max_epochs=int(config["train"]["max_epochs"]),
                patience=int(config["train"]["patience"]),
                gradient_clip_norm=float(config["train"]["gradient_clip_norm"]),
                lambda_cvar=lambda_cvar,
                seed=seed,
            )
            result = train_single_head(packs["train"], packs["val"], train_config)
            results[lambda_cvar][seed] = result

            n_params = count_parameters(result.head)
            for split in ("train", "val", "test"):
                evaluation = evaluate_on_pack(result.head, packs[split])
                append_row(
                    RegistryRow(
                        universe=universe, rung="1", arm=config["arm"],
                        seed=seed, fold=fold.name, split=split,
                        n_params=n_params, cfg_hash=arm_hash,
                        note=(
                            f"lambda_cvar={lambda_cvar}; best_epoch="
                            f"{result.best_epoch}; mean_cash="
                            f"{evaluation['mean_cash_weight']:.3f}"
                        ),
                        **evaluation["metrics"],
                    ),
                    registry_path,
                )

            history = result.history
            improved = float(history["loss"].iloc[-1]) < float(history["loss"].iloc[0])
            stability_notes.append(
                {
                    "lambda_cvar": lambda_cvar, "seed": seed,
                    "epochs": len(history), "best_epoch": result.best_epoch,
                    "loss_improved": improved,
                    "val_sharpe": result.best_val_sharpe,
                    "finite": bool(np.isfinite(history.to_numpy()).all()),
                }
            )

        seed_sharpes = [results[lambda_cvar][s].best_val_sharpe for s in seeds]
        logger.info(
            "  lambda_cvar=%.1f : val Sharpe %.3f +/- %.3f over %d seeds",
            lambda_cvar, float(np.mean(seed_sharpes)), float(np.std(seed_sharpes)),
            len(seeds),
        )

    summary = {
        "stability": stability_notes,
        "all_finite": all(n["finite"] for n in stability_notes),
        "all_improved": all(n["loss_improved"] for n in stability_notes),
        "val_sharpe_by_lambda": {
            str(lam): {
                "mean": float(np.mean([results[lam][s].best_val_sharpe for s in seeds])),
                "sd": float(np.std([results[lam][s].best_val_sharpe for s in seeds])),
            }
            for lam in lambda_grid
        },
    }
    return summary, artifacts, cached_states, results


# --------------------------------------------------------------------------- #
# Step 3 - the C2 shuffle-future test on the real pipeline
# --------------------------------------------------------------------------- #
def rebuild_states_on(
    log_returns: pd.DataFrame,
    reservoir_columns: list[str],
    template: CachedStates,
    n_days: int,
) -> CachedStates:
    """Re-run the frozen reservoir pipeline on a (possibly shuffled) world.

    Identical to the P3 build: z-score the log returns on the trailing window,
    drive the SCR from the first fully finite row, apply the same washout. The
    stored input scale is reused rather than recalibrated - it is a global
    reservoir hyperparameter shared across folds, exactly as deployed.
    """
    reservoir_config = template.reservoir_config
    inputs = rolling_zscore(
        log_returns[reservoir_columns], window=252, min_periods=126
    ).to_numpy()

    finite_rows = np.isfinite(inputs).all(axis=1)
    first_row = int(np.argmax(finite_rows))

    weights = build_scr(
        n_reservoir=int(reservoir_config["n_reservoir"]),
        n_inputs=len(reservoir_columns),
        spectral_radius=float(reservoir_config["spectral_radius"]),
        input_scale=float(reservoir_config["input_scale"]),
        leak_rate=float(reservoir_config["leak_rate"]),
        bias_scale=float(reservoir_config.get("bias_scale", 0.0)),
    )
    computed, _ = run_reservoir(weights, inputs[first_row:])

    states = np.full((n_days, weights.n_reservoir), np.nan, dtype=np.float32)
    states[first_row:] = computed.astype(np.float32)
    usable = build_usable_mask(n_days, first_row, int(reservoir_config["washout"]))

    return CachedStates(
        states=states,
        dates=log_returns.index,
        usable=usable,
        reservoir_config=dict(reservoir_config),
        cfg_hash="rebuilt-for-c2",
        kind="scr",
    )


def run_c2(
    universe: str,
    artifacts: UniverseArtifacts,
    cached_states: CachedStates,
    trained_head,
    config: dict,
) -> dict:
    """Shuffle the future, rebuild the world, and check performance collapses."""
    logger.info("=== %s : C2 shuffle-future ===", universe)
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    fold = folds_for(universe)[0]

    log_returns = truncate_before_holdout(
        pd.read_parquet(processed_dir / f"{universe}_returns_log.parquet")
    )
    reservoir_columns = list(
        pd.read_parquet(processed_dir / f"{universe}_reservoir_input.parquet").columns
    )
    risky_columns = [c for c in artifacts.returns_simple.columns if c != "CASH"]

    # The cut: the last day of the embargoed training window.
    packs = build_fold_packs(artifacts, cached_states, fold)
    cut_position = int(artifacts.dates.get_loc(packs["train"].dates[-1]))

    # The C2 quantity is the CONDITIONING EDGE - the head's Sharpe minus its
    # own static counterpart's, within the same world. Raw Sharpe would not
    # collapse for a mostly-static strategy (shuffling preserves drift), and
    # the etf10 head's defensive tilt is exactly such a strategy; the edge
    # isolates the part of performance that leakage could fake: timing.
    actual = conditioning_edge(trained_head, packs["val"])

    # Sanity first: rebuilding on the UNSHUFFLED data must reproduce the cached
    # states (float32 storage tolerance) - C5 in action on the deployed path.
    unshuffled = rebuild_states_on(
        log_returns, reservoir_columns, cached_states, len(artifacts.dates)
    )
    probe = np.flatnonzero(unshuffled.usable)[:100]
    if not np.allclose(
        unshuffled.states[probe],
        cached_states.states[: len(artifacts.dates)][probe],
        atol=1e-4,
    ):
        raise AssertionError(
            "Rebuilding states on unshuffled data does not reproduce the cache "
            "- the C2 rebuild path diverges from the deployed pipeline."
        )

    permuted_edges = []
    permuted_dynamic_sharpes = []
    for permutation_seed in range(int(config["c2"]["n_permutations"])):
        shuffled_log = permute_rows_after(
            log_returns, cut_position, permutation_seed, columns=reservoir_columns
        )
        shuffled_simple = permute_rows_after(
            artifacts.returns_simple, cut_position, permutation_seed,
            columns=risky_columns,
        )

        shuffled_states = rebuild_states_on(
            shuffled_log, reservoir_columns, cached_states, len(artifacts.dates)
        )
        shuffled_artifacts = UniverseArtifacts(
            universe=universe,
            returns_simple=shuffled_simple,
            ledger_panel=shuffled_simple.fillna(0.0),
            availability=artifacts.availability,
            risk_free=artifacts.risk_free,
            dates=artifacts.dates,
            columns=artifacts.columns,
            cash_index=artifacts.cash_index,
        )
        shuffled_val = build_fold_packs(shuffled_artifacts, shuffled_states, fold)["val"]
        shuffled_result = conditioning_edge(trained_head, shuffled_val)
        permuted_edges.append(shuffled_result["edge"])
        permuted_dynamic_sharpes.append(shuffled_result["dynamic"])

    verdict = collapse_verdict(
        actual["edge"], permuted_edges, float(config["c2"]["noise_band"])
    )
    logger.info(
        "  actual: dynamic %.3f | static %.3f | edge %.3f",
        actual["dynamic"], actual["static"], actual["edge"],
    )
    logger.info(
        "  shuffled: mean edge %.3f (sd %.3f, n=%d; mean raw Sharpe %.3f - "
        "drift, not information) | band %.2f -> %s",
        verdict.mean_permuted, float(np.std(verdict.permuted_sharpes)),
        len(verdict.permuted_sharpes), float(np.mean(permuted_dynamic_sharpes)),
        verdict.noise_band,
        "COLLAPSED (clean)" if verdict.collapsed else "SURVIVED (LEAK!)",
    )
    return {
        "actual_dynamic_sharpe": actual["dynamic"],
        "actual_static_sharpe": actual["static"],
        "actual_edge": actual["edge"],
        "permuted_edges": list(verdict.permuted_sharpes),
        "permuted_mean_raw_sharpe": float(np.mean(permuted_dynamic_sharpes)),
        "mean_permuted_edge": verdict.mean_permuted,
        "collapsed": verdict.collapsed,
    }


# --------------------------------------------------------------------------- #
# Loss-curve figure
# --------------------------------------------------------------------------- #
def plot_loss_curves(
    universe: str,
    results: dict[float, dict[int, TrainResult]],
    runs_dir: Path,
) -> Path:
    """Validation (solid) and training (faint) Sharpe per epoch, per lambda."""
    lambda_values = sorted(results)
    figure, axes = plt.subplots(
        1, len(lambda_values), figsize=(5.2 * len(lambda_values), 4.2),
        sharey=True, squeeze=False,
    )

    for column, lambda_cvar in enumerate(lambda_values):
        axis = axes[0][column]
        for row, (seed, result) in enumerate(sorted(results[lambda_cvar].items())):
            colour = SEED_COLOURS[row % len(SEED_COLOURS)]
            axis.plot(
                result.history["epoch"], result.history["val_sharpe_ann"],
                color=colour, lw=1.2, label=f"seed {seed}",
            )
            axis.plot(
                result.history["epoch"], result.history["train_sharpe_ann"],
                color=colour, lw=0.7, alpha=0.35,
            )
            axis.axvline(result.best_epoch, color=colour, lw=0.6, ls=":", alpha=0.5)
        axis.set_title(f"lambda_cvar = {lambda_cvar}", fontsize=10)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("annualised excess Sharpe")
    axes[0][0].legend(fontsize=8, frameon=False)
    figure.suptitle(
        f"Rung 1 - {universe} fold F01: validation (solid) and train (faint) "
        "Sharpe by epoch; dotted line = epoch kept",
        fontsize=10, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"loss_curves_{universe}.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
# Entry point and the G1 verdict
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rung 1 trainings and gate G1.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rung1.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars "
             "(etf10 from 2006, sector11 from 1999), so they are run and "
             "reported independently and never aggregated.",
    )
    parser.add_argument("--skip-c2", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    registry_path = PROJECT_ROOT / "results" / "registry.csv"
    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    universes = [args.universe]

    report: dict = {"toy": run_toy_protocol(config), "universes": {}}

    for universe in universes:
        training_summary, artifacts, cached_states, results = train_universe(
            universe, config, registry_path
        )
        curve_path = plot_loss_curves(universe, results, runs_dir)
        logger.info("  loss curves: %s", curve_path.name)

        universe_report = {"training": training_summary}
        if not args.skip_c2:
            c2_head = results[float(config["c2"]["lambda_cvar"])][
                int(config["c2"]["seed_head"])
            ].head
            universe_report["c2"] = run_c2(
                universe, artifacts, cached_states, c2_head, config
            )
        report["universes"][universe] = universe_report

    # ---- the G1 verdict ---------------------------------------------------
    checks = {
        "toy_gradients": report["toy"]["finite_difference_pass"],
        "toy_overfit": report["toy"]["overfit_pass"],
    }
    for universe in universes:
        universe_report = report["universes"][universe]
        checks[f"{universe}_stable"] = (
            universe_report["training"]["all_finite"]
            and universe_report["training"]["all_improved"]
        )
        if "c2" in universe_report:
            checks[f"{universe}_c2_collapsed"] = universe_report["c2"]["collapsed"]

    report["g1_checks"] = checks
    report["g1_pass"] = all(checks.values())

    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / "g1_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info("=============== GATE G1 ===============")
    for check_name, passed in checks.items():
        logger.info("  %-28s %s", check_name, "PASS" if passed else "FAIL")
    logger.info("  overall: %s", "G1 PASS" if report["g1_pass"] else "G1 FAIL")
    logger.info("  report: %s", report_path)
    return 0 if report["g1_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
