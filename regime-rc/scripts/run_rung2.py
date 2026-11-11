"""Rung 2: train the mixture of experts and judge gate G2.

    python scripts/run_rung2.py --universe etf10
    python scripts/run_rung2.py --universe etf10

What gate G2 asks (the plan's P6 exit criteria)
-----------------------------------------------
1. **Every expert is alive**: each holds at least 5 percent of the total gate
   mass. A starved run is retrained once at the escalated balance weight
   (Table 3's governance) before the verdict is taken.
2. **The specialisation question gets an unambiguous answer**: either the
   experts differentiate (largest pairwise correlation between their weight
   schedules below 0.9), or the documented negative finding - "the market's
   decision space is effectively one regime at this K" - with a K-reduction
   decision recorded. The Specification's own Section 8.7 makes the negative
   branch a legitimate, publishable outcome; hiding it would be the bug.

Artifacts
---------
* registry rows per (universe, seed, split), rung "2", arm ``moe_k<K>``,
  with the gate-entropy and importance-share columns filled;
* ``results/runs/rung2/`` - occupancy timelines, expert weights, loss curves,
  and the machine-readable ``g2_report.json``;
* figures F6 (regime-occupancy timeline) and F7 (the expert dashboard).
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.experts import count_parameters  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import (  # noqa: E402
    MixtureTrainConfig,
    MixtureTrainResult,
    evaluate_on_pack,
    mixture_diagnostics,
    train_mixture,
)
from rrc.utils.io import load_yaml  # noqa: E402

# Reuse rung 1's state-cache locator rather than duplicating it.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("rung2")

EXPERT_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def build_train_config(config: dict, lambda_balance: float, seed: int) -> MixtureTrainConfig:
    """Assemble one run's training configuration from the YAML."""
    return MixtureTrainConfig(
        n_experts=int(config["mixture"]["n_experts"]),
        learning_rate=float(config["train"]["learning_rate"]),
        max_epochs=int(config["train"]["max_epochs"]),
        patience=int(config["train"]["patience"]),
        gradient_clip_norm=float(config["train"]["gradient_clip_norm"]),
        lambda_cvar=float(config["loss"]["lambda_cvar"]),
        lambda_balance=lambda_balance,
        gate_temperature=float(config["mixture"]["gate_temperature"]),
        gate_noise_initial=float(config["mixture"]["gate_noise_initial"]),
        seed=seed,
    )


def train_one_seed(
    packs: dict, config: dict, seed: int
) -> tuple[MixtureTrainResult, dict, bool]:
    """Train one seed, escalating the balance weight once if any expert starves.

    Aliveness is judged on the TRAINING window's noise-free memberships - the
    span the balance loss actually governed. Returns the (possibly escalated)
    result, its training-window diagnostics, and whether escalation happened.
    """
    threshold = float(config["starvation_threshold"])

    result = train_mixture(
        packs["train"], packs["val"],
        build_train_config(config, float(config["loss"]["lambda_balance"]), seed),
    )
    diagnostics = mixture_diagnostics(result.model, packs["train"])

    if min(diagnostics["importance_shares"]) >= threshold:
        return result, diagnostics, False

    logger.warning(
        "  seed %d: expert starvation (min share %.3f < %.2f) - retraining at "
        "the escalated balance weight %.2f (Table 3 governance).",
        seed, min(diagnostics["importance_shares"]), threshold,
        float(config["loss"]["lambda_balance_escalated"]),
    )
    result = train_mixture(
        packs["train"], packs["val"],
        build_train_config(
            config, float(config["loss"]["lambda_balance_escalated"]), seed
        ),
    )
    diagnostics = mixture_diagnostics(result.model, packs["train"])
    return result, diagnostics, True


def run_universe(universe: str, config: dict, registry_path: Path) -> dict:
    """Train all seeds on fold F01, write registry rows, save artifacts."""
    logger.info("=== %s : rung 2, K = %d ===", universe, config["mixture"]["n_experts"])
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    states_dir = PROJECT_ROOT / config["paths"]["states_dir"]
    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_universe_artifacts(processed_dir, universe)
    cached_states = find_cached_states(states_dir, universe)
    fold = folds_for(universe)[0]
    packs = build_fold_packs(artifacts, cached_states, fold)

    n_experts = int(config["mixture"]["n_experts"])
    arm = f"{config['arm_prefix']}_k{n_experts}"
    seeds = [int(s) for s in config["seeds"]]

    per_seed_summaries = []
    seed_zero_bundle = None

    for seed in seeds:
        result, train_diagnostics, escalated = train_one_seed(packs, config, seed)

        identity = {
            "universe": universe, "rung": "2", "arm": arm,
            "mixture": {k: config["mixture"][k] for k in sorted(config["mixture"])},
            "train": {k: config["train"][k] for k in sorted(config["train"])},
            "lambda_cvar": float(config["loss"]["lambda_cvar"]),
            "lambda_balance": result.config.lambda_balance,  # escalation-aware
            "reservoir_cfg_hash": cached_states.cfg_hash,
        }
        arm_hash = config_hash(identity)
        n_params = count_parameters(result.model)

        for split in ("train", "val", "test"):
            evaluation = evaluate_on_pack(result.model, packs[split])
            split_diagnostics = mixture_diagnostics(result.model, packs[split])
            append_row(
                RegistryRow(
                    universe=universe, rung="2", arm=arm, seed=seed,
                    fold=fold.name, split=split, n_params=n_params,
                    cfg_hash=arm_hash,
                    mean_gate_entropy=split_diagnostics["gate_entropy"],
                    importance_shares=json.dumps(
                        [round(s, 4) for s in split_diagnostics["importance_shares"]]
                    ),
                    note=(
                        f"K={n_experts}; lambda_balance={result.config.lambda_balance}"
                        f"{' (escalated)' if escalated else ''}; "
                        f"best_epoch={result.best_epoch}; "
                        f"max_pair_corr={train_diagnostics['max_pairwise_correlation']:.3f}"
                    ),
                    **evaluation["metrics"],
                ),
                registry_path,
            )

        per_seed_summaries.append(
            {
                "seed": seed,
                "escalated": escalated,
                "val_sharpe": result.best_val_sharpe,
                "min_share": min(train_diagnostics["importance_shares"]),
                "shares": train_diagnostics["importance_shares"],
                "max_pairwise_correlation": train_diagnostics["max_pairwise_correlation"],
                "gate_entropy": train_diagnostics["gate_entropy"],
                "standalone": train_diagnostics["standalone"],
            }
        )
        if seed == seeds[0]:
            seed_zero_bundle = (result, train_diagnostics)

    _save_artifacts(universe, packs, seed_zero_bundle, runs_dir)
    _plot_f6_occupancy(universe, packs, seed_zero_bundle, fold)
    _plot_f7_dashboard(
        universe, packs, seed_zero_bundle, per_seed_summaries, artifacts.columns
    )

    return _judge_universe(universe, per_seed_summaries, config)


def _judge_universe(universe: str, per_seed: list[dict], config: dict) -> dict:
    """The G2 verdict for one universe, honest in both directions."""
    threshold = float(config["starvation_threshold"])
    correlation_line = float(config["specialisation_correlation_threshold"])

    all_alive = all(entry["min_share"] >= threshold for entry in per_seed)
    median_max_correlation = float(
        np.median([entry["max_pairwise_correlation"] for entry in per_seed])
    )
    specialised = median_max_correlation < correlation_line

    if specialised:
        branch = (
            f"SPECIALISED: median max pairwise correlation "
            f"{median_max_correlation:.3f} < {correlation_line}"
        )
    else:
        branch = (
            f"NEGATIVE FINDING: experts converged (median max pairwise "
            f"correlation {median_max_correlation:.3f} >= {correlation_line}) - "
            f"the market's decision space at K={config['mixture']['n_experts']} "
            "is effectively fewer regimes; record a K-reduction decision "
            "(Spec 8.7's honest branch)."
        )

    val_sharpes = [entry["val_sharpe"] for entry in per_seed]
    logger.info(
        "  verdict: alive=%s | %s | val Sharpe %.3f +/- %.3f | entropy %.3f",
        all_alive, branch.split(":")[0],
        float(np.mean(val_sharpes)), float(np.std(val_sharpes)),
        float(np.mean([entry["gate_entropy"] for entry in per_seed])),
    )
    return {
        "per_seed": per_seed,
        "all_alive": all_alive,
        "median_max_correlation": median_max_correlation,
        "specialised": bool(specialised),
        "branch": branch,
        "val_sharpe_mean": float(np.mean(val_sharpes)),
        "val_sharpe_sd": float(np.std(val_sharpes)),
    }


# --------------------------------------------------------------------------- #
# Artifacts and figures
# --------------------------------------------------------------------------- #
def _save_artifacts(universe, packs, seed_zero_bundle, runs_dir: Path) -> None:
    """Persist the seed-0 routing internals for the F6/F7 figure builders."""
    result, _ = seed_zero_bundle

    occupancy_frames = []
    for split in ("train", "val", "test"):
        diagnostics = mixture_diagnostics(result.model, packs[split])
        frame = diagnostics["memberships"].copy()
        frame["split"] = split
        occupancy_frames.append(frame)
    occupancy = pd.concat(occupancy_frames)
    occupancy.to_parquet(runs_dir / f"occupancy_{universe}.parquet")

    train_diagnostics = mixture_diagnostics(result.model, packs["train"])
    np.savez_compressed(
        runs_dir / f"expert_weights_{universe}.npz",
        expert_weights=train_diagnostics["expert_weights"],
    )
    result.history.to_csv(runs_dir / f"history_{universe}_seed0.csv", index=False)


def _plot_f6_occupancy(universe, packs, seed_zero_bundle, fold) -> None:
    """F6 - the regime-occupancy timeline: who is in charge, when."""
    result, _ = seed_zero_bundle
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ("train", "val", "test"):
        diagnostics = mixture_diagnostics(result.model, packs[split])
        frames.append(diagnostics["memberships"])
    occupancy = pd.concat(frames)
    n_experts = occupancy.shape[1]

    figure, axis = plt.subplots(figsize=(13, 4.6))
    axis.stackplot(
        occupancy.index,
        [occupancy.iloc[:, k].to_numpy() for k in range(n_experts)],
        labels=[f"expert {k}" for k in range(n_experts)],
        colors=EXPERT_COLOURS[:n_experts],
        alpha=0.85,
    )
    for boundary_year in (fold.val_year, fold.test_year):
        axis.axvline(pd.Timestamp(year=boundary_year, month=1, day=1),
                     color="black", lw=1.0, ls="--")
    axis.set_ylim(0, 1)
    axis.set_ylabel("gate membership pi")
    axis.set_title(
        f"F6 - {universe} regime occupancy (seed 0, K={n_experts}): gate mass "
        f"per expert over train | val {fold.val_year} | test {fold.test_year}",
        fontsize=10, loc="left",
    )
    axis.legend(ncol=n_experts, fontsize=8, frameon=False, loc="upper left")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures_dir / f"F6_occupancy_{universe}.png", dpi=170,
                   bbox_inches="tight")
    plt.close(figure)


def _plot_f7_dashboard(
    universe, packs, seed_zero_bundle, per_seed_summaries, asset_names
) -> None:
    """F7 - the expert dashboard: what each expert holds, and how distinct."""
    result, train_diagnostics = seed_zero_bundle
    figures_dir = PROJECT_ROOT / "results" / "figures"

    expert_weights = train_diagnostics["expert_weights"]  # (J, K, N)
    n_signals, n_experts, n_assets = expert_weights.shape

    figure = plt.figure(figsize=(13, 2.1 * n_experts + 4.4))
    grid = figure.add_gridspec(n_experts + 1, 2, width_ratios=[2.4, 1.0])

    for k in range(n_experts):
        axis = figure.add_subplot(grid[k, :])
        image = axis.imshow(
            expert_weights[:, k].T, aspect="auto", cmap="viridis",
            vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        axis.set_ylabel(f"expert {k}\nasset", fontsize=8)
        axis.set_yticks(range(n_assets))
        axis.set_yticklabels(asset_names, fontsize=6)
        if k == 0:
            axis.set_title(
                f"F7 - {universe} expert dashboard (seed 0): each expert's "
                "proposed weights over the training window",
                fontsize=10, loc="left",
            )
        if k == n_experts - 1:
            axis.set_xlabel("signal day (training window)", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.8, label="weight")

    # Pairwise-correlation matrix.
    axis = figure.add_subplot(grid[n_experts, 0])
    correlation = np.eye(n_experts)
    for pair, value in train_diagnostics["pairwise_weight_correlations"].items():
        first, second = (int(i) for i in pair.split("-"))
        correlation[first, second] = correlation[second, first] = value
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    for row in range(n_experts):
        for column in range(n_experts):
            axis.text(column, row, f"{correlation[row, column]:.2f}",
                      ha="center", va="center", fontsize=8,
                      color="white" if abs(correlation[row, column]) > 0.6 else "black")
    axis.set_xticks(range(n_experts)); axis.set_yticks(range(n_experts))
    axis.set_title("pairwise weight correlation\n(G2 line: below 0.9)", fontsize=9)
    figure.colorbar(image, ax=axis, shrink=0.8)

    # Importance shares across seeds.
    axis = figure.add_subplot(grid[n_experts, 1])
    for entry in per_seed_summaries:
        axis.plot(range(n_experts), entry["shares"], "o-", lw=0.9, alpha=0.7,
                  label=f"seed {entry['seed']}")
    axis.axhline(0.05, color="red", ls="--", lw=1.0, label="aliveness line")
    axis.set_xticks(range(n_experts))
    axis.set_xlabel("expert"); axis.set_ylabel("importance share")
    axis.set_title("gate mass per expert, all seeds", fontsize=9)
    axis.legend(fontsize=6, frameon=False)
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(figures_dir / f"F7_experts_{universe}.png", dpi=170,
                   bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rung 2 mixture trainings and gate G2.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rung2.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars "
             "(etf10 from 2006, sector11 from 1999), so they are run and "
             "reported independently and never aggregated.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    registry_path = PROJECT_ROOT / "results" / "registry.csv"
    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    universes = [args.universe]

    report: dict = {"n_experts": int(config["mixture"]["n_experts"]), "universes": {}}
    for universe in universes:
        report["universes"][universe] = run_universe(universe, config, registry_path)

    # G2 passes when every expert is alive everywhere and the specialisation
    # question has a documented answer - which, by construction, it now has in
    # either direction. Aliveness is the only hard failure.
    all_alive = all(entry["all_alive"] for entry in report["universes"].values())
    report["g2_pass"] = bool(all_alive)

    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / "g2_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info("=============== GATE G2 ===============")
    for universe, entry in report["universes"].items():
        logger.info("  %-9s alive=%s | %s", universe, entry["all_alive"], entry["branch"])
    logger.info("  overall: %s", "G2 PASS" if report["g2_pass"] else "G2 FAIL (starvation)")
    logger.info("  report: %s", report_path)
    return 0 if report["g2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
