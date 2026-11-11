"""The thesis week: rungs 3a/3b/3c and 4, and gate G3.

    python scripts/run_rung3.py --universe etf10

What runs (per universe, folds F01-F03, seeds 0-4)
--------------------------------------------------
=========  ==============  ====================================================
rung       arm             model
=========  ==============  ====================================================
2 (top-up) moe_k3          the rung-2 mixture on folds F02/F03, so every
                           comparison below has the same three folds
3a         sent_gate       signals -> the gate only
3b         sent_experts    signals -> the experts only
3c         sent_both       signals -> both (the ceiling reference, A8)
4          hmm_gate        filtered-HMM routing, experts retrained (R6)
4          vix_gate        VIX-tercile routing, experts retrained (R6)
=========  ==============  ====================================================

Every arm trains through the *same* loop, loss, schedule and seeds - the
equal-conditions rule as literal code reuse. Exact parameter counts are logged
per arm (A8's honesty requirement).

After training, the A10 agreement metrics are computed per fold between the
learned gate's regimes (rung 2, seed 0) and each statistical labelling:
AMI/ARI, dwell times, transition rates, occupancy-conditioned Sharpe, and the
decision-divergence D under identical experts. Figure F8 shows the agreement
bars; the headline table carries bootstrap CIs on the concatenated test years.

Gate G3 passes when the tables and metrics exist, complete, on all three folds
- the research question is then answerable in whichever direction the numbers
point, which is the plan's own criterion.
"""

from __future__ import annotations

import argparse
import dataclasses
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

from rrc.baselines.hmm_gate import (  # noqa: E402
    build_observations,
    fit_hmm_gate,
    memberships_at,
)
from rrc.baselines.vix_gate import memberships_at as vix_memberships_at  # noqa: E402
from rrc.baselines.vix_gate import vix_tercile_labels  # noqa: E402
from rrc.eval.agreement import (  # noqa: E402
    decision_divergence,
    dwell_and_transition,
    hard_labels,
    label_agreement,
    occupancy_conditioned_sharpe,
)
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for, train_dates  # noqa: E402
from rrc.eval.stats import sharpe_confidence_interval  # noqa: E402
from rrc.models.experts import count_parameters  # noqa: E402
from rrc.models.gating import (  # noqa: E402
    FixedRoutingMixture,
    MixtureOfExperts,
    SignalRoutedMixture,
)
from rrc.optim.fold_data import (  # noqa: E402
    UniverseArtifacts,
    build_fold_packs,
    load_universe_artifacts,
)
from rrc.optim.train_loop import (  # noqa: E402
    MixtureTrainConfig,
    evaluate_on_pack,
    mixture_diagnostics,
    train_mixture,
)
from rrc.utils.io import load_yaml  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("rung3")

# (rung, arm) -> how to build the model. K, widths etc. are filled at runtime.
LEARNED_ARMS = {
    ("2", "moe_k3"): dict(signals_to_gate=False, signals_to_experts=False),
    ("3a", "sent_gate"): dict(signals_to_gate=True, signals_to_experts=False),
    ("3b", "sent_experts"): dict(signals_to_gate=False, signals_to_experts=True),
    ("3c", "sent_both"): dict(signals_to_gate=True, signals_to_experts=True),
}
STATISTICAL_ARMS = ["hmm_gate", "vix_gate"]


def model_factory(rung: str, arm: str, pack, config: dict, hmm_states: int | None = None):
    """Build the right model for an arm, sized from the pack."""
    n_state = pack.states.shape[1]
    n_assets = pack.logit_mask.shape[1]
    n_experts = int(config["mixture"]["n_experts"])
    temperature = float(config["mixture"]["gate_temperature"])

    if (rung, arm) in LEARNED_ARMS:
        flags = LEARNED_ARMS[(rung, arm)]
        if not (flags["signals_to_gate"] or flags["signals_to_experts"]):
            return lambda: MixtureOfExperts(
                n_state, n_assets, n_experts, gate_temperature=temperature
            )
        n_signals = pack.signals.shape[1]
        return lambda: SignalRoutedMixture(
            n_state, n_assets, n_experts, n_signals,
            gate_temperature=temperature, **flags,
        )
    # Rung 4: no gate; K comes from the statistical labelling itself.
    return lambda: FixedRoutingMixture(n_state, n_assets, hmm_states or n_experts)


def statistical_memberships(
    arm: str,
    artifacts: UniverseArtifacts,
    fold,
    packs: dict,
    config: dict,
) -> tuple[dict, int, str]:
    """Fixed membership matrices per split for a rung-4 arm.

    Returns the per-split memberships, the state count actually used, and a
    provenance note for the registry.
    """
    if arm == "hmm_gate":
        observations = build_observations(
            artifacts.returns_simple[[c for c in artifacts.columns if c != "CASH"]],
            artifacts.availability,
            artifacts.risk_free,
        )
        gate = fit_hmm_gate(
            observations,
            train_dates(artifacts.dates, fold),
            n_states=int(config["hmm"]["n_states"]),
            random_state=int(config["hmm"]["random_state"]),
            n_iter=int(config["hmm"]["n_iter"]),
        )
        note = f"hmm states={gate.n_states}" + (
            f" (reduced from {gate.reduced_from})" if gate.reduced_from else ""
        )
        per_split = {}
        for split, pack in packs.items():
            window_start = int(artifacts.dates.get_loc(pack.dates[0]))
            signal_absolute = pack.execution_positions - 1 + window_start
            per_split[split] = memberships_at(gate, artifacts.dates, signal_absolute)
        return per_split, gate.n_states, note

    if arm == "vix_gate":
        raw_vix = pd.read_parquet(
            PROJECT_ROOT / config["paths"]["raw_dir"] / "fred_VIXCLS.parquet"
        )["VIXCLS"]
        labels = vix_tercile_labels(raw_vix, artifacts.dates)
        per_split = {}
        for split, pack in packs.items():
            window_start = int(artifacts.dates.get_loc(pack.dates[0]))
            signal_absolute = pack.execution_positions - 1 + window_start
            per_split[split] = vix_memberships_at(labels, signal_absolute)
        return per_split, 3, "vix terciles (causal, trailing 252d)"

    raise ValueError(f"Unknown statistical arm {arm!r}.")


def attach_fixed_memberships(packs: dict, per_split: dict) -> dict:
    """Return packs carrying the statistical gate's memberships."""
    return {
        split: dataclasses.replace(
            pack,
            fixed_memberships=torch.as_tensor(per_split[split], dtype=torch.float64),
        )
        for split, pack in packs.items()
    }


def run_arm(
    universe: str,
    rung: str,
    arm: str,
    fold,
    packs: dict,
    artifacts: UniverseArtifacts,
    cached_states,
    config: dict,
    registry_path: Path,
) -> dict:
    """Train all seeds of one (rung, arm, fold); write registry rows."""
    hmm_states_used = None
    note_suffix = ""

    if arm in STATISTICAL_ARMS:
        per_split, hmm_states_used, note_suffix = statistical_memberships(
            arm, artifacts, fold, packs, config
        )
        packs = attach_fixed_memberships(packs, per_split)

    factory = model_factory(rung, arm, packs["train"], config, hmm_states_used)

    identity = {
        "universe": universe, "rung": rung, "arm": arm,
        "mixture": {k: config["mixture"][k] for k in sorted(config["mixture"])},
        "train": {k: config["train"][k] for k in sorted(config["train"])},
        "loss": {k: config["loss"][k] for k in sorted(config["loss"])},
        "hmm": {k: config["hmm"][k] for k in sorted(config["hmm"])} if arm == "hmm_gate" else None,
        "reservoir_cfg_hash": cached_states.cfg_hash,
    }
    arm_hash = config_hash(identity)

    seed_results = []
    for seed in [int(s) for s in config["seeds"]]:
        train_config = MixtureTrainConfig(
            n_experts=int(config["mixture"]["n_experts"]),
            learning_rate=float(config["train"]["learning_rate"]),
            max_epochs=int(config["train"]["max_epochs"]),
            patience=int(config["train"]["patience"]),
            gradient_clip_norm=float(config["train"]["gradient_clip_norm"]),
            lambda_cvar=float(config["loss"]["lambda_cvar"]),
            lambda_balance=float(config["loss"]["lambda_balance"]),
            gate_temperature=float(config["mixture"]["gate_temperature"]),
            gate_noise_initial=float(config["mixture"]["gate_noise_initial"]),
            seed=seed,
        )
        result = train_mixture(packs["train"], packs["val"], train_config, model=factory)
        n_params = count_parameters(result.model)

        per_split_metrics = {}
        for split in ("train", "val", "test"):
            evaluation = evaluate_on_pack(result.model, packs[split])
            diagnostics = mixture_diagnostics(result.model, packs[split])
            per_split_metrics[split] = {
                "net_returns": evaluation["net_returns"],
                "metrics": evaluation["metrics"],
            }
            append_row(
                RegistryRow(
                    universe=universe, rung=rung, arm=arm, seed=seed,
                    fold=fold.name, split=split, n_params=n_params,
                    cfg_hash=arm_hash,
                    mean_gate_entropy=diagnostics["gate_entropy"],
                    importance_shares=json.dumps(
                        [round(s, 4) for s in diagnostics["importance_shares"]]
                    ),
                    note=f"{note_suffix}; best_epoch={result.best_epoch}".strip("; "),
                    **evaluation["metrics"],
                ),
                registry_path,
            )

        seed_results.append(
            {
                "seed": seed,
                "model": result.model,
                "val_sharpe": result.best_val_sharpe,
                "test_returns": per_split_metrics["test"]["net_returns"],
                "test_sharpe": per_split_metrics["test"]["metrics"]["excess_sharpe"],
                "n_params": n_params,
            }
        )

    val_sharpes = [r["val_sharpe"] for r in seed_results]
    logger.info(
        "  %-4s %-13s %s: val %.3f +/- %.3f | params %d %s",
        rung, arm, fold.name,
        float(np.mean(val_sharpes)), float(np.std(val_sharpes)),
        seed_results[0]["n_params"], note_suffix,
    )
    return {"per_seed": seed_results, "arm_hash": arm_hash, "note": note_suffix}


# --------------------------------------------------------------------------- #
# Agreement metrics (A10) and figure F8
# --------------------------------------------------------------------------- #
def agreement_for_fold(
    universe: str,
    fold,
    packs: dict,
    artifacts: UniverseArtifacts,
    learned_model,
    config: dict,
) -> dict:
    """AMI/ARI, dwell, divergence between learned and statistical regimes."""
    val_test = {"val": packs["val"], "test": packs["test"]}
    learned_memberships = {}
    learned_expert_weights = {}
    for split, pack in val_test.items():
        with torch.no_grad():
            memberships, expert_weights = learned_model.route_pack(pack, 0.0)
        learned_memberships[split] = memberships.numpy()
        learned_expert_weights[split] = expert_weights.numpy()

    results: dict = {}
    for arm in STATISTICAL_ARMS:
        per_split, _, _ = statistical_memberships(arm, artifacts, fold, val_test, config)
        entry: dict = {}
        for split in val_test:
            learned_hard = hard_labels(learned_memberships[split])
            statistical_hard = hard_labels(per_split[split])
            entry[split] = {
                **label_agreement(learned_hard, statistical_hard),
                "learned_dwell": dwell_and_transition(learned_hard),
                "statistical_dwell": dwell_and_transition(statistical_hard),
                "decision_divergence": decision_divergence(
                    learned_memberships[split], per_split[split],
                    learned_expert_weights[split],
                ),
            }
        results[arm] = entry

    # Occupancy-conditioned Sharpe of the learned system on the test year.
    test_pack = packs["test"]
    with torch.no_grad():
        test_returns = evaluate_on_pack(learned_model, test_pack)["net_returns"]
    results["occupancy_sharpe_test"] = occupancy_conditioned_sharpe(
        test_returns,
        pd.Series(test_pack.risk_free.numpy(), index=test_pack.dates),
        hard_labels(learned_memberships["test"]),
        test_pack.execution_positions - 1,
    )
    return results


def plot_f8(universe: str, agreement_by_fold: dict, runs_dir: Path) -> None:
    """F8 - regime agreement: AMI bars and decision divergence per fold."""
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    folds = sorted(agreement_by_fold)
    figure, (ami_axis, divergence_axis) = plt.subplots(1, 2, figsize=(12, 4.4))

    bar_width = 0.35
    positions = np.arange(len(folds))
    for offset, arm, colour in (
        (-bar_width / 2, "hmm_gate", "#0072B2"),
        (bar_width / 2, "vix_gate", "#D55E00"),
    ):
        ami_values = [agreement_by_fold[f][arm]["test"]["ami"] for f in folds]
        divergence_values = [
            agreement_by_fold[f][arm]["test"]["decision_divergence"] for f in folds
        ]
        ami_axis.bar(positions + offset, ami_values, bar_width,
                     label=arm, color=colour, alpha=0.85)
        divergence_axis.bar(positions + offset, divergence_values, bar_width,
                            label=arm, color=colour, alpha=0.85)

    for axis, title, ceiling in (
        (ami_axis, "adjusted mutual information vs learned regimes\n"
                   "(0 = carve the market independently, 1 = identically)", 1.0),
        (divergence_axis, "decision divergence D = mean ||w_learned - w_stat||_1\n"
                          "(identical experts; 0 = same trades, 2 = disjoint)", 2.0),
    ):
        axis.set_xticks(positions)
        axis.set_xticklabels(folds)
        axis.set_ylim(0, ceiling)
        axis.set_title(f"{title}", fontsize=9, loc="left")
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8, frameon=False)

    figure.suptitle(
        f"F8 - {universe}: do decision-defined regimes carve the market "
        "differently from statistical ones? (test years, seed 0)",
        fontsize=10, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(figures_dir / f"F8_agreement_{universe}.png", dpi=170,
                   bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# The headline table
# --------------------------------------------------------------------------- #
def headline_table(universe: str, results: dict) -> pd.DataFrame:
    """Per-arm summary over the three folds, with a bootstrap CI on the
    concatenated test stream (seed-0, matching the P4 convention)."""
    rows = []
    for (rung, arm), by_fold in results.items():
        val_matrix = np.array(
            [[r["val_sharpe"] for r in by_fold[f]["per_seed"]] for f in sorted(by_fold)]
        )
        test_matrix = np.array(
            [[r["test_sharpe"] for r in by_fold[f]["per_seed"]] for f in sorted(by_fold)]
        )

        # Seed-0 concatenated test stream across the folds; the caller pairs
        # it with the true r_f series when computing the bootstrap CI.
        seed_zero_stream = pd.concat(
            [by_fold[f]["per_seed"][0]["test_returns"] for f in sorted(by_fold)]
        )
        rows.append(
            {
                "rung": rung, "arm": arm,
                "val_sharpe_mean": float(val_matrix.mean()),
                "val_sharpe_sd": float(val_matrix.std()),
                "test_sharpe_mean": float(test_matrix.mean()),
                "test_sharpe_sd": float(test_matrix.std()),
                "n_params": by_fold[sorted(by_fold)[0]]["per_seed"][0]["n_params"],
                "_stream": seed_zero_stream,
            }
        )

    table = pd.DataFrame(rows)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rungs 3a/3b/3c + 4 and gate G3.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rung3.yaml"))
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
    runs_dir.mkdir(parents=True, exist_ok=True)
    universes = [args.universe]

    report: dict = {"universes": {}}

    for universe in universes:
        logger.info("=== %s : thesis week ===", universe)
        processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
        artifacts = load_universe_artifacts(processed_dir, universe)
        cached_states = find_cached_states(
            PROJECT_ROOT / config["paths"]["states_dir"], universe
        )

        wanted_folds = [f for f in folds_for(universe) if f.name in config["folds"]]
        arm_results: dict = {}
        packs_by_fold: dict = {}

        for fold in wanted_folds:
            packs = build_fold_packs(artifacts, cached_states, fold)
            packs_by_fold[fold.name] = packs

            for rung, arm in list(LEARNED_ARMS) + [("4", a) for a in STATISTICAL_ARMS]:
                # Rung 2 already has F01 rows from P6; only top up F02/F03.
                if (rung, arm) == ("2", "moe_k3") and fold.name == "F01":
                    continue
                arm_results.setdefault((rung, arm), {})[fold.name] = run_arm(
                    universe, rung, arm, fold, packs, artifacts, cached_states,
                    config, registry_path,
                )

        # Rung-2 F01 comparison numbers: retrain seed set quickly is wasteful;
        # instead reuse P6's registry rows? For the in-run table we retrain F01
        # once here - it is seconds and keeps this script self-contained.
        fold_f01 = wanted_folds[0]
        arm_results[("2", "moe_k3")][fold_f01.name] = run_arm(
            universe, "2", "moe_k3", fold_f01, packs_by_fold[fold_f01.name],
            artifacts, cached_states, config, registry_path,
        )

        # ---- agreement metrics, using rung-2 seed 0 as the learned system --
        agreement_by_fold = {}
        for fold in wanted_folds:
            learned_model = arm_results[("2", "moe_k3")][fold.name]["per_seed"][0]["model"]
            agreement_by_fold[fold.name] = agreement_for_fold(
                universe, fold, packs_by_fold[fold.name], artifacts,
                learned_model, config,
            )
        plot_f8(universe, agreement_by_fold, runs_dir)

        # ---- headline table with the bootstrap CI -------------------------
        table = headline_table(universe, arm_results)
        risk_free_full = artifacts.risk_free
        ci_rows = []
        for _, row in table.iterrows():
            stream = row["_stream"]
            interval = sharpe_confidence_interval(
                stream.to_numpy(),
                risk_free_full.loc[stream.index].to_numpy(),
                level=float(config["bootstrap"]["level"]),
                n_resamples=int(config["bootstrap"]["n_resamples"]),
                mean_block_length=float(config["bootstrap"]["mean_block_length"]),
                seed=int(config["bootstrap"]["seed"]),
            )
            ci_rows.append({"ci_low": interval.low, "ci_high": interval.high,
                            "ci_point_seed0": interval.point})
        table = pd.concat([table.drop(columns="_stream"),
                           pd.DataFrame(ci_rows)], axis=1)
        table.to_csv(runs_dir / f"headline_{universe}.csv", index=False)
        logger.info("\n%s\n", table.round(3).to_string(index=False))

        report["universes"][universe] = {
            "table": table.to_dict(orient="records"),
            "agreement": agreement_by_fold,
        }

    # ---- G3: complete tables on >= 3 folds, agreement computed ------------
    complete = all(
        len(entry["table"]) == len(LEARNED_ARMS) + len(STATISTICAL_ARMS)
        and len(entry["agreement"]) >= 3
        for entry in report["universes"].values()
    )
    report["g3_pass"] = bool(complete)
    (runs_dir / "g3_report.json").write_text(json.dumps(report, indent=2, default=str))

    logger.info("=============== GATE G3 ===============")
    logger.info("  tables complete on %s folds, agreement metrics computed: %s",
                config["folds"], "yes" if complete else "NO")
    logger.info("  overall: %s", "G3 PASS" if complete else "G3 FAIL")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
