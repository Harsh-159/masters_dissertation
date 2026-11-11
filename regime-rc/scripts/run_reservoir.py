"""Build, calibrate, cache and health-check the reservoir states.

    python scripts/run_reservoir.py --universe etf10
    python scripts/run_reservoir.py --with-esn       # also build the ESN variant
    python scripts/run_reservoir.py --force          # ignore existing caches

What it does, in order
----------------------
1. Loads the reservoir-input panel built in P1 (z-scored log returns of the
   assets alive since the universe start).
2. **Calibrates the input scale** on the training portion only, so the pre-tanh
   activations sit in the curved part of ``tanh``.
3. Drives the frozen reservoir once over the whole sample and caches the states,
   keyed by a hash of the reservoir configuration.
4. Runs the health diagnostics and writes figures F2 (memory capacity vs
   spectral radius) and F3 (pre-tanh saturation histogram).
5. If ``--with-esn``, builds the random reservoir too and reports the canonical
   correlation between the two state spaces.

Everything after this point in the project reads the cached states and never
re-drives the reservoir.
"""

from __future__ import annotations

import argparse
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

from rrc.reservoir.cache import (  # noqa: E402
    CachedStates,
    build_usable_mask,
    load_or_none,
    reservoir_config_hash,
    save_states,
)
from rrc.reservoir.diagnostics import (  # noqa: E402
    canonical_correlations,
    memory_capacity,
    memory_capacity_curve,
    saturation_statistics,
    state_pca_spectrum,
)
from rrc.reservoir.esn import build_esn  # noqa: E402
from rrc.reservoir.scr import build_scr, calibrate_input_scale, run_reservoir  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("reservoir")

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_reservoir_inputs(processed_dir: Path, universe: str) -> pd.DataFrame:
    """Load the z-scored reservoir-input panel produced by P1."""
    path = processed_dir / f"{universe}_reservoir_input.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found; run scripts/run_data.py first.")
    return pd.read_parquet(path)


def first_fully_finite_row(panel: pd.DataFrame) -> int:
    """Index of the first row where every input channel is finite.

    The reservoir cannot start before this: a single NaN propagates through the
    recurrence and destroys every state after it. Rows before this point are
    warm-up of the trailing z-score, not missing data.
    """
    finite_rows = np.isfinite(panel.to_numpy()).all(axis=1)
    if not finite_rows.any():
        raise ValueError("No fully finite row in the reservoir-input panel.")
    return int(np.argmax(finite_rows))


def build_states_for_universe(
    universe: str,
    config: dict,
    processed_dir: Path,
    cache_dir: Path,
    kind: str = "scr",
    esn_seed: int | None = None,
    force: bool = False,
) -> tuple[CachedStates, np.ndarray]:
    """Calibrate, drive, and cache one reservoir. Returns states and pre-activations."""
    reservoir_config = dict(config["reservoir"])
    scaling_config = config["input_scaling"]

    panel = load_reservoir_inputs(processed_dir, universe)
    start_row = first_fully_finite_row(panel)
    driving_inputs = panel.to_numpy()[start_row:]

    logger.info(
        "[%s/%s] panel %d x %d; reservoir driven from row %d (%s).",
        universe, kind, *panel.shape, start_row, panel.index[start_row].date(),
    )

    # --- 1. Calibrate the input scale on the training portion only ----------
    calibration_rows = int(len(driving_inputs) * scaling_config["calibration_fraction"])
    input_scale, achieved_std = calibrate_input_scale(
        inputs=driving_inputs[:calibration_rows],
        n_reservoir=reservoir_config["n_reservoir"],
        spectral_radius=reservoir_config["spectral_radius"],
        leak_rate=reservoir_config["leak_rate"],
        target_pre_activation_std=scaling_config["target_pre_activation_std"],
        max_iterations=scaling_config["max_iterations"],
        tolerance=scaling_config["tolerance"],
    )
    logger.info(
        "  calibrated input scale v = %.4f (pre-tanh std %.3f, target %.2f)",
        input_scale, achieved_std, scaling_config["target_pre_activation_std"],
    )

    # --- 2. Identify this exact configuration -------------------------------
    full_config = {
        "universe": universe,
        "kind": kind,
        "n_inputs": int(panel.shape[1]),
        "input_scale": round(input_scale, 10),
        "esn_seed": esn_seed,
        **reservoir_config,
    }
    cfg_hash = reservoir_config_hash(full_config)

    existing = None if force else load_or_none(cache_dir, universe, kind, cfg_hash)
    if existing is not None:
        logger.info("  cache hit (%s); reusing.", cfg_hash[:12])
        # Pre-activations are not cached; recompute only if a caller needs them.
        weights = _build_weights(kind, full_config, panel.shape[1], esn_seed)
        _, pre_activations = run_reservoir(weights, driving_inputs)
        return existing, pre_activations

    # --- 3. Drive the reservoir over the whole sample ------------------------
    weights = _build_weights(kind, full_config, panel.shape[1], esn_seed)
    computed_states, pre_activations = run_reservoir(weights, driving_inputs)

    # --- 4. Align to the full calendar and mark the usable region ------------
    n_days = len(panel)
    states = np.full((n_days, weights.n_reservoir), np.nan, dtype=np.float64)
    states[start_row:] = computed_states
    usable = build_usable_mask(n_days, start_row, reservoir_config["washout"])

    cached = CachedStates(
        states=states.astype(np.float32),
        dates=panel.index,
        usable=usable,
        reservoir_config=full_config,
        cfg_hash=cfg_hash,
        kind=kind,
    )
    cached.assert_usable_states_are_finite()
    save_states(cached, cache_dir)

    logger.info(
        "  %d usable days of %d (%s to %s); spectral radius %.6f",
        cached.n_usable, n_days,
        cached.usable_dates().min().date(), cached.usable_dates().max().date(),
        weights.spectral_radius(),
    )
    return cached, pre_activations


def _build_weights(kind: str, full_config: dict, n_inputs: int, esn_seed: int | None):
    """Construct the reservoir of the requested kind."""
    if kind == "scr":
        return build_scr(
            n_reservoir=full_config["n_reservoir"],
            n_inputs=n_inputs,
            spectral_radius=full_config["spectral_radius"],
            input_scale=full_config["input_scale"],
            leak_rate=full_config["leak_rate"],
            bias_scale=full_config["bias_scale"],
        )
    if kind == "esn":
        return build_esn(
            n_reservoir=full_config["n_reservoir"],
            n_inputs=n_inputs,
            spectral_radius=full_config["spectral_radius"],
            input_scale=full_config["input_scale"],
            leak_rate=full_config["leak_rate"],
            seed=esn_seed if esn_seed is not None else 0,
            bias_scale=full_config["bias_scale"],
        )
    raise ValueError(f"Unknown reservoir kind {kind!r}.")


# --------------------------------------------------------------------------- #
# Figures F2 and F3
# --------------------------------------------------------------------------- #
def figure_f2_memory_capacity(config: dict) -> Path:
    """F2 - memory capacity against spectral radius.

    A reservoir-health figure, on synthetic i.i.d. input. The expected shape is
    capacity rising towards the edge of stability; seeing it confirms the
    construction behaves like a reservoir before any finance is attempted.
    """
    reservoir_config = config["reservoir"]
    diagnostics_config = config["diagnostics"]
    memory_config = diagnostics_config["memory_capacity"]

    operating_leak = reservoir_config["leak_rate"]
    reference_leak = memory_config["reference_leak_rate"]
    probe_scale = memory_config["probe_input_scale"]

    sweep_kwargs = dict(
        spectral_radii=diagnostics_config["spectral_radius_sweep"],
        n_reservoir=reservoir_config["n_reservoir"],
        # Probed near-linear on purpose; see memory_capacity_curve's docstring.
        input_scale=probe_scale,
        max_delay=memory_config["max_delay"],
        n_train=memory_config["n_train"],
        n_test=memory_config["n_test"],
        washout=reservoir_config["washout"],
        seed=memory_config["seed"],
    )

    logger.info("F2: sweeping spectral radius at the operating leak rate ...")
    radii, operating_capacity = memory_capacity_curve(leak_rate=operating_leak, **sweep_kwargs)

    logger.info("F2: reference sweep at leak = %.1f (no leaky integration) ...", reference_leak)
    _, reference_capacity = memory_capacity_curve(leak_rate=reference_leak, **sweep_kwargs)

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.plot(
        radii, operating_capacity, "o-", color="#0072B2", lw=1.8,
        label=f"operating: alpha = {operating_leak}",
    )
    axis.plot(
        radii, reference_capacity, "s--", color="#999999", lw=1.4,
        label=f"reference: alpha = {reference_leak} (no leaking)",
    )
    axis.axvline(
        reservoir_config["spectral_radius"], color="#D55E00", ls=":", lw=1.2,
        label=f"configured rho = {reservoir_config['spectral_radius']}",
    )
    axis.set_xlabel("spectral radius")
    axis.set_ylabel("memory capacity (sum of recoverable fractions)")
    axis.set_title(
        f"F2 - memory capacity vs spectral radius (N_r = {reservoir_config['n_reservoir']}, "
        f"probe input scale {probe_scale})\n"
        "Rising with rho confirms the reservoir remembers. Leaky integration "
        "costs raw memory but smooths noisy input.\n"
        "Reservoir health on i.i.d. input - NOT a finance result.",
        fontsize=9.5, loc="left",
    )
    axis.grid(alpha=0.3)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "F2_memory_capacity.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_f3_saturation(pre_activations_by_universe: dict[str, np.ndarray]) -> Path:
    """F3 - pre-tanh activation histogram, with the saturation target marked."""
    n_panels = len(pre_activations_by_universe)
    figure, axes = plt.subplots(n_panels, 1, figsize=(10, 3.4 * n_panels), squeeze=False)

    for row, (universe, pre_activations) in enumerate(pre_activations_by_universe.items()):
        axis = axes[row][0]
        statistics = saturation_statistics(pre_activations)
        sample = pre_activations.ravel()

        axis.hist(sample, bins=200, density=True, color="#0072B2", alpha=0.6)
        for boundary in (-2.0, 2.0):
            axis.axvline(boundary, color="#D55E00", ls="--", lw=1.0)

        axis.set_title(
            f"{universe} - pre-tanh activations: std {statistics['std']:.3f} "
            f"(target 0.70), saturated {100 * statistics['saturated_fraction']:.2f}%",
            fontsize=10, loc="left",
        )
        axis.set_xlabel("pre-activation value")
        axis.set_ylabel("density")
        axis.grid(alpha=0.25)

    figure.suptitle(
        "F3 - reservoir saturation. Dashed lines at +/-2, beyond which tanh is "
        "nearly flat.", fontsize=10, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "F3_saturation.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Build states for the requested universes and run the health checks."""
    parser = argparse.ArgumentParser(description="Build and cache reservoir states.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "reservoir.yaml"))
    parser.add_argument(
        "--universe", required=True, choices=["etf10", "sector11"],
        help="Which universe to run. Required, and never both: the two "
             "universes are separate testbeds on different calendars "
             "(etf10 from 2006, sector11 from 1999), so they are run and "
             "reported independently and never aggregated.",
    )
    parser.add_argument("--with-esn", action="store_true", help="Also build the ESN variant.")
    parser.add_argument("--force", action="store_true", help="Ignore existing caches.")
    parser.add_argument("--no-figures", action="store_true", help="Skip F2/F3.")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    cache_dir = PROJECT_ROOT / config["paths"]["state_cache_dir"]

    universes = [args.universe]
    pre_activations_by_universe: dict[str, np.ndarray] = {}

    for universe in universes:
        logger.info("=== %s ===", universe)
        cached, pre_activations = build_states_for_universe(
            universe, config, processed_dir, cache_dir, kind="scr", force=args.force
        )
        pre_activations_by_universe[universe] = pre_activations

        statistics = saturation_statistics(pre_activations)
        logger.info(
            "  saturation: std %.3f, |p99| %.2f, saturated %.2f%%",
            statistics["std"], statistics["abs_p99"],
            100 * statistics["saturated_fraction"],
        )

        spectrum = state_pca_spectrum(
            cached.usable_states(), n_components=config["diagnostics"]["pca_components"]
        )
        components_for_90 = int(np.searchsorted(np.cumsum(spectrum), 0.90) + 1)
        logger.info(
            "  state PCA: top component %.1f%% of variance; %d components reach 90%%",
            100 * spectrum[0], components_for_90,
        )

        if args.with_esn:
            esn_cached, _ = build_states_for_universe(
                universe, config, processed_dir, cache_dir,
                kind="esn", esn_seed=config["esn"]["seeds"][0], force=args.force,
            )
            shared_rows = cached.usable & esn_cached.usable
            correlations = canonical_correlations(
                cached.states[shared_rows], esn_cached.states[shared_rows],
                n_components=config["diagnostics"]["canonical_components"],
            )
            logger.info(
                "  SCR vs ESN canonical correlations (top %d): mean %.3f, min %.3f",
                len(correlations), correlations.mean(), correlations.min(),
            )

    if not args.no_figures:
        logger.info("=== figures ===")
        f3_path = figure_f3_saturation(pre_activations_by_universe)
        logger.info("  %s", f3_path.name)
        f2_path = figure_f2_memory_capacity(config)
        logger.info("  %s", f2_path.name)

    logger.info("=== RESERVOIR STATES CACHED ===")
    logger.info("Every later phase reads these; the reservoir is never re-driven.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
