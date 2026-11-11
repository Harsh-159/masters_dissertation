"""The outer reservoir-hyperparameter search - random search.

Why random search, not NSGA-II, out here
-----------------------------------------
The plan named NSGA-II here with random search as the sanctioned fallback.
The fallback is the better tool for this problem - a handful of reservoir
hyperparameters that ultimately collapse to **one frozen configuration**:

* only ~2-3 of the ~5 hyperparameters actually move the result, and random
  search covers those better than a grid at equal budget (Bergstra & Bengio,
  2012);
* each evaluation is a noisy training run, so a lean, well-spread sample is
  worth more than an evolutionary process chasing fragile fitness differences;
* every distinct configuration counts as a deflated-Sharpe trial, so ~50 draws
  keep the deflation honest where ~360 (pop 24 x 15 gen) would harshen it for no
  measured gain;
* the multi-objective Pareto front NSGA-II would return is discarded the moment
  a single config is frozen.

The search still *reports* the non-dominated (validation Sharpe, validation
CVaR) set, so the multi-objective picture is kept for the write-up; it just
freezes one point from it by a documented rule.

What one evaluation costs
-------------------------
Each candidate reservoir config needs its state cache rebuilt (states depend on
N_r, rho, alpha and the input scale), then the gradient system trained on a small
fold subset to read off validation Sharpe and CVaR. States are cached by config
hash, so a repeated draw is free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rrc.eval.registry import config_hash
from rrc.eval.slicing import folds_for
from rrc.backtest.metrics import excess_sharpe
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts
from rrc.optim.train_loop import MixtureTrainConfig, evaluate_on_pack, train_mixture
from rrc.reservoir.cache import (
    CachedStates,
    build_usable_mask,
    load_or_none,
    reservoir_config_hash,
    save_states,
)
from rrc.reservoir.scr import build_scr, calibrate_input_scale, run_reservoir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HyperCandidate:
    """One evaluated reservoir configuration and its validation objectives."""

    params: dict          # the searched values (rho, alpha, N_r, input target, tau_g)
    val_sharpe: float     # mean validation excess Sharpe over the evaluated folds
    val_cvar: float       # mean validation CVaR_95 (a positive loss; lower is better)
    reservoir_cfg_hash: str


# --------------------------------------------------------------------------- #
# Sampling the search space
# --------------------------------------------------------------------------- #
def sample_candidates(ranges: dict, n_random: int, seed: int) -> list[dict]:
    """Draw ``n_random`` hyperparameter dictionaries uniformly from the ranges.

    ``ranges`` maps each searched parameter to ``[low, high]``. ``n_reservoir``
    is drawn as an integer; the rest are continuous. This is the whole "random"
    of random search - deliberately simple and fully reproducible from ``seed``.
    """
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n_random):
        candidates.append(
            {
                "spectral_radius": float(rng.uniform(*ranges["spectral_radius"])),
                "leak_rate": float(rng.uniform(*ranges["leak_rate"])),
                "target_pre_activation_std": float(
                    rng.uniform(*ranges["target_pre_activation_std"])
                ),
                "gate_temperature": float(rng.uniform(*ranges["gate_temperature"])),
                "n_reservoir": int(rng.integers(ranges["n_reservoir"][0], ranges["n_reservoir"][1] + 1)),
            }
        )
    return candidates


# --------------------------------------------------------------------------- #
# Building states for one candidate (reuses the P3 reservoir primitives)
# --------------------------------------------------------------------------- #
def build_states_for_config(
    universe: str,
    params: dict,
    fixed: dict,
    processed_dir: Path,
    cache_dir: Path,
) -> CachedStates:
    """Calibrate, drive and cache the reservoir for one candidate config.

    Mirrors ``scripts/run_reservoir.py`` exactly - same calibration on the
    training portion only, same washout, same cache keying - but takes the
    searched parameters as arguments. Cached by config hash, so re-evaluating a
    repeated draw is a cache hit.
    """
    panel = pd.read_parquet(processed_dir / f"{universe}_reservoir_input.parquet")
    inputs = panel.to_numpy()
    finite = np.isfinite(inputs).all(axis=1)
    if not finite.any():
        raise ValueError(f"{universe}: no fully finite reservoir-input row.")
    start = int(np.argmax(finite))
    driving = inputs[start:]

    calibration_rows = int(len(driving) * fixed["calibration_fraction"])
    input_scale, _ = calibrate_input_scale(
        inputs=driving[:calibration_rows],
        n_reservoir=params["n_reservoir"],
        spectral_radius=params["spectral_radius"],
        leak_rate=params["leak_rate"],
        target_pre_activation_std=params["target_pre_activation_std"],
        max_iterations=fixed["max_iterations"],
        tolerance=fixed["tolerance"],
    )

    full_config = {
        "universe": universe,
        "kind": "scr",
        "n_inputs": int(panel.shape[1]),
        "input_scale": round(input_scale, 10),
        "esn_seed": None,
        "n_reservoir": params["n_reservoir"],
        "spectral_radius": params["spectral_radius"],
        "leak_rate": params["leak_rate"],
        "bias_scale": fixed["bias_scale"],
        "washout": fixed["washout"],
    }
    cfg_hash = reservoir_config_hash(full_config)

    existing = load_or_none(cache_dir, universe, "scr", cfg_hash)
    if existing is not None:
        return existing

    weights = build_scr(
        n_reservoir=params["n_reservoir"],
        n_inputs=int(panel.shape[1]),
        spectral_radius=params["spectral_radius"],
        input_scale=input_scale,
        leak_rate=params["leak_rate"],
        bias_scale=fixed["bias_scale"],
    )
    computed, _ = run_reservoir(weights, driving)

    n_days = len(panel)
    states = np.full((n_days, weights.n_reservoir), np.nan, dtype=np.float64)
    states[start:] = computed
    cached = CachedStates(
        states=states.astype(np.float32),
        dates=panel.index,
        usable=build_usable_mask(n_days, start, fixed["washout"]),
        reservoir_config=full_config,
        cfg_hash=cfg_hash,
        kind="scr",
    )
    cached.assert_usable_states_are_finite()
    save_states(cached, cache_dir)
    return cached


# --------------------------------------------------------------------------- #
# Evaluating one candidate on validation
# --------------------------------------------------------------------------- #
def evaluate_candidate(
    universe: str,
    params: dict,
    fixed: dict,
    processed_dir: Path,
    cache_dir: Path,
    fold_names: list[str],
    seeds: list[int],
    n_experts: int,
    lambda_cvar: float,
    lambda_balance: float,
    lambda_turn: float,
    max_epochs: int,
    patience: int,
    lambda_div: float = 0.0,
) -> HyperCandidate:
    """Score one configuration on the CONCATENATED validation stream, over seeds.

    Two corrections over the first version of this search, both forced by
    measurement:

    **Multiple seeds.** The first pass scored every candidate at a single seed
    and froze the maximum, which produced a winner at validation Sharpe 0.817
    that collapsed to 0.298 under five seeds - the entire advantage came from
    one seed on one fold. Max-over-N at a single seed is precisely where
    selection bias lives, so fitness is now averaged over ``seeds``.

    **Concatenated Sharpe.** Fitness was the mean of per-fold Sharpes. Sharpe is
    a ratio: averaging it across years implicitly rebases wealth each January and
    weights a calm year equally with a crisis year (on sector11 the two
    aggregations differ by 0.48). Each seed's fold validation streams are now
    stitched into one continuous series and scored once, exactly as the
    baselines and the literature are scored.
    """
    cached_states = build_states_for_config(universe, params, fixed, processed_dir, cache_dir)
    artifacts = load_universe_artifacts(processed_dir, universe)
    folds = [f for f in folds_for(universe) if f.name in fold_names]
    packs_by_fold = {f.name: build_fold_packs(artifacts, cached_states, f) for f in folds}

    per_seed_sharpe, per_seed_cvar = [], []
    for seed in seeds:
        streams, risk_frees, cvars = [], [], []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            result = train_mixture(
                packs["train"],
                packs["val"],
                MixtureTrainConfig(
                    n_experts=n_experts,
                    lambda_cvar=lambda_cvar,
                    lambda_balance=lambda_balance,
                    lambda_turn=lambda_turn,
                    lambda_div=lambda_div,
                    gate_temperature=params["gate_temperature"],
                    max_epochs=max_epochs,
                    patience=patience,
                    seed=seed,
                ),
            )
            evaluation = evaluate_on_pack(result.model, packs["val"])
            streams.append(evaluation["net_returns"].to_numpy())
            risk_frees.append(packs["val"].risk_free.numpy())
            cvars.append(evaluation["metrics"]["cvar_95"])

        returns = np.concatenate(streams)
        per_seed_sharpe.append(excess_sharpe(returns, np.concatenate(risk_frees)))
        per_seed_cvar.append(float(np.mean(cvars)))

    return HyperCandidate(
        params=params,
        val_sharpe=float(np.mean(per_seed_sharpe)),
        val_cvar=float(np.mean(per_seed_cvar)),
        reservoir_cfg_hash=cached_states.cfg_hash,
    )


# --------------------------------------------------------------------------- #
# The search and the freeze decision
# --------------------------------------------------------------------------- #
def non_dominated(candidates: list[HyperCandidate]) -> list[HyperCandidate]:
    """The Pareto-non-dominated set on (higher Sharpe, lower CVaR).

    A candidate is dominated if another is at least as good on both objectives
    and strictly better on one. Reported for the write-up; the freeze picks one
    point from it.
    """
    kept = []
    for candidate in candidates:
        dominated = any(
            other.val_sharpe >= candidate.val_sharpe
            and other.val_cvar <= candidate.val_cvar
            and (other.val_sharpe > candidate.val_sharpe or other.val_cvar < candidate.val_cvar)
            for other in candidates
        )
        if not dominated:
            kept.append(candidate)
    return kept


def select_frozen(candidates: list[HyperCandidate]) -> HyperCandidate:
    """The single configuration to freeze: highest validation Sharpe on the front.

    The rule is stated in advance so the choice is not made by eyeballing: take
    the non-dominated set, then the maximum-Sharpe point within it. Ties break by
    lower CVaR, then by a smaller reservoir (Occam).
    """
    if not candidates:
        raise ValueError("No candidates to select from.")
    front = non_dominated(candidates)
    return max(
        front,
        key=lambda c: (c.val_sharpe, -c.val_cvar, -c.params["n_reservoir"]),
    )


def random_search(
    universe: str,
    ranges: dict,
    fixed: dict,
    processed_dir: Path,
    cache_dir: Path,
    fold_names: list[str],
    n_random: int,
    search_seed: int,
    eval_seeds: list[int],
    n_experts: int,
    lambda_cvar: float,
    lambda_balance: float,
    lambda_turn: float,
    max_epochs: int,
    patience: int,
    lambda_div: float = 0.0,
) -> list[HyperCandidate]:
    """Evaluate ``n_random`` sampled configurations; return them all (unsorted).

    The caller freezes one with :func:`select_frozen` and logs the whole set so
    the deflated-Sharpe trial count reflects every configuration tried.
    """
    candidates = sample_candidates(ranges, n_random, search_seed)
    evaluated: list[HyperCandidate] = []
    for index, params in enumerate(candidates, start=1):
        result = evaluate_candidate(
            universe, params, fixed, processed_dir, cache_dir, fold_names,
            eval_seeds, n_experts, lambda_cvar, lambda_balance, lambda_turn,
            max_epochs, patience, lambda_div=lambda_div,
        )
        evaluated.append(result)
        logger.info(
            "  [%s %d/%d] rho=%.3f a=%.3f v*=%.2f tau=%.2f N=%d -> "
            "val Sharpe %.3f, CVaR %.4f",
            universe, index, n_random, params["spectral_radius"], params["leak_rate"],
            params["target_pre_activation_std"], params["gate_temperature"],
            params["n_reservoir"], result.val_sharpe, result.val_cvar,
        )
    return evaluated
