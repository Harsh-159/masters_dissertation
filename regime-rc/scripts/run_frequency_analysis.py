"""Level 2: which reservoir timescales does the decision objective read?

    python scripts/run_frequency_analysis.py --universe sector11

The theory this rests on
------------------------
The Simple Cycle Reservoir wires its units into a ring: unit ``i`` receives from
unit ``i-1`` with a shared weight ``rho``, and unit 0 closes the loop. With
leaking, the (linearised) state update is

    x_t = A x_{t-1} + alpha W_in u_t ,      A = (1 - alpha) I + alpha rho P

where ``P`` is the cyclic shift. **``A`` is circulant**, so it is diagonalised
exactly by the discrete Fourier basis - the ring's natural modes are spatial
oscillations around the loop, like harmonics on a string. This is the structure
behind Fong, Li and Tino (Chaos 2025): a linear cycle reservoir performs a
Fourier decomposition of its driving signal.

What each mode means, concretely
--------------------------------
Mode ``k`` has eigenvalue ``lambda_k = (1 - alpha) + alpha*rho*exp(-2i*pi*k/N)``,
and therefore a memory timescale ``tau_k = -1 / ln|lambda_k|`` in days - how long
an impulse into that mode takes to decay by ``1/e``. At the frozen configuration
(N=300, rho=0.90, alpha=0.30) the modes span:

    k =   0   ->  tau = 32.8 days   (slowest: the running-sum mode)
    k =  50   ->  tau =  7.0 days
    k = 100   ->  tau =  2.0 days
    k = 150   ->  tau =  1.2 days   (fastest, Nyquist)

So the reservoir is not an opaque bag of 300 numbers. It is a bank of 300
filters with known memory lengths, and a linear readout **chooses which
timescales to listen to**.

The question, and the prediction
--------------------------------
Level 1 found that the learned gate organises its regimes around 5-day return
direction while the HMM organises around volatility (dwell 9-13 weeks). If that
is right, the gate should put its weight on the SHORT-memory modes, and the
timescale the HMM operates on should sit beyond the slowest mode available.

Stating it before the run: **the gate's spectral mass should concentrate at
small tau.** If instead it is flat, or sits at long tau, the Level-1 reading is
wrong and this says so.

How the measurement is made honest
----------------------------------
Three quantities, not one:

``gate weight power``
    |DFT of the gate's weight row|^2 - which modes the readout *points at*.

``state power``
    Variance of the real cached reservoir states projected onto the same basis -
    which modes actually *carry signal*. This matters because the input is
    injected through a fixed sign pattern, so the modes are not excited equally,
    and because tanh is not linear.

``variance contribution`` (the one to quote)
    The product. How much of the gate logit's variance actually comes from each
    mode. This is what the gate effectively uses, and it accounts for both
    reachability and the nonlinearity empirically rather than by assumption.

Against a null: the same model at initialisation, untrained. A random readout
has flat expected power, so any structure in the trained spectrum is learned.

The caveat, stated plainly
--------------------------
The Fourier diagonalisation is exact for the LINEAR system. This reservoir has
tanh. The input scale is calibrated to a pre-activation std of 0.7, which is the
mildly-curved part of tanh rather than the saturated part, so a first-order
reading is defensible - but it is a first-order reading, and the empirical state
spectrum is included precisely so the conclusion does not rest on the
linearisation alone.
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

from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.models.gating import MixtureOfExperts  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402
from rrc.utils.seeds import set_all_seeds  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("frequency")

N_EXPERTS = 3          # the gate is inert at K = 1; nothing to read there

# Memory-timescale bands, in trading days. Chosen to be interpretable rather
# than equal-width: "under three days" and "a trading month" are the units a
# reader thinks in.
BANDS = [(0.0, 3.0, "fast  (< 3 days)"),
         (3.0, 10.0, "medium (3-10 days)"),
         (10.0, 1e9, "slow  (> 10 days)")]


def mode_timescales(n_reservoir: int, spectral_radius: float, leak_rate: float):
    """Memory timescale tau_k of each ring mode, for the rfft mode ordering."""
    k = np.arange(n_reservoir // 2 + 1)
    eigenvalues = (1.0 - leak_rate) + leak_rate * spectral_radius * np.exp(
        -2j * np.pi * k / n_reservoir
    )
    return -1.0 / np.log(np.abs(eigenvalues))


def band_shares(power: np.ndarray, timescales: np.ndarray) -> dict[str, float]:
    """Fraction of spectral power falling in each memory-timescale band."""
    total = power.sum()
    if total <= 0:
        return {name: np.nan for *_, name in BANDS}
    return {
        name: float(power[(timescales >= low) & (timescales < high)].sum() / total)
        for low, high, name in BANDS
    }


def analyse_fold(packs, n_state, n_assets, seed, timescales):
    """Gate weight power, state power and variance contribution for one fold."""
    trained = train_mixture(
        packs["train"], packs["val"],
        MixtureTrainConfig(n_experts=N_EXPERTS, lambda_cvar=0.0, lambda_balance=0.01,
                           lambda_turn=0.0, lambda_div=0.0, seed=seed),
        model=lambda: MixtureOfExperts(n_state, n_assets, N_EXPERTS),
    )
    # The untrained null: same architecture, same seed, no gradient steps.
    set_all_seeds(seed)
    untrained = MixtureOfExperts(n_state, n_assets, N_EXPERTS)

    def gate_power(model):
        weights = model.gating.linear.weight.detach().numpy()      # (K, N)
        return (np.abs(np.fft.rfft(weights, axis=1)) ** 2).mean(axis=0)

    # Which modes actually carry signal, measured on the real test states.
    states = packs["test"].states.detach().numpy()                 # (T, N)
    projected = np.fft.rfft(states - states.mean(axis=0), axis=1)
    state_power = (np.abs(projected) ** 2).mean(axis=0)

    trained_gate = gate_power(trained.model)
    return {
        "trained_gate": trained_gate,
        "untrained_gate": gate_power(untrained),
        "state": state_power,
        # What the gate logit's variance is actually made of.
        "contribution": trained_gate * state_power,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Which reservoir timescales does the gate read?")
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--folds", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "reservoir.yaml"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    reservoir_cfg = load_yaml(Path(args.config))["reservoir"]
    n_reservoir = int(reservoir_cfg["n_reservoir"])
    timescales = mode_timescales(n_reservoir, float(reservoir_cfg["spectral_radius"]),
                                 float(reservoir_cfg["leak_rate"]))

    logger.info("=== ring modes at the frozen reservoir (N=%d, rho=%.2f, alpha=%.2f) ===",
                n_reservoir, reservoir_cfg["spectral_radius"], reservoir_cfg["leak_rate"])
    logger.info("  memory timescales span %.1f to %.1f days across %d modes",
                timescales.min(), timescales.max(), len(timescales))

    artifacts = load_universe_artifacts(data_root / "data" / "processed", args.universe)
    cached = find_cached_states(data_root / "data" / "states", args.universe)
    fold_names = (args.folds.split(",") if args.folds
                  else [f.name for f in folds_for(args.universe)])
    folds = [f for f in folds_for(args.universe) if f.name in fold_names]
    seeds = [int(v) for v in args.seeds.split(",")]

    accumulated = {key: [] for key in ("trained_gate", "untrained_gate", "state", "contribution")}
    for fold in folds:
        packs = build_fold_packs(artifacts, cached, fold)
        n_state = packs["train"].states.shape[1]
        n_assets = packs["train"].logit_mask.shape[1]
        for seed in seeds:
            result = analyse_fold(packs, n_state, n_assets, seed, timescales)
            for key, value in result.items():
                accumulated[key].append(value / value.sum())   # normalise, then pool
        logger.info("  %s done", fold.name)

    spectra = {key: np.mean(value, axis=0) for key, value in accumulated.items()}

    logger.info("")
    logger.info("=== where the spectral mass sits, by memory timescale ===")
    table = pd.DataFrame({name: band_shares(power, timescales)
                          for name, power in spectra.items()})
    for line in (table * 100).round(1).to_string().splitlines():
        logger.info("  %s", line)
    logger.info("  (percent of power; 'contribution' is the one to quote)")

    logger.info("")
    logger.info("=== weighted-average memory timescale actually used ===")
    for name, power in spectra.items():
        weighted = float((power * timescales).sum() / power.sum())
        logger.info("  %-15s %5.1f days", name, weighted)

    out_dir = data_root / "results" / "runs" / "frequency"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mode": np.arange(len(timescales)), "tau_days": timescales,
                  **spectra}).to_csv(out_dir / f"spectra_{args.universe}.csv", index=False)
    table.to_csv(out_dir / f"bands_{args.universe}.csv")
    logger.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
