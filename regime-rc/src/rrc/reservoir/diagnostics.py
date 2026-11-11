"""Reservoir health diagnostics - is the dynamical system working at all?

Why these exist, and what they are *not*
----------------------------------------
When a strategy underperforms there are two very different explanations: the
reservoir is a poor dynamical system, or the reservoir is fine and the financial
signal simply is not there. Those demand opposite responses, and once results
start arriving it is very hard to tell them apart.

So the reservoir is checked **before any finance touches it**, on synthetic
tasks with known answers:

* **Memory capacity** - drive the reservoir with pure i.i.d. noise and ask how
  many past inputs a linear readout can still recover. This measures the
  dynamical system, nothing else.
* **Saturation** - are the pre-tanh activations sitting in the curved part of
  the nonlinearity, or is the reservoir effectively linear (inputs too small) or
  effectively binary (inputs too large)?
* **State PCA spectrum** - how many directions of the state space actually carry
  variance. A reservoir of 300 units whose states live in 5 dimensions is a
  5-unit reservoir wearing a large coat.
* **Canonical correlation** between SCR and ESN states - do the deterministic
  and random constructions span comparable information?

**A warning the plan states explicitly and is worth repeating: memory capacity is
a reservoir-health check, not a finance result.** A higher memory capacity does
not mean a better strategy. Reading strategy meaning into these numbers is a
mistake; they exist only to rule out a broken reservoir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from rrc.reservoir.scr import ReservoirWeights, build_scr, run_reservoir

logger = logging.getLogger(__name__)

# Small ridge on the memory-capacity readout, purely for numerical stability of
# the least-squares solve; far too small to affect the measurement.
MEMORY_READOUT_RIDGE = 1e-8


@dataclass(frozen=True)
class MemoryCapacityResult:
    """Outcome of the memory-capacity task.

    Attributes
    ----------
    per_delay
        ``(max_delay,)`` recoverable fraction at each delay k = 1..max_delay.
    total
        Sum over delays - the memory capacity. Theoretically bounded above by
        the number of reservoir units.
    max_delay
        The longest delay probed.
    """

    per_delay: np.ndarray
    total: float
    max_delay: int


def memory_capacity(
    weights: ReservoirWeights,
    max_delay: int = 150,
    n_train: int = 3000,
    n_test: int = 2000,
    washout: int = 250,
    seed: int = 0,
) -> MemoryCapacityResult:
    """Measure how far back a linear readout can recover the driving input.

    The standard task: drive the reservoir with i.i.d. uniform noise, then for
    each delay ``k`` fit a linear readout from the state ``x_t`` to the input
    ``u_{t-k}``. The squared correlation achieved at delay ``k`` is the fraction
    of that input still recoverable; summing over delays gives the capacity.

    Evaluated **out of sample** - the readout is fitted on one segment and
    scored on another. The common in-sample convention inflates the number when
    the reservoir has many units relative to the sample, and this project would
    rather report a slightly lower honest figure than a flattering one.

    Parameters
    ----------
    weights
        A reservoir with exactly **one** input channel; the task drives it with
        a scalar noise series.
    max_delay
        Longest delay probed.
    n_train, n_test
        Lengths of the fitting and scoring segments.
    washout
        Steps discarded at the start so the arbitrary ``x_0`` has faded.
    seed
        Seed for the driving noise.

    Returns
    -------
    MemoryCapacityResult
    """
    if weights.n_inputs != 1:
        raise ValueError(
            f"The memory-capacity task drives the reservoir with a scalar noise "
            f"series, so it needs a 1-input reservoir; got {weights.n_inputs}."
        )

    generator = np.random.default_rng(seed)
    n_steps = washout + max_delay + n_train + n_test
    driving_input = generator.uniform(-1.0, 1.0, size=(n_steps, 1))

    states, _ = run_reservoir(weights, driving_input)

    # Start far enough in that every delayed target exists and washout is past.
    first_row = washout + max_delay
    usable_states = states[first_row:]
    flat_input = driving_input[:, 0]

    # Target matrix: column k holds the input delayed by k+1 steps.
    targets = np.column_stack(
        [flat_input[first_row - delay : first_row - delay + len(usable_states)]
         for delay in range(1, max_delay + 1)]
    )

    train_states, test_states = usable_states[:n_train], usable_states[n_train:]
    train_targets, test_targets = targets[:n_train], targets[n_train:]

    readout = _ridge_solve(train_states, train_targets, MEMORY_READOUT_RIDGE)
    predictions = test_states @ readout

    per_delay = np.array(
        [_squared_correlation(predictions[:, k], test_targets[:, k])
         for k in range(max_delay)]
    )
    return MemoryCapacityResult(
        per_delay=per_delay, total=float(per_delay.sum()), max_delay=max_delay
    )


def memory_capacity_curve(
    spectral_radii: np.ndarray | list[float],
    n_reservoir: int,
    leak_rate: float,
    input_scale: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory capacity as a function of spectral radius - the data behind F2.

    The expected shape is rising towards the edge of stability: a radius near 1
    gives long, slowly-fading memory; a small radius forgets quickly. Seeing that
    shape confirms the reservoir behaves like a reservoir.

    **Use a small ``input_scale`` here** (around 0.05), not the operating scale
    calibrated for the real data. Memory capacity asks how much of a past input a
    *linear* readout can still recover. Driving the reservoir hard pushes it into
    the curved part of ``tanh``, where it trades linear memory for nonlinear
    computation - measured at the operating scale the curve flattens and can even
    invert, which looks like a broken reservoir but is really a mismeasured one.
    Probing near-linear is the standard convention and keeps the diagnostic
    answering the question it was designed for.

    The leak rate matters too: leaky integration low-passes the state and blurs
    individual delays, so a small ``leak_rate`` gives markedly lower capacity
    than ``leak_rate=1.0``. That is a deliberate trade for noisy financial data,
    not a defect.
    """
    radii = np.asarray(spectral_radii, dtype=float)
    capacities = np.zeros(len(radii))

    for index, radius in enumerate(radii):
        weights = build_scr(
            n_reservoir=n_reservoir,
            n_inputs=1,
            spectral_radius=float(radius),
            input_scale=input_scale,
            leak_rate=leak_rate,
        )
        capacities[index] = memory_capacity(weights, **kwargs).total
        logger.info("  rho = %.3f -> memory capacity %.1f", radius, capacities[index])

    return radii, capacities


def saturation_statistics(pre_activations: np.ndarray) -> dict[str, float]:
    """Describe where the pre-tanh activations sit on the nonlinearity.

    The target is a standard deviation near 0.7. Interpreting the output:

    * spread far **below** target - the reservoir is operating in the linear
      region of ``tanh`` and is an expensive linear filter;
    * spread far **above** target, with a high saturated fraction - units are
      pinned at +/-1 and can no longer distinguish inputs.

    Returns
    -------
    dict
        Mean, standard deviation, absolute 99th percentile, and the fraction of
        activations whose ``tanh`` exceeds 0.99 in magnitude.
    """
    values = np.asarray(pre_activations, dtype=float)
    activations = np.tanh(values)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "abs_p99": float(np.percentile(np.abs(values), 99)),
        "saturated_fraction": float((np.abs(activations) > 0.99).mean()),
    }


def state_pca_spectrum(states: np.ndarray, n_components: int | None = None) -> np.ndarray:
    """Explained-variance ratios of the state covariance, largest first.

    Answers "how many dimensions is this reservoir really using". A rapidly
    collapsing spectrum means most units are redundant, and the effective state
    space is far smaller than ``N_r``.
    """
    matrix = np.asarray(states, dtype=float)
    centred = matrix - matrix.mean(axis=0, keepdims=True)

    singular_values = np.linalg.svd(centred, compute_uv=False)
    variances = singular_values**2
    total = variances.sum()
    if total <= 0:
        return np.zeros_like(variances)

    ratios = variances / total
    return ratios if n_components is None else ratios[:n_components]


def canonical_correlations(
    states_a: np.ndarray, states_b: np.ndarray, n_components: int = 10
) -> np.ndarray:
    """Canonical correlations between two state spaces, largest first.

    Two reservoirs cannot be compared entry by entry - their units are in
    different, arbitrary bases. What *can* be compared is the information the
    two state spaces linearly span. Canonical correlation analysis finds the
    directions in each that align best; values near 1 mean the two reservoirs
    encode essentially the same thing in different coordinates.

    This is the SCR-versus-ESN sanity check: if the deterministic construction
    spanned much less than the random one, the choice of SCR would be costing
    real representational power and would need justifying.
    """
    first = np.asarray(states_a, dtype=float)
    second = np.asarray(states_b, dtype=float)
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"state matrices must cover the same rows; got {first.shape[0]} and "
            f"{second.shape[0]}."
        )

    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)

    # Orthonormal bases for each column space; the singular values of the
    # cross-product are then exactly the canonical correlations.
    basis_a, _ = np.linalg.qr(first)
    basis_b, _ = np.linalg.qr(second)
    correlations = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)

    return np.clip(correlations[:n_components], 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ridge_solve(design: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    """Solve a ridge-regularised least-squares problem for all targets at once."""
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ targets)


def _squared_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    """Squared Pearson correlation, clipped to [0, 1]."""
    if np.std(prediction) <= 0 or np.std(target) <= 0:
        return 0.0
    correlation = float(np.corrcoef(prediction, target)[0, 1])
    return float(np.clip(correlation**2, 0.0, 1.0))
