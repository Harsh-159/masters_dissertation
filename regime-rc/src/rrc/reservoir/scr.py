"""The Simple Cycle Reservoir - the project's primary, deterministic reservoir.

What a reservoir is, and what it is for here
--------------------------------------------
A reservoir is a fixed recurrent dynamical system driven by the input series. It
is **never trained**: only the small readout heads on top of it are. Its job is
to turn the daily return vector into a high-dimensional state ``x_t`` that
encodes the *whole fading history* of the series, not just today's value.

The state update is the leaky-integrator echo state network form::

    x_t = (1 - alpha) * x_{t-1} + alpha * tanh(W_in u_t + W x_{t-1} + b)

Because ``x_{t-1}`` feeds back, unrolling the recurrence shows ``x_t`` is a
nonlinear function of the entire past input sequence, with older inputs decaying
away. The memory length is set by the spectral radius and the leak rate, not by
an explicit lookback window - which is a genuine advantage over feature
engineering, where you must choose "the last 20 days" by hand.

Why a *cycle* reservoir rather than a random one (locked decision D2)
--------------------------------------------------------------------
The Simple Cycle Reservoir wires the units into a single ring with **one shared
weight**, and gives every input connection the same magnitude with signs drawn
from an aperiodic sequence. There is no randomness anywhere. Three consequences:

1. **Reservoir-draw variance is exactly zero.** With a random reservoir, every
   reported number is an average over reservoir draws as well as head seeds, and
   the seed budget multiplies. Here "the reservoir" is a single, reproducible
   object.
2. **The theory is current and strong.** SCRs are proven universal approximators
   of fading-memory filters (Li, Fong & Tino, JMLR 2024), and linear SCRs at the
   edge of stability perform a Fourier decomposition of the driving input
   (Fong, Li & Tino, Chaos 2025) - which is what anchors the regime story.
3. It is the supervisor's own line of work, which matters for the viva.

The random ESN in :mod:`rrc.reservoir.esn` exists purely as a robustness check.

The echo state property
-----------------------
For the state to depend on the *input history* rather than on the arbitrary
initial condition ``x_0 = 0``, the influence of ``x_0`` must fade. The practical
handle is the spectral radius ``rho(W) < 1``. The ring construction makes this
exact rather than approximate: the eigenvalues of an ``N``-cycle with shared
weight ``r`` are ``r`` times the ``N``-th roots of unity, so every eigenvalue has
modulus exactly ``|r|`` and the spectral radius **is** ``r`` by construction. No
rescaling step, no eigenvalue solve, no numerical drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np

logger = logging.getLogger(__name__)

# Rodan and Tino's convention: a digit below 5 gives a negative sign.
_PI_SIGN_THRESHOLD = 5


@dataclass(frozen=True)
class ReservoirWeights:
    """The frozen parameters of a reservoir. Never trained, only constructed.

    Attributes
    ----------
    input_weights
        ``W_in``, shape ``(n_reservoir, n_inputs)``.
    recurrent_weights
        ``W``, shape ``(n_reservoir, n_reservoir)``.
    bias
        ``b``, shape ``(n_reservoir,)``.
    leak_rate
        ``alpha`` in the state update; 1.0 means no leaking.
    kind
        ``"scr"`` or ``"esn"``, recorded so a cached state file can never be
        mistaken for one built by the other construction.
    """

    input_weights: np.ndarray
    recurrent_weights: np.ndarray
    bias: np.ndarray
    leak_rate: float
    kind: str

    @property
    def n_reservoir(self) -> int:
        """Number of reservoir units."""
        return self.recurrent_weights.shape[0]

    @property
    def n_inputs(self) -> int:
        """Number of driving input channels."""
        return self.input_weights.shape[1]

    def spectral_radius(self) -> float:
        """Largest absolute eigenvalue of ``W``, computed numerically.

        For the SCR this should equal the requested radius to machine precision;
        the test suite asserts exactly that.
        """
        return float(np.max(np.abs(np.linalg.eigvals(self.recurrent_weights))))


# --------------------------------------------------------------------------- #
# The aperiodic sign pattern
# --------------------------------------------------------------------------- #
def pi_digits(count: int) -> list[int]:
    """Return the first ``count`` decimal digits of pi after the point.

    Computed rather than hard-coded, using Machin's formula::

        pi = 16 * arctan(1/5) - 4 * arctan(1/239)

    so correctness does not depend on a transcribed constant. The digits are the
    source of the reservoir's aperiodic sign pattern; any fixed aperiodic
    sequence would satisfy the theory, but using pi makes the construction
    reproducible by anyone without shipping a table of numbers.
    """
    if count <= 0:
        return []

    with localcontext() as context:
        context.prec = count + 20  # guard digits against accumulated rounding
        pi_value = 16 * _arctan_reciprocal(5) - 4 * _arctan_reciprocal(239)
        digit_text = str(pi_value).split(".")[1]

    return [int(character) for character in digit_text[:count]]


def _arctan_reciprocal(denominator: int) -> Decimal:
    """``arctan(1/denominator)`` by its alternating Taylor series."""
    reciprocal = Decimal(1) / Decimal(denominator)
    denominator_squared = Decimal(denominator) ** 2

    term = reciprocal
    total = term
    index = 1
    while True:
        term = term / denominator_squared
        contribution = term / (2 * index + 1)
        if contribution == 0:
            break
        total += -contribution if index % 2 == 1 else contribution
        index += 1
    return +total


def pi_sign_sequence(length: int) -> np.ndarray:
    """An aperiodic +/-1 sequence of the requested length, driven by pi's digits.

    Aperiodicity matters. A periodic sign pattern would make groups of reservoir
    units respond identically to the input, collapsing the effective
    dimensionality of the state space - the reservoir would be smaller than it
    looks.
    """
    return np.array(
        [1.0 if digit >= _PI_SIGN_THRESHOLD else -1.0 for digit in pi_digits(length)],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_cycle_matrix(n_reservoir: int, spectral_radius: float) -> np.ndarray:
    """Build the ``N x N`` cycle matrix with a single shared weight.

    Unit ``i`` receives from unit ``i-1``, and unit ``0`` receives from the last
    unit, closing the ring. Every nonzero entry equals ``spectral_radius``.

    Parameters
    ----------
    n_reservoir
        Number of units.
    spectral_radius
        The shared weight, which *is* the spectral radius by construction.
        Must be below 1 for the echo state property to hold.

    Returns
    -------
    numpy.ndarray
        The recurrent weight matrix.
    """
    if n_reservoir < 2:
        raise ValueError(f"n_reservoir must be at least 2, got {n_reservoir}.")
    if not 0 < spectral_radius < 1:
        raise ValueError(
            f"spectral_radius must lie strictly in (0, 1) for the echo state "
            f"property; got {spectral_radius}."
        )

    recurrent = np.zeros((n_reservoir, n_reservoir), dtype=np.float64)
    previous_unit = np.arange(-1, n_reservoir - 1)  # -1 wraps to the last unit
    recurrent[np.arange(n_reservoir), previous_unit] = spectral_radius
    return recurrent


def build_scr_input_weights(
    n_reservoir: int, n_inputs: int, input_scale: float
) -> np.ndarray:
    """Build ``W_in = v * D * C`` - the deterministic input mapping.

    ``C`` assigns reservoir units cyclically to input channels (unit ``i``
    listens to input ``i mod n_inputs``), so each unit is driven by exactly one
    asset and each asset drives an equal share of the reservoir. ``D`` is the
    diagonal of aperiodic +/-1 signs, and ``v`` is the input scale.

    Every input connection therefore has the same magnitude and differs only in
    sign and in which unit it feeds - the "minimum complexity" idea: strip the
    reservoir of every degree of freedom that is not doing work.
    """
    if n_inputs < 1:
        raise ValueError(f"n_inputs must be at least 1, got {n_inputs}.")

    signs = pi_sign_sequence(n_reservoir)
    input_weights = np.zeros((n_reservoir, n_inputs), dtype=np.float64)
    assigned_input = np.arange(n_reservoir) % n_inputs
    input_weights[np.arange(n_reservoir), assigned_input] = input_scale * signs
    return input_weights


def build_scr(
    n_reservoir: int,
    n_inputs: int,
    spectral_radius: float,
    input_scale: float,
    leak_rate: float,
    bias_scale: float = 0.0,
) -> ReservoirWeights:
    """Construct a Simple Cycle Reservoir. Fully deterministic - no seed needed.

    Parameters
    ----------
    n_reservoir
        Reservoir size ``N_r`` (default 300, searched over 100-500).
    n_inputs
        Number of driving channels, i.e. assets in the reservoir-input universe.
    spectral_radius
        Shared ring weight, and the spectral radius (default 0.9).
    input_scale
        ``v``. Normally set by :func:`calibrate_input_scale` rather than by hand.
    leak_rate
        ``alpha`` (default 0.3). Smaller values retain more of the previous
        state and lengthen memory.
    bias_scale
        Constant bias added before the nonlinearity; 0 by default, since the
        inputs are already standardised and centred.

    Returns
    -------
    ReservoirWeights
        The frozen reservoir.
    """
    if not 0 < leak_rate <= 1:
        raise ValueError(f"leak_rate must lie in (0, 1], got {leak_rate}.")

    return ReservoirWeights(
        input_weights=build_scr_input_weights(n_reservoir, n_inputs, input_scale),
        recurrent_weights=build_cycle_matrix(n_reservoir, spectral_radius),
        bias=np.full(n_reservoir, bias_scale, dtype=np.float64),
        leak_rate=float(leak_rate),
        kind="scr",
    )


# --------------------------------------------------------------------------- #
# Running the reservoir
# --------------------------------------------------------------------------- #
def run_reservoir(
    weights: ReservoirWeights,
    inputs: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive the reservoir with an input sequence and return its state path.

    Shared by both reservoir constructions - the SCR and the random ESN differ
    only in their weights, never in their dynamics.

    Parameters
    ----------
    weights
        The frozen reservoir.
    inputs
        ``(n_steps, n_inputs)`` driving series. Must be finite: a single NaN
        would propagate through the recurrence and poison every subsequent
        state, so this is checked rather than trusted.
    initial_state
        Starting state; defaults to zeros. The echo state property guarantees
        its influence fades, which is exactly what the washout period discards.

    Returns
    -------
    states, pre_activations
        ``(n_steps, n_reservoir)`` each. ``pre_activations`` is the argument of
        ``tanh`` before it is applied, kept because the saturation diagnostic
        (figure F3) and the input-scale calibration are both defined on it.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be 2-D (n_steps, n_inputs), got {inputs.shape}.")
    if inputs.shape[1] != weights.n_inputs:
        raise ValueError(
            f"reservoir expects {weights.n_inputs} input channels, got {inputs.shape[1]}."
        )
    if not np.isfinite(inputs).all():
        raise ValueError(
            "inputs contain non-finite values. A single NaN propagates through "
            "the recurrence and destroys every state after it, so the caller "
            "must slice to the finite region first."
        )

    n_steps = inputs.shape[0]
    states = np.zeros((n_steps, weights.n_reservoir), dtype=np.float64)
    pre_activations = np.zeros((n_steps, weights.n_reservoir), dtype=np.float64)

    state = (
        np.zeros(weights.n_reservoir, dtype=np.float64)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64).copy()
    )
    input_drive = inputs @ weights.input_weights.T  # (n_steps, n_reservoir), precomputed

    for step in range(n_steps):
        pre_activation = input_drive[step] + weights.recurrent_weights @ state + weights.bias
        state = (1.0 - weights.leak_rate) * state + weights.leak_rate * np.tanh(pre_activation)

        pre_activations[step] = pre_activation
        states[step] = state

    return states, pre_activations


# --------------------------------------------------------------------------- #
# Input-scale calibration by the saturation rule
# --------------------------------------------------------------------------- #
def calibrate_input_scale(
    inputs: np.ndarray,
    n_reservoir: int,
    spectral_radius: float,
    leak_rate: float,
    target_pre_activation_std: float = 0.7,
    max_iterations: int = 10,
    tolerance: float = 0.02,
) -> tuple[float, float]:
    """Choose the input scale ``v`` so the reservoir uses its nonlinearity.

    Why this is not guesswork
    -------------------------
    ``tanh`` is essentially linear near zero and flat far from it. Raw daily log
    returns are of order 0.01, so an unscaled reservoir would sit entirely in
    the linear region and would be an expensive linear filter. Scale the inputs
    too far the other way and every unit saturates at +/-1, destroying the
    state's ability to distinguish inputs.

    The rule fixes the target directly: choose ``v`` so the **standard deviation
    of the pre-tanh activations is about 0.7**. That places the bulk of the
    activations in the curved part of the function.

    Because the state feeds back into its own pre-activation, ``v`` and the
    resulting spread are coupled, so this solves the fixed point by simple
    iteration: run, measure, rescale, repeat. It converges in a handful of
    passes.

    Parameters
    ----------
    inputs
        The finite driving series to calibrate on. Use the **training fold
        only** - calibrating on the whole sample would let the input scale carry
        information about the test period.
    n_reservoir, spectral_radius, leak_rate
        Reservoir hyperparameters held fixed during calibration.
    target_pre_activation_std
        Desired spread of the pre-tanh activations (0.7 by default; 0.3-1.5 is
        the accepted range in the hyperparameter search).
    max_iterations
        Cap on refinement passes.
    tolerance
        Relative tolerance on the achieved spread.

    Returns
    -------
    input_scale, achieved_std
        The calibrated ``v`` and the pre-activation spread it produced.
    """
    input_scale = 1.0
    achieved_std = float("nan")

    for iteration in range(1, max_iterations + 1):
        weights = build_scr(
            n_reservoir=n_reservoir,
            n_inputs=inputs.shape[1],
            spectral_radius=spectral_radius,
            input_scale=input_scale,
            leak_rate=leak_rate,
        )
        _, pre_activations = run_reservoir(weights, inputs)
        achieved_std = float(np.std(pre_activations))

        if achieved_std <= 0:
            raise ValueError(
                "Pre-activations have zero spread; the input series is constant."
            )

        relative_error = abs(achieved_std - target_pre_activation_std) / target_pre_activation_std
        logger.debug(
            "calibration pass %d: v=%.4f -> std(pre-tanh)=%.4f (target %.2f)",
            iteration, input_scale, achieved_std, target_pre_activation_std,
        )
        if relative_error <= tolerance:
            break

        input_scale *= target_pre_activation_std / achieved_std

    return float(input_scale), achieved_std
