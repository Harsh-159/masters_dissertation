"""The random Echo State Network - used only as a robustness check.

Why this exists at all
----------------------
The Simple Cycle Reservoir is the project's primary construction (decision D2),
chosen because it is deterministic and because its universality results are the
theoretical anchor of the regime story. That choice invites an obvious question
from an examiner:

    "Are your results an artefact of that particular hand-built reservoir?"

The answer has to be empirical. This module builds the *conventional* random
reservoir - sparse recurrent weights, Gaussian input weights, rescaled to the
same spectral radius - so the whole ladder can be re-run on it at the frozen
best configuration and the answer reported. Three reservoir seeds ``{0, 1, 2}``
are enough to show the result is not a fluke of one draw.

How it differs from the SCR
---------------------------
=================  ==========================  ============================
                   SCR (primary)               random ESN (robustness)
=================  ==========================  ============================
recurrent weights  a ring, one shared weight   sparse random, density 0.1
input weights      one per unit, +/-v          dense Gaussian
spectral radius    exact by construction       achieved by rescaling
randomness         none                        seeded
=================  ==========================  ============================

Everything else - the state update, the leak rate, the washout, the caching - is
identical, because they share :func:`rrc.reservoir.scr.run_reservoir`. That
shared path is deliberate: any difference in results must come from the weights,
not from a second implementation of the dynamics.
"""

from __future__ import annotations

import logging

import numpy as np

from rrc.reservoir.scr import ReservoirWeights

logger = logging.getLogger(__name__)

# Fraction of recurrent connections that are nonzero (Appendix B default).
DEFAULT_DENSITY = 0.1


def build_esn(
    n_reservoir: int,
    n_inputs: int,
    spectral_radius: float,
    input_scale: float,
    leak_rate: float,
    seed: int,
    density: float = DEFAULT_DENSITY,
    bias_scale: float = 0.0,
) -> ReservoirWeights:
    """Construct a random sparse echo state network.

    Parameters
    ----------
    n_reservoir
        Reservoir size.
    n_inputs
        Number of driving channels.
    spectral_radius
        Target spectral radius. Unlike the SCR this is *achieved by rescaling*:
        the random matrix is drawn, its largest eigenvalue computed, and the
        whole matrix divided through. The result is exact only up to the
        eigenvalue solver's precision.
    input_scale
        Standard deviation of the Gaussian input weights.
    leak_rate
        ``alpha`` in the state update.
    seed
        Reservoir draw seed. Results are reported over ``{0, 1, 2}``.
    density
        Fraction of nonzero recurrent connections.
    bias_scale
        Constant bias before the nonlinearity.

    Returns
    -------
    ReservoirWeights
        The frozen reservoir, tagged ``kind="esn"``.
    """
    if not 0 < spectral_radius < 1:
        raise ValueError(
            f"spectral_radius must lie strictly in (0, 1); got {spectral_radius}."
        )
    if not 0 < density <= 1:
        raise ValueError(f"density must lie in (0, 1], got {density}.")
    if not 0 < leak_rate <= 1:
        raise ValueError(f"leak_rate must lie in (0, 1], got {leak_rate}.")

    generator = np.random.default_rng(seed)

    # Sparse recurrent matrix: draw dense, then blank all but `density` of it.
    recurrent = generator.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    keep = generator.random((n_reservoir, n_reservoir)) < density
    recurrent *= keep

    achieved_radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
    if achieved_radius <= 0:
        raise ValueError(
            f"The drawn recurrent matrix has spectral radius 0 (seed {seed}, "
            f"density {density}); it is too sparse to be a usable reservoir."
        )
    recurrent *= spectral_radius / achieved_radius

    input_weights = generator.normal(0.0, input_scale, size=(n_reservoir, n_inputs))

    logger.debug(
        "ESN seed %d: density %.2f, rescaled spectral radius %.4f -> %.4f",
        seed, density, achieved_radius, spectral_radius,
    )

    return ReservoirWeights(
        input_weights=input_weights,
        recurrent_weights=recurrent,
        bias=np.full(n_reservoir, bias_scale, dtype=np.float64),
        leak_rate=float(leak_rate),
        kind="esn",
    )
