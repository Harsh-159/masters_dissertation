"""Seed discipline: one call that makes a run reproducible.

Why this module exists
----------------------
Results in this project are reported as mean +/- standard deviation over
head-initialisation seeds, and every seed is written to the registry. That is
only meaningful if seeding is *complete* - if any source of randomness is left
unseeded, two runs recorded under "seed 3" are not the same experiment, and the
error bars silently become fiction.

There are three independent random number generators in play (Python's
:mod:`random`, NumPy's, and PyTorch's), plus PyTorch's choice of algorithm,
which can itself be non-deterministic even with a fixed seed. :func:`set_all_seeds`
handles all four in one call, and returns a record of exactly what it did so the
run can log it.

An honest caveat about ``PYTHONHASHSEED``
-----------------------------------------
Python fixes its string-hash randomisation at interpreter start-up. Setting the
environment variable from inside a running process therefore does **not**
retroactively make the current process's hashing deterministic - it only affects
subprocesses spawned afterwards. We set it anyway (it is free, and this project
does spawn parallel workers for the walk-forward evaluation), but the honest
statement is that determinism of *this* process does not depend on it: nothing in
the numerical path relies on set or dict iteration order.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

# PyTorch requires this environment variable to be set before CUDA is
# initialised in order to make cuBLAS operations deterministic. It is harmless
# on CPU-only and Apple-silicon machines, so we set it unconditionally.
_CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"


@dataclass(frozen=True)
class SeedRecord:
    """A record of what :func:`set_all_seeds` configured, for logging.

    Attributes
    ----------
    seed
        The integer seed applied to every generator.
    deterministic_algorithms
        Whether PyTorch was asked to use deterministic algorithm implementations.
    warn_only
        If ``True``, PyTorch warns instead of raising when an operation has no
        deterministic implementation.
    generators_seeded
        Names of the random number generators that were seeded.
    """

    seed: int
    deterministic_algorithms: bool
    warn_only: bool
    generators_seeded: tuple[str, ...] = field(
        default=("python.random", "numpy.random", "torch")
    )

    def as_dict(self) -> dict:
        """Return a flat dictionary suitable for logging into a config or note."""
        return {
            "seed": self.seed,
            "deterministic_algorithms": self.deterministic_algorithms,
            "warn_only": self.warn_only,
            "generators_seeded": list(self.generators_seeded),
        }


def set_all_seeds(
    seed: int,
    deterministic_algorithms: bool = True,
    warn_only: bool = False,
) -> SeedRecord:
    """Seed every random number generator used in this project.

    Call this once, at the top of every script, before anything random happens.

    Parameters
    ----------
    seed
        The seed to apply to Python, NumPy, and PyTorch.
    deterministic_algorithms
        If ``True`` (the default), instruct PyTorch to select deterministic
        implementations of its operations. This is what turns "same seed" into
        "same numbers": a fixed seed alone is not sufficient, because some
        kernels accumulate in a non-deterministic order.
    warn_only
        If ``True``, PyTorch emits a warning rather than raising when an
        operation has no deterministic implementation. Keep this ``False`` for
        results of record - a silent fallback to a non-deterministic kernel is
        exactly the kind of thing that should stop a run, not warn about it.

    Returns
    -------
    SeedRecord
        A record of what was configured, for writing into the run's log or the
        registry's free-text note.

    Examples
    --------
    >>> record = set_all_seeds(0)
    >>> record.seed
    0
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}.")
    seed = int(seed)

    # 1. Environment. Set before anything else so subprocesses inherit it.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_DETERMINISTIC_WORKSPACE)

    # 2. Python's own generator (used by any stdlib shuffling or sampling).
    random.seed(seed)

    # 3. NumPy's global generator. Note that new-style np.random.Generator
    #    objects must be seeded explicitly where they are constructed; this
    #    call covers the legacy global functions (np.random.rand and friends),
    #    which is what most library code still uses internally.
    np.random.seed(seed)

    # 4. PyTorch, on every device backend that might be present.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        # Deterministic convolution algorithm selection; benchmark mode picks
        # the fastest algorithm per shape at runtime, which is non-deterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 5. Algorithm-level determinism.
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

    return SeedRecord(
        seed=seed,
        deterministic_algorithms=deterministic_algorithms,
        warn_only=warn_only,
    )


def numpy_generator(seed: int) -> np.random.Generator:
    """Return an explicitly seeded NumPy generator.

    Prefer this over the legacy global ``np.random.*`` functions for any new
    code: an explicit generator makes the source of randomness visible at the
    call site and cannot be disturbed by unrelated library code re-seeding the
    global state.

    Parameters
    ----------
    seed
        Seed for the generator.

    Returns
    -------
    numpy.random.Generator
        A PCG64-backed generator.
    """
    return np.random.default_rng(seed)


def torch_generator(seed: int) -> torch.Generator:
    """Return an explicitly seeded PyTorch generator (CPU).

    Useful for passing to ``DataLoader`` or to initialisation functions that
    accept a ``generator=`` argument, so that a component's randomness is
    reproducible independently of global state.

    Parameters
    ----------
    seed
        Seed for the generator.

    Returns
    -------
    torch.Generator
        A CPU generator seeded with ``seed``.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
