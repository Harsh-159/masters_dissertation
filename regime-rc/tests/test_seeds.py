"""Tests for the seed discipline.

The claim these protect is the one the error bars in the results chapter rest
on: "seed 3" names the same experiment every time it is run. That requires all
three generators to be seeded, so each is checked independently - a partial
seeding bug (say, NumPy seeded but PyTorch not) is exactly the kind of thing
that would otherwise pass unnoticed and quietly widen every reported standard
deviation.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from rrc.utils.seeds import SeedRecord, numpy_generator, set_all_seeds, torch_generator


def draw_from_every_generator() -> tuple[float, float, float]:
    """Draw one number from each of Python, NumPy, and PyTorch."""
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


def test_same_seed_reproduces_all_three_generators():
    """The whole point: identical seed, identical numbers everywhere."""
    set_all_seeds(1234)
    first_draw = draw_from_every_generator()

    set_all_seeds(1234)
    second_draw = draw_from_every_generator()

    assert first_draw == second_draw


def test_different_seeds_produce_different_draws():
    """Guards against a seeding call that silently does nothing."""
    set_all_seeds(0)
    draw_with_seed_zero = draw_from_every_generator()

    set_all_seeds(1)
    draw_with_seed_one = draw_from_every_generator()

    assert draw_with_seed_zero != draw_with_seed_one


@pytest.mark.parametrize(
    "generator_name, draw",
    [
        ("python.random", lambda: random.random()),
        ("numpy.random", lambda: float(np.random.rand())),
        ("torch", lambda: float(torch.rand(1).item())),
    ],
)
def test_each_generator_is_seeded_independently(generator_name, draw):
    """Each generator is checked on its own, so partial seeding cannot hide."""
    set_all_seeds(7)
    first = draw()
    set_all_seeds(7)
    second = draw()
    assert first == second, f"{generator_name} was not reproducibly seeded"


def test_seed_record_describes_what_was_configured():
    """The record is what gets logged, so it must be faithful."""
    record = set_all_seeds(42)

    assert isinstance(record, SeedRecord)
    assert record.seed == 42
    assert record.deterministic_algorithms is True
    assert record.warn_only is False
    assert set(record.generators_seeded) == {"python.random", "numpy.random", "torch"}

    as_dict = record.as_dict()
    assert as_dict["seed"] == 42
    assert as_dict["generators_seeded"] == ["python.random", "numpy.random", "torch"]


def test_non_integer_seed_is_rejected():
    """A float seed usually means a config type error; fail rather than coerce."""
    with pytest.raises(TypeError, match="seed must be an integer"):
        set_all_seeds(1.5)


def test_boolean_seed_is_rejected():
    """`True` is an int in Python; reject it rather than silently seed with 1."""
    with pytest.raises(TypeError, match="seed must be an integer"):
        set_all_seeds(True)


def test_explicit_numpy_generator_is_reproducible():
    """Explicit generators are preferred for new code; check they behave."""
    first = numpy_generator(3).normal(size=5)
    second = numpy_generator(3).normal(size=5)
    np.testing.assert_array_equal(first, second)


def test_explicit_torch_generator_is_reproducible():
    """Same for the PyTorch generator handed to init functions and loaders."""
    first = torch.rand(5, generator=torch_generator(3))
    second = torch.rand(5, generator=torch_generator(3))
    torch.testing.assert_close(first, second)


def test_deterministic_algorithms_flag_is_actually_enabled():
    """Verify the global torch setting really was switched on."""
    set_all_seeds(0, deterministic_algorithms=True)
    assert torch.are_deterministic_algorithms_enabled() is True
