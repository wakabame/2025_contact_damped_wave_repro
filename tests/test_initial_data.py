"""The initial data of Section 6, in particular the Example 2 discrepancy."""

from __future__ import annotations

import numpy as np
import pytest

from contact_damped_wave.initial_data import (
    example1_eta0,
    example1_v0,
    example2_eta0,
    example2_v0,
    grid,
    initial_data,
)
from contact_damped_wave.params import EXAMPLE1, EXAMPLE2


def test_example1_matches_the_paper_formula() -> None:
    x = grid(EXAMPLE1)
    eta0 = example1_eta0(x)
    assert eta0 == pytest.approx(1.0 + 0.5 * np.sin(10 * np.pi * x) ** 2)
    assert eta0.min() == pytest.approx(1.0)
    assert eta0.max() == pytest.approx(1.5)
    assert eta0[0] == pytest.approx(1.0)
    assert eta0[-1] == pytest.approx(1.0)
    assert np.all(example1_v0(x) == -50.0)


def test_example1_initial_data_is_symmetric() -> None:
    eta0 = example1_eta0(grid(EXAMPLE1))
    assert eta0 == pytest.approx(eta0[::-1], abs=1e-12)


def test_example2_figure_variant_matches_figure_4a() -> None:
    """Continuous, endpoints at h = 1, value 2 at x = 0.2, 0.5, 0.8, peak 3 at 0.35."""
    x = grid(EXAMPLE2)
    eta0 = example2_eta0(x, "figure")
    assert eta0[0] == pytest.approx(1.0)
    assert eta0[-1] == pytest.approx(1.0)
    for point, value in ((0.2, 2.0), (0.5, 2.0), (0.8, 2.0), (0.65, 1.0)):
        assert eta0[np.argmin(np.abs(x - point))] == pytest.approx(value, abs=1e-6)
    assert eta0.max() == pytest.approx(3.0)
    assert x[eta0.argmax()] == pytest.approx(0.35, abs=1e-3)
    # continuity: the profile is Lipschitz with constant max(5, pi / 0.3) = 10.472
    lipschitz = np.abs(np.diff(eta0)).max() / EXAMPLE2.dx
    assert lipschitz == pytest.approx(np.pi / 0.3, rel=1e-3)
    # Theorem 2.1 assumes eta_0 >= c > 0
    assert eta0.min() > 0.0


def test_example2_paper_literal_variant_is_discontinuous() -> None:
    """Documents plan.md item (b): the printed formula cannot be what was run."""
    x = grid(EXAMPLE2)
    eta0 = example2_eta0(x, "paper-literal")
    assert np.abs(np.diff(eta0)).max() > 1.0  # jump of ~1.2 at x = 0.8
    assert eta0[0] == pytest.approx(0.0)  # violates eta_0 >= c > 0


def test_example2_velocity() -> None:
    x = grid(EXAMPLE2)
    v0 = example2_v0(x)
    assert np.all(v0[x < 0.6] == -50.0)
    assert np.all(v0[x >= 0.6] == -0.5)


def test_initial_data_dispatch() -> None:
    for example in (1, 2):
        params, x, eta0, v0 = initial_data(example)
        assert x.shape == eta0.shape == v0.shape == (params.N + 1,)
    with pytest.raises(ValueError):
        initial_data(3)
    with pytest.raises(ValueError):
        example2_eta0(grid(EXAMPLE2), "nonsense")  # type: ignore[arg-type]
