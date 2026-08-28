"""The initial data: Section 6 (in particular the Example 2 discrepancy) and ours."""

from __future__ import annotations

import numpy as np
import pytest

from contact_damped_wave.initial_data import (
    EXAMPLE3_CENTER,
    example1_eta0,
    example1_v0,
    example2_eta0,
    example2_v0,
    example3_eta0,
    example3_v0,
    grid,
    initial_data,
)
from contact_damped_wave.params import EXAMPLE1, EXAMPLE2, EXAMPLE3


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


def test_example3_is_smooth_and_compatible() -> None:
    """Example 3 (ours) must meet the clamped endpoints in both eta and v."""
    x = grid(EXAMPLE3)
    eta0 = example3_eta0(x, EXAMPLE3.h)
    v0 = example3_v0(x)
    assert eta0[0] == pytest.approx(EXAMPLE3.h)
    assert eta0[-1] == pytest.approx(EXAMPLE3.h)
    assert eta0.max() == pytest.approx(EXAMPLE3.h + 1.5)
    assert np.all(eta0 > 0.0)  # eta_0 >= c > 0 of Theorem 2.1
    # v^0 vanishes at both clamped endpoints, unlike the paper's data, so no
    # initial layer is created there.
    assert v0[0] == pytest.approx(0.0, abs=1e-12)
    assert v0[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(v0 <= 0.0)
    # The downward velocity is concentrated on the left of x = 0.35 and the
    # right third starts essentially at rest.
    assert v0.min() == pytest.approx(-68.8, abs=0.5)
    assert x[v0.argmin()] < EXAMPLE3_CENTER
    assert np.abs(v0[x >= 0.7]).max() < 0.1
    # Smooth: adjacent-node differences stay at the O(dx) level everywhere,
    # unlike Example 2's velocity, which jumps by 49.5 across one cell.
    assert np.abs(np.diff(v0)).max() < 600.0 * EXAMPLE3.dx


def test_example3_scales_with_h_and_length() -> None:
    params = EXAMPLE3.replace(h=2.0, length=2.0, dx=2 / 5000)
    x = grid(params)
    eta0 = example3_eta0(x, params.h, params.length)
    v0 = example3_v0(x, params.length)
    assert eta0[0] == pytest.approx(params.h)
    assert eta0[-1] == pytest.approx(params.h)
    assert eta0.max() == pytest.approx(params.h + 1.5)
    assert v0[0] == pytest.approx(0.0, abs=1e-12)
    assert v0[-1] == pytest.approx(0.0, abs=1e-12)


def test_initial_data_dispatch() -> None:
    for example in (1, 2, 3):
        params, x, eta0, v0 = initial_data(example)
        assert x.shape == eta0.shape == v0.shape == (params.N + 1,)
    with pytest.raises(ValueError):
        initial_data(4)
    with pytest.raises(ValueError):
        example2_eta0(grid(EXAMPLE2), "nonsense")  # type: ignore[arg-type]
