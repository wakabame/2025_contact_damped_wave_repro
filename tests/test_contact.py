"""Behaviour at the obstacle: energy balance, symmetry, penetration, contact set."""

from __future__ import annotations

import numpy as np
import pytest

from contact_damped_wave.diagnostics import (
    components,
    contact_area,
    contact_intervals,
    contact_mask,
    energy_balance,
    first_contact_time,
    penetration_depth,
)
from contact_damped_wave.initial_data import initial_data
from contact_damped_wave.params import Params
from contact_damped_wave.solver import solve


def _run(example: int, params: Params, **kwargs):
    _, _, eta0, v0 = initial_data(example, params)
    return solve(params, eta0, v0, **kwargs)


@pytest.mark.parametrize("initial_step", ["backward", "forward"])
def test_energy_balance_closes_to_machine_precision(coarse_example1, initial_step) -> None:
    """``E^{i+1} - E^i = -dt (Q_visc + Q_con + Q_num)`` is an exact identity."""
    result = _run(1, coarse_example1, initial_step=initial_step)
    balance = energy_balance(result)
    assert balance.relative_drift < 1e-9


def test_dissipation_rates_are_non_negative(coarse_example1) -> None:
    """Non-negativity of ``D_con`` is assumption (A1); the other two are structural.

    ``Q_visc`` and ``Q_num`` are sums of squares, hence exactly non-negative.
    ``Q_con = -dx sum P^i v^{i+1/2}`` can dip very slightly negative because the
    penalty is explicit (``P^i`` is built from ``v^{i-1/2}``), which matters only
    at the few steps where a node reverses direction.  The dip is an ``O(dt)``
    artefact: relative to the dissipated energy it is below 1e-8 already at this
    coarse resolution and shrinks with ``dt``.
    """
    result = _run(1, coarse_example1)
    assert result.viscous_dissipation.min() >= 0.0
    assert result.numerical_dissipation.min() >= 0.0
    increments = result.contact_dissipation * coarse_example1.dt
    positive = increments[increments > 0.0].sum()
    negative = -increments[increments < 0.0].sum()
    assert positive > 0.0
    assert negative / positive < 1e-6


def test_energy_is_non_increasing(coarse_example1) -> None:
    result = _run(1, coarse_example1)
    assert np.all(np.diff(result.energy) <= 1e-9 * result.energy[0])
    assert result.energy[-1] < 0.05 * result.energy[0]


def test_contact_dissipation_only_while_in_contact(coarse_example1) -> None:
    """(A2): ``supp(D_con) subset supp(F_con) subset {eta <= 0}``."""
    result = _run(1, coarse_example1)
    active = result.contact_dissipation > 1e-12 * result.energy[0]
    # rate index i belongs to the step (i-1) -> i, whose penalty is built from level i-1
    assert np.all(result.contact_fraction[:-1][active[1:]] > 0.0)


def test_example1_stays_symmetric(coarse_example1) -> None:
    """The initial data of Example 1 is symmetric about x = l / 2, so the solution is too."""
    result = _run(1, coarse_example1)
    # Not bit-exact: the banded Cholesky solve is not symmetric in floating point,
    # so a few ulps accumulate over the time steps.
    assert np.abs(result.eta - result.eta[:, ::-1]).max() < 1e-10
    mask = contact_mask(result)
    assert np.array_equal(mask, mask[:, ::-1])


def test_penetration_scales_like_eps_times_impact_speed(coarse_example1) -> None:
    """The penalized velocity obeys ``v' = -v / eps`` while penetrating, so ``|eta| ~ |v0| eps``.

    ``dt`` is refined together with ``eps`` to keep ``dt / eps < 1``; at
    ``dt / eps >= 2`` the explicit penalty is unstable and the relation breaks.
    """
    depths = {}
    for eps in (4e-3, 2e-3, 1e-3):
        params = coarse_example1.replace(eps=eps, dx=1 / 2000, dt=1 / 2000, T=0.1)
        assert params.is_penalty_stable()
        depths[eps] = penetration_depth(_run(1, params))
    for eps, depth in depths.items():
        assert depth == pytest.approx(50.0 * eps, rel=0.2), depths
    assert depths[1e-3] < depths[2e-3] < depths[4e-3]


def test_example1_contact_set_is_one_symmetric_component(coarse_example1) -> None:
    result = _run(1, coarse_example1)
    comps = components(result, min_size=5)
    assert len(comps) == 1
    biggest = comps[0]
    assert biggest.x_min == pytest.approx(1.0 - biggest.x_max, abs=2 * coarse_example1.dx)
    assert 0.015 < biggest.t_min < 0.05
    assert contact_area(result) == pytest.approx(biggest.area, rel=0.05)


def test_example2_contact_set_has_two_components(coarse_example2) -> None:
    """Section 6.2: "the contact set can consist of multiple disconnected components"."""
    result = _run(2, coarse_example2)
    comps = components(result, min_size=20)
    assert len(comps) == 2
    large, small = comps
    assert large.x_max < small.x_min  # left/large block and right/small block
    assert large.t_min < small.t_min
    assert 0.3 < large.area / contact_area(result) < 1.0


def test_contact_intervals_and_first_contact(coarse_example1) -> None:
    result = _run(1, coarse_example1)
    start = first_contact_time(result)
    assert start is not None and 0.015 < start < 0.05
    index = result.snapshot_index(0.1)
    intervals = contact_intervals(result, index)
    assert len(intervals) == 1
    left, right = intervals[0]
    assert left < 0.5 < right
    assert np.all(result.eta[index][(result.x > left) & (result.x < right)] < 0.0)


def test_contact_mask_modes_are_nested(coarse_example1) -> None:
    result = _run(1, coarse_example1)
    negative = contact_mask(result, mode="negative")
    threshold = contact_mask(result, mode="threshold", tol=1e-3)
    assert np.all(negative <= threshold)
    with pytest.raises(ValueError):
        contact_mask(result, mode="other")  # type: ignore[arg-type]


def test_no_contact_means_no_first_contact_time() -> None:
    params = Params(length=1.0, T=0.1, dx=1 / 200, dt=1 / 200, alpha=0.01, eps=5e-3, h=1.0)
    x = np.linspace(0.0, 1.0, params.N + 1)
    result = solve(params, np.full_like(x, 1.0), np.zeros_like(x))
    assert first_contact_time(result) is None
    assert contact_area(result) == 0.0
    assert components(result) == []
