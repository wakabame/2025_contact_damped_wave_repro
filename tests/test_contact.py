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
    at the few steps where a node reverses direction -- which is why it is named
    ``contact_work`` rather than a dissipation.  The dip is an ``O(dt)`` artefact:
    relative to the energy it removes it is below 1e-8 already at this coarse
    resolution and shrinks with ``dt``.
    """
    result = _run(1, coarse_example1)
    assert result.viscous_dissipation.min() >= 0.0
    assert result.numerical_dissipation.min() >= 0.0
    increments = result.contact_work * coarse_example1.dt
    positive = increments[increments > 0.0].sum()
    negative = -increments[increments < 0.0].sum()
    assert positive > 0.0
    assert negative / positive < 1e-6


def test_energy_is_non_increasing(coarse_example1) -> None:
    result = _run(1, coarse_example1)
    assert np.all(np.diff(result.energy) <= 1e-9 * result.energy[0])
    assert result.energy[-1] < 0.05 * result.energy[0]


def test_contact_work_matches_an_independent_reconstruction(coarse_example1) -> None:
    """(A2): the penalty force lives on ``{eta < 0 and v < 0}`` and produces ``Q_con``.

    With ``store_every=1`` the snapshots contain every level, so ``P^i`` and the
    work rate can be rebuilt without looking at the solver's internals: entry
    ``k`` of ``contact_work`` belongs to the step ``k-1 -> k``, whose penalty is
    built from ``eta`` and the backward velocity at level ``k-1`` and multiplied
    by the forward velocity, i.e. the backward velocity at level ``k``.
    """
    params = coarse_example1
    result = _run(1, params)
    eta, v = result.eta, result.v
    penalty = np.where(eta[:-1] < 0.0, np.maximum(0.0, -v[:-1]) / params.eps, 0.0)
    reconstructed = -params.dx * np.sum(penalty * v[1:], axis=1)
    recorded = result.contact_work[1:]
    scale = np.abs(recorded).max()
    assert scale > 0.0  # the run does reach the obstacle
    assert np.allclose(reconstructed, recorded, rtol=1e-10, atol=1e-12 * scale)
    # Steps at which no node is both penetrating and moving down carry exactly
    # zero force, hence exactly zero recorded work -- that is (A2).
    force_possible = np.any((eta[:-1] < 0.0) & (v[:-1] < 0.0), axis=1)
    assert np.all(recorded[~force_possible] == 0.0)


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
        assert params.is_penalty_monotone()
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


def test_time_weights_sum_to_the_time_span(coarse_example1) -> None:
    """Regression: weights come from ``result.t``, not from ``dt * store_every``."""
    from contact_damped_wave.diagnostics import time_weights

    for store_every in (1, 7, 13, 10_000):
        result = _run(1, coarse_example1, store_every=store_every)
        weights = time_weights(result)
        assert weights.shape == result.t.shape
        assert np.all(weights >= 0.0)
        assert weights.sum() == pytest.approx(coarse_example1.T)


def test_contact_area_is_stable_under_snapshot_stride(coarse_example1) -> None:
    """Regression: a stride that does not divide M used to inflate the area."""
    reference = contact_area(_run(1, coarse_example1))
    for store_every in (2, 7, 13):
        strided = contact_area(_run(1, coarse_example1, store_every=store_every))
        assert strided == pytest.approx(reference, rel=0.05), store_every
    # A stride larger than the whole run keeps only t = 0 and t = T; the area must
    # stay bounded by the domain measure instead of being counted stride times.
    coarse = contact_area(_run(1, coarse_example1, store_every=10_000))
    assert 0.0 <= coarse <= coarse_example1.length * coarse_example1.T


def test_drift_of_a_rest_state_is_reported_honestly() -> None:
    """Regression: dividing by an exactly zero initial budget gave ~1e279.

    For a rest state both the energy and the drift are pure round-off, so the
    meaningful statement is that the *absolute* drift is negligible and that the
    relative figure is at least finite rather than astronomically large.
    """
    params = Params(length=1.0, T=0.1, dx=1 / 200, dt=1 / 200, alpha=0.01, eps=5e-3, h=1.0)
    x = np.linspace(0.0, 1.0, params.N + 1)
    balance = energy_balance(solve(params, np.full_like(x, 1.0), np.zeros_like(x)))
    assert balance.absolute_drift < 1e-20
    assert np.isfinite(balance.relative_drift)
    assert balance.relative_drift < 1e3


def test_relative_drift_is_meaningful_when_there_is_energy(coarse_example1) -> None:
    result = _run(1, coarse_example1)
    balance = energy_balance(result)
    assert balance.relative_drift < 1e-9
    assert balance.absolute_drift < 1e-6 * balance.energy[0]


def test_diagonal_front_is_one_component(coarse_example1) -> None:
    """``connectivity=1`` cuts a front that advances one node per step into pieces."""
    result = _run(1, coarse_example1)
    assert len(components(result, min_size=5, connectivity=2)) == 1
    # A strictly diagonal band in the (t, x) plane makes the claim sharp: one
    # component with connectivity=2, n isolated cells with connectivity=1.
    mask = np.zeros((result.t.size, result.x.size), dtype=bool)
    n = min(result.t.size, result.x.size, 40)
    mask[np.arange(n), np.arange(n)] = True
    assert len(components(result, mask, connectivity=2)) == 1
    assert len(components(result, mask, connectivity=1)) == n


def test_initial_data_meets_the_boundary_condition_for_any_h(coarse_example1) -> None:
    """Regression: built-in data used to hard-code endpoint height 1.

    Clamping data that disagrees with ``h`` injects ``~ (eta0(0) - h)^2 / dx`` of
    elastic energy, which diverges under refinement; ``solve`` now warns about it.
    """
    import warnings

    for h in (0.5, 1.0, 2.0):
        params = coarse_example1.replace(h=h)
        _, x, eta0, v0 = initial_data(1, params)
        assert eta0[0] == pytest.approx(h)
        assert eta0[-1] == pytest.approx(h)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = solve(params, eta0, v0)
        assert result.energy[0] == pytest.approx(_run(1, coarse_example1).energy[0], rel=1e-9)


def test_clamping_inconsistent_initial_data_warns(coarse_example2) -> None:
    """The ``paper-literal`` data has ``eta^0(0) = 0 != h``; that must not pass silently."""
    _, _, eta0, v0 = initial_data(2, coarse_example2, "paper-literal")
    with pytest.warns(RuntimeWarning, match="boundary condition"):
        solve(coarse_example2, eta0, v0)


def test_example3_contact_front_travels_to_the_right(coarse_example3) -> None:
    """Example 3 (ours): one contact interval that translates, and never bounces.

    This is what distinguishes it from the paper's two examples, whose contact
    sets are triangles shrinking towards a single point.
    """
    result = _run(3, coarse_example3)
    mask = contact_mask(result)
    touching = np.flatnonzero(mask.any(axis=1))
    assert touching.size > 0
    # The contact set is a single connected band in the (t, x) plane ...
    assert len(components(result, mask, min_size=20)) == 1
    # ... which at (almost) every time is a single interval in x ...
    single_interval = sum(len(contact_intervals(result, int(i), mask)) == 1 for i in touching)
    assert single_interval > 0.95 * touching.size
    # ... whose two edges both move to the right, monotonically in the mean.
    left = np.array([result.x[np.flatnonzero(mask[i])[0]] for i in touching])
    right = np.array([result.x[np.flatnonzero(mask[i])[-1]] for i in touching])
    assert left[-1] > left[0] + 0.5
    assert right[-1] > right[0] + 0.3
    # A quarter of the string is still free at the right end when contact ends.
    assert right.max() < 0.95

    # Inelastic contact: the penetration stays at the O(eps) level predicted by
    # the penetration ODE v' = -v / eps, i.e. about |v^0|_max * eps.
    _, _, _, v0 = initial_data(3, coarse_example3)
    assert penetration_depth(result) < 2.0 * np.abs(v0).max() * coarse_example3.eps
