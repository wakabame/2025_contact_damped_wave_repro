"""Correctness of the scheme itself, away from and at the obstacle."""

from __future__ import annotations

import numpy as np
import pytest

from contact_damped_wave.params import Params
from contact_damped_wave.solver import solve

ALPHA = 0.01
AMPLITUDE = 0.1
H = 1.0


def _decay_rates(alpha: float = ALPHA, mode: int = 1, length: float = 1.0) -> tuple[float, float]:
    """Roots of ``lambda^2 + alpha (k pi / l)^2 lambda + (k pi / l)^2 = 0``."""
    mu = (mode * np.pi / length) ** 2
    root = np.roots([1.0, alpha * mu, mu])[0]
    return float(root.real), float(abs(root.imag))


def _exact(t: float, x: np.ndarray) -> np.ndarray:
    """``h + a sin(pi x) e^{sigma t} cos(omega t)``: a solution with no contact."""
    sigma, omega = _decay_rates()
    return H + AMPLITUDE * np.sin(np.pi * x) * np.exp(sigma * t) * np.cos(omega * t)


def _exact_v0(x: np.ndarray) -> np.ndarray:
    sigma, _ = _decay_rates()
    return AMPLITUDE * np.sin(np.pi * x) * sigma


def _run_smooth(n: int, final_time: float = 0.5) -> tuple[Params, np.ndarray, np.ndarray]:
    params = Params(length=1.0, T=final_time, dx=1 / n, dt=1 / n, alpha=ALPHA, eps=5e-2, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    result = solve(params, _exact(0.0, x), _exact_v0(x))
    return params, x, result.eta[-1]


def test_no_contact_in_the_smooth_reference_solution() -> None:
    params = Params(length=1.0, T=0.5, dx=1 / 400, dt=1 / 400, alpha=ALPHA, eps=5e-2, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    result = solve(params, _exact(0.0, x), _exact_v0(x))
    assert result.min_eta.min() > 0.5
    assert np.all(result.contact_fraction == 0.0)
    assert np.all(result.contact_work == 0.0)


def test_first_order_convergence_to_the_exact_solution() -> None:
    """The scheme is first order in dt (implicit stiffness term, one-sided ``partial_txx``)."""
    final_time = 0.5
    errors = []
    for n in (200, 400, 800, 1600):
        _, x, eta = _run_smooth(n, final_time)
        errors.append(np.abs(eta - _exact(final_time, x)).max())
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 0.9), f"observed rates {rates}"
    assert np.all(rates < 1.4), f"observed rates {rates}"


def test_rest_state_is_preserved() -> None:
    params = Params(length=1.0, T=0.2, dx=1 / 200, dt=1 / 200, alpha=ALPHA, eps=5e-3, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    result = solve(params, np.full_like(x, H), np.zeros_like(x))
    assert np.abs(result.eta - H).max() < 1e-12
    assert result.energy.max() < 1e-20


def test_scheme_satisfies_the_printed_difference_equation() -> None:
    """One-step residual of the paper's stencil, rebuilt from the snapshots.

    On the interior nodes the scheme must satisfy, exactly,

        (eta^{i+1} - 2 eta^i + eta^{i-1}) / dt^2
            = alpha (D eta^{i+1} - D eta^i) / dt + D eta^{i+1} + P^i,

    with ``D`` the second central difference and ``P^i`` the explicit penalty
    built from level ``i``.  The run is chosen so that contact actually occurs,
    which exercises the penalty branch of the equation as well.
    """
    params = Params(length=1.0, T=0.05, dx=1 / 200, dt=1 / 200, alpha=ALPHA, eps=1.25e-2, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    eta0 = H + 0.5 * np.sin(np.pi * x) ** 2
    result = solve(params, eta0, np.full_like(x, -50.0))
    assert result.contact_fraction.max() > 0.0  # the penalty branch is exercised

    def second_difference(u: np.ndarray) -> np.ndarray:
        return (u[2:] - 2.0 * u[1:-1] + u[:-2]) / params.dx**2

    eta, dt = result.eta, params.dt
    ghost = eta[0] - dt * result.v[0]  # eta^{-1} of the backward initial step
    levels = np.concatenate(([ghost], eta))
    worst = 0.0
    for i in range(1, levels.shape[0] - 1):
        previous, current, new = levels[i - 1], levels[i], levels[i + 1]
        penalty = np.where(current < 0.0, np.maximum(0.0, -(current - previous) / dt), 0.0)
        lhs = (new - 2.0 * current + previous)[1:-1] / dt**2
        rhs = (
            params.alpha * (second_difference(new) - second_difference(current)) / dt
            + second_difference(new)
            + penalty[1:-1] / params.eps
        )
        worst = max(worst, float(np.abs(lhs - rhs).max() / np.abs(rhs).max()))
    assert worst < 1e-11


def test_boundary_values_are_clamped() -> None:
    params = Params(length=1.0, T=0.1, dx=1 / 200, dt=1 / 200, alpha=ALPHA, eps=5e-3, h=2.5)
    x = np.linspace(0.0, 1.0, params.N + 1)
    result = solve(params, np.full_like(x, 2.5), np.full_like(x, -3.0))
    assert np.all(result.eta[:, 0] == 2.5)
    assert np.all(result.eta[:, -1] == 2.5)
    assert np.all(result.v[:, 0] == 0.0)
    assert np.all(result.v[:, -1] == 0.0)


def test_stored_velocity_at_time_zero_is_the_initial_velocity() -> None:
    """``initial_step="backward"`` is chosen so that ``v^0`` is exactly ``v0``."""
    params = Params(length=1.0, T=0.1, dx=1 / 200, dt=1 / 200, alpha=ALPHA, eps=5e-3, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    v0 = -3.0 * np.sin(np.pi * x)
    result = solve(params, np.full_like(x, H), v0)
    assert result.v[0] == pytest.approx(v0, abs=1e-12)


def test_store_every_keeps_first_and_last_levels() -> None:
    params = Params(length=1.0, T=0.1, dx=1 / 200, dt=1 / 200, alpha=ALPHA, eps=5e-3, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    full = solve(params, np.full_like(x, H), -np.sin(np.pi * x))
    strided = solve(params, np.full_like(x, H), -np.sin(np.pi * x), store_every=7)
    assert strided.t[0] == 0.0
    assert strided.t[-1] == pytest.approx(params.T)
    assert strided.eta[-1] == pytest.approx(full.eta[-1], abs=1e-14)
    assert strided.energy == pytest.approx(full.energy, abs=1e-12)


def test_result_roundtrip(tmp_path) -> None:
    params = Params(length=1.0, T=0.05, dx=1 / 100, dt=1 / 100, alpha=ALPHA, eps=5e-2, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    result = solve(params, _exact(0.0, x), _exact_v0(x))
    from contact_damped_wave.solver import Result

    # Bit-exact: an archive that only round-trips approximately is not a checkpoint.
    written = result.save(tmp_path / "r.npz")
    loaded = Result.load(written)
    assert loaded.params == params
    for field in (
        "x",
        "t",
        "eta",
        "v",
        "t_full",
        "energy",
        "viscous_dissipation",
        "contact_work",
        "numerical_dissipation",
        "min_eta",
        "contact_fraction",
    ):
        assert np.array_equal(getattr(loaded, field), getattr(result, field)), field
    assert loaded.store_every == result.store_every
    assert loaded.initial_step == result.initial_step
    assert loaded.balance_start_index == result.balance_start_index

    # A name without the extension must still be loadable from the returned path.
    written_bare = result.save(tmp_path / "bare")
    assert written_bare.suffix == ".npz"
    assert np.array_equal(Result.load(written_bare).eta, result.eta)


@pytest.mark.parametrize("bad", [{"store_every": 0}, {"initial_step": "middle"}])
def test_invalid_options_rejected(bad: dict) -> None:
    params = Params(length=1.0, T=0.05, dx=1 / 100, dt=1 / 100, alpha=ALPHA, eps=5e-2, h=H)
    x = np.linspace(0.0, 1.0, params.N + 1)
    with pytest.raises(ValueError):
        solve(params, np.full_like(x, H), np.zeros_like(x), **bad)


def test_shape_mismatch_rejected() -> None:
    params = Params(length=1.0, T=0.05, dx=1 / 100, dt=1 / 100, alpha=ALPHA, eps=5e-2, h=H)
    with pytest.raises(ValueError):
        solve(params, np.ones(10), np.zeros(10))
