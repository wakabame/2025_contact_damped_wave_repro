"""Finite difference solver for the penalized contact problem (Section 6).

The paper discretizes

    partial_tt eta - alpha partial_txx eta - partial_xx eta
        = (1 / eps) chi_{eta < 0} (partial_t eta)^-

as (Section 6, with ``i`` the time index and ``j`` the space index)::

    (eta^{i+1}_j - 2 eta^i_j + eta^{i-1}_j) / dt^2
      - (alpha / dt) [ (D eta^{i+1})_j - (D eta^i)_j ]
      - (D eta^{i+1})_j
      = (1 / eps) chi_{eta^i_j < 0} ( (eta^i_j - eta^{i-1}_j) / dt )^-

where ``(D u)_j = (u_{j+1} - 2 u_j + u_{j-1}) / dx^2`` and ``f^- = max(0, -f)``.
The penalization force is evaluated *explicitly*, from the previous time step.
Rearranging, every step is one symmetric positive definite tridiagonal solve::

    [I - (alpha dt + dt^2) D] eta^{i+1}
        = 2 eta^i - eta^{i-1} - alpha dt (D eta^i) + dt^2 P^i

with ``P^i_j = (1 / eps) chi_{eta^i_j < 0} ( (eta^i_j - eta^{i-1}_j) / dt )^-``.
The matrix is constant in time, so it is Cholesky-factorized once.

Boundary values are held at ``eta^i_0 = eta^i_N = h`` (see ``plan.md`` item (a)).

Discrete energy identity
------------------------
Writing ``v^{i+1/2} = (eta^{i+1} - eta^i) / dt`` and testing the scheme with
``dx v^{i+1/2}`` (summation by parts is exact because ``v`` vanishes at the two
clamped endpoints) gives, for

    E^i = (dx/2) sum_j (v^{i-1/2}_j)^2 + (dx/2) sum_j (delta_x eta^i_j)^2,

the *exact* identity

    E^{i+1} - E^i = -dt [ Q_visc + Q_num + Q_con ],
    Q_visc = alpha dx sum_j (delta_x v^{i+1/2}_j)^2                    >= 0,
    Q_num  = (dx / 2 dt) sum_j (v^{i+1/2}_j - v^{i-1/2}_j)^2
             + (dt / 2) dx sum_j (delta_x v^{i+1/2}_j)^2               >= 0,
    Q_con  = -dx sum_j P^i_j v^{i+1/2}_j .

``Q_visc`` and ``Q_num`` are sums of squares, hence non-negative.  ``Q_visc`` is
the discrete counterpart of ``alpha int |partial_tx eta|^2`` in the energy
balance (1.4) and ``Q_num`` is the purely numerical dissipation of the scheme
(``O(dt)``, it vanishes in the limit).

``Q_con`` is the *work* extracted by the penalty force, the discrete counterpart
of ``int D_con``.  It is **not** non-negative step by step: ``P^i`` is built from
the old velocity ``v^{i-1/2}`` but multiplied by the new one ``v^{i+1/2}``, so at
the few steps where a node reverses direction ``Q_con`` can dip slightly below
zero.  The dip is an ``O(dt)`` artefact of the explicit penalization -- relative
to the energy it removes it is below ``4e-9`` at ``dx = dt = 1/500`` and below
``4e-14`` at the paper's resolution -- so the field is called ``contact_work``
rather than a dissipation.  Its time integral is positive.

:func:`solve` records the three rates at every step so that the balance closes to
machine precision.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded

from .params import Params

__all__ = ["InitialStep", "Result", "solve"]

InitialStep = Literal["backward", "forward"]


@dataclass
class Result:
    """Output of :func:`solve`.

    Field snapshots (``eta``, ``v``) are stored every ``store_every`` steps,
    while the scalar diagnostics are recorded at *every* step.

    Attributes
    ----------
    params:
        The parameters used.
    x:
        Spatial grid, shape ``(N + 1,)``.
    t:
        Times of the stored snapshots, shape ``(n_stored,)``.
    eta:
        Displacement snapshots, shape ``(n_stored, N + 1)``.
    v:
        Velocity snapshots ``v^i = (eta^i - eta^{i-1}) / dt``, same shape as
        ``eta``.  This is the backward difference that the penalty term uses;
        ``v[0]`` is exactly the prescribed ``v0`` for ``initial_step="backward"``.
    t_full:
        Every time level ``t^0, ..., t^M``, shape ``(M + 1,)``.
    energy:
        Discrete energy ``0.5 dx sum (v^i)^2 + 0.5 dx sum ((delta_x eta^i))^2``.
    viscous_dissipation, contact_work, numerical_dissipation:
        The rates ``Q_visc``, ``Q_con`` and ``Q_num`` of the discrete energy
        identity above.  Only the first and last are non-negative; see the module
        docstring for why ``contact_work`` can dip slightly negative.  Entry ``i``
        belongs to the step ``i - 1 -> i``, so that
        ``E^i = E^{i_0} - dt * sum_{k>i_0} (Q_visc + Q_num + Q_con)[k]`` where
        ``i_0 = balance_start_index``.
    min_eta:
        ``min_j eta^i_j``; the penetration depth is ``max(0, -min_eta)``.
    contact_fraction:
        Fraction of nodes with ``eta^i_j < 0`` (the set where the penalty acts).
    """

    params: Params
    x: np.ndarray
    t: np.ndarray
    eta: np.ndarray
    v: np.ndarray
    t_full: np.ndarray
    energy: np.ndarray
    viscous_dissipation: np.ndarray
    contact_work: np.ndarray
    numerical_dissipation: np.ndarray
    min_eta: np.ndarray
    contact_fraction: np.ndarray
    store_every: int
    initial_step: InitialStep
    balance_start_index: int = 0

    def snapshot_index(self, time: float) -> int:
        """Index of the stored snapshot closest to ``time``."""
        return int(np.argmin(np.abs(self.t - time)))

    def snapshot(self, time: float) -> tuple[float, np.ndarray]:
        """Return ``(actual_time, eta)`` of the stored snapshot closest to ``time``."""
        idx = self.snapshot_index(time)
        return float(self.t[idx]), self.eta[idx]

    def save(self, path: str | Path) -> Path:
        """Save the result to a ``.npz`` archive and return the path actually written.

        ``numpy.savez_compressed`` appends ``.npz`` when the name lacks it, so the
        returned path -- not the argument -- is what :meth:`load` must be given.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            t=self.t,
            eta=self.eta,
            v=self.v,
            t_full=self.t_full,
            energy=self.energy,
            viscous_dissipation=self.viscous_dissipation,
            contact_work=self.contact_work,
            numerical_dissipation=self.numerical_dissipation,
            min_eta=self.min_eta,
            contact_fraction=self.contact_fraction,
            store_every=self.store_every,
            initial_step=self.initial_step,
            balance_start_index=self.balance_start_index,
            params=np.array(
                [
                    self.params.length,
                    self.params.T,
                    self.params.dx,
                    self.params.dt,
                    self.params.alpha,
                    self.params.eps,
                    self.params.h,
                ]
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Result:
        """Load a result previously written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            length, t_final, dx, dt, alpha, eps, h = (float(value) for value in data["params"])
            return cls(
                params=Params(length=length, T=t_final, dx=dx, dt=dt, alpha=alpha, eps=eps, h=h),
                x=data["x"],
                t=data["t"],
                eta=data["eta"],
                v=data["v"],
                t_full=data["t_full"],
                energy=data["energy"],
                viscous_dissipation=data["viscous_dissipation"],
                contact_work=data["contact_work"],
                numerical_dissipation=data["numerical_dissipation"],
                min_eta=data["min_eta"],
                contact_fraction=data["contact_fraction"],
                store_every=int(data["store_every"]),
                initial_step=str(data["initial_step"]),  # type: ignore[arg-type]
                balance_start_index=int(data["balance_start_index"]),
            )


def _laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """Second central difference of ``u`` on the interior nodes ``1, ..., N-1``."""
    return (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2


def _penalty(eta: np.ndarray, eta_prev: np.ndarray, params: Params) -> np.ndarray:
    """``P^i = (1/eps) chi_{eta^i < 0} ((eta^i - eta^{i-1}) / dt)^-`` on all nodes."""
    velocity = (eta - eta_prev) / params.dt
    return np.where(eta < 0.0, np.maximum(0.0, -velocity) / params.eps, 0.0)


def _energy(eta: np.ndarray, velocity: np.ndarray, dx: float) -> float:
    kinetic = 0.5 * dx * float(np.sum(velocity**2))
    elastic = 0.5 * dx * float(np.sum((np.diff(eta) / dx) ** 2))
    return kinetic + elastic


def _budget_rates(
    v_back: np.ndarray, v_fwd: np.ndarray, penalty: np.ndarray, params: Params
) -> tuple[float, float, float]:
    """Return ``(Q_visc, Q_con, Q_num)`` of the discrete energy identity.

    ``v_back = v^{i-1/2}``, ``v_fwd = v^{i+1/2}`` and ``penalty = P^i``.  Both
    velocities vanish at the clamped endpoints, so the summation by parts used to
    derive the identity is exact and ``E^{i+1} - E^i = -dt (Q_visc + Q_con + Q_num)``
    holds to machine precision.
    """
    dx, dt = params.dx, params.dt
    grad_v_fwd = float(np.sum((np.diff(v_fwd) / dx) ** 2)) * dx
    viscous = params.alpha * grad_v_fwd
    numerical = 0.5 * dx * float(np.sum((v_fwd - v_back) ** 2)) / dt + 0.5 * dt * grad_v_fwd
    contact = -dx * float(np.sum(penalty * v_fwd))
    return viscous, contact, numerical


def solve(
    params: Params,
    eta0: np.ndarray,
    v0: np.ndarray,
    *,
    store_every: int = 1,
    initial_step: InitialStep = "backward",
    progress: bool = False,
) -> Result:
    """Run the scheme of Section 6.

    Parameters
    ----------
    params:
        Discretization parameters.
    eta0, v0:
        Initial displacement and velocity sampled on the ``N + 1`` grid nodes.
        The endpoint values of ``eta0`` are overwritten by ``params.h`` and the
        endpoint values of ``v0`` by ``0`` (the endpoints are clamped by (1.3)).
    store_every:
        Keep one field snapshot every ``store_every`` steps.  The first and last
        time levels are always stored.
    initial_step:
        How to start the three-level recursion, which the paper does not specify
        (``plan.md`` item (d)).  ``"backward"`` (default) sets
        ``eta^{-1} = eta^0 - dt v^0`` so that the backward difference at step 0
        reproduces ``v^0`` exactly; ``"forward"`` sets ``eta^1 = eta^0 + dt v^0``
        and starts the recursion at ``i = 1``.
    progress:
        Print a one-line progress message every 10% of the steps.
    """
    n_nodes = params.N + 1
    eta0 = np.asarray(eta0, dtype=float).reshape(-1)
    v0 = np.asarray(v0, dtype=float).reshape(-1)
    if eta0.shape != (n_nodes,) or v0.shape != (n_nodes,):
        raise ValueError(
            f"eta0 and v0 must have shape ({n_nodes},), got {eta0.shape} and {v0.shape}"
        )
    if store_every < 1:
        raise ValueError(f"store_every must be >= 1, got {store_every}")
    if initial_step not in ("backward", "forward"):
        raise ValueError(f"initial_step must be 'backward' or 'forward', got {initial_step!r}")

    dx, dt, h = params.dx, params.dt, params.h
    n_steps = params.M

    eta0 = eta0.copy()
    v0 = v0.copy()
    mismatch = max(abs(eta0[0] - h), abs(eta0[-1] - h))
    if mismatch > 1e-12 * max(1.0, abs(h)):
        # Clamping (1.3) onto data that does not meet it puts a jump across the
        # first/last cell whose elastic energy ~ mismatch^2 / dx grows without
        # bound as the grid is refined.  Only "paper-literal" data does this.
        warnings.warn(
            f"eta0 disagrees with the boundary condition eta = h = {h:g} by {mismatch:g} "
            f"at an endpoint; clamping it adds about {mismatch**2 / dx:.3g} of spurious "
            "elastic energy, which diverges as dx -> 0",
            RuntimeWarning,
            stacklevel=2,
        )
    eta0[0] = eta0[-1] = h
    v0[0] = v0[-1] = 0.0

    # Constant SPD tridiagonal matrix I - (alpha dt + dt^2) D on the interior nodes.
    coefficient = params.implicit_coefficient
    off = -coefficient / dx**2
    diag = 1.0 + 2.0 * coefficient / dx**2
    n_interior = params.N - 1
    banded = np.zeros((2, n_interior))
    banded[0, 1:] = off
    banded[1, :] = diag
    factor = cholesky_banded(banded, lower=False)

    stored_indices = list(range(0, n_steps + 1, store_every))
    if stored_indices[-1] != n_steps:
        stored_indices.append(n_steps)
    index_to_slot = {step: slot for slot, step in enumerate(stored_indices)}
    n_stored = len(stored_indices)

    eta_out = np.empty((n_stored, n_nodes))
    v_out = np.empty((n_stored, n_nodes))
    energy = np.empty(n_steps + 1)
    viscous = np.zeros(n_steps + 1)
    contact = np.zeros(n_steps + 1)
    numerical = np.zeros(n_steps + 1)
    min_eta = np.empty(n_steps + 1)
    contact_fraction = np.empty(n_steps + 1)

    def record_state(step: int, eta: np.ndarray, velocity: np.ndarray) -> None:
        energy[step] = _energy(eta, velocity, dx)
        min_eta[step] = float(eta.min())
        contact_fraction[step] = float(np.count_nonzero(eta < 0.0)) / n_nodes
        slot = index_to_slot.get(step)
        if slot is not None:
            eta_out[slot] = eta
            v_out[slot] = velocity

    if initial_step == "backward":
        eta_prev = eta0 - dt * v0
        eta_cur = eta0
        first_step = 0
        balance_start_index = 0
    else:
        eta_prev = eta0
        eta_cur = eta0 + dt * v0
        eta_cur[0] = eta_cur[-1] = h
        first_step = 1
        # The 0 -> 1 increment is prescribed, not produced by the scheme, so the
        # energy identity only holds from level 1 on.
        balance_start_index = 1
        record_state(0, eta0, v0)
    eta_prev[0] = eta_prev[-1] = h

    rhs = np.empty(n_interior)
    report_every = max(1, n_steps // 10)

    for step in range(first_step, n_steps + 1):
        v_back = (eta_cur - eta_prev) / dt
        record_state(step, eta_cur, v_back)
        if step == n_steps:
            break

        penalty = _penalty(eta_cur, eta_prev, params)
        lap_cur = _laplacian(eta_cur, dx)
        rhs[:] = (
            2.0 * eta_cur[1:-1]
            - eta_prev[1:-1]
            - params.alpha * dt * lap_cur
            + dt**2 * penalty[1:-1]
        )
        # Dirichlet contributions of the implicit Laplacian.
        rhs[0] -= off * h
        rhs[-1] -= off * h

        eta_next = np.empty(n_nodes)
        eta_next[0] = eta_next[-1] = h
        eta_next[1:-1] = cho_solve_banded((factor, False), rhs)

        v_fwd = (eta_next - eta_cur) / dt
        viscous[step + 1], contact[step + 1], numerical[step + 1] = _budget_rates(
            v_back, v_fwd, penalty, params
        )
        eta_prev, eta_cur = eta_cur, eta_next

        if progress and step % report_every == 0:
            print(f"  step {step}/{n_steps}  t={step * dt:.4f}  min eta={eta_cur.min():+.4e}")

    return Result(
        params=params,
        x=np.linspace(0.0, params.length, n_nodes),
        t=np.array([step * dt for step in stored_indices]),
        eta=eta_out,
        v=v_out,
        t_full=np.arange(n_steps + 1) * dt,
        energy=energy,
        viscous_dissipation=viscous,
        contact_work=contact,
        numerical_dissipation=numerical,
        min_eta=min_eta,
        contact_fraction=contact_fraction,
        store_every=store_every,
        initial_step=initial_step,
        balance_start_index=balance_start_index,
    )
