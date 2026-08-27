"""Discretization parameters for the contact problem of arXiv:2412.06185.

The paper (Section 6) fixes ``l = 1``, ``dt = dx = 1/5000``, ``alpha = 0.01`` and
``eps = 0.0005`` for both numerical examples; only the final time ``T`` and the
initial data differ.  The endpoint height ``h`` is not stated numerically in the
text -- the boundary condition is written as ``eta^i_0 = eta^i_N = 0`` while the
PDE (1.3) prescribes ``eta(t, 0) = eta(t, l) = h > 0`` and Figures 2/4 clearly
show the endpoints sitting at 1.  We follow the figures and the PDE, see
``plan.md`` section 4 item (a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = ["EXAMPLE1", "EXAMPLE2", "Params"]

_GRID_TOL = 1e-9


@dataclass(frozen=True)
class Params:
    """Parameters of the finite difference scheme of Section 6.

    Attributes
    ----------
    length:
        Length ``l`` of the string; the spatial domain is ``(0, length)``.
    T:
        Final time.
    dx, dt:
        Spatial and temporal step sizes.
    alpha:
        Viscoelasticity coefficient in front of ``partial_txx eta``.
    eps:
        Penalization parameter; the repulsive force is
        ``(1 / eps) * chi_{eta < 0} * (partial_t eta)^-``.
    h:
        Prescribed height of both endpoints.
    """

    length: float = 1.0
    T: float = 0.3
    dx: float = 1.0 / 5000.0
    dt: float = 1.0 / 5000.0
    alpha: float = 0.01
    eps: float = 5e-4
    h: float = 1.0

    def __post_init__(self) -> None:
        for name in ("length", "T", "dx", "dt", "alpha", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        if not math.isfinite(self.h) or self.h <= 0.0:
            # eta_0 >= c > 0 is assumed in Theorem 2.1, and h > 0 in (1.3).
            raise ValueError(f"h must be a finite positive number, got {self.h!r}")
        if self.N < 2:
            raise ValueError(f"length / dx = {self.length / self.dx} gives fewer than 2 cells")
        if self.M < 1:
            raise ValueError(f"T / dt = {self.T / self.dt} gives fewer than 1 time step")
        for name, total, step in (("length", self.length, self.dx), ("T", self.T, self.dt)):
            ratio = total / step
            if abs(ratio - round(ratio)) > _GRID_TOL * max(1.0, ratio):
                raise ValueError(
                    f"{name} / {'dx' if name == 'length' else 'dt'} = {ratio} is not an integer; "
                    "choose step sizes that divide the domain exactly"
                )

    @property
    def N(self) -> int:
        """Number of spatial cells; the grid has ``N + 1`` nodes ``x_0, ..., x_N``."""
        return round(self.length / self.dx)

    @property
    def M(self) -> int:
        """Number of time steps; the solution is computed at ``t^0, ..., t^M``."""
        return round(self.T / self.dt)

    @property
    def penalty_ratio(self) -> float:
        """``dt / eps``.

        During penetration (``eta < 0`` and ``v < 0``) the explicit penalty term
        updates the velocity roughly as ``v <- (1 - dt / eps) v``.  The paper's
        setting gives ``dt / eps = 0.4``, i.e. a monotone decay to zero (no
        bounce).  Values ``>= 1`` oscillate and ``>= 2`` are unstable.
        """
        return self.dt / self.eps

    @property
    def implicit_coefficient(self) -> float:
        """``alpha * dt + dt**2``: the factor multiplying the implicit Laplacian."""
        return self.alpha * self.dt + self.dt**2

    def is_penalty_stable(self) -> bool:
        """Whether the explicit penalty force decays monotonically (``dt / eps < 1``)."""
        return self.penalty_ratio < 1.0

    def replace(self, **changes: float) -> Params:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)

    def summary(self) -> str:
        return (
            f"l={self.length:g} T={self.T:g} dx={self.dx:g} dt={self.dt:g} "
            f"alpha={self.alpha:g} eps={self.eps:g} h={self.h:g} "
            f"N={self.N} M={self.M} dt/eps={self.penalty_ratio:g}"
        )


#: Section 6.1, Example 1.
EXAMPLE1 = Params(T=0.3)

#: Section 6.2, Example 2.
EXAMPLE2 = Params(T=0.5)
