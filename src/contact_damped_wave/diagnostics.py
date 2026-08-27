"""Post-processing of a :class:`~contact_damped_wave.solver.Result`.

Quantities computed here are the ones the paper's figures and theorems talk
about: the contact set in the ``(t, x)`` plane, its connected components, and
the energy balance (1.4)

    d/dt [ (1/2) int |partial_t eta|^2 + (1/2) int |partial_x eta|^2 ]
        + int |partial_tx eta|^2 + int D_con = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from .solver import Result

__all__ = [
    "ContactMode",
    "EnergyBalance",
    "components",
    "contact_area",
    "contact_intervals",
    "contact_mask",
    "energy_balance",
    "first_contact_time",
    "penetration_depth",
    "summarize",
]

ContactMode = Literal["negative", "threshold"]


def contact_mask(result: Result, mode: ContactMode = "negative", tol: float = 1e-12) -> np.ndarray:
    """Boolean contact set on the stored snapshots, shape ``(n_stored, N + 1)``.

    ``mode="negative"`` marks ``eta < 0``, i.e. exactly the set where the penalty
    force acts in the scheme; ``mode="threshold"`` marks ``eta <= tol``, which is
    closer to ``{eta = 0}`` in the limit problem.  The paper does not say which
    one Figures 3 and 5 use (``plan.md`` item (e)), so both are provided.
    """
    if mode == "negative":
        return result.eta < 0.0
    if mode == "threshold":
        return result.eta <= tol
    raise ValueError(f"mode must be 'negative' or 'threshold', got {mode!r}")


@dataclass(frozen=True)
class Component:
    """A connected component of the contact set in the ``(t, x)`` plane."""

    size: int
    t_min: float
    t_max: float
    x_min: float
    x_max: float
    area: float

    def describe(self) -> str:
        return (
            f"t in [{self.t_min:.4f}, {self.t_max:.4f}], "
            f"x in [{self.x_min:.4f}, {self.x_max:.4f}], area={self.area:.5f}"
        )


def components(
    result: Result, mask: np.ndarray | None = None, min_size: int = 1
) -> list[Component]:
    """Connected components of the contact set, largest first.

    Components smaller than ``min_size`` grid cells are discarded; this filters
    out single-node speckle produced by the penalization near the contact front.
    """
    mask = contact_mask(result) if mask is None else mask
    labels, count = ndimage.label(mask)
    cell_area = result.params.dx * result.params.dt * result.store_every
    out: list[Component] = []
    for label in range(1, count + 1):
        rows, cols = np.nonzero(labels == label)
        size = rows.size
        if size < min_size:
            continue
        out.append(
            Component(
                size=size,
                t_min=float(result.t[rows.min()]),
                t_max=float(result.t[rows.max()]),
                x_min=float(result.x[cols.min()]),
                x_max=float(result.x[cols.max()]),
                area=size * cell_area,
            )
        )
    out.sort(key=lambda component: component.size, reverse=True)
    return out


def contact_intervals(
    result: Result, index: int, mask: np.ndarray | None = None
) -> list[tuple[float, float]]:
    """Contact intervals in ``x`` at the stored snapshot ``index``."""
    mask = contact_mask(result) if mask is None else mask
    row = mask[index]
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (float(result.x[start]), float(result.x[stop - 1]))
        for start, stop in zip(edges[::2], edges[1::2], strict=True)
    ]


def contact_area(result: Result, mask: np.ndarray | None = None) -> float:
    """Measure of the contact set in the ``(t, x)`` plane."""
    mask = contact_mask(result) if mask is None else mask
    return float(np.count_nonzero(mask)) * result.params.dx * result.params.dt * result.store_every


def first_contact_time(result: Result) -> float | None:
    """First time level at which some node satisfies ``eta < 0``, else ``None``."""
    hits = np.flatnonzero(result.contact_fraction > 0.0)
    return float(result.t_full[hits[0]]) if hits.size else None


def penetration_depth(result: Result) -> float:
    """Largest penetration ``max_i max(0, -min_j eta^i_j)``.

    The penalized solution is expected to satisfy ``eta >= -C eps``.
    """
    return float(max(0.0, -result.min_eta.min()))


@dataclass(frozen=True)
class EnergyBalance:
    """Discrete counterpart of the energy identity (1.4).

    ``viscous_cumulative`` and ``contact_cumulative`` are the discrete versions of
    ``int_0^t alpha int |partial_tx eta|^2`` and ``int_0^t int D_con``;
    ``numerical_cumulative`` is the ``O(dt)`` dissipation of the scheme itself.
    With all three included the budget closes to machine precision, see the
    module docstring of :mod:`contact_damped_wave.solver`.
    """

    t: np.ndarray
    energy: np.ndarray
    viscous_cumulative: np.ndarray
    contact_cumulative: np.ndarray
    numerical_cumulative: np.ndarray
    start_index: int = 0

    @property
    def total(self) -> np.ndarray:
        """``E(t) + int_0^t (viscous + contact + numerical)``, constant in exact arithmetic."""
        return (
            self.energy
            + self.viscous_cumulative
            + self.contact_cumulative
            + self.numerical_cumulative
        )

    @property
    def physical_cumulative(self) -> np.ndarray:
        """Dissipation attributable to the PDE (viscous + contact)."""
        return self.viscous_cumulative + self.contact_cumulative

    @property
    def relative_drift(self) -> float:
        """Largest relative deviation of :attr:`total` from its value at ``start_index``."""
        total = self.total[self.start_index :]
        reference = total[0]
        return float(np.abs(total - reference).max() / max(abs(reference), 1e-300))


def energy_balance(result: Result) -> EnergyBalance:
    """Energy and time-integrated dissipation at every time level.

    The rates recorded by the solver are exact per-step increments, so the
    cumulative sums are plain left sums (no quadrature error is introduced).
    """
    dt = result.params.dt
    start = result.balance_start_index

    def cumulative(rate: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rate)
        out[start + 1 :] = np.cumsum(rate[start + 1 :]) * dt
        return out

    return EnergyBalance(
        t=result.t_full,
        energy=result.energy,
        viscous_cumulative=cumulative(result.viscous_dissipation),
        contact_cumulative=cumulative(result.contact_dissipation),
        numerical_cumulative=cumulative(result.numerical_dissipation),
        start_index=start,
    )


def summarize(result: Result, mask: np.ndarray | None = None, min_size: int = 1) -> str:
    """Human-readable summary used by the CLI."""
    mask = contact_mask(result) if mask is None else mask
    comps = components(result, mask, min_size=min_size)
    balance = energy_balance(result)
    start = first_contact_time(result)
    lines = [
        f"parameters      : {result.params.summary()}",
        f"initial step    : {result.initial_step}, store_every={result.store_every}",
        f"energy          : E(0)={balance.energy[0]:.6g} -> E(T)={balance.energy[-1]:.6g}",
        f"  dissipated    : viscous={balance.viscous_cumulative[-1]:.6g}, "
        f"contact={balance.contact_cumulative[-1]:.6g}, "
        f"numerical={balance.numerical_cumulative[-1]:.6g}",
        f"  balance drift : {balance.relative_drift:.3e} (relative)",
        f"energy monotone : {bool(np.all(np.diff(result.energy) <= 1e-9 * abs(result.energy[0])))}",
        f"  min rates     : viscous={result.viscous_dissipation.min():.4g}, "
        f"contact={result.contact_dissipation.min():.4g}, "
        f"numerical={result.numerical_dissipation.min():.4g} (all should be >= 0)",
        f"first contact   : {'none' if start is None else f'{start:.4f}'}",
        f"penetration     : {penetration_depth(result):.4e} "
        f"({penetration_depth(result) / result.params.eps:.2f} x eps)",
        f"contact area    : {contact_area(result, mask):.6g}",
        f"components      : {len(comps)} (min_size={min_size})",
    ]
    lines.extend(f"  #{i + 1} {c.describe()}" for i, c in enumerate(comps[:6]))
    return "\n".join(lines)
