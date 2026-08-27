"""Figures reproducing those of arXiv:2412.06185, Section 6.

* :func:`plot_snapshots` -- Figures 2 and 4 (``eta`` at six times).
* :func:`plot_contact_set` -- left panels of Figures 3 and 5.
* :func:`plot_velocity_field` -- right panels of Figures 3 and 5.
* :func:`plot_energy` -- extra diagnostic, not in the paper.

The paper's figures come from MATLAB, so the ``jet`` colormap is used by default
for the velocity field to make side-by-side comparison easy.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .diagnostics import ContactMode, contact_mask, energy_balance  # noqa: E402
from .solver import Result  # noqa: E402

__all__ = [
    "plot_contact_set",
    "plot_energy",
    "plot_snapshots",
    "plot_velocity_field",
]

_OBSTACLE_COLOR = "#4a90d9"
_CURVE_COLOR = "#e8622a"


def plot_snapshots(
    result: Result,
    times: Sequence[float],
    *,
    ylim: tuple[float, float] = (-0.5, 2.0),
    ncols: int = 2,
    title: str | None = None,
) -> Figure:
    """Displacement at the requested times, laid out like Figures 2 and 4."""
    nrows = -(-len(times) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows), squeeze=False)
    for ax, requested in zip(axes.ravel(), times, strict=False):
        actual, eta = result.snapshot(requested)
        ax.axhline(0.0, color=_OBSTACLE_COLOR, lw=1.2)
        ax.plot(result.x, eta, color=_CURVE_COLOR, lw=1.4)
        ax.set_xlim(0.0, result.params.length)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\eta$")
        ax.set_title(f"$t = {actual:g}$", fontsize=10)
    for ax in axes.ravel()[len(times) :]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _field_axes(ax, result: Result) -> None:
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_xlim(float(result.t[0]), float(result.t[-1]))
    ax.set_ylim(0.0, result.params.length)


def plot_contact_set(
    result: Result,
    *,
    mode: ContactMode = "negative",
    tol: float = 1e-12,
    title: str | None = None,
) -> Figure:
    """Contact set in the ``(t, x)`` plane, black where the string touches."""
    mask = contact_mask(result, mode=mode, tol=tol)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.imshow(
        mask.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["white", "black"]),
        vmin=0,
        vmax=1,
        extent=(float(result.t[0]), float(result.t[-1]), 0.0, result.params.length),
    )
    ax.grid(True, color="0.85", lw=0.5)
    ax.set_axisbelow(False)
    _field_axes(ax, result)
    ax.set_title(title or f"contact set ({mode})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_velocity_field(
    result: Result,
    *,
    vlim: tuple[float, float] | None = None,
    cmap: str = "jet",
    title: str | None = None,
) -> Figure:
    """Velocity ``partial_t eta`` in the ``(t, x)`` plane."""
    v = result.v
    vmin, vmax = vlim if vlim is not None else (float(v.min()), float(v.max()))
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    image = ax.imshow(
        v.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(float(result.t[0]), float(result.t[-1]), 0.0, result.params.length),
    )
    fig.colorbar(image, ax=ax)
    _field_axes(ax, result)
    ax.set_title(title or r"velocity $\partial_t\eta$", fontsize=10)
    fig.tight_layout()
    return fig


def plot_energy(result: Result, *, title: str | None = None) -> Figure:
    """Energy and cumulative dissipation; a check of the balance (1.4)."""
    balance = energy_balance(result)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(balance.t, balance.energy, label="energy $E(t)$")
    ax.plot(balance.t, balance.viscous_cumulative, label="viscous dissipation (cum.)")
    ax.plot(balance.t, balance.contact_cumulative, label="contact work (cum.)")
    ax.plot(balance.t, balance.numerical_cumulative, label="numerical dissipation (cum.)")
    ax.plot(balance.t, balance.total, "k--", lw=1.0, label="sum (should be constant)")
    ax.set_xlabel("t")
    ax.set_ylabel("energy")
    ax.set_xlim(float(balance.t[0]), float(balance.t[-1]))
    ax.legend(fontsize=8)
    ax.grid(True, color="0.9")
    ax.set_title(title or f"energy balance (drift {balance.relative_drift:.2e})", fontsize=10)
    fig.tight_layout()
    return fig
