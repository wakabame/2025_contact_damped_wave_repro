"""Animated GIF of a solved run.

:func:`animate` writes a three-panel GIF:

* the string ``eta(t, .)`` above the obstacle, with the part that is currently in
  contact drawn in a separate colour;
* the contact set in the ``(t, x)`` plane, revealed up to the current time;
* the energy balance (1.4), with a cursor at the current time.

Only Pillow is required -- the frames are written with
:class:`matplotlib.animation.PillowWriter`, so no external encoder (ffmpeg,
ImageMagick) has to be installed.

The animation is driven by the *stored* snapshots of a
:class:`~contact_damped_wave.solver.Result`, so a run made with a large
``store_every`` animates just as well as a fully stored one; ``frames``
subsamples them further to keep the GIF small.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .diagnostics import ContactMode, contact_mask, energy_balance  # noqa: E402
from .solver import Result  # noqa: E402

__all__ = ["animate", "frame_indices"]

_OBSTACLE_COLOR = "#4a90d9"
_CURVE_COLOR = "#e8622a"
_CONTACT_COLOR = "#111111"
_GROUND_COLOR = "#dbe6f2"


def frame_indices(n_stored: int, frames: int) -> np.ndarray:
    """Indices of ``frames`` snapshots spread evenly over ``n_stored`` of them.

    The first and last snapshots are always included; if fewer snapshots exist
    than frames were asked for, every snapshot is used exactly once.
    """
    if n_stored < 1:
        raise ValueError(f"n_stored must be >= 1, got {n_stored}")
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    if frames >= n_stored:
        return np.arange(n_stored)
    return np.unique(np.linspace(0, n_stored - 1, frames).round().astype(int))


def animate(
    result: Result,
    path: str | Path,
    *,
    frames: int = 200,
    fps: int = 20,
    ylim: tuple[float, float] | None = None,
    contact_mode: ContactMode = "negative",
    tol: float = 1e-12,
    title: str | None = None,
    dpi: int = 90,
    progress: bool = False,
) -> Path:
    """Write a GIF of ``result`` to ``path`` and return the path written.

    Parameters
    ----------
    frames:
        Number of GIF frames, subsampled from the stored snapshots.
    fps:
        Frames per second of the GIF.
    ylim:
        Vertical range of the displacement panel; defaults to a range that fits
        the whole run with a little margin.
    contact_mode, tol:
        Criterion for the contact set, as in
        :func:`~contact_damped_wave.diagnostics.contact_mask`.
    dpi:
        Resolution of the frames.  The GIF is ``dpi`` times the figure size in
        pixels, so this is the main knob for the file size.
    progress:
        Print a line every 10% of the frames while encoding.
    """
    path = Path(path)
    if path.suffix.lower() != ".gif":
        path = path.with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = contact_mask(result, mode=contact_mode, tol=tol)
    indices = frame_indices(result.t.size, frames)
    balance = energy_balance(result)
    x, length = result.x, result.params.length
    t_first, t_last = float(result.t[0]), float(result.t[-1])
    if ylim is None:
        top = float(result.eta.max())
        ylim = (-0.1 * top, 1.05 * top)

    figure = plt.figure(figsize=(9.0, 6.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    ax_string = figure.add_subplot(grid[0, :])
    ax_contact = figure.add_subplot(grid[1, 0])
    ax_energy = figure.add_subplot(grid[1, 1])
    if title:
        figure.suptitle(title)

    # --- displacement panel -------------------------------------------------
    ax_string.axhspan(ylim[0], 0.0, color=_GROUND_COLOR, zorder=0)
    ax_string.axhline(0.0, color=_OBSTACLE_COLOR, lw=1.6, zorder=2)
    ax_string.plot(x, result.eta[0], color="0.75", lw=1.0, ls="--", zorder=1, label=r"$\eta(0,x)$")
    (curve,) = ax_string.plot(x, result.eta[0], color=_CURVE_COLOR, lw=1.8, zorder=3)
    (touching,) = ax_string.plot(
        [],
        [],
        color=_CONTACT_COLOR,
        lw=4.0,
        zorder=4,
        solid_capstyle="butt",
        label="in contact",
    )
    clock = ax_string.text(
        0.985,
        0.9,
        "",
        transform=ax_string.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8"},
    )
    ax_string.set_xlim(0.0, length)
    ax_string.set_ylim(*ylim)
    ax_string.set_xlabel("x")
    ax_string.set_ylabel(r"$\eta$")
    ax_string.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # --- contact set panel --------------------------------------------------
    ax_contact.imshow(
        mask.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["white", _CONTACT_COLOR]),
        vmin=0,
        vmax=1,
        extent=(t_first, t_last, 0.0, length),
        zorder=1,
    )
    # The future is hidden by an opaque patch that shrinks frame by frame; this
    # is O(1) per frame, unlike re-uploading the (n_stored, N + 1) image.
    veil = Rectangle((t_first, 0.0), t_last - t_first, length, facecolor="white", zorder=2)
    ax_contact.add_patch(veil)
    contact_cursor = ax_contact.axvline(t_first, color=_CURVE_COLOR, lw=1.2, zorder=3)
    ax_contact.set_xlim(t_first, t_last)
    ax_contact.set_ylim(0.0, length)
    ax_contact.set_xlabel("t")
    ax_contact.set_ylabel("x")
    ax_contact.set_title(f"contact set ({contact_mode})", fontsize=10)

    # --- energy panel -------------------------------------------------------
    # Show the *complete* budget: viscous + contact + numerical.  The numerical
    # term is 14-18% of the loss at the paper's resolution, so leaving it out
    # would make the two curves visibly fail to mirror each other.
    dissipated = balance.dissipated_cumulative
    ax_energy.plot(balance.t, balance.energy, color="0.85", lw=1.0)
    ax_energy.plot(balance.t, dissipated, color="0.85", lw=1.0)
    (energy_line,) = ax_energy.plot([], [], color="#2d6ca2", lw=1.6, label="energy $E(t)$")
    (dissipated_line,) = ax_energy.plot(
        [], [], color="#b5651d", lw=1.6, label="dissipated (visc. + contact + num.)"
    )
    energy_cursor = ax_energy.axvline(t_first, color=_CURVE_COLOR, lw=1.2)
    ax_energy.set_xlim(t_first, t_last)
    ax_energy.set_ylim(0.0, 1.05 * float(max(balance.energy.max(), dissipated[-1])))
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("energy")
    ax_energy.legend(fontsize=8, loc="center right")
    ax_energy.grid(True, color="0.92")

    # The scalar diagnostics live on the full time grid, the snapshots on the
    # stored one; map a stored index to the matching full index once.
    full_index = np.rint(result.t / result.params.dt).astype(int)
    full_index = np.clip(full_index, 0, balance.t.size - 1)
    report_every = max(1, indices.size // 10)

    def draw(frame: int):
        i = int(indices[frame])
        eta = result.eta[i]
        curve.set_ydata(eta)
        touching.set_data(np.where(mask[i], x, np.nan), np.where(mask[i], eta, np.nan))
        time = float(result.t[i])
        in_contact = float(np.count_nonzero(mask[i])) * result.params.dx
        clock.set_text(f"t = {time:6.3f}\ncontact = {in_contact:5.3f}")
        veil.set_x(time)
        veil.set_width(max(t_last - time, 0.0))
        contact_cursor.set_xdata([time, time])
        energy_cursor.set_xdata([time, time])
        upto = full_index[i] + 1
        energy_line.set_data(balance.t[:upto], balance.energy[:upto])
        dissipated_line.set_data(balance.t[:upto], dissipated[:upto])
        if progress and frame % report_every == 0:
            print(f"  frame {frame}/{indices.size}  t={time:.4f}")
        return curve, touching, clock, veil, contact_cursor, energy_cursor, energy_line

    movie = animation.FuncAnimation(figure, draw, frames=indices.size, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    return path
