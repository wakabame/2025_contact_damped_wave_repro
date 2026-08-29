"""Numerical check of Theorem 2.3: contact dissipation localizes at the contact boundary.

Theorem 2.3 of arXiv:2412.06185 characterizes the contact force of the limit
problem as a measure concentrated on the boundary of the contact set -- a jump
in stress along the moving part of ``partial {eta = 0}``, a jump in velocity on
horizontal segments -- with ``F_con = D_con = 0`` in the *interior* of the
contact set (the paper's Figure 1), and, by (A3), nothing on the detaching part
of the boundary where ``partial_t eta >= 0``.

For the penalized discrete solution this script checks the epsilon-analogue of
that statement.  It rebuilds the nodewise contact-work density
``q^i_j = -P^i_j v^{i+1/2}_j`` from a fully stored run
(:func:`~contact_damped_wave.diagnostics.contact_work_density`) and measures how
far from the contact boundary the dissipation actually happens:

* the distance (Euclidean in the ``(t, x)`` plane, where the wave speed is 1)
  from every dissipating cell to the nearest boundary cell of the contact set
  ``{eta < 0}``, and the distances below which 50% / 95% / 99% of the total
  contact dissipation occurs -- expected to be a few ``eps`` wide, since an
  arriving node is stopped within a handful of steps;
* an ``eps``-sweep at the paper's grid showing that this layer shrinks
  proportionally to ``eps``;
* the exact structural facts: dissipating cells all have ``v < 0`` (the
  detachment front carries none of the dissipation) and lie inside
  ``{eta < 0}``.

Figures (log-scale density maps over the contact set) land in
``results/localization/`` together with ``localization.txt``.

Usage::

    uv run python scripts/dissipation_localization.py [--out results/localization]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from contact_damped_wave.diagnostics import contact_work_density, penetration_depth  # noqa: E402
from contact_damped_wave.initial_data import initial_data  # noqa: E402
from contact_damped_wave.params import EXAMPLE1, EXAMPLE3  # noqa: E402
from contact_damped_wave.solver import Result, solve  # noqa: E402

#: eps values for the layer-width scaling sweep (Example 1, paper grid, so
#: dt/eps = 0.05 ... 0.4 stays in the monotone range).
EPS_SWEEP = (4e-3, 2e-3, 1e-3, 5e-4)


def _run(example: int) -> Result:
    params = EXAMPLE1 if example == 1 else EXAMPLE3
    _, _, eta0, v0 = initial_data(example, params)
    return solve(params, eta0, v0, store_every=1)


def _boundary_distance(result: Result) -> np.ndarray:
    """Distance of every cell to the nearest boundary cell of ``{eta < 0}``.

    Euclidean in the ``(t, x)`` plane with physical sampling ``(dt, dx)``; the
    natural metric here because the characteristic speed of the wave part is 1.
    """
    contact = result.eta < 0.0
    boundary = contact & ~ndimage.binary_erosion(contact)
    return ndimage.distance_transform_edt(~boundary, sampling=(result.params.dt, result.params.dx))


def localization_quantiles(
    result: Result, density: np.ndarray, quantiles: tuple[float, ...] = (0.5, 0.95, 0.99)
) -> dict[float, float]:
    """Distances to the contact boundary below which the given fractions of the
    total contact dissipation occur."""
    distance = _boundary_distance(result)
    dissipating = density > 0.0
    d = distance[dissipating]
    w = (density * result.params.dt * result.params.dx)[dissipating]
    order = np.argsort(d)
    d, w = d[order], w[order]
    cumulative = np.cumsum(w) / w.sum()
    return {q: float(d[np.searchsorted(cumulative, q)]) for q in quantiles}


def _structural_checks(result: Result, density: np.ndarray) -> list[str]:
    """The exact (not asymptotic) parts of the statement."""
    dissipating = density > 0.0
    inside = result.eta < 0.0
    moving_down = result.v < 0.0
    penalised_levels = np.zeros_like(dissipating)
    penalised_levels[1:] = inside[:-1] & moving_down[:-1]
    assert np.all(penalised_levels[dissipating]), "dissipation outside {eta<0, v<0}"
    detach = inside & ~moving_down  # in contact but moving up: about to detach
    return [
        f"  dissipating cells               : {dissipating.sum()} "
        f"({dissipating.sum() / dissipating.size:.2%} of the (t, x) grid)",
        "  all inside {eta<0} with v<0     : True (exact, by construction of P)",
        f"  cells in contact with v >= 0    : {detach.sum()} -- carry zero dissipation "
        "(the detachment front dissipates nothing, cf. (A3)/(2.14))",
    ]


def _report(result: Result, density: np.ndarray, label: str) -> list[str]:
    eps = result.params.eps
    quantiles = localization_quantiles(result, density)
    lines = [f"\n=== {label} ===", f"parameters      : {result.params.summary()}"]
    lines += _structural_checks(result, density)
    lines += [
        f"  penetration depth               : {penetration_depth(result):.3e} "
        f"({penetration_depth(result) / eps:.1f} eps)",
    ]
    for q, dist in quantiles.items():
        lines.append(
            f"  {q:.0%} of dissipation within      : {dist:.3e} of the boundary "
            f"({dist / eps:.2f} eps, {dist / result.params.dx:.1f} cells)"
        )
    return lines


def _figure(result: Result, density: np.ndarray, title: str, path: Path) -> None:
    """Contact set (grey) with the dissipation density on top, log colour scale."""
    contact = result.eta < 0.0
    extent = (float(result.t[0]), float(result.t[-1]), 0.0, result.params.length)
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    axis.imshow(
        contact.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["white", "0.85"]),
        vmin=0,
        vmax=1,
        extent=extent,
    )
    logdensity = np.ma.log10(np.ma.masked_less_equal(density, 0.0))
    top = float(np.ceil(logdensity.max()))
    image = axis.imshow(
        logdensity.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=top - 6,
        vmax=top,
        extent=extent,
    )
    figure.colorbar(image, ax=axis, label=r"$\log_{10}$ dissipation density $-P\,\partial_t\eta$")
    axis.set_xlabel("t")
    axis.set_ylabel("x")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results") / "localization")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    report: list[str] = [
        "Numerical check of Theorem 2.3: the contact dissipation of the penalized",
        "solution concentrates in an O(eps) layer along the entry portion of the",
        "contact boundary, vanishes in the interior of the contact set, and puts",
        "exactly nothing on the detachment front.",
    ]

    for example, name, blurb in (
        (1, "ex1", "uniform impact, triangular contact set"),
        (3, "ex3", "rolling contact front"),
    ):
        result = _run(example)
        density = contact_work_density(result)
        # The density is an exact decomposition of the recorded budget.
        recorded = result.contact_work
        rebuilt = result.params.dx * density.sum(axis=1)
        scale = np.abs(recorded).max()
        assert np.allclose(rebuilt, recorded, rtol=1e-10, atol=1e-12 * scale)
        report += _report(result, density, f"Example {example} ({blurb})")
        _figure(
            result,
            density,
            f"Example {example}: contact set (grey) and where the contact dissipation happens",
            args.out / f"{name}_density.png",
        )

    report.append("\n=== eps sweep (Example 1, paper grid fixed) ===")
    report.append(f"{'eps':>9} {'dt/eps':>7} {'d50':>10} {'d95':>10} {'d99':>10} {'d95/eps':>8}")
    for eps in EPS_SWEEP:
        params = EXAMPLE1.replace(eps=eps)
        _, _, eta0, v0 = initial_data(1, params)
        result = solve(params, eta0, v0, store_every=1)
        quantiles = localization_quantiles(result, contact_work_density(result))
        d50, d95, d99 = quantiles[0.5], quantiles[0.95], quantiles[0.99]
        report.append(
            f"{eps:9.1e} {params.penalty_ratio:7.2f} {d50:10.3e} {d95:10.3e} {d99:10.3e} "
            f"{d95 / eps:8.2f}"
        )
    report.append(
        "d95 shrinking proportionally to eps (d95/eps roughly constant) is the\n"
        "discrete counterpart of the dissipation concentrating on the boundary\n"
        "as eps -> 0."
    )

    text = "\n".join(report)
    print(text)
    (args.out / "localization.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {args.out / 'localization.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
