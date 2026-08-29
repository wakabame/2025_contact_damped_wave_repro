"""Convergence of the contact set with respect to dx = dt and to eps.

Section 6 of the paper states that "we tested the numerical convergence with
respect to both the time and space discretization parameters, as well as the
penalization parameter eps" but reports no data.  This script produces that
table for both examples, in the paper's own setting: the grid sweep refines
``dx = dt`` *jointly* and the eps sweep varies ``eps`` at a fixed grid, each
compared against the finest member of its own sweep.  It is a self-convergence
check, not a separation of the time, space and penalization errors -- that
would need independent ``dx``/``dt`` sweeps and an eps refinement with
``dt/eps -> 0``, which the explicit penalty couples (``dt < eps`` is required,
see ``Params.penalty_ratio``).

Usage::

    uv run python scripts/convergence_study.py [--example 1|2] [--out results/convergence]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from contact_damped_wave.diagnostics import (
    components,
    contact_area,
    energy_balance,
    first_contact_time,
    penetration_depth,
)
from contact_damped_wave.initial_data import initial_data
from contact_damped_wave.params import EXAMPLE1, EXAMPLE2
from contact_damped_wave.solver import solve

#: Grid refinements; eps is fixed at the paper value.  The explicit penalty needs
#: ``dt < eps`` (see ``Params.penalty_ratio``), so refining dx = dt at fixed eps
#: is what actually resolves the ``eps``-thick contact layer; coarser steps are
#: kept in the table and flagged so the failure is visible rather than hidden.
GRID_STEPS = (1 / 1000, 1 / 2000, 1 / 2500, 1 / 5000, 1 / 10000)

#: Penalization values; dx = dt is fixed at the paper value (dt/eps < 1 throughout).
EPS_VALUES = (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4)

#: Target number of stored snapshots, to bound memory on the finest grids.
TARGET_SNAPSHOTS = 750

_HEADER = (
    f"{'dx=dt':>9} {'eps':>9} {'dt/eps':>7} {'t_contact':>10} {'area':>10} "
    f"{'penetr.':>10} {'comps':>6} {'E(T)':>10} {'drift':>9}  note"
)


def _row(example: int, step: float, eps: float, min_size: int) -> tuple[str, dict]:
    base = EXAMPLE1 if example == 1 else EXAMPLE2
    params = base.replace(dx=step, dt=step, eps=eps)
    _, _, eta0, v0 = initial_data(example, params)
    store_every = max(1, params.M // TARGET_SNAPSHOTS)
    result = solve(params, eta0, v0, store_every=store_every)
    balance = energy_balance(result)
    # Scale the speckle filter with the grid so that it always covers at least the
    # same area; ceil, not round, since the flag means "discard below this area".
    scaled_min_size = max(
        1, math.ceil(min_size * (base.dx / step) * (base.dt / (step * store_every)))
    )
    comps = components(result, min_size=scaled_min_size)
    start = first_contact_time(result)
    record = {
        "example": example,
        "dx": step,
        "dt": step,
        "eps": eps,
        "penalty_ratio": params.penalty_ratio,
        "first_contact": start,
        "contact_area": contact_area(result),
        "penetration": penetration_depth(result),
        "n_components": len(comps),
        "components": [
            {"t_min": c.t_min, "t_max": c.t_max, "x_min": c.x_min, "x_max": c.x_max, "area": c.area}
            for c in comps
        ],
        "store_every": store_every,
        "penalty_monotone": params.is_penalty_monotone(),
        "penalty_linearly_stable": params.is_penalty_linearly_stable(),
        "energy_initial": float(result.energy[0]),
        "energy_final": float(result.energy[-1]),
        "balance_drift": balance.relative_drift,
        "eta_final": result.eta[-1].tolist(),
    }
    if not params.is_penalty_linearly_stable():
        note = "UNSTABLE (dt/eps >= 2)"
    elif not params.is_penalty_monotone():
        note = "NON-MONOTONE (dt/eps >= 1)"
    else:
        note = ""
    line = (
        f"{step:9.2e} {eps:9.2e} {params.penalty_ratio:7.2f} "
        f"{'n/a' if start is None else f'{start:10.4f}':>10} "
        f"{record['contact_area']:10.5f} {record['penetration']:10.3e} "
        f"{len(comps):6d} {record['energy_final']:10.4g} {record['balance_drift']:9.1e}"
        f"  {note}"
    )
    return line, record


def _eta_differences(records: list[dict]) -> list[str]:
    """Sup-norm distance of the final profile to the last (reference) run."""
    finest = records[-1]
    fine_eta = np.array(finest["eta_final"])
    fine_x = np.linspace(0.0, 1.0, fine_eta.size)
    lines = []
    for record in records[:-1]:
        eta = np.array(record["eta_final"])
        x = np.linspace(0.0, 1.0, eta.size)
        error = float(np.abs(eta - np.interp(x, fine_x, fine_eta)).max())
        lines.append(
            f"  dx=dt={record['dx']:.2e} eps={record['eps']:.2e}: |eta(T) - ref|_inf = {error:.4e}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=int, choices=(1, 2), default=None)
    parser.add_argument("--out", type=Path, default=Path("results") / "convergence")
    parser.add_argument("--min-component-size", type=int, default=5)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    examples = (1, 2) if args.example is None else (args.example,)
    report: list[str] = []
    everything: list[dict] = []

    for example in examples:
        base = EXAMPLE1 if example == 1 else EXAMPLE2
        report.append(f"\n=== Example {example} ===")

        report.append(f"\ngrid refinement (eps = {base.eps:g} fixed)")
        report.append(_HEADER)
        grid_records = []
        for step in GRID_STEPS:
            line, record = _row(example, step, base.eps, args.min_component_size)
            report.append(line)
            grid_records.append(record)
            everything.append({**record, "sweep": "grid"})
        report.extend(_eta_differences(grid_records))

        report.append(f"\npenalization (dx = dt = {base.dx:g} fixed)")
        report.append(_HEADER)
        eps_records = []
        for eps in EPS_VALUES:
            line, record = _row(example, base.dx, eps, args.min_component_size)
            report.append(line)
            eps_records.append(record)
            everything.append({**record, "sweep": "eps"})
        report.extend(_eta_differences(eps_records))

    text = "\n".join(report)
    print(text)
    (args.out / "convergence.txt").write_text(text + "\n", encoding="utf-8")
    for record in everything:
        record.pop("eta_final", None)
    (args.out / "convergence.json").write_text(json.dumps(everything, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'convergence.txt'} and {args.out / 'convergence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
