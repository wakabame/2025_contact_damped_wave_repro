"""Command line interface: ``uv run cdw run --example 1 --out results/ex1``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from .diagnostics import contact_mask, summarize
from .initial_data import initial_data
from .params import EXAMPLE1, EXAMPLE2, Params
from .plotting import plot_contact_set, plot_energy, plot_snapshots, plot_velocity_field
from .solver import solve

__all__ = ["main"]

#: Snapshot times and axis limits of Figures 2 and 4.
FIGURE_SETTINGS = {
    1: {"times": (0.0, 0.02, 0.04, 0.06, 0.2, 0.3), "ylim": (-0.5, 2.0), "vlim": (-60.0, 10.0)},
    2: {"times": (0.0, 0.04, 0.08, 0.16, 0.28, 0.32), "ylim": (-0.5, 3.0), "vlim": (-50.0, 40.0)},
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdw",
        description="Reproduce the numerical examples of arXiv:2412.06185 (Section 6).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one of the two examples and write figures")
    run.add_argument("--example", type=int, choices=(1, 2), required=True)
    run.add_argument(
        "--out", type=Path, default=None, help="output directory (default results/exN)"
    )
    run.add_argument("--dx", type=float, default=None)
    run.add_argument("--dt", type=float, default=None)
    run.add_argument("--eps", type=float, default=None)
    run.add_argument("--alpha", type=float, default=None)
    run.add_argument("--final-time", type=float, default=None, dest="final_time", metavar="T")
    run.add_argument("--h", type=float, default=None, help="endpoint height (default 1.0)")
    run.add_argument(
        "--initial",
        choices=("figure", "paper-literal"),
        default="figure",
        help="Example 2 initial displacement variant (see plan.md item (b))",
    )
    run.add_argument(
        "--initial-step",
        choices=("backward", "forward"),
        default="backward",
        help="how to start the three-level recursion (see plan.md item (d))",
    )
    run.add_argument("--store-every", type=int, default=1, help="snapshot stride")
    run.add_argument(
        "--contact-mode",
        choices=("negative", "threshold"),
        default="negative",
        help="criterion for the contact set (see plan.md item (e))",
    )
    run.add_argument(
        "--min-component-size",
        type=int,
        default=1,
        help="ignore contact components smaller than this many cells when reporting",
    )
    run.add_argument("--no-data", action="store_true", help="do not write the .npz archive")
    run.add_argument("--quiet", action="store_true")
    return parser


def _resolve_params(args: argparse.Namespace) -> Params:
    base = EXAMPLE1 if args.example == 1 else EXAMPLE2
    changes: dict[str, float] = {}
    for attr, field in (
        ("dx", "dx"),
        ("dt", "dt"),
        ("eps", "eps"),
        ("alpha", "alpha"),
        ("final_time", "T"),
        ("h", "h"),
    ):
        value = getattr(args, attr)
        if value is not None:
            changes[field] = value
    return base.replace(**changes)


def _run(args: argparse.Namespace) -> int:
    params = _resolve_params(args)
    out = args.out or Path("results") / f"ex{args.example}"
    out.mkdir(parents=True, exist_ok=True)
    settings = FIGURE_SETTINGS[args.example]

    if not params.is_penalty_stable():
        print(
            f"warning: dt/eps = {params.penalty_ratio:g} >= 1; the explicit penalty force "
            "will oscillate instead of damping monotonically.",
            file=sys.stderr,
        )

    _, x, eta0, v0 = initial_data(args.example, params, args.initial)
    if not args.quiet:
        print(f"Example {args.example}: {params.summary()}")
        print(f"initial data: {args.initial}, initial step: {args.initial_step}")

    started = time.perf_counter()
    result = solve(
        params,
        eta0,
        v0,
        store_every=args.store_every,
        initial_step=args.initial_step,
        progress=not args.quiet,
    )
    elapsed = time.perf_counter() - started

    mask = contact_mask(result, mode=args.contact_mode)
    report = summarize(result, mask, min_size=args.min_component_size)
    print(f"solved in {elapsed:.2f} s")
    print(report)
    (out / "summary.txt").write_text(
        f"Example {args.example} (initial={args.initial}, contact_mode={args.contact_mode})\n"
        f"{report}\nwall time: {elapsed:.2f} s\n",
        encoding="utf-8",
    )

    fig_number = 2 if args.example == 1 else 4
    field_number = 3 if args.example == 1 else 5
    figures = {
        f"fig{fig_number}_snapshots.png": plot_snapshots(
            result,
            settings["times"],
            ylim=settings["ylim"],
            title=f"Example {args.example}: solution at different times",
        ),
        f"fig{field_number}_contact_set.png": plot_contact_set(
            result,
            mode=args.contact_mode,
            title=f"Example {args.example}: contact set ({args.contact_mode})",
        ),
        f"fig{field_number}_velocity.png": plot_velocity_field(
            result,
            vlim=settings["vlim"],
            title=rf"Example {args.example}: velocity $\partial_t\eta$",
        ),
        "energy.png": plot_energy(result, title=f"Example {args.example}: energy balance"),
    }
    for name, figure in figures.items():
        figure.savefig(out / name, dpi=150)
        plt.close(figure)
        if not args.quiet:
            print(f"wrote {out / name}")

    if not args.no_data:
        data_path = Path("results") / "data" / f"ex{args.example}_{args.initial}.npz"
        result.save(data_path)
        if not args.quiet:
            print(f"wrote {data_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
