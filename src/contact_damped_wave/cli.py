"""Command line interface.

    uv run cdw run     --example 1 --out results/ex1     # figures + diagnostics
    uv run cdw animate --example 3 --out results/ex3     # animated GIF

Examples 1 and 2 are the paper's (Section 6.1, 6.2); Example 3 is our own
showcase built on the same solver (``plan.md`` Phase 7).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from .animation import animate
from .diagnostics import contact_mask, summarize
from .initial_data import initial_data
from .params import EXAMPLE1, EXAMPLE2, EXAMPLE3, Params
from .plotting import plot_contact_set, plot_energy, plot_snapshots, plot_velocity_field
from .solver import Result, solve

__all__ = ["main"]

#: Snapshot times, axis limits and figure names per example.  Examples 1 and 2
#: follow Figures 2-5 of the paper; Example 3 is ours, so its figures are named
#: plainly.
FIGURE_SETTINGS = {
    1: {
        "times": (0.0, 0.02, 0.04, 0.06, 0.2, 0.3),
        "ylim": (-0.5, 2.0),
        "vlim": (-60.0, 10.0),
        "names": ("fig2_snapshots", "fig3_contact_set", "fig3_velocity"),
        "label": "Example 1 (paper Fig. 2, 3)",
    },
    2: {
        "times": (0.0, 0.04, 0.08, 0.16, 0.28, 0.32),
        "ylim": (-0.5, 3.0),
        "vlim": (-50.0, 40.0),
        "names": ("fig4_snapshots", "fig5_contact_set", "fig5_velocity"),
        "label": "Example 2 (paper Fig. 4, 5)",
    },
    3: {
        "times": (0.0, 0.05, 0.12, 0.24, 0.36, 0.5),
        "ylim": (-0.3, 2.7),
        "vlim": (-70.0, 12.0),
        "names": ("snapshots", "contact_set", "velocity"),
        "label": "Example 3 (ours): rolling contact front",
    },
}

_BASE_PARAMS = {1: EXAMPLE1, 2: EXAMPLE2, 3: EXAMPLE3}


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Options shared by ``run`` and ``animate``: which problem, on which grid."""
    parser.add_argument("--example", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory (default results/exN)"
    )
    parser.add_argument("--dx", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--final-time", type=float, default=None, dest="final_time", metavar="T")
    parser.add_argument("--h", type=float, default=None, help="endpoint height (default 1.0)")
    parser.add_argument(
        "--initial",
        choices=("figure", "paper-literal"),
        default="figure",
        help="Example 2 initial displacement variant (see plan.md item (b))",
    )
    parser.add_argument(
        "--initial-step",
        choices=("backward", "forward"),
        default="backward",
        help="how to start the three-level recursion (see plan.md item (d))",
    )
    parser.add_argument(
        "--contact-mode",
        choices=("negative", "threshold"),
        default="negative",
        help="criterion for the contact set (see plan.md item (e))",
    )
    parser.add_argument("--quiet", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdw",
        description=("Numerical examples of arXiv:2412.06185 (Section 6) plus one of our own."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an example and write figures")
    _add_common_arguments(run)
    run.add_argument("--store-every", type=int, default=1, help="snapshot stride")
    run.add_argument(
        "--min-component-size",
        type=int,
        default=1,
        help="ignore contact components smaller than this many cells when reporting",
    )
    run.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2),
        default=2,
        help="2 (default) counts diagonal neighbours as connected, 1 does not",
    )
    run.add_argument("--no-data", action="store_true", help="do not write the .npz archive")

    gif = sub.add_parser("animate", help="run an example and write an animated GIF")
    _add_common_arguments(gif)
    gif.add_argument(
        "--store-every",
        type=int,
        default=5,
        help="snapshot stride; the animation is subsampled from the stored snapshots",
    )
    gif.add_argument("--frames", type=int, default=200, help="number of GIF frames")
    gif.add_argument("--fps", type=int, default=20, help="frames per second")
    gif.add_argument("--dpi", type=int, default=90, help="frame resolution (drives the file size)")
    gif.add_argument("--name", default=None, help="file name (default animation.gif)")
    return parser


def _resolve_params(args: argparse.Namespace) -> Params:
    base = _BASE_PARAMS[args.example]
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


def _solve_from_args(args: argparse.Namespace) -> tuple[Params, Result, float]:
    """Resolve the parameters, warn about ``dt/eps``, solve, and time it."""
    params = _resolve_params(args)
    if not params.is_penalty_stable():
        print(
            f"warning: dt/eps = {params.penalty_ratio:g} >= 1; the explicit penalty force "
            "will oscillate instead of damping monotonically.",
            file=sys.stderr,
        )
    _, _, eta0, v0 = initial_data(args.example, params, args.initial)
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
    if not args.quiet:
        print(f"solved in {elapsed:.2f} s")
    return params, result, elapsed


def _out_dir(args: argparse.Namespace) -> Path:
    out = args.out or Path("results") / f"ex{args.example}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run(args: argparse.Namespace) -> int:
    params, result, elapsed = _solve_from_args(args)
    out = _out_dir(args)
    settings = FIGURE_SETTINGS[args.example]

    mask = contact_mask(result, mode=args.contact_mode)
    report = summarize(
        result, mask, min_size=args.min_component_size, connectivity=args.connectivity
    )
    print(report)
    (out / "summary.txt").write_text(
        f"Example {args.example} (initial={args.initial}, contact_mode={args.contact_mode})\n"
        f"{report}\nwall time: {elapsed:.2f} s\n",
        encoding="utf-8",
    )

    snapshots, contact, velocity = settings["names"]
    figures = {
        f"{snapshots}.png": plot_snapshots(
            result,
            settings["times"],
            ylim=settings["ylim"],
            title=f"Example {args.example}: solution at different times",
        ),
        f"{contact}.png": plot_contact_set(
            result,
            mode=args.contact_mode,
            title=f"Example {args.example}: contact set ({args.contact_mode})",
        ),
        f"{velocity}.png": plot_velocity_field(
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
        # Name the archive after everything that changes the run, so that two
        # invocations with different parameters cannot silently overwrite each
        # other, and keep it next to the figures the same invocation produced.
        stem = (
            f"ex{args.example}_{args.initial}_{args.initial_step}"
            f"_dx{params.dx:g}_dt{params.dt:g}_eps{params.eps:g}"
            f"_alpha{params.alpha:g}_h{params.h:g}_T{params.T:g}"
        )
        data_path = result.save(out / f"{stem}.npz")
        if not args.quiet:
            print(f"wrote {data_path}")
    return 0


def _animate(args: argparse.Namespace) -> int:
    _, result, _ = _solve_from_args(args)
    out = _out_dir(args)
    settings = FIGURE_SETTINGS[args.example]

    started = time.perf_counter()
    path = animate(
        result,
        out / (args.name or "animation.gif"),
        frames=args.frames,
        fps=args.fps,
        ylim=settings["ylim"],
        contact_mode=args.contact_mode,
        title=settings["label"],
        dpi=args.dpi,
        progress=not args.quiet,
    )
    size_mb = path.stat().st_size / 1e6
    print(
        f"wrote {path} ({size_mb:.1f} MB, {args.frames} frames at {args.fps} fps) "
        f"in {time.perf_counter() - started:.1f} s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "animate":
        return _animate(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
