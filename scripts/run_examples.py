"""Run every example at the paper's parameters and write every figure.

Usage::

    uv run python scripts/run_examples.py              # figures only (~2 s)
    uv run python scripts/run_examples.py --animate    # figures + GIFs (~5 min)

Examples 1 and 2 are the paper's (Section 6.1, 6.2), plus the initial data
exactly as printed there for comparison; Example 3 is our own showcase
(``plan.md`` section 9).  The GIFs are slow to encode, so they are opt-in.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

#: ``cdw run`` invocations.  ``--min-component-size`` filters the single-node
#: speckle the penalization leaves near the contact front, which would otherwise
#: be reported as extra connected components.
RUNS = [
    ["run", "--example", "1", "--min-component-size", "5"],
    ["run", "--example", "2", "--min-component-size", "20"],
    ["run", "--example", "3", "--min-component-size", "20"],
    # The initial data as literally printed in the paper, for comparison.
    [
        "run",
        "--example",
        "2",
        "--initial",
        "paper-literal",
        "--out",
        "results/ex2_paper_literal",
        "--min-component-size",
        "20",
    ],
]

#: ``cdw animate`` invocations; ``store_every`` and ``frames`` are chosen to keep
#: each GIF around 1.5-2 MB.
ANIMATIONS = [
    [
        "animate",
        "--example",
        "1",
        "--frames",
        "140",
        "--fps",
        "18",
        "--dpi",
        "80",
        "--store-every",
        "3",
    ],
    [
        "animate",
        "--example",
        "2",
        "--frames",
        "140",
        "--fps",
        "18",
        "--dpi",
        "80",
        "--store-every",
        "4",
    ],
    [
        "animate",
        "--example",
        "3",
        "--frames",
        "180",
        "--fps",
        "18",
        "--dpi",
        "85",
        "--store-every",
        "5",
    ],
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animate", action="store_true", help="also write the GIFs (slow: a few minutes)"
    )
    args = parser.parse_args()

    for command in RUNS + (ANIMATIONS if args.animate else []):
        print(f"\n$ cdw {' '.join(command)}")
        result = subprocess.run(["cdw", *command], check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
