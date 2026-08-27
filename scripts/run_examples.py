"""Run both examples at the paper's parameters and write every figure.

Usage::

    uv run python scripts/run_examples.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    commands = [
        ["cdw", "run", "--example", "1", "--min-component-size", "5"],
        ["cdw", "run", "--example", "2", "--min-component-size", "20"],
        # The initial data as literally printed in the paper, for comparison.
        [
            "cdw",
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
    for command in commands:
        print(f"\n$ {' '.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
