"""Initial data of the two numerical examples (arXiv:2412.06185, Section 6).

Example 1 (Section 6.1) is unambiguous::

    eta^0 = 1 + (1/2) sin^2(10 pi x),   v^0 = -50.

Example 2 (Section 6.2) is printed in the paper as::

    eta^0 = x                       for 0   <= x < 0.2,
            sin(pi (x - 0.2) / 0.3) for 0.2 <= x < 0.8,
            2 - x                   for 0.8 <= x < 1,

which is discontinuous at ``x = 0.2`` (0.2 -> 0) and at ``x = 0.8`` (0 -> 1.2),
violates the standing assumption ``eta_0 >= c > 0`` of Theorem 2.1 (it gives
``eta^0(0) = 0``) and does not match Figure 4(a), where the profile is
continuous, equals 1 at both endpoints, equals 2 at ``x = 0.2, 0.5, 0.8``, peaks
at 3 near ``x = 0.35`` and has a local minimum 1 near ``x = 0.65``.  The
``"figure"`` variant below is the reading consistent with Figure 4(a) and is the
default; the ``"paper-literal"`` variant reproduces the printed formula verbatim
so that the discrepancy can be documented.  See ``plan.md`` section 4 item (b).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .params import EXAMPLE1, EXAMPLE2, Params

__all__ = [
    "Example2Variant",
    "example1_eta0",
    "example1_v0",
    "example2_eta0",
    "example2_v0",
    "grid",
    "initial_data",
]

Example2Variant = Literal["figure", "paper-literal"]


def grid(params: Params) -> np.ndarray:
    """Uniform grid ``x_0 = 0, ..., x_N = length`` with ``N + 1`` nodes."""
    return np.linspace(0.0, params.length, params.N + 1)


def example1_eta0(x: np.ndarray) -> np.ndarray:
    """``eta^0 = 1 + (1/2) sin^2(10 pi x)`` (Section 6.1)."""
    return 1.0 + 0.5 * np.sin(10.0 * np.pi * x) ** 2


def example1_v0(x: np.ndarray) -> np.ndarray:
    """``v^0 = -50`` (Section 6.1)."""
    return np.full_like(x, -50.0)


def example2_eta0(x: np.ndarray, variant: Example2Variant = "figure") -> np.ndarray:
    """Initial displacement of Section 6.2.

    Parameters
    ----------
    variant:
        ``"figure"`` (default) uses the continuous profile consistent with
        Figure 4(a)::

            eta^0 = 1 + 5 x                          for 0   <= x < 0.2,
                    2 + sin(pi (x - 0.2) / 0.3)      for 0.2 <= x < 0.8,
                    6 - 5 x                          for 0.8 <= x <= 1.

        ``"paper-literal"`` uses the formula exactly as printed in the paper.
    """
    if variant == "figure":
        return np.piecewise(
            x,
            [x < 0.2, (x >= 0.2) & (x < 0.8), x >= 0.8],
            [
                lambda s: 1.0 + 5.0 * s,
                lambda s: 2.0 + np.sin(np.pi * (s - 0.2) / 0.3),
                lambda s: 6.0 - 5.0 * s,
            ],
        )
    if variant == "paper-literal":
        return np.piecewise(
            x,
            [x < 0.2, (x >= 0.2) & (x < 0.8), x >= 0.8],
            [
                lambda s: s,
                lambda s: np.sin(np.pi * (s - 0.2) / 0.3),
                lambda s: 2.0 - s,
            ],
        )
    raise ValueError(f"unknown Example 2 variant {variant!r}")


def example2_v0(x: np.ndarray) -> np.ndarray:
    """``v^0 = -50`` on ``[0, 0.6)`` and ``-0.5`` on ``[0.6, 1]`` (Section 6.2)."""
    return np.where(x < 0.6, -50.0, -0.5)


def initial_data(
    example: int,
    params: Params | None = None,
    variant: Example2Variant = "figure",
) -> tuple[Params, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(params, x, eta0, v0)`` for Example ``1`` or ``2``.

    ``variant`` is ignored for Example 1, whose initial data is unambiguous.
    """
    if example == 1:
        params = params if params is not None else EXAMPLE1
        x = grid(params)
        return params, x, example1_eta0(x), example1_v0(x)
    if example == 2:
        params = params if params is not None else EXAMPLE2
        x = grid(params)
        return params, x, example2_eta0(x, variant), example2_v0(x)
    raise ValueError(f"example must be 1 or 2, got {example!r}")
