"""Initial data of the numerical examples (arXiv:2412.06185, Section 6, plus our own).

Example 1 (Section 6.1) is unambiguous::

    eta^0 = 1 + (1/2) sin^2(10 pi x),   v^0 = -50.

Example 2 (Section 6.2) is printed in the paper as::

    eta^0 = x                       for 0   <= x < 0.2,
            sin(pi (x - 0.2) / 0.3) for 0.2 <= x < 0.8,
            2 - x                   for 0.8 <= x < 1,

which is discontinuous at ``x = 0.2`` (0.2 -> 0) and at ``x = 0.8`` (0 -> 1.2),
*negative* on ``(0.5, 0.8)`` -- the sine branch reaches ``-1`` at ``x = 0.65``,
so 30% of the string starts below the obstacle -- violates the standing
assumption ``eta_0 >= c > 0`` of Theorem 2.1 (it gives ``eta^0(0) = 0``, and
``eta^0 < 0`` inside) and does not match Figure 4(a), where the profile is
continuous, equals 1 at both endpoints, equals 2 at ``x = 0.2, 0.5, 0.8``, peaks
at 3 near ``x = 0.35`` and has a local minimum 1 near ``x = 0.65``.  The
``"figure"`` variant below is the reading consistent with Figure 4(a) and is the
default; the ``"paper-literal"`` variant reproduces the printed formula verbatim
so that the discrepancy can be documented.  See ``docs/notes.md`` §2 (b).

All profiles except ``"paper-literal"`` are written as ``h + shape(x)`` with
``shape(0) = shape(l) = 0``, so they stay compatible with the clamped endpoints
(1.3) for any ``h``.  Using data whose endpoints disagree with ``h`` puts a jump
of size ``|eta^0(0) - h|`` across the first cell, whose elastic energy
``~ (eta^0(0) - h)^2 / dx`` blows up as the grid is refined; :func:`solve` warns
when that happens.  The paper's value is ``h = 1``, for which these formulas
reduce exactly to the printed ones.

Example 3 (:func:`example3_eta0`, :func:`example3_v0`) is **not** from the paper.
It is our own showcase, built on the solver once the two paper examples were
reproduced: a single arch given a downward velocity concentrated on its left
half, which makes the string roll onto the obstacle from left to right instead
of slapping down all at once.  Unlike the paper's data it is smooth and fully
compatible with the clamped endpoints (``v^0(0) = v^0(l) = 0``), so it carries no
initial layer at the boundary.  See ``docs/notes.md`` §6.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .params import EXAMPLE1, EXAMPLE2, EXAMPLE3, Params

__all__ = [
    "Example2Variant",
    "example1_eta0",
    "example1_v0",
    "example2_eta0",
    "example2_v0",
    "example3_eta0",
    "example3_v0",
    "grid",
    "initial_data",
]

Example2Variant = Literal["figure", "paper-literal"]


def grid(params: Params) -> np.ndarray:
    """Uniform grid ``x_0 = 0, ..., x_N = length`` with ``N + 1`` nodes."""
    return np.linspace(0.0, params.length, params.N + 1)


def example1_eta0(x: np.ndarray, h: float = 1.0) -> np.ndarray:
    """``eta^0 = h + (1/2) sin^2(10 pi x)`` (Section 6.1, printed with ``h = 1``)."""
    return h + 0.5 * np.sin(10.0 * np.pi * x) ** 2


def example1_v0(x: np.ndarray) -> np.ndarray:
    """``v^0 = -50`` (Section 6.1)."""
    return np.full_like(x, -50.0)


def example2_eta0(x: np.ndarray, variant: Example2Variant = "figure", h: float = 1.0) -> np.ndarray:
    """Initial displacement of Section 6.2.

    Parameters
    ----------
    variant:
        ``"figure"`` (default) uses the continuous profile consistent with
        Figure 4(a), written so that it meets the clamped endpoints for any
        ``h`` (for ``h = 1`` it is ``1 + 5x``, ``2 + sin(...)``, ``6 - 5x``)::

            eta^0 = h + 5 x                          for 0   <= x < 0.2,
                    h + 1 + sin(pi (x - 0.2) / 0.3)  for 0.2 <= x < 0.8,
                    h + 5 (1 - x)                    for 0.8 <= x <= 1.

        ``"paper-literal"`` uses the formula exactly as printed in the paper and
        ignores ``h``; its endpoint value ``eta^0(0) = 0`` disagrees with the
        boundary condition, which :func:`solve` reports.
    """
    if variant == "figure":
        return np.piecewise(
            x,
            [x < 0.2, (x >= 0.2) & (x < 0.8), x >= 0.8],
            [
                lambda s: h + 5.0 * s,
                lambda s: h + 1.0 + np.sin(np.pi * (s - 0.2) / 0.3),
                lambda s: h + 5.0 * (1.0 - s),
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


#: Shape parameters of Example 3, see :func:`example3_eta0` / :func:`example3_v0`.
EXAMPLE3_AMPLITUDE = 1.5
EXAMPLE3_SPEED = 110.0
EXAMPLE3_CENTER = 0.35
EXAMPLE3_WIDTH = 0.10


def example3_eta0(x: np.ndarray, h: float = 1.0, length: float = 1.0) -> np.ndarray:
    """``eta^0 = h + 1.5 sin(pi x / l)``: a single arch of height ``h + 1.5``.

    Not from the paper; see the module docstring and ``docs/notes.md`` §6.
    """
    return h + EXAMPLE3_AMPLITUDE * np.sin(np.pi * x / length)


def example3_v0(x: np.ndarray, length: float = 1.0) -> np.ndarray:
    """Downward velocity concentrated on the left half of the arch.

    ``v^0 = -110 sin(pi x / l) * (1 - tanh((x - 0.35) / 0.1)) / 2``, i.e. a smooth
    step that switches off around ``x = 0.35``, multiplied by a sine so that it
    vanishes at both clamped endpoints.  Peak speed is about ``-69`` near
    ``x = 0.27``, while the right third of the string starts essentially at rest.

    The left part therefore reaches the obstacle first and sticks to it
    (inelastic contact), and the disturbance released by that impact travels
    right, laying the string down progressively: the contact set is a band in the
    ``(t, x)`` plane that translates at roughly the characteristic speed instead
    of the triangles of the paper's two examples.
    """
    envelope = np.sin(np.pi * x / length)
    step = 0.5 * (1.0 - np.tanh((x - EXAMPLE3_CENTER) / EXAMPLE3_WIDTH))
    return -EXAMPLE3_SPEED * envelope * step


def initial_data(
    example: int,
    params: Params | None = None,
    variant: Example2Variant = "figure",
) -> tuple[Params, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(params, x, eta0, v0)`` for Example ``1``, ``2`` or ``3``.

    ``variant`` only applies to Example 2; the data of Examples 1 and 3 is
    unambiguous.
    """
    if example == 1:
        params = params if params is not None else EXAMPLE1
        x = grid(params)
        return params, x, example1_eta0(x, params.h), example1_v0(x)
    if example == 2:
        params = params if params is not None else EXAMPLE2
        x = grid(params)
        return params, x, example2_eta0(x, variant, params.h), example2_v0(x)
    if example == 3:
        params = params if params is not None else EXAMPLE3
        x = grid(params)
        return params, x, example3_eta0(x, params.h, params.length), example3_v0(x, params.length)
    raise ValueError(f"example must be 1, 2 or 3, got {example!r}")
