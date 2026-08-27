"""Shared fixtures.  Tests use coarse grids so that the suite runs in seconds.

Every coarse configuration keeps ``dt / eps < 1``, the stability constraint of
the explicit penalty force (see :attr:`Params.penalty_ratio`).
"""

from __future__ import annotations

import pytest

from contact_damped_wave.params import EXAMPLE1, EXAMPLE2, Params


@pytest.fixture
def coarse_example1() -> Params:
    return EXAMPLE1.replace(dx=1 / 500, dt=1 / 500, eps=4e-3)


@pytest.fixture
def coarse_example2() -> Params:
    return EXAMPLE2.replace(dx=1 / 1000, dt=1 / 1000, eps=2e-3)
