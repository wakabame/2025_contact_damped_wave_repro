from __future__ import annotations

import pytest

from contact_damped_wave.params import EXAMPLE1, EXAMPLE2, Params


def test_paper_settings() -> None:
    """Section 6: l = 1, dt = dx = 1/5000, alpha = 0.01, eps = 0.0005."""
    for params, n_steps in ((EXAMPLE1, 1500), (EXAMPLE2, 2500)):
        assert params.length == 1.0
        assert params.dx == params.dt == 1 / 5000
        assert params.alpha == 0.01
        assert params.eps == 5e-4
        assert params.N == 5000
        assert params.M == n_steps
    assert EXAMPLE1.T == 0.3
    assert EXAMPLE2.T == 0.5


def test_implicit_coefficient_matches_hand_computation() -> None:
    """``(alpha dt + dt^2) / dx^2 = 51``, i.e. the matrix tridiag(-51, 103, -51)."""
    assert EXAMPLE1.implicit_coefficient / EXAMPLE1.dx**2 == pytest.approx(51.0)


def test_penalty_ratio_and_stability() -> None:
    assert EXAMPLE1.penalty_ratio == pytest.approx(0.4)
    assert EXAMPLE1.is_penalty_stable()
    assert not EXAMPLE1.replace(eps=EXAMPLE1.dt).is_penalty_stable()


@pytest.mark.parametrize(
    "changes",
    [
        {"dx": 0.0},
        {"dt": -1.0},
        {"eps": 0.0},
        {"h": 0.0},
        {"h": -1.0},
        {"dx": 0.3},  # 1 / 0.3 is not an integer
        {"T": 0.3, "dt": 0.7},  # T / dt < 1
    ],
)
def test_invalid_parameters_rejected(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Params(**{**{"length": 1.0, "T": 0.3, "dx": 1 / 100, "dt": 1 / 100}, **changes})
