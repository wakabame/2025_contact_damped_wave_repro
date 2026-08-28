"""The GIF writer.  Kept on a very coarse grid: encoding frames is the slow part."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from contact_damped_wave.animation import animate, frame_indices
from contact_damped_wave.initial_data import initial_data
from contact_damped_wave.params import EXAMPLE3
from contact_damped_wave.solver import solve


def test_frame_indices_spans_the_run() -> None:
    indices = frame_indices(101, 11)
    assert indices[0] == 0
    assert indices[-1] == 100
    assert indices.size == 11
    assert np.all(np.diff(indices) > 0)


def test_frame_indices_never_repeats_a_snapshot() -> None:
    # Asking for more frames than there are snapshots must not stutter.
    indices = frame_indices(5, 50)
    assert np.array_equal(indices, np.arange(5))
    for bad in (0, -1):
        with pytest.raises(ValueError):
            frame_indices(10, bad)
        with pytest.raises(ValueError):
            frame_indices(bad, 10)


@pytest.fixture(scope="module")
def tiny_result():
    params = EXAMPLE3.replace(dx=1 / 200, dt=1 / 200, eps=1e-2, T=0.2)
    _, _, eta0, v0 = initial_data(3, params)
    return solve(params, eta0, v0, store_every=2)


def test_animate_writes_a_gif(tiny_result, tmp_path: Path) -> None:
    path = animate(tiny_result, tmp_path / "movie.gif", frames=6, fps=5, dpi=40)
    assert path == tmp_path / "movie.gif"
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 6


def test_animate_appends_the_gif_suffix(tiny_result, tmp_path: Path) -> None:
    path = animate(tiny_result, tmp_path / "sub" / "movie", frames=3, fps=5, dpi=40)
    assert path == tmp_path / "sub" / "movie.gif"
    assert path.exists()
