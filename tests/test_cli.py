"""End-to-end smoke tests of the ``cdw`` command line interface."""

from __future__ import annotations

from contact_damped_wave.cli import main


def test_run_writes_figures_summary_and_archive(tmp_path) -> None:
    code = main(
        [
            "run",
            "--example",
            "1",
            "--out",
            str(tmp_path),
            "--dx",
            "0.005",
            "--dt",
            "0.005",
            "--eps",
            "0.0125",
            "--final-time",
            "0.06",
            "--store-every",
            "2",
            "--quiet",
        ]
    )
    assert code == 0
    for name in (
        "summary.txt",
        "fig2_snapshots.png",
        "fig3_contact_set.png",
        "fig3_velocity.png",
        "energy.png",
    ):
        assert (tmp_path / name).exists(), name
    archives = list(tmp_path.glob("*.npz"))
    assert len(archives) == 1
    # The archive name encodes the snapshot stride as well: two runs differing
    # only in --store-every store different data and must not share a name.
    assert "_se2" in archives[0].name


def test_animate_writes_a_gif(tmp_path) -> None:
    code = main(
        [
            "animate",
            "--example",
            "1",
            "--out",
            str(tmp_path),
            "--dx",
            "0.005",
            "--dt",
            "0.005",
            "--eps",
            "0.0125",
            "--final-time",
            "0.06",
            "--frames",
            "4",
            "--fps",
            "4",
            "--dpi",
            "40",
            "--quiet",
        ]
    )
    assert code == 0
    assert (tmp_path / "animation.gif").stat().st_size > 0
