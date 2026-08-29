"""Side-by-side comparison images: the paper's figures above, ours below.

For each of Figures 2-5 of arXiv:2412.06185 this script crops the figure out of
the rendered page images in ``paper/pages/`` and stacks it on top of the
corresponding figure from ``results/ex{1,2}/``, so that the visual agreement
claimed in the README can be checked at a glance.  The four outputs land in
``results/comparison/`` and are embedded in the README.

The page images are not tracked in git.  To recreate them, fetch the PDF (see
the README) and run this script with pypdfium2 available::

    uv run --with pypdfium2 python scripts/make_comparison.py

If ``paper/pages/page{21..24}.png`` already exist they are used as they are.

Usage::

    uv run python scripts/run_examples.py            # produce results/ex{1,2} first
    uv run python scripts/make_comparison.py [--out results/comparison]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

PAGES_DIR = Path("paper") / "pages"
PDF_PATH = Path("paper") / "2412.06185v2.pdf"
#: Width the pages were originally rendered at; crops below are in these pixels.
PAGE_SIZE = (910, 1287)

#: page number -> (crop box in PAGE_SIZE pixels, paper figure label)
PAPER_FIGURES = {
    21: ((80, 235, 860, 1195), "Figure 2"),
    22: ((140, 335, 770, 580), "Figure 3"),
    23: ((80, 90, 860, 1060), "Figure 4"),
    24: ((140, 160, 770, 410), "Figure 5"),
}

#: output name -> (page number, our figure paths; two paths are placed side by side)
COMPARISONS = {
    "fig2_snapshots": (21, ["results/ex1/fig2_snapshots.png"]),
    "fig3_contact_velocity": (
        22,
        ["results/ex1/fig3_contact_set.png", "results/ex1/fig3_velocity.png"],
    ),
    "fig4_snapshots": (23, ["results/ex2/fig4_snapshots.png"]),
    "fig5_contact_velocity": (
        24,
        ["results/ex2/fig5_contact_set.png", "results/ex2/fig5_velocity.png"],
    ),
}


def _render_pages() -> None:
    """Render pages 21-24 of the paper's PDF into ``paper/pages/``."""
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{PAGES_DIR}/page{{21..24}}.png not found and pypdfium2 is not installed.\n"
            "Fetch the PDF (see the README) and run:\n"
            "    uv run --with pypdfium2 python scripts/make_comparison.py"
        ) from error
    if not PDF_PATH.exists():
        raise SystemExit(f"{PDF_PATH} not found; fetch it as described in the README.")
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(PDF_PATH)
    for number in PAPER_FIGURES:
        page = document[number - 1]
        scale = PAGE_SIZE[0] / page.get_size()[0]
        image = page.render(scale=scale).to_pil().convert("RGB")
        image.save(PAGES_DIR / f"page{number}.png")
        print(f"rendered {PAGES_DIR / f'page{number}.png'}")


def _paper_crop(number: int) -> Image.Image:
    box, _ = PAPER_FIGURES[number]
    image = Image.open(PAGES_DIR / f"page{number}.png").convert("RGB")
    if image.size != PAGE_SIZE:
        # Pages rendered at another resolution: rescale the crop box.
        factor = image.size[0] / PAGE_SIZE[0]
        box = tuple(round(value * factor) for value in box)
    return image.crop(box)


def _ours(paths: list[str]) -> Image.Image:
    images = [Image.open(path).convert("RGB") for path in paths]
    if len(images) == 1:
        return images[0]
    # Two panels side by side, scaled to a common height, like the paper's layout.
    height = min(image.height for image in images)
    scaled = [
        image.resize((round(image.width * height / image.height), height), Image.LANCZOS)
        for image in images
    ]
    combined = Image.new("RGB", (sum(image.width for image in scaled), height), "white")
    offset = 0
    for image in scaled:
        combined.paste(image, (offset, 0))
        offset += image.width
    return combined


def _stack(paper: Image.Image, ours: Image.Image, label: str, path: Path) -> None:
    """Paper figure above, ours below, scaled to a common width, with row titles."""
    width = 9.0
    heights = [width * image.height / image.width for image in (paper, ours)]
    title_pad = 0.45
    figure = plt.figure(figsize=(width, sum(heights) + 2 * title_pad))
    grid = figure.add_gridspec(2, 1, height_ratios=heights, hspace=2 * title_pad / sum(heights))
    for axis_index, (image, title) in enumerate(
        (
            (paper, f"arXiv:2412.06185, {label} (Muha & Trifunović)"),
            (ours, "this reproduction"),
        )
    ):
        axis = figure.add_subplot(grid[axis_index])
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.set_axis_off()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results") / "comparison")
    args = parser.parse_args(argv)

    if not all((PAGES_DIR / f"page{number}.png").exists() for number in PAPER_FIGURES):
        _render_pages()
    missing = [
        path for _, paths in COMPARISONS.values() for path in paths if not Path(path).exists()
    ]
    if missing:
        raise SystemExit(
            "missing reproduction figures (run `uv run python scripts/run_examples.py` first):\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, (number, paths) in COMPARISONS.items():
        _, label = PAPER_FIGURES[number]
        _stack(_paper_crop(number), _ours(paths), label, args.out / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
