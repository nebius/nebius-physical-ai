#!/usr/bin/env python3
"""Render published-image GPU coverage from repository sources of truth.

The compatibility matrix defines each image/GPU cell and the public publishing
plan defines which images belong in the chart. Unknown cell wording is an
error, so the generated SVG cannot silently guess at compatibility.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / "docs/workbench/image-gpu-compatibility-matrix.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs/assets/image-gpu-coverage.svg"

COLUMNS = (
    ("L40S", "sm_89"),
    ("H100 / H200", "sm_90"),
    ("RTX PRO 6000", "sm_120"),
    ("B200", "sm_100"),
    ("B300", "sm_103"),
)

# Cell prefix -> (class, compact cell label). Order matters because the first
# matching prefix wins; keep longer phrases ahead of their prefixes.
CELL_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("verified", "verified", "verified"),
    ("historical predecessor only", "historical", "predecessor"),
    ("historical evidence", "historical", "historical"),
    ("supported", "supported", "supported"),
    ("blocked", "blocked", "blocked"),
    ("not routed", "unrouted", "not routed"),
    ("unverified runtime", "unverified", "unverified"),
    ("unverified", "unverified", "unverified"),
    ("built, no gpu result", "unverified", "unverified"),
    ("cpu", "cpu", "CPU-only"),
)

FILL = {
    "verified": "#1a7f37",
    "historical": "#57a773",
    "supported": "#c3e6cd",
    "blocked": "#cf222e",
    "unrouted": "#e5903d",
    "unverified": "#d4a72c",
    "cpu": "#d8dee4",
}
TEXT = {
    "verified": "#ffffff",
    "historical": "#ffffff",
    "supported": "#11301d",
    "blocked": "#ffffff",
    "unrouted": "#2d1600",
    "unverified": "#2d2200",
    "cpu": "#3b444d",
}
BANDS = (
    ("No blocked platform", "clean"),
    ("Blocked on at least one platform", "blocked"),
    ("No verified GPU result anywhere", "unverified"),
    ("CPU-only, GPU-agnostic", "cpu"),
)

LABEL_W = 250
CELL_W = 158
CELL_H = 26
ROW_GAP = 2
BAND_H = 30
TOP = 96
LEGEND_H = 80

Grid = dict[str, list[tuple[str, str, str]]]
Band = tuple[str, str, list[str]]


def published_images() -> dict[str, str]:
    """Read the accepted public publishing plan.

    Args:
        None.

    Returns:
        Public image names mapped to their accepted release tags.

    Raises:
        None.
    """
    sys.path.insert(0, str(REPO_ROOT / "npa/src"))
    from npa.deploy import images as deploy_images

    resolve = deploy_images.public_release_tag_for_tool
    return {
        deploy_images.CONTAINER_IMAGE_NAMES[tool]: resolve(tool)
        for tool in deploy_images.publicly_publishable_tools()
    }


def matrix_rows() -> dict[str, list[str]]:
    """Parse the compatibility matrix's main table.

    Args:
        None.

    Returns:
        Image names mapped to their five GPU-platform cells.

    Raises:
        None.
    """
    section = MATRIX.read_text(encoding="utf-8").split("## Compatibility matrix")[1]
    section = section.split("###")[0]
    rows: dict[str, list[str]] = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[0].startswith("---") or cells[0] == "Image":
            continue
        name = re.sub(r"[`*]", "", cells[0]).split("(")[0].strip()
        rows[name] = [cell.replace("**", "") for cell in cells[1:]]
    return rows


def classify(cell: str, image: str) -> tuple[str, str]:
    """Classify one matrix cell, failing closed on unknown wording.

    Args:
        cell: The matrix cell's Markdown content without bold markers.
        image: The image name used in any classification error.

    Returns:
        The color class and compact label for the cell.

    Raises:
        SystemExit: If the matrix introduces unrecognized cell wording.
    """
    lowered = cell.lower()
    for prefix, kind, label in CELL_CLASSES:
        if lowered.startswith(prefix):
            return kind, label
    raise SystemExit(
        f"ERROR: {image} has cell {cell!r} that this chart cannot classify. "
        "Add it to CELL_CLASSES rather than letting the chart guess."
    )


def band_of(kinds: list[str]) -> str:
    """Choose the mutually exclusive summary band for an image row.

    Args:
        kinds: Classified cell kinds in GPU-column order.

    Returns:
        The row's summary band key.

    Raises:
        None.
    """
    if all(kind == "cpu" for kind in kinds):
        return "cpu"
    if any(kind in {"blocked", "unrouted"} for kind in kinds):
        return "blocked"
    if all(kind == "unverified" for kind in kinds):
        return "unverified"
    return "clean"


def _tooltip_text(cell: str) -> str:
    """Remove citation markup while retaining the matrix's human wording."""
    without_links = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", cell)
    return re.sub(r"\s*\[\d+]", "", without_links)


def _classified_grid(published: dict[str, str], rows: dict[str, list[str]]) -> Grid:
    """Build chart cells for every published image."""
    grid: Grid = {}
    for image in published:
        if image not in rows:
            raise SystemExit(f"ERROR: {image} is published but absent from the matrix")
        grid[image] = [
            (*classify(cell, image), _tooltip_text(cell)) for cell in rows[image]
        ]
    return grid


def _ordered_bands(grid: Grid) -> list[Band]:
    """Group image rows into stable, nonempty summary bands."""
    ordered: list[Band] = []
    for band_label, band_key in BANDS:
        members = sorted(
            image
            for image, cells in grid.items()
            if band_of([kind for kind, _, _ in cells]) == band_key
        )
        if members:
            ordered.append((band_label, band_key, members))
    return ordered


def _canvas_height(ordered: list[Band]) -> int:
    """Calculate chart height from its rendered band membership."""
    rows_height = sum(
        BAND_H + len(members) * (CELL_H + ROW_GAP)
        for _, _, members in ordered
    )
    return TOP + LEGEND_H + rows_height + 44


def _header_elements(width: int, image_count: int) -> list[str]:
    """Return the chart title and source annotation elements."""
    return [
        f'<rect width="{width}" height="100%" fill="#ffffff"/>',
        '<text x="16" y="30" font-size="17" font-weight="600" fill="#1f2328">'
        "Published GHCR images on Nebius GPU platforms</text>",
        '<text x="16" y="52" font-size="12.5" fill="#59636e">'
        f"{image_count} images in the public publishing plan. Generated from "
        "docs/workbench/image-gpu-compatibility-matrix.md.</text>",
    ]


def _column_header_elements(x0: int) -> list[str]:
    """Return GPU name and compute-capability column headers."""
    out: list[str] = []
    for index, (name, sm) in enumerate(COLUMNS):
        center = x0 + index * CELL_W + CELL_W / 2
        out.extend(
            [
                f'<text x="{center:.0f}" y="{TOP - 26}" font-size="12.5" '
                f'font-weight="600" fill="#1f2328" text-anchor="middle">'
                f"{escape(name)}</text>",
                f'<text x="{center:.0f}" y="{TOP - 11}" font-size="11" '
                f'fill="#59636e" text-anchor="middle">{sm}</text>',
            ]
        )
    return out


def _cell_elements(
    image: str, index: int, cell: tuple[str, str, str], y: int
) -> list[str]:
    """Return SVG elements for one colored compatibility cell."""
    kind, label, detail = cell
    x = 16 + LABEL_W + index * CELL_W
    column = f"{COLUMNS[index][0]} {COLUMNS[index][1]}"
    return [
        f'<g><title>{escape(f"{image} on {column}: {detail}")}</title>',
        f'<rect x="{x}" y="{y}" width="{CELL_W - 4}" height="{CELL_H}" '
        f'rx="3" fill="{FILL[kind]}"/>',
        f'<text x="{x + (CELL_W - 4) / 2:.0f}" y="{y + 17}" font-size="11.5" '
        f'fill="{TEXT[kind]}" text-anchor="middle">{escape(label)}</text>',
        "</g>",
    ]


def _row_elements(
    image: str, cells: list[tuple[str, str, str]], y: int
) -> list[str]:
    """Return the image label and all compatibility cells for one row."""
    out = [
        f'<text x="16" y="{y + 17}" font-size="12" fill="#1f2328" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
        f"{escape(image)}</text>"
    ]
    for index, cell in enumerate(cells):
        out.extend(_cell_elements(image, index, cell, y))
    return out


def _band_elements(
    band: Band, grid: Grid, y: int, width: int
) -> tuple[list[str], int]:
    """Return one band and the next available vertical position."""
    label, _key, members = band
    out = [
        f'<text x="16" y="{y + 19}" font-size="12.5" font-weight="600" '
        f'fill="#1f2328">{escape(label)} &#183; {len(members)}</text>',
        f'<line x1="16" y1="{y + 26}" x2="{width - 16}" y2="{y + 26}" '
        'stroke="#d1d9e0" stroke-width="1"/>',
    ]
    y += BAND_H
    for image in members:
        out.extend(_row_elements(image, grid[image], y))
        y += CELL_H + ROW_GAP
    return out, y + 8


def _legend_elements(y: int) -> list[str]:
    """Return legend and generated-date elements."""
    lines = (
        "verified = current-release run &#183; supported = compatible toolchain, no run "
        "&#183; historical = earlier release or predecessor",
        "blocked = upstream/physical limit &#183; not routed = launcher exclusion &#183; "
        "unverified = current release lacks cell evidence. Hover for detail.",
        "Rendering needs RT cores (L40S or RTX PRO 6000). Current public runtimes "
        "are linux/amd64, so aarch64 gpu-gb300 is uncovered.",
    )
    out = [
        f'<text x="16" y="{y + offset * 17}" font-size="11.5" fill="#59636e">'
        f"{line}</text>"
        for offset, line in enumerate(lines)
    ]
    rendered = datetime.date.today().isoformat()
    out.append(
        f'<text x="16" y="{y + len(lines) * 17 + 4}" font-size="10.5" '
        f'fill="#818b98">Rendered {rendered} by '
        "npa/scripts/render_gpu_coverage_chart.py</text>"
    )
    return out


def render(published: dict[str, str], rows: dict[str, list[str]]) -> str:
    """Render an SVG for public images and compatibility rows.

    Args:
        published: Public image names mapped to accepted release tags.
        rows: Matrix image names mapped to GPU-platform cells.

    Returns:
        A complete SVG document.

    Raises:
        SystemExit: If a public image or matrix cell cannot be represented.
    """
    grid = _classified_grid(published, rows)
    ordered = _ordered_bands(grid)
    width = LABEL_W + CELL_W * len(COLUMNS) + 32
    height = _canvas_height(ordered)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="-apple-system, '
        "BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\">"
    ]
    out.extend(_header_elements(width, len(published)))
    out.extend(_column_header_elements(16 + LABEL_W))
    y = TOP
    for band in ordered:
        elements, y = _band_elements(band, grid, y, width)
        out.extend(elements)
    out.extend(_legend_elements(y + 6))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _without_render_date(svg: str) -> str:
    """Normalize the intentionally volatile rendered date."""
    return re.sub(r"Rendered \d{4}-\d{2}-\d{2}", "Rendered", svg)


def main(argv: list[str] | None = None) -> int:
    """Render the chart, or check the committed chart for drift.

    Args:
        argv: Optional command-line arguments; defaults to ``sys.argv``.

    Returns:
        Zero on success, or one when ``--check`` detects drift.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    svg = render(published_images(), matrix_rows())
    if args.check:
        if not args.output.exists():
            print(f"ERROR: {args.output} does not exist", file=sys.stderr)
            return 1
        if _without_render_date(args.output.read_text(encoding="utf-8")) != (
            _without_render_date(svg)
        ):
            print(f"ERROR: {args.output} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
