#!/usr/bin/env python3
"""render_gpu_coverage_chart.py - draw the published image x Nebius GPU chart.

The compatibility matrix in docs/workbench/image-gpu-compatibility-matrix.md is
the source of truth for what each image can do on each GPU, and
publicly_publishable_tools() is the source of truth for which images are public.
This renders those two together as an SVG so the shape of the coverage - which
bands are solid, which columns are thin - is readable at a glance.

Nothing here decides anything. An unrecognized cell is an error rather than a
guess, so the chart cannot quietly disagree with the table it came from.

USAGE
  render_gpu_coverage_chart.py [--output PATH] [--check]

  --check  re-render and diff against the committed SVG; exit 1 on drift. Use
           this after editing the matrix table.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

# Cell prefix -> (class, label drawn in the cell). Order matters: the first
# matching prefix wins, so keep the longer phrases above their prefixes.
CELL_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("verified", "verified", "verified"),
    ("historical evidence", "historical", "historical"),
    ("supported", "supported", "supported"),
    ("blocked", "blocked", "blocked"),
    ("not routed", "unrouted", "not routed"),
    ("built, no gpu result", "unbuilt", "no GPU run"),
    ("cpu", "cpu", "CPU-only"),
)

FILL = {
    "verified": "#1a7f37",
    "historical": "#57a773",
    "supported": "#c3e6cd",
    "blocked": "#cf222e",
    "unrouted": "#e5903d",
    "unbuilt": "#d4a72c",
    "cpu": "#d8dee4",
}
TEXT = {
    "verified": "#ffffff",
    "historical": "#ffffff",
    "supported": "#11301d",
    "blocked": "#ffffff",
    "unrouted": "#2d1600",
    "cpu": "#3b444d",
    "unbuilt": "#2d2200",
}

BANDS = (
    ("No blocked platform", "clean"),
    ("Blocked on at least one platform", "blocked"),
    ("Built, no GPU result anywhere", "unbuilt"),
    ("CPU-only, GPU-agnostic", "cpu"),
)

LABEL_W = 250
CELL_W = 158
CELL_H = 26
ROW_GAP = 2
BAND_H = 30
TOP = 96
LEGEND_H = 80


def published_images() -> dict[str, str]:
    sys.path.insert(0, str(REPO_ROOT / "npa/src"))
    from npa.deploy import images as deploy_images

    resolve = deploy_images.public_release_tag_for_tool
    return {
        deploy_images.CONTAINER_IMAGE_NAMES[tool]: resolve(tool)
        for tool in deploy_images.publicly_publishable_tools()
    }


def matrix_rows() -> dict[str, list[str]]:
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
    lowered = cell.lower()
    for prefix, kind, label in CELL_CLASSES:
        if lowered.startswith(prefix):
            return kind, label
    raise SystemExit(
        f"ERROR: {image} has cell {cell!r} that this chart cannot classify. "
        "Add it to CELL_CLASSES rather than letting the chart guess."
    )


def band_of(kinds: list[str]) -> str:
    if all(kind == "cpu" for kind in kinds):
        return "cpu"
    if all(kind == "unbuilt" for kind in kinds):
        return "unbuilt"
    if any(kind in {"blocked", "unrouted"} for kind in kinds):
        return "blocked"
    return "clean"


def render(published: dict[str, str], rows: dict[str, list[str]]) -> str:
    # Each cell keeps the matrix's own wording as a tooltip, so the chart can
    # compress "blocked (cu128 NVRTC cannot JIT sm_103)" to "blocked" in the
    # swatch without losing the reason.
    grid: dict[str, list[tuple[str, str, str]]] = {}
    for image in published:
        if image not in rows:
            raise SystemExit(f"ERROR: {image} is published but absent from the matrix")
        grid[image] = [
            (*classify(cell, image), re.sub(r"\s*\[\d+\]", "", cell))
            for cell in rows[image]
        ]

    ordered: list[tuple[str, str, list[str]]] = []
    for band_label, band_key in BANDS:
        members = sorted(
            image
            for image in grid
            if band_of([kind for kind, _, _ in grid[image]]) == band_key
        )
        if members:
            ordered.append((band_label, band_key, members))  # type: ignore[arg-type]

    width = LABEL_W + CELL_W * len(COLUMNS) + 32
    height = TOP + LEGEND_H + sum(
        BAND_H + len(members) * (CELL_H + ROW_GAP) for _, _, members in ordered
    ) + 44

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="-apple-system, BlinkMacSystemFont, '
        f'\'Segoe UI\', Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="16" y="30" font-size="17" font-weight="600" fill="#1f2328">'
        "Published GHCR images on Nebius GPU platforms</text>",
        f'<text x="16" y="52" font-size="12.5" fill="#59636e">'
        f"{len(published)} images in the public publishing plan. Generated from "
        f"docs/workbench/image-gpu-compatibility-matrix.md.</text>",
    ]

    x0 = 16 + LABEL_W
    for index, (name, sm) in enumerate(COLUMNS):
        cx = x0 + index * CELL_W + CELL_W / 2
        out.append(
            f'<text x="{cx:.0f}" y="{TOP - 26}" font-size="12.5" font-weight="600" '
            f'fill="#1f2328" text-anchor="middle">{escape(name)}</text>'
        )
        out.append(
            f'<text x="{cx:.0f}" y="{TOP - 11}" font-size="11" fill="#59636e" '
            f'text-anchor="middle">{sm}</text>'
        )

    y = TOP
    for band_label, _band_key, members in ordered:
        out.append(
            f'<text x="16" y="{y + 19}" font-size="12.5" font-weight="600" fill="#1f2328">'
            f"{escape(band_label)} &#183; {len(members)}</text>"
        )
        out.append(
            f'<line x1="16" y1="{y + 26}" x2="{width - 16}" y2="{y + 26}" '
            f'stroke="#d1d9e0" stroke-width="1"/>'
        )
        y += BAND_H
        for image in members:
            out.append(
                f'<text x="16" y="{y + 17}" font-size="12" fill="#1f2328" '
                f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
                f"{escape(image)}</text>"
            )
            for index, (kind, label, detail) in enumerate(grid[image]):
                cx = x0 + index * CELL_W
                column = f"{COLUMNS[index][0]} {COLUMNS[index][1]}"
                out.append(f'<g><title>{escape(f"{image} on {column}: {detail}")}</title>')
                out.append(
                    f'<rect x="{cx}" y="{y}" width="{CELL_W - 4}" height="{CELL_H}" '
                    f'rx="3" fill="{FILL[kind]}"/>'
                )
                out.append(
                    f'<text x="{cx + (CELL_W - 4) / 2:.0f}" y="{y + 17}" font-size="11.5" '
                    f'fill="{TEXT[kind]}" text-anchor="middle">{escape(label)}</text>'
                )
                out.append("</g>")
            y += CELL_H + ROW_GAP
        y += 8

    legend_y = y + 6
    legend_lines = (
        "verified = a real capability run on that part &#183; supported = the toolchain can "
        "execute there, no run recorded &#183; historical = a run on an earlier candidate",
        "blocked = an upstream or physical limit &#183; not routed = the launcher never sends "
        "work there &#183; no GPU run = built, nothing measured. Hover a cell for the reason.",
        "Rendering needs RT cores, which only L40S and RTX PRO 6000 have. Every image is "
        "linux/amd64, so aarch64 gpu-gb300 is uncovered.",
    )
    for offset, line in enumerate(legend_lines):
        out.append(
            f'<text x="16" y="{legend_y + offset * 17}" font-size="11.5" fill="#59636e">'
            f"{line}</text>"
        )
    out.append(
        f'<text x="16" y="{legend_y + len(legend_lines) * 17 + 4}" font-size="10.5" '
        f'fill="#818b98">Rendered {dt.date.today().isoformat()} by '
        "npa/scripts/render_gpu_coverage_chart.py</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    svg = render(published_images(), matrix_rows())

    if args.check:
        if not args.output.exists():
            print(f"ERROR: {args.output} does not exist", file=sys.stderr)
            return 1
        current = args.output.read_text(encoding="utf-8")
        # The render date is the one line that legitimately moves on its own.
        strip = lambda text: re.sub(r"Rendered \d{4}-\d{2}-\d{2}", "Rendered", text)  # noqa: E731
        if strip(current) != strip(svg):
            print(
                f"ERROR: {args.output} is stale; re-run without --check",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output} is up to date")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
