"""Render a synthetic media fixture through the production offline report renderer."""

from pathlib import Path
import tempfile

from PIL import Image

from npa.workflows.preview_html import image_preview, write_preview


def _frames():
    frames = []
    for index, color in enumerate(((200, 40, 50), (40, 180, 80), (50, 70, 200))):
        data = image_preview(Image.new("RGB", (80, 60), color))
        frames.append(
            {
                "label": f"Recorded fixture step {index}",
                "images": [
                    {"label": name, "data": data}
                    for name in ("Front RGB", "Front depth", "Rear RGB", "Rear depth")
                ],
                "points": [[0, 0, 0, 255, 0, 0], [1, 2, 3, 0, 255, 0]],
            }
        )
    return frames


def _render():
    groups = [
        {"title": name, "note": "Synthetic browser fixture.", "frames": _frames()}
        for name in ("Camera rig", "Independent timeline")
    ]
    with tempfile.TemporaryDirectory(prefix="npa-preview-fixture-") as temporary:
        output = Path(temporary) / "index.html"
        write_preview(
            output,
            title="Workflow report fixture",
            summary="Synthetic test media; not workflow execution evidence.",
            metrics={f"Measurement {index}": index for index in range(6)},
            groups=groups,
            details={
                "quality_passed": False,
                "unsafe_text": "</pre><script>bad()</script>",
            },
        )
        return output.read_text()


if __name__ == "__main__":
    print(_render())
