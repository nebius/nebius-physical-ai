"""Export a customer-safe foundation-training architecture preview with public footage."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from .demo_report import _video_command
from .foundation_media import public_clips


def _evidence(media):
    return {
        "schema": "npa.foundation.preview.v1",
        "artifact_kind": "architecture-preview",
        "training_executed": False,
        "model": "NVIDIA GR00T N1.7 3B (intended public model)",
        "runtime": "Soperator / Slurm + Pyxis + torchrun (intended)",
        "media": media,
        "validation": {
            "gpu_training": "not executed",
            "model_backbone_access": "pending",
            "benchmark_results": "not available",
            "visual_topology": "illustrative",
        },
    }


def _write_html(output, evidence):
    embedded = json.loads(json.dumps(evidence))
    for clip in embedded["media"]["clips"]:
        encoded = base64.b64encode((output / clip["file"]).read_bytes()).decode("ascii")
        clip["data"] = "data:video/mp4;base64," + encoded
    payload = json.dumps(embedded, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("foundation_report.html").read_text()
    (output / "index.html").write_text(
        template.replace("__FOUNDATION_EVIDENCE__", payload)
    )
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    (output / "attribution.txt").write_text(
        "Public LIBERO demonstrations from lerobot/libero. License: Apache-2.0.\n"
        "https://www.apache.org/licenses/LICENSE-2.0\n"
        + evidence["media"]["source_url"]
        + "\n"
        "Clips extracted at the source episode timestamps and transcoded to H.264.\n"
        "These are dataset demonstrations, not outputs from a trained model in this run.\n"
        "Source revision, paths, timestamps, and file hashes: evidence.json.\n"
        "Original LIBERO project: https://github.com/Lifelong-Robot-Learning/LIBERO\n"
    )


def _capture(output, video):
    from playwright.sync_api import sync_playwright

    errors = []
    with sync_playwright() as playwright:
        with playwright.chromium.launch() as browser:
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto((output / "index.html").as_uri() + "?capture=1")
            page.locator('body[data-ready="true"]').wait_for()
            page.evaluate("t => window.renderAt(t)", 3)
            page.screenshot(path=str(output / "poster.png"))
            if video:
                _encode(page, output)
            if errors:
                raise RuntimeError("foundation preview has browser errors")


def _encode(page, output):
    with subprocess.Popen(
        _video_command(output), stdin=subprocess.PIPE, stderr=subprocess.PIPE
    ) as encoder:
        try:
            for frame in range(60 * 24):
                page.evaluate("t => window.renderAt(t)", frame / 24)
                encoder.stdin.write(page.screenshot())
            encoder.stdin.close()
            error = encoder.stderr.read()
            if encoder.wait():
                raise RuntimeError("preview video encoding failed: " + error.decode())
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise


def export_preview(output: Path, cache: Path, *, video: bool) -> dict:
    """Build the offline preview and record the same view as an MP4 walkthrough.

    Args:
        output: New destination for the shareable artifacts and public clips.
        cache: Download cache containing only public immutable dataset inputs.
        video: Whether to encode a sixty-second, 1080p walkthrough.
    Returns:
        Public evidence identifying media provenance and execution limitations.
    Raises:
        FileExistsError: Output already exists; prior artifacts are preserved.
        RuntimeError: Browser rendering or video encoding fails.
    """
    output.mkdir(parents=True, exist_ok=False)
    evidence = _evidence(public_clips(cache, output))
    _write_html(output, evidence)
    _capture(output, video)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2))
    return evidence


def main() -> None:
    """Run the public-data preview exporter without accessing private runtime inputs.

    Args:
        None.
    Returns:
        None.
    Raises:
        SystemExit: Command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    export_preview(args.output_path, args.cache_path, video=not args.html_only)


if __name__ == "__main__":
    main()
