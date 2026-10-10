"""Export an allowlisted evidence report and record its deterministic HTML walkthrough."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess

from npa.workbench.dataset.storage import read_json_uri


def _history(workspace, phase):
    history = []
    paths = sorted(
        (workspace / phase).glob("*/decision.json"), key=lambda p: int(p.parent.name)
    )
    for path in paths:
        gate = read_json_uri(str(path))
        candidate = read_json_uri(str(path.with_name("candidate.json")))
        evaluation = read_json_uri(str(path.with_name("evaluation.json")))
        rollouts = read_json_uri(evaluation["rollouts_uri"])
        history.append(
            {
                "iteration": int(path.parent.name),
                "decision": gate["decision"],
                "rates": gate["success_rates"],
                "thresholds": gate["thresholds"],
                "systems": evaluation["systems"],
                "loss": candidate["training_loss"],
                "checkpoint": candidate["checkpoint"]["sha256"],
                "rollouts": rollouts["systems"],
            }
        )
    if not history or history[-1]["decision"] != "promote_checkpoint":
        raise ValueError("report requires a measured successful final gate")
    return history


def _summary(workspace, steps):
    curated = read_json_uri(str(workspace / "curation/episodes.json"))
    split = read_json_uri(str(workspace / "splits/index.json"))
    deployed = read_json_uri(str(workspace / "deployment/result.json"))
    summary = {
        "schema": "npa.policy.demo-report.v1",
        "runtime": "local-reference",
        "model": "Damped least-squares behavior cloning",
        "environment": "Analytical Cartesian reacher",
        "curation": {
            key: curated[key]
            for key in (
                "engine",
                "input_count",
                "selected_count",
                "inhouse_disagreement_count",
            )
        },
        "splits": {name: item["groups"] for name, item in split["partitions"].items()},
        "steps": [
            {key: step[key] for key in ("state", "status", "seconds")} for step in steps
        ],
        "pretrain": _history(workspace, "pretrain"),
        "finetune": _history(workspace, "finetune"),
        "deployment": {
            "systems": deployed["systems"],
            "checkpoint": deployed["checkpoint"]["sha256"],
        },
        "previews": [
            base64.b64encode((workspace / "previews" / name).read_bytes()).decode()
            for name in ("0000.png", "0001.png", "0002.png", "0120.png")
        ],
    }
    if summary["deployment"]["checkpoint"] != summary["finetune"][-1]["checkpoint"]:
        raise ValueError("terminal test did not use the approved checkpoint")
    return summary


def _video_command(output):
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-framerate",
        "24",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "demo.mp4"),
    ]


def _encode_video(page, output):
    with subprocess.Popen(
        _video_command(output), stdin=subprocess.PIPE, stderr=subprocess.PIPE
    ) as encoder:
        try:
            for frame in range(45 * 24):
                page.evaluate("t => window.setStoryTime(t)", frame / 24)
                encoder.stdin.write(page.screenshot())
            encoder.stdin.close()
            error = encoder.stderr.read()
            if encoder.wait():
                raise RuntimeError(f"demo video encoding failed: {error.decode()}")
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise


def _capture(output, *, video):
    from playwright.sync_api import sync_playwright

    errors = []
    with sync_playwright() as playwright:
        with playwright.chromium.launch() as browser:
            page = browser.new_page(
                viewport={"width": 1440, "height": 810}, device_scale_factor=1
            )
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto((output / "index.html").as_uri() + "?capture=1")
            page.locator('body[data-ready="true"]').wait_for()
            page.evaluate("window.setStoryTime(38)")
            page.screenshot(path=str(output / "poster.png"))
            if video:
                _encode_video(page, output)
            if errors:
                raise RuntimeError("demo HTML has JavaScript errors")


def export(workspace: Path, output: Path, steps: list, *, video: bool) -> dict:
    """Write a self-contained HTML report, JSON evidence, poster, and optional MP4.

    Args:
        workspace: Private reference evidence directory.
        output: Public artifact destination, separate from the private subdirectory.
        steps: Completed stage names, statuses, and observed durations.
        video: Whether to record a forty-five-second MP4 walkthrough.
    Returns:
        Allowlisted evidence with no worker paths, credentials, or request payloads.
    Raises:
        ValueError: Required gate or checkpoint evidence is inconsistent.
        RuntimeError: Browser rendering or video encoding fails.
    """
    summary = _summary(workspace, steps)
    payload = json.dumps(summary, separators=(",", ":"), allow_nan=False)
    template = Path(__file__).with_name("demo_report.html").read_text()
    (output / "index.html").write_text(
        template.replace("__DEMO_EVIDENCE__", payload.replace("<", "\\u003c"))
    )
    (output / "evidence.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    _capture(output, video=video)
    artifacts = ["index.html", "evidence.json", "poster.png"] + (
        ["demo.mp4"] if video else []
    )
    hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in artifacts
    }
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2))
    return summary
