"""Compose measured native rollout videos over the offline evidence dashboard for MP4 handoff."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from .public_vla_data import write_json
from .public_vla_export import file_sha256


def export_video(report: Path) -> Path:
    """Export paired rollouts at the simulator's 20 Hz control rate in a 1080p film.

    Args:
        report: Completed public report containing HTML and native MP4 files.
    Returns:
        Path to demo.mp4, with complete paired examples and an explicit selection rule.
    Raises:
        ValueError: Real training evidence or native video timing is missing.
        RuntimeError: Browser rendering or video validation fails.
        subprocess.CalledProcessError: FFmpeg cannot encode the evidence film.
    """
    evidence = json.loads((report / "evidence.json").read_text())
    if evidence.get("training_executed") is not True:
        raise ValueError("video export requires measured training evidence")
    with tempfile.TemporaryDirectory(prefix="public-vla-film-") as temporary:
        scenes = _scenes(report, evidence, Path(temporary))
        listing = Path(temporary) / "scenes.txt"
        listing.write_text(
            "".join(f"file '{scene['path'].name}'\n" for scene in scenes)
        )
        output = report / "demo.mp4"
        _encode(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        duration = sum(scene["seconds"] for scene in scenes)
        _validate_video(output, duration)
        _provenance(report, scenes, output, duration)
    return output


def _scenes(report, evidence, temporary):
    phases = ("baseline", "generalist", "specialist")
    scenes = []
    for index, (episode, reason) in enumerate(_episodes(evidence)):
        videos = [report / evidence["evaluation"][p]["videos"][episode] for p in phases]
        seconds = math.ceil(max(_simulation_seconds(path) for path in videos)) + 1
        poster = temporary / f"poster-{index:02d}.png"
        boxes = _poster(report, episode, poster)
        if index == 0:
            (report / "poster.png").write_bytes(poster.read_bytes())
        output = temporary / f"scene-{index:02d}.mp4"
        _encode(_ffmpeg(poster, videos, boxes, output, seconds))
        scenes.append(
            {
                "path": output,
                "episode": episode,
                "selection_reason": reason,
                "seconds": seconds,
                "sources": {path.name: file_sha256(path) for path in videos},
            }
        )
    return scenes


def _episodes(evidence):
    phases = ("baseline", "generalist", "specialist")
    evaluations = evidence["evaluation"]
    count = min(len(evaluations[phase]["videos"]) for phase in phases)
    outcomes = [
        tuple(evaluations[p]["successes"][i] for p in phases) for i in range(count)
    ]
    selected = []
    rules = [
        ("generalist improvement", lambda b, g, s: g and not b),
        ("task-adaptation regression", lambda b, g, s: g and not s),
        ("shared success", lambda b, g, s: b and g and s),
        ("task-adaptation improvement", lambda b, g, s: s and not g),
        ("additional paired case", lambda b, g, s: True),
    ]
    for reason, matches in rules:
        for episode, outcome in enumerate(outcomes):
            if episode not in [i for i, _ in selected] and matches(*outcome):
                selected.append((episode, reason))
                break
        if len(selected) == min(3, count):
            break
    selected.extend(
        (episode, "additional paired case")
        for episode in range(count)
        if episode not in [i for i, _ in selected]
    )
    return selected[:3]


def _simulation_seconds(path):
    stream = _probe(path)
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    if numerator / denominator != 80:
        raise ValueError("expected the pinned native evaluator's 80 fps recording")
    return int(stream["nb_read_frames"]) / 20


def _encode(command):
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
    )


def _provenance(report, scenes, output, duration):
    write_json(
        report / "video-provenance.json",
        {
            "duration_seconds": duration,
            "playback": "20 simulation steps per second; native 80 fps recordings slowed 4x; hold final frame to align each paired scene",
            "scenes": [
                {key: value for key, value in scene.items() if key != "path"}
                for scene in scenes
            ],
            "output_sha256": file_sha256(output),
            "training_executed": True,
        },
    )
    write_json(
        report / "checksums.json",
        {
            p.name: file_sha256(p)
            for p in report.iterdir()
            if p.is_file() and p.name != "checksums.json"
        },
    )


def _poster(report, episode, poster):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            bypass_csp=True,
            offline=True,
        )
        page.goto((report / "index.html").resolve().as_uri())
        page.wait_for_function("window.demoReady === true")
        page.select_option("#episode", str(episode))
        page.add_style_tag(
            content="details{display:none}main{padding:16px 44px}.hero{margin:12px 0}h1{font-size:42px;letter-spacing:-1.2px}.lede{font-size:13px;max-width:460px}.stat{padding:10px 20px}.stat strong{font-size:26px}.film{height:400px}.pipeline{margin:12px 0}.screenhead{padding:10px 15px}.screenfoot{padding:8px 15px}.lower{margin-top:12px}.panel{padding:10px 16px}.footer{margin-top:8px}svg.chart{height:100px}.label{display:none}"
        )
        page.evaluate(
            "document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.controls=false})"
        )
        page.wait_for_function(
            "[...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking)"
        )
        boxes = page.locator("video").evaluate_all(
            "vs=>vs.map(v=>{const b=v.getBoundingClientRect();const s=Math.min(b.width/v.videoWidth,b.height/v.videoHeight);return{x:Math.round(b.x+(b.width-v.videoWidth*s)/2),y:Math.round(b.y+(b.height-v.videoHeight*s)/2),width:Math.round(v.videoWidth*s/2)*2,height:Math.round(v.videoHeight*s/2)*2}})"
        )
        footer = page.locator(".footer").bounding_box()
        if footer is None or footer["y"] + footer["height"] > 1080:
            raise RuntimeError("the dashboard does not fit the 1080p evidence frame")
        page.evaluate(
            "document.querySelectorAll('.screen').forEach(s=>s.querySelector('.screenfoot span').textContent=s.querySelector('.label').textContent)"
        )
        page.screenshot(path=str(poster))
        browser.close()
        return boxes


def _ffmpeg(poster, videos, boxes, output, seconds):
    command = ["ffmpeg", "-y", "-loop", "1", "-framerate", "24", "-i", str(poster)]
    for video in videos:
        command += ["-i", str(video)]
    filters = []
    for index, box in enumerate(boxes, 1):
        filters.append(
            f"[{index}:v]setpts=4*(PTS-STARTPTS),scale={box['width']}:{box['height']},fps=24,tpad=stop_mode=clone:stop_duration={seconds}[v{index}]"
        )
        previous = "0:v" if index == 1 else f"composite{index - 1}"
        filters.append(
            f"[{previous}][v{index}]overlay={box['x']}:{box['y']}:shortest=1[composite{index}]"
        )
    return command + [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[composite3]",
        "-an",
        "-t",
        str(seconds),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _probe(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command))["streams"][0]


def _validate_video(path, seconds):
    stream = _probe(path)
    if (stream["width"], stream["height"], int(stream["nb_read_frames"])) != (
        1920,
        1080,
        seconds * 24,
    ):
        raise RuntimeError(
            "encoded evidence film has unexpected dimensions or frame count"
        )


def main() -> None:
    """Export a shareable MP4 from an already collected public report.

    Args:
        None; --input-path selects the completed report directory.
    Returns:
        None.
    Raises:
        RuntimeError: Browser or FFmpeg validation fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    args = parser.parse_args()
    print(export_video(args.input_path).resolve())


if __name__ == "__main__":
    main()
