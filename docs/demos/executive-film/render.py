"""Render an exact two-minute executive film from hash-verified local run artifacts."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from graphics import _draw_overlay, _draw_titles, _rectangles
from soundtrack import _duration, _mix

_ROOT = Path(__file__).parent


def _hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ], text=True))


def _load_assets(path):
    assets = json.loads(path.read_text())
    for asset in assets.values():
        source = Path(asset["path"]).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        asset["path"] = str(source.resolve())
    return assets


def _validate(storyboard, assets):
    if sum(scene["duration"] for scene in storyboard["scenes"]) != 120:
        raise ValueError("The full storyboard must last exactly 120 seconds")
    required = {role for scene in storyboard["scenes"] for role in scene["assets"]}
    if required - assets.keys():
        raise ValueError(f"Missing asset roles: {sorted(required - assets.keys())}")
    for role in sorted(required):
        asset = assets[role]
        path = Path(asset["path"]).expanduser().resolve(strict=True)
        if _hash(path) != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch for asset {role}")
        if asset["kind"] not in {"image", "video"}:
            raise ValueError(f"Unsupported asset kind for {role}")
        crop = asset.get("crop")
        if crop and (len(crop) != 4 or any(type(v) is not int or v < 0 for v in crop)
                     or crop[2] == 0 or crop[3] == 0):
            raise ValueError(f"Invalid pixel crop for {role}")
        streams = _probe(path)["streams"]
        visual = next((s for s in streams if s["codec_type"] == "video"), None)
        if visual is None:
            raise ValueError(f"No decodable visual stream for {role}")
        if crop and (crop[0] + crop[2] > visual["width"] or crop[1] + crop[3] > visual["height"]):
            raise ValueError(f"Crop exceeds source bounds for {role}")
        asset["path"] = str(path)
    return required


def _input(asset):
    if asset["kind"] == "image":
        return ["-loop", "1", "-framerate", "30", "-i", asset["path"]]
    return ["-stream_loop", "-1", "-i", asset["path"]]


def _media_filter(index, asset, rectangle, duration):
    x, y, width, height = rectangle
    content_height = height if width == 1920 else height - 56
    filters = ["setpts=PTS-STARTPTS", "fps=30"]
    if asset.get("crop"):
        crop_x, crop_y, crop_width, crop_height = asset["crop"]
        filters.append(f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}")
    filters += [f"scale={width}:{content_height}:force_original_aspect_ratio=decrease:force_divisible_by=2",
                f"pad={width}:{height}:(ow-iw)/2:({content_height}-ih)/2:color=0x080d12", "setsar=1"]
    if asset["kind"] == "image":
        filters.append(f"zoompan=z='1+0.018*on/{30 * duration}':"
                       f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={width}x{height}:fps=30")
    filters += [f"trim=duration={duration}", "format=yuv420p"]
    layer = f"[{index}:v]{','.join(filters)}[media{index}]"
    overlay = f"[base{index - 1}][media{index}]overlay={x}:{y}:shortest=1[base{index}]"
    return [layer, overlay]


def _filter_graph(scene, assets):
    duration = scene["duration"]
    graph = ["[0:v]format=yuv420p[base0]"]
    for index, (role, rectangle) in enumerate(zip(scene["assets"], _rectangles(scene), strict=True), 1):
        graph += _media_filter(index, assets[role], rectangle, duration)
    count = len(scene["assets"])
    graph.append(f"[{count + 1}:v]format=rgba[graphic]")
    graph.append(f"[base{count}][graphic]overlay=0:0:shortest=1[framed]")
    graph.append(f"[{count + 2}:v]format=rgba,fade=t=in:st=0.15:d=0.5:alpha=1[titles]")
    graph.append("[framed][titles]overlay=x='-55*exp(-6*t)':y=0:shortest=1[composed]")
    graph.append("color=c=0xdcff46:s=1920x4:r=30[bar]")
    graph.append(f"[composed][bar]overlay=x='-1920+1920*min(t/{duration},1)':y=1076:shortest=1,"
                 f"fade=t=in:d=0.18,fade=t=out:st={duration - 0.18}:d=0.18,format=yuv420p[out]")
    return ";\n".join(graph)


def _render_scene(scene, index, total, assets, directory):
    overlay = directory / f"{scene['id']}.png"
    target = directory / f"{scene['id']}.mp4"
    _draw_overlay(scene, index, total, overlay)
    titles = directory / f"{scene['id']}-titles.png"
    _draw_titles(scene, titles)
    graph = directory / f"{scene['id']}.ffmpeg"
    graph.write_text(_filter_graph(scene, assets))
    command = ["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "2",
               "-f", "lavfi", "-i", "color=c=0x080d12:s=1920x1080:r=30"]
    for role in scene["assets"]:
        command += _input(assets[role])
    command += ["-loop", "1", "-framerate", "30", "-i", str(overlay),
                "-loop", "1", "-framerate", "30", "-i", str(titles),
                "-filter_complex_script", str(graph), "-map", "[out]", "-an",
                "-t", str(scene["duration"]), "-c:v", "libx264", "-preset", "fast",
                "-crf", "18", "-threads", "4", "-r", "30", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)
    print(f"Rendered {scene['id']}", flush=True)
    return target


def _join(parts, sound, directory):
    listing = directory / "segments.txt"
    listing.write_text("".join(f"file {p.name}\n" for p in parts))
    target = directory / "workbench-executive-film.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(listing),
        "-i", str(sound), "-map", "0:v", "-map", "1:a", "-c", "copy", "-t", "120",
        "-metadata", "title=Intelligence that moves | Nebius Physical AI Workbench",
        "-movflags", "+faststart", str(target),
    ], check=True)
    return target


def _timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _seconds(timestamp):
    hours, minutes, tail = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(tail.replace(",", "."))


def _captions(scenes, voice_dir, directory):
    captions, elapsed = [], 0
    for scene in scenes:
        duration = _duration(voice_dir / f"{scene['id']}.mp3")
        speed = max(1, duration / (scene["duration"] - 0.85))
        source = voice_dir / f"{scene['id']}.srt"
        for block in source.read_text().strip().split("\n\n"):
            lines = block.splitlines()
            if len(lines) < 3:
                continue
            start, end = lines[1].split(" --> ")
            start = elapsed + 0.35 + _seconds(start) / speed
            end = min(elapsed + scene["duration"], elapsed + 0.35 + _seconds(end) / speed)
            captions.append(f"{len(captions) + 1}\n{_timestamp(start)} --> {_timestamp(end)}\n"
                            + "\n".join(lines[2:]) + "\n")
        elapsed += scene["duration"]
    (directory / "workbench-executive-film.srt").write_text("\n".join(captions))


def _evidence(target, storyboard, assets, directory):
    probe = _probe(target)
    duration = float(probe["format"]["duration"])
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    if not math.isclose(duration, 120, abs_tol=0.05) or video["nb_frames"] != "3600":
        raise ValueError("Final duration or frame count does not match the storyboard")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"], check=True)
    manifest = {"title": storyboard["title"], "duration_seconds": duration,
                "frames": 3600, "width": video["width"], "height": video["height"],
                "fps": video["r_frame_rate"], "sha256": _hash(target),
                "font_sha256": _hash(_ROOT / "fonts/Manrope.ttf"),
                "storyboard_sha256": _hash(_ROOT / "storyboard.json"),
                "renderer_sha256": {p.name: _hash(p) for p in _ROOT.glob("*.py")},
                "narration_sha256": _hash(directory / "narration.wav"),
                "score_sha256": _hash(directory / "original-score.wav"),
                "captions_sha256": _hash(directory / "workbench-executive-film.srt"),
                "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                "assets": {role: {"sha256": asset["sha256"], "kind": asset["kind"],
                                  "crop": asset.get("crop")} for role, asset in assets.items()},
                "editorial": "Saved run outputs; crops, loops and presentation zooms; no new model inference claimed."}
    (directory / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "3", "-i", str(target),
                    "-frames:v", "1", str(directory / "poster.png")], check=True)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Private JSON mapping asset roles to local paths, kinds and SHA-256 digests.")
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Verify all media without rendering.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not all(shutil.which(binary) for binary in ("ffmpeg", "ffprobe")):
        parser.error("Install FFmpeg with libx264 and ffprobe before rendering")
    storyboard = json.loads((_ROOT / "storyboard.json").read_text())
    assets = _load_assets(args.assets)
    _validate(storyboard, assets)
    if args.check:
        print("All storyboard assets verified")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = storyboard["scenes"]
    sound = _mix(scenes, args.voice_dir, args.output_dir)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(_render_scene, scene, index, len(scenes), assets,
                               args.output_dir) for index, scene in enumerate(scenes)]
        parts = [result.result() for result in pending]
    target = _join(parts, sound, args.output_dir)
    _captions(scenes, args.voice_dir, args.output_dir)
    _evidence(target, storyboard, assets, args.output_dir)
    print(f"Verified 120-second film: {target}")


if __name__ == "__main__":
    _main()
