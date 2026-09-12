"""Render an exact two-minute executive film from hash-verified local run artifacts."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from film_cache import (
    _build_cached,
    _cache_path,
    _cached,
    _fingerprint,
    _hash,
    _render_lock,
)
from film_player import _write_player
from film_profiles import (
    _PROFILES,
    _audio_inputs,
    _environment,
    _scaled_rectangle,
    _scene_inputs,
)
from film_voice import _verify_narration
from graphics import _draw_overlay, _draw_titles, _rectangles
from soundtrack import _duration, _mix

_ROOT = Path(__file__).parent


def _source_hashes():
    return {str(path.relative_to(_ROOT)): _hash(path)
            for path in [*_ROOT.glob("*.py"), _ROOT / "fonts/Manrope.ttf"]}


_LOADED_SOURCES = _source_hashes()


def _assert_sources_unchanged():
    if _source_hashes() != _LOADED_SOURCES:
        raise ValueError("Renderer source changed during this render; rerun after finishing code edits")


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


def _validate_storyboard(storyboard):
    identifiers = [scene["id"] for scene in storyboard["scenes"]]
    if len(set(identifiers)) != len(identifiers) or any(
        re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None for name in identifiers
    ):
        raise ValueError("Scene IDs must be unique lowercase names without path separators")
    if any(type(scene["duration"]) is not int or scene["duration"] <= 0
           for scene in storyboard["scenes"]):
        raise ValueError("Scene durations must be positive whole seconds")


def _validate(storyboard, assets, scene_id=None):
    if sum(scene["duration"] for scene in storyboard["scenes"]) != 120:
        raise ValueError("The full storyboard must last exactly 120 seconds")
    selected, _ = _selection(storyboard, scene_id)
    required = {role for _, scene in selected for role in scene["assets"]}
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


def _media_filter(index, asset, rectangle, duration, profile):
    x, y, width, height = rectangle
    label_height = round(56 * profile["width"] / 1920)
    content_height = height if width == profile["width"] else height - label_height
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


def _filter_graph(scene, assets, profile=None):
    profile = profile or _PROFILES["final"]
    width, height = profile["width"], profile["height"]
    duration = scene["duration"]
    graph = ["[0:v]format=yuv420p[base0]"]
    for index, (role, rectangle) in enumerate(zip(scene["assets"], _rectangles(scene), strict=True), 1):
        graph += _media_filter(index, assets[role], _scaled_rectangle(rectangle, profile), duration, profile)
    count = len(scene["assets"])
    graph.append(f"[{count + 1}:v]scale={width}:{height},format=rgba[graphic]")
    graph.append(f"[base{count}][graphic]overlay=0:0:shortest=1[framed]")
    graph.append(f"[{count + 2}:v]scale={width}:{height},format=rgba,fade=t=in:st=0.15:d=0.5:alpha=1[titles]")
    graph.append(f"[framed][titles]overlay=x='-{55 * width / 1920}*exp(-6*t)':y=0:shortest=1[composed]")
    bar_height = round(4 * width / 1920)
    graph.append(f"color=c=0xdcff46:s={width}x{bar_height}:r=30[bar]")
    graph.append(f"[composed][bar]overlay=x='-{width}+{width}*min(t/{duration},1)':y={height - bar_height}:shortest=1,"
                 f"fade=t=in:d=0.18,fade=t=out:st={duration - 0.18}:d=0.18,format=yuv420p[out]")
    return ";\n".join(graph)


def _render_scene(scene, index, total, assets, directory, profile=None):
    profile = profile or _PROFILES["final"]
    overlay = directory / f"{scene['id']}.png"
    target = directory / f"{scene['id']}.mp4"
    _draw_overlay(scene, index, total, overlay)
    titles = directory / f"{scene['id']}-titles.png"
    _draw_titles(scene, titles)
    graph = directory / f"{scene['id']}.ffmpeg"
    graph.write_text(_filter_graph(scene, assets, profile))
    command = ["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "2",
               "-f", "lavfi", "-i", f"color=c=0x080d12:s={profile['width']}x{profile['height']}:r=30"]
    for role in scene["assets"]:
        command += _input(assets[role])
    command += ["-loop", "1", "-framerate", "30", "-i", str(overlay),
                "-loop", "1", "-framerate", "30", "-i", str(titles),
                "-filter_complex_script", str(graph), "-map", "[out]", "-an",
                "-t", str(scene["duration"]), "-c:v", "libx264", "-preset", profile["preset"],
                "-crf", str(profile["crf"]), "-threads", "4", "-r", "30", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)
    print(f"Rendered {scene['id']}", flush=True)
    return target


def _join(parts, sound, directory, duration=120, offset=0, title="Intelligence that moves"):
    links = [directory / f"segment-{index:02d}.mp4" for index in range(len(parts))]
    for source, target in zip(parts, links, strict=True):
        os.link(source, target)
    listing = directory / "segments.txt"
    listing.write_text("".join(f"file {p.name}\n" for p in links))
    target = directory / "workbench-executive-film.mp4"
    audio_codec = ["-c:a", "copy"] if duration == 120 else ["-c:a", "aac", "-b:a", "256k"]
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(listing),
        "-ss", str(offset), "-i", str(sound), "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", *audio_codec, "-t", str(duration),
        "-metadata", f"title={title} | Nebius Physical AI Workbench",
        "-movflags", "+faststart", str(target),
    ], check=True)
    for link in links:
        link.unlink()
    listing.unlink()
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


def _manifest(target, storyboard, storyboard_path, assets, directory, probe):
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return {"title": storyboard["title"], "duration_seconds": float(probe["format"]["duration"]),
                "frames": int(video["nb_frames"]), "width": video["width"], "height": video["height"],
                "fps": video["r_frame_rate"], "sha256": _hash(target),
                "font_sha256": _hash(_ROOT / "fonts/Manrope.ttf"),
                "storyboard_sha256": _hash(storyboard_path),
                "selected_scenes": [scene["id"] for scene in storyboard["scenes"]],
                "renderer_sha256": {name: digest for name, digest in _LOADED_SOURCES.items() if name.endswith(".py")},
                "narration_sha256": _hash(directory / "narration.wav"),
                "score_sha256": _hash(directory / "original-score.wav"),
                "captions_sha256": _hash(directory / "workbench-executive-film.srt"),
                "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                "assets": {role: {"sha256": asset["sha256"], "kind": asset["kind"],
                                  "crop": asset.get("crop")} for role, asset in assets.items()},
                "editorial": "Separate saved run outputs; editorial assembly applies crops, loops and presentation zooms."}


def _evidence(target, storyboard, assets, directory, storyboard_path=None):
    probe = _probe(target)
    duration = float(probe["format"]["duration"])
    expected = sum(scene["duration"] for scene in storyboard["scenes"])
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    if not math.isclose(duration, expected, abs_tol=0.05) or int(video["nb_frames"]) != expected * 30:
        raise ValueError("Final duration or frame count does not match the storyboard")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"], check=True)
    manifest = _manifest(target, storyboard, storyboard_path or _ROOT / "storyboard.json", assets, directory, probe)
    (directory / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(min(3, expected / 2)), "-i", str(target),
                    "-frames:v", "1", str(directory / "poster.png")], check=True)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Private JSON mapping asset roles to local paths, kinds and SHA-256 digests.")
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storyboard", type=Path, default=_ROOT / "storyboard.json")
    parser.add_argument("--profile", choices=_PROFILES, default="final")
    parser.add_argument("--scene", help="Render one scene with its original audio position and captions.")
    parser.add_argument("--plan", action="store_true", help="Show which visual scenes need encoding without rendering.")
    parser.add_argument("--check", action="store_true", help="Verify all media without rendering.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not all(shutil.which(binary) for binary in ("ffmpeg", "ffprobe")):
        parser.error("Install FFmpeg with libx264 and ffprobe before rendering")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def _selection(storyboard, scene_id):
    indexed = list(enumerate(storyboard["scenes"]))
    if scene_id is None:
        return indexed, 0
    selected = [(index, scene) for index, scene in indexed if scene["id"] == scene_id]
    if not selected:
        raise ValueError(f"Unknown scene {scene_id}; use the scene ID from the storyboard")
    offset = sum(scene["duration"] for _, scene in indexed[:selected[0][0]])
    return selected, offset


def _visual_plan(args, storyboard, assets, environment):
    selected, _ = _selection(storyboard, args.scene)
    root = args.output_dir / ".cache"
    profile = _PROFILES[args.profile]
    return [{"scene": scene["id"], "cached": _cached(_cache_path(root, "scenes",
             _scene_inputs(scene, index, len(storyboard["scenes"]), assets, profile, environment)))}
            for index, scene in selected]


def _scene_job(args, scene, index, total, assets, environment):
    _assert_sources_unchanged()
    profile = _PROFILES[args.profile]
    inputs = _scene_inputs(scene, index, total, assets, profile, environment)
    directory, reused = _build_cached(args.output_dir / ".cache", "scenes", inputs,
        lambda staging: _verified_scene(scene, index, total, assets, staging, profile))
    if reused:
        print(f"Reused {scene['id']}", flush=True)
    return directory / f"{scene['id']}.mp4", reused


def _verified_scene(scene, index, total, assets, staging, profile):
    _render_scene(scene, index, total, assets, staging, profile)
    _assert_sources_unchanged()
    for role in scene["assets"]:
        if _hash(assets[role]["path"]) != assets[role]["sha256"]:
            raise ValueError(f"Source media changed while rendering {scene['id']}; rerun after the edit")


def _assemble(args, storyboard, assets, parts, audio, selected, offset, staging):
    _assert_sources_unchanged()
    scenes = [scene for _, scene in selected]
    duration = sum(scene["duration"] for scene in scenes)
    target = _join(parts, audio / "mix.m4a", staging, duration, offset, storyboard["title"])
    for name in ["narration.wav", "original-score.wav"]:
        os.link(audio / name, staging / name)
    _captions(scenes, args.voice_dir, staging)
    snapshot = {**storyboard, "scenes": scenes, "duration": duration}
    recipe = staging / "render-storyboard.json"
    recipe.write_text(json.dumps(snapshot, indent=2) + "\n")
    _evidence(target, snapshot, assets, staging, recipe)
    _write_player(staging)
    _assert_sources_unchanged()


def _assembly_inputs(args, parts, audio, offset, selected, environment, storyboard):
    return {"parts": [_hash(path) for path in parts], "sound": _hash(audio / "mix.m4a"),
            "captions": [_hash(args.voice_dir / f"{scene['id']}.srt") for _, scene in selected],
            "offset": offset, "storyboard": _fingerprint(storyboard), "environment": environment,
            "code": _LOADED_SOURCES}


def _publish(directory, destination):
    destination.mkdir(parents=True, exist_ok=True)
    names = ["workbench-executive-film.mp4", "workbench-executive-film.srt", "poster.png",
             "narration.wav", "original-score.wav", "render-storyboard.json", "render-manifest.json",
             "watch.html"]
    for name in names:
        source, target = directory / name, destination / name
        if target.is_file() and _hash(source) == _hash(target):
            continue
        temporary = destination / f".{name}.pending"
        shutil.copy2(source, temporary)
        temporary.replace(target)


def _render_film(args, storyboard, assets, environment):
    started = time.perf_counter()
    selected, offset = _selection(storyboard, args.scene)
    root = args.output_dir / ".cache"
    audio, audio_reused = _build_cached(root, "audio",
        _audio_inputs(storyboard["scenes"], args.voice_dir, environment),
        lambda staging: _mix(storyboard["scenes"], args.voice_dir, staging, root))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(_scene_job, args, scene, index, len(storyboard["scenes"]),
                               assets, environment) for index, scene in selected]
        results = [result.result() for result in pending]
    parts = [path for path, _ in results]
    assembled, assembly_reused = _build_cached(root, "assembly",
        _assembly_inputs(args, parts, audio, offset, selected, environment, storyboard),
        lambda staging: _assemble(args, storyboard, assets, parts, audio, selected, offset, staging))
    destination = args.output_dir / args.profile
    if args.scene:
        destination /= args.scene
    _assert_sources_unchanged()
    _publish(assembled, destination)
    report = {"profile": args.profile, "scenes_reused": sum(reused for _, reused in results),
              "scenes_rendered": sum(not reused for _, reused in results), "audio_reused": audio_reused,
              "assembly_reused": assembly_reused, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    (destination / "render-timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    print(f"Verified film: {destination / 'workbench-executive-film.mp4'}")


def _main():
    args = _arguments()
    storyboard = json.loads(args.storyboard.read_text())
    _validate_storyboard(storyboard)
    assets = _load_assets(args.assets)
    required = _validate(storyboard, assets, args.scene)
    assets = {role: assets[role] for role in sorted(required)}
    if args.check:
        print("Selected scene assets verified" if args.scene else "All storyboard assets verified")
        return
    environment = _environment()
    if args.plan:
        print(json.dumps(_visual_plan(args, storyboard, assets, environment), indent=2))
        return
    with _render_lock(args.voice_dir), _render_lock(args.output_dir):
        selected, _ = _selection(storyboard, args.scene)
        _verify_narration([scene for _, scene in selected], args.voice_dir)
        _render_film(args, storyboard, assets, environment)


if __name__ == "__main__":
    try:
        _main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Render stopped: {error}") from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Render command failed with exit code {error.returncode}; see its output above") from None
