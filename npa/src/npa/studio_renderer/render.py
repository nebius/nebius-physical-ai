"""Render a storyboard of any whole-second duration from verified local artifacts."""

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

from film_brief import _LAYOUTS, _validate_brief
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
    return {
        str(path.relative_to(_ROOT)): _hash(path)
        for path in [
            *_ROOT.glob("*.py"),
            _ROOT / "fonts/Manrope.ttf",
            _ROOT / "brand/nebius-logo.png",
            _ROOT / "brand/source.json",
        ]
    }


_LOADED_SOURCES = _source_hashes()


def _assert_sources_unchanged():
    if _source_hashes() != _LOADED_SOURCES:
        raise ValueError(
            "Renderer source changed during this render; rerun after finishing code edits"
        )


def _probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def _load_assets(path):
    assets = json.loads(path.read_text())
    for asset in assets.values():
        source = Path(asset["path"]).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        asset["path"] = str(source.resolve())
    return assets


def _validate_storyboard(storyboard):
    _validate_duration(storyboard)
    if type(storyboard.get("unique_sources", False)) is not bool:
        raise ValueError("Storyboard unique_sources must be a boolean")
    identifiers = [scene["id"] for scene in storyboard["scenes"]]
    if len(set(identifiers)) != len(identifiers) or any(
        re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None for name in identifiers
    ):
        raise ValueError(
            "Scene IDs must be unique lowercase names without path separators"
        )
    for scene in storyboard["scenes"]:
        if "layout" in scene:
            _validate_layout(scene)


def _validate_layout(scene):
    count = _LAYOUTS.get(scene["layout"])
    if count is None:
        raise ValueError(f"Unknown scene layout: {scene['layout']}")
    if len(scene["assets"]) != count or len(scene["labels"]) != count:
        raise ValueError(
            f"Scene {scene['id']} needs {count} asset roles and labels for {scene['layout']}"
        )
    if scene["layout"] == "immersive" and len(scene["title"]) > 2:
        raise ValueError("Immersive scenes support at most two headline lines")
    if scene["layout"] == "film":
        _validate_film_titles(scene)
    if scene["layout"] == "architecture":
        _validate_architecture(scene)
    for field, maximum in (
        ("source_notes", 2),
        ("review_steps", 3),
        ("pipeline_labels", 3),
    ):
        if len(scene.get(field, [])) > maximum:
            raise ValueError(f"Scene {scene['id']} supports at most {maximum} {field}")


def _validate_architecture(scene):
    nodes = scene.get("compute_nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 3:
        raise ValueError("Architecture needs one to three compute nodes")
    for node in nodes:
        if not isinstance(node, dict) or not all(
            isinstance(node.get(key), str) and node[key].strip()
            for key in ("title", "purpose")
        ):
            raise ValueError("Each compute node needs a title and purpose")
        models = node.get("models")
        if (
            not isinstance(models, list)
            or len(models) > 3
            or any(not isinstance(value, str) for value in models)
        ):
            raise ValueError("Compute nodes support up to three model labels")
    storage = scene.get("storage_node")
    if not isinstance(storage, dict) or not all(
        isinstance(storage.get(key), str) and storage[key].strip()
        for key in ("title", "detail")
    ):
        raise ValueError("Architecture needs a storage title and detail")
    for key in ("controller", "architecture_note"):
        if not isinstance(scene.get(key), str) or not scene[key].strip():
            raise ValueError(f"Architecture needs {key}")


def _validate_film_titles(scene):
    if len(scene["title"]) > 2:
        raise ValueError("Film scenes support at most two headline lines")
    if scene.get("title_position", "bottom-left") not in {"bottom-left", "center"}:
        raise ValueError("Film title_position must be bottom-left or center")
    if type(scene.get("brand", False)) is not bool:
        raise ValueError("Film brand must be a boolean")
    if scene.get("transition", "fade") not in {"fade", "cut"}:
        raise ValueError("Film transition must be fade or cut")
    delay = scene.get("title_delay_seconds", 0.5)
    if type(delay) not in (int, float) or not math.isfinite(delay):
        raise ValueError("Film title timing must use finite seconds")
    duration = scene.get("title_duration_seconds", scene["duration"] - delay)
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in (delay, duration)
    ):
        raise ValueError("Film title timing must use finite seconds")
    if delay < 0 or duration < 0.5 or delay + duration > scene["duration"]:
        raise ValueError(
            "Film titles must fit inside the scene with at least 0.5 seconds of display"
        )


def _validate_duration(storyboard):
    scenes = storyboard["scenes"]
    if not scenes:
        raise ValueError("The storyboard must contain at least one scene")
    if any(
        type(scene["duration"]) is not int or scene["duration"] <= 0 for scene in scenes
    ):
        raise ValueError("Scene durations must be positive whole seconds")
    duration = sum(scene["duration"] for scene in scenes)
    declared = storyboard.get("duration", duration)
    if type(declared) is not int or declared != duration:
        raise ValueError("Storyboard duration must equal the sum of scene durations")
    if "brief" in storyboard:
        _validate_brief(storyboard["brief"])
        if storyboard["brief"]["duration_seconds"] != duration:
            raise ValueError(
                "Scene durations must match the creative brief duration_seconds"
            )


def _validate(storyboard, assets, scene_id=None):
    _validate_duration(storyboard)
    selected, _ = _selection(storyboard, scene_id)
    required = {role for _, scene in selected for role in scene["assets"]}
    if required - assets.keys():
        raise ValueError(f"Missing asset roles: {sorted(required - assets.keys())}")
    _validate_unique_sources(storyboard, assets, selected)
    for role in sorted(required):
        asset = assets[role]
        path = Path(asset["path"]).expanduser().resolve(strict=True)
        if _hash(path) != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch for asset {role}")
        if asset["kind"] not in {"image", "video"}:
            raise ValueError(f"Unsupported asset kind for {role}")
        if asset.get("playback", "loop") not in ("loop", "hold"):
            raise ValueError(f"Unsupported playback mode for {role}")
        if asset.get("fit", "contain") not in ("contain", "cover"):
            raise ValueError(f"Unsupported fit mode for {role}")
        crop = asset.get("crop")
        if crop and (
            len(crop) != 4
            or any(type(v) is not int or v < 0 for v in crop)
            or crop[2] == 0
            or crop[3] == 0
        ):
            raise ValueError(f"Invalid pixel crop for {role}")
        probe = _probe(path)
        streams = probe["streams"]
        visual = next((s for s in streams if s["codec_type"] == "video"), None)
        if visual is None:
            raise ValueError(f"No decodable visual stream for {role}")
        if crop and (
            crop[0] + crop[2] > visual["width"] or crop[1] + crop[3] > visual["height"]
        ):
            raise ValueError(f"Crop exceeds source bounds for {role}")
        _validate_trim(asset, probe, role)
        asset["path"] = str(path)
    return required


def _validate_unique_sources(storyboard, assets, selected):
    enabled = storyboard.get("unique_sources", False)
    if type(enabled) is not bool:
        raise ValueError("Storyboard unique_sources must be a boolean")
    if not enabled:
        return
    seen = {}
    for _, scene in selected:
        for role in scene["assets"]:
            digest = assets[role]["sha256"]
            if digest in seen:
                previous_scene, previous_role = seen[digest]
                raise ValueError(
                    f"Repeated source media: {previous_scene}/{previous_role} and "
                    f"{scene.get('id', 'unnamed')}/{role}. unique_sources requires "
                    "different source bytes, including when trims or filenames differ."
                )
            seen[digest] = (scene.get("id", "unnamed"), role)


def _validate_trim(asset, probe, role):
    if "trim" not in asset:
        return
    trim = asset["trim"]
    if (
        asset["kind"] != "video"
        or not isinstance(trim, list)
        or len(trim) != 2
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in trim
        )
        or not 0 <= trim[0] < trim[1]
    ):
        raise ValueError(
            f"Invalid trim for {role}: use [start_seconds, end_seconds] on a video"
        )
    if asset.get("playback", "hold") != "hold":
        raise ValueError(f"Trimmed video {role} requires hold playback")
    duration = float(probe.get("format", {}).get("duration", "nan"))
    if not math.isfinite(duration) or trim[1] > duration:
        raise ValueError(f"Trim exceeds known source duration for {role}")


def _input(asset):
    if asset["kind"] == "image":
        return ["-loop", "1", "-framerate", "30", "-i", asset["path"]]
    if "trim" in asset:
        start, end = asset["trim"]
        return ["-ss", str(start), "-t", str(end - start), "-i", asset["path"]]
    if asset.get("playback") == "hold":
        return ["-i", asset["path"]]
    return ["-stream_loop", "-1", "-i", asset["path"]]


def _media_filter(index, asset, rectangle, duration, profile):
    x, y, width, height = rectangle
    label_height = round(56 * profile["width"] / 1920)
    content_height = height if width == profile["width"] else height - label_height
    filters = ["setpts=PTS-STARTPTS", "fps=30"]
    if asset.get("crop"):
        crop_x, crop_y, crop_width, crop_height = asset["crop"]
        filters.append(f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}")
    if asset.get("fit", "contain") == "cover":
        filters += [
            f"scale={width}:{content_height}:force_original_aspect_ratio=increase:force_divisible_by=2",
            f"crop={width}:{content_height}",
        ]
    else:
        filters.append(
            f"scale={width}:{content_height}:force_original_aspect_ratio=decrease:force_divisible_by=2"
        )
    filters += [
        f"pad={width}:{height}:(ow-iw)/2:({content_height}-ih)/2:color=0x080d12",
        "setsar=1",
    ]
    if asset["kind"] == "image":
        filters.append(
            f"zoompan=z='1+0.018*on/{30 * duration}':"
            f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={width}x{height}:fps=30"
        )
    elif asset.get("playback") == "hold" or "trim" in asset:
        filters.append(f"tpad=stop_mode=clone:stop_duration={duration}")
    filters += [f"trim=duration={duration}", "format=yuv420p"]
    layer = f"[{index}:v]{','.join(filters)}[media{index}]"
    overlay = f"[base{index - 1}][media{index}]overlay={x}:{y}:shortest=1[base{index}]"
    return [layer, overlay]


def _filter_graph(scene, assets, profile=None):
    profile = profile or _PROFILES["final"]
    width, height = profile["width"], profile["height"]
    duration = scene["duration"]
    graph = ["[0:v]format=yuv420p[base0]"]
    for index, (role, rectangle) in enumerate(
        zip(scene["assets"], _rectangles(scene), strict=True), 1
    ):
        graph += _media_filter(
            index,
            assets[role],
            _scaled_rectangle(rectangle, profile),
            duration,
            profile,
        )
    count = len(scene["assets"])
    graph.append(f"[{count + 1}:v]scale={width}:{height},format=rgba[graphic]")
    graph.append(f"[base{count}][graphic]overlay=0:0:shortest=1[framed]")
    if scene["layout"] == "film":
        return _film_filter_graph(graph, scene, count, width, height)
    graph.append(
        f"[{count + 2}:v]scale={width}:{height},format=rgba,fade=t=in:st=0.15:d=0.5:alpha=1[titles]"
    )
    graph.append(
        f"[framed][titles]overlay=x='-{55 * width / 1920}*exp(-6*t)':y=0:shortest=1[composed]"
    )
    bar_height = round(4 * width / 1920)
    graph.append(f"color=c=0xdcff46:s={width}x{bar_height}:r=30[bar]")
    graph.append(
        f"[composed][bar]overlay=x='-{width}+{width}*min(t/{duration},1)':y={height - bar_height}:shortest=1,"
        f"fade=t=in:d=0.18,fade=t=out:st={duration - 0.18}:d=0.18,format=yuv420p[out]"
    )
    return ";\n".join(graph)


def _film_filter_graph(graph, scene, count, width, height):
    delay = scene.get("title_delay_seconds", 0.5)
    duration = scene.get("title_duration_seconds", scene["duration"] - delay)
    fade = min(0.6, duration / 2)
    graph.append(
        f"[{count + 2}:v]scale={width}:{height},format=rgba,"
        f"fade=t=in:st={delay}:d={fade}:alpha=1,"
        f"fade=t=out:st={delay + duration - fade}:d={fade}:alpha=1[titles]"
    )
    transition = ""
    if scene.get("transition", "fade") == "fade":
        transition = f"fade=t=in:d=0.3,fade=t=out:st={scene['duration'] - 0.3}:d=0.3,"
    graph.append(
        f"[framed][titles]overlay=0:0:shortest=1,{transition}format=yuv420p[out]"
    )
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
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-filter_complex_threads",
        "2",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x080d12:s={profile['width']}x{profile['height']}:r=30",
    ]
    for role in scene["assets"]:
        command += _input(assets[role])
    command += [
        "-loop",
        "1",
        "-framerate",
        "30",
        "-i",
        str(overlay),
        "-loop",
        "1",
        "-framerate",
        "30",
        "-i",
        str(titles),
        "-filter_complex_script",
        str(graph),
        "-map",
        "[out]",
        "-an",
        "-t",
        str(scene["duration"]),
        "-c:v",
        "libx264",
        "-preset",
        profile["preset"],
        "-crf",
        str(profile["crf"]),
        "-threads",
        "4",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True)
    print(f"Rendered {scene['id']}", flush=True)
    return target


def _join(parts, sound, directory, duration, offset=0, title="Workbench film"):
    links = [directory / f"segment-{index:02d}.mp4" for index in range(len(parts))]
    for source, target in zip(parts, links, strict=True):
        os.link(source, target)
    listing = directory / "segments.txt"
    listing.write_text("".join(f"file {p.name}\n" for p in links))
    target = directory / "film.mp4"
    complete_audio = offset == 0 and math.isclose(
        duration, _duration(sound), abs_tol=0.05
    )
    audio_codec = (
        ["-c:a", "copy"] if complete_audio else ["-c:a", "aac", "-b:a", "256k"]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(listing),
            "-ss",
            str(offset),
            "-i",
            str(sound),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            *audio_codec,
            "-t",
            str(duration),
            "-metadata",
            f"title={title} | Nebius Physical AI Workbench",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )
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
            end = min(
                elapsed + scene["duration"], elapsed + 0.35 + _seconds(end) / speed
            )
            captions.append(
                f"{len(captions) + 1}\n{_timestamp(start)} --> {_timestamp(end)}\n"
                + "\n".join(lines[2:])
                + "\n"
            )
        elapsed += scene["duration"]
    (directory / "film.srt").write_text("\n".join(captions))


def _manifest(target, storyboard, storyboard_path, assets, directory, probe):
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    brief = storyboard.get("brief", storyboard.get("source_brief"))
    return {
        "title": storyboard["title"],
        "duration_seconds": float(probe["format"]["duration"]),
        "frames": int(video["nb_frames"]),
        "width": video["width"],
        "height": video["height"],
        "fps": video["r_frame_rate"],
        "sha256": _hash(target),
        "font_sha256": _hash(_ROOT / "fonts/Manrope.ttf"),
        "branding": {
            "logo_sha256": _hash(_ROOT / "brand/nebius-logo.png"),
            "source_manifest_sha256": _hash(_ROOT / "brand/source.json"),
            "source_page": "https://nebius.com/media-kit",
        },
        "storyboard_sha256": _hash(storyboard_path),
        "creative_brief": brief,
        "creative_brief_sha256": _fingerprint(brief) if brief else None,
        "selected_scenes": [scene["id"] for scene in storyboard["scenes"]],
        "renderer_sha256": {
            name: digest
            for name, digest in _LOADED_SOURCES.items()
            if name.endswith(".py")
        },
        "narration_sha256": _hash(directory / "narration.wav"),
        "score_sha256": _hash(directory / "original-score.wav"),
        "captions_sha256": _hash(directory / "film.srt"),
        "ffmpeg_version": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "assets": {
            role: {
                "sha256": asset["sha256"],
                "kind": asset["kind"],
                "crop": asset.get("crop"),
                "playback": asset.get("playback", "loop")
                if asset["kind"] == "video"
                else None,
                "provenance": asset.get("provenance"),
            }
            for role, asset in assets.items()
        },
        "editorial": "Separate saved run outputs; editorial assembly applies crops, configured loops or final-frame holds, and presentation zooms.",
    }


def _evidence(target, storyboard, assets, directory, storyboard_path=None):
    probe = _probe(target)
    duration = float(probe["format"]["duration"])
    expected = sum(scene["duration"] for scene in storyboard["scenes"])
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    if (
        not math.isclose(duration, expected, abs_tol=0.05)
        or int(video["nb_frames"]) != expected * 30
    ):
        raise ValueError("Final duration or frame count does not match the storyboard")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"],
        check=True,
    )
    manifest = _manifest(
        target,
        storyboard,
        storyboard_path or _ROOT / "storyboard.json",
        assets,
        directory,
        probe,
    )
    (directory / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(min(3, expected / 2)),
            "-i",
            str(target),
            "-frames:v",
            "1",
            str(directory / "poster.png"),
        ],
        check=True,
    )


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Private JSON mapping asset roles to local paths, kinds and SHA-256 digests.",
    )
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument(
        "--music-path",
        type=Path,
        help="Optional local score covering the complete film; defaults to the synthesized bed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storyboard", type=Path, default=_ROOT / "storyboard.json")
    parser.add_argument("--profile", choices=_PROFILES, default="final")
    parser.add_argument(
        "--scene",
        help="Render one scene with its original audio position and captions.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show which visual scenes need encoding without rendering.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify all media without rendering."
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not all(shutil.which(binary) for binary in ("ffmpeg", "ffprobe")):
        parser.error("Install FFmpeg with libx264 and ffprobe before rendering")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def _selection(storyboard, scene_id):
    footer = storyboard.get("footer", storyboard.get("title", "PHYSICAL AI WORKBENCH"))
    indexed = list(
        enumerate({"footer": footer, **scene} for scene in storyboard["scenes"])
    )
    if scene_id is None:
        return indexed, 0
    selected = [(index, scene) for index, scene in indexed if scene["id"] == scene_id]
    if not selected:
        raise ValueError(
            f"Unknown scene {scene_id}; use the scene ID from the storyboard"
        )
    offset = sum(scene["duration"] for _, scene in indexed[: selected[0][0]])
    return selected, offset


def _visual_plan(args, storyboard, assets, environment):
    selected, _ = _selection(storyboard, args.scene)
    root = args.output_dir / ".cache"
    profile = _PROFILES[args.profile]
    return [
        {
            "scene": scene["id"],
            "cached": _cached(
                _cache_path(
                    root,
                    "scenes",
                    _scene_inputs(
                        scene,
                        index,
                        len(storyboard["scenes"]),
                        assets,
                        profile,
                        environment,
                    ),
                )
            ),
        }
        for index, scene in selected
    ]


def _scene_job(args, scene, index, total, assets, environment):
    _assert_sources_unchanged()
    profile = _PROFILES[args.profile]
    inputs = _scene_inputs(scene, index, total, assets, profile, environment)
    directory, reused = _build_cached(
        args.output_dir / ".cache",
        "scenes",
        inputs,
        lambda staging: _verified_scene(scene, index, total, assets, staging, profile),
    )
    if reused:
        print(f"Reused {scene['id']}", flush=True)
    return directory / f"{scene['id']}.mp4", reused


def _verified_scene(scene, index, total, assets, staging, profile):
    _render_scene(scene, index, total, assets, staging, profile)
    _assert_sources_unchanged()
    for role in scene["assets"]:
        if _hash(assets[role]["path"]) != assets[role]["sha256"]:
            raise ValueError(
                f"Source media changed while rendering {scene['id']}; rerun after the edit"
            )


def _assemble(args, storyboard, assets, parts, audio, selected, offset, staging):
    _assert_sources_unchanged()
    scenes = [scene for _, scene in selected]
    duration = sum(scene["duration"] for scene in scenes)
    target = _join(
        parts, audio / "mix.m4a", staging, duration, offset, storyboard["title"]
    )
    for name in ["narration.wav", "original-score.wav"]:
        os.link(audio / name, staging / name)
    _captions(scenes, args.voice_dir, staging)
    snapshot = {**storyboard, "scenes": scenes, "duration": duration}
    if "brief" in snapshot and duration != snapshot["brief"]["duration_seconds"]:
        snapshot["source_brief"] = snapshot.pop("brief")
    recipe = staging / "render-storyboard.json"
    recipe.write_text(json.dumps(snapshot, indent=2) + "\n")
    _evidence(target, snapshot, assets, staging, recipe)
    _write_player(staging)
    _assert_sources_unchanged()


def _assembly_inputs(
    args, parts, audio, offset, selected, environment, storyboard, assets
):
    return {
        "parts": [_hash(path) for path in parts],
        "sound": _hash(audio / "mix.m4a"),
        "captions": [
            _hash(args.voice_dir / f"{scene['id']}.srt") for _, scene in selected
        ],
        "asset_provenance": {
            role: asset.get("provenance") for role, asset in assets.items()
        },
        "offset": offset,
        "storyboard": _fingerprint(storyboard),
        "environment": environment,
        "code": _LOADED_SOURCES,
    }


def _publish(directory, destination):
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "film.mp4",
        "film.srt",
        "poster.png",
        "narration.wav",
        "original-score.wav",
        "render-storyboard.json",
        "render-manifest.json",
        "watch.html",
    ]
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
    music = getattr(args, "music_path", None)
    music_options = {"music_path": music} if music else {}
    audio, audio_reused = _build_cached(
        root,
        "audio",
        _audio_inputs(storyboard["scenes"], args.voice_dir, environment, music),
        lambda staging: _mix(
            storyboard["scenes"], args.voice_dir, staging, root, **music_options
        ),
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [
            pool.submit(
                _scene_job,
                args,
                scene,
                index,
                len(storyboard["scenes"]),
                assets,
                environment,
            )
            for index, scene in selected
        ]
        results = [result.result() for result in pending]
    parts = [path for path, _ in results]
    assembled, assembly_reused = _build_cached(
        root,
        "assembly",
        _assembly_inputs(
            args, parts, audio, offset, selected, environment, storyboard, assets
        ),
        lambda staging: _assemble(
            args, storyboard, assets, parts, audio, selected, offset, staging
        ),
    )
    destination = args.output_dir / args.profile
    if args.scene:
        destination /= args.scene
    _assert_sources_unchanged()
    _publish(assembled, destination)
    report = {
        "profile": args.profile,
        "scenes_reused": sum(reused for _, reused in results),
        "scenes_rendered": sum(not reused for _, reused in results),
        "audio_reused": audio_reused,
        "assembly_reused": assembly_reused,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (destination / "render-timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    print(f"Verified film: {destination / 'film.mp4'}")


def _main():
    args = _arguments()
    storyboard = json.loads(args.storyboard.read_text())
    _validate_storyboard(storyboard)
    assets = _load_assets(args.assets)
    required = _validate(storyboard, assets, args.scene)
    assets = {role: assets[role] for role in sorted(required)}
    if args.check:
        print(
            "Selected scene assets verified"
            if args.scene
            else "All storyboard assets verified"
        )
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
    except (ValueError, TypeError, FileNotFoundError) as error:
        raise SystemExit(f"Render stopped: {error}") from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Render command failed with exit code {error.returncode}; see its output above"
        ) from None
