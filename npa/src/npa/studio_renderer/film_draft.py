"""Render one verified scene without waiting for narration or assembling a full film."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import render
from edit import _project
from film_cache import _hash, _render_lock
from film_profiles import _environment


def _inputs(project, scene_id):
    storyboard = json.loads(project["storyboard"].read_text())
    render._validate_storyboard(storyboard)
    assets = render._load_assets(project["assets"])
    required = render._validate(storyboard, assets, scene_id)
    selected, _ = render._selection(storyboard, scene_id)
    index, scene = selected[0]
    return storyboard, index, scene, {role: assets[role] for role in sorted(required)}


def _verify_clip(clip, duration):
    probe = render._probe(clip)
    streams = probe["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    if any(stream["codec_type"] == "audio" for stream in streams):
        raise ValueError("Draft scene unexpectedly contains audio")
    if int(video["nb_frames"]) != duration * 30 or video["r_frame_rate"] != "30/1":
        raise ValueError("Draft frame count or frame rate does not match the scene")
    if (video["width"], video["height"]) != (960, 540):
        raise ValueError("Draft scene must use the 540p preview profile")


def _publish(clip, report, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".draft-", dir=destination) as temporary:
        staging = Path(temporary)
        shutil.copy2(clip, staging / "workbench-draft.mp4")
        (staging / "draft.json").write_text(json.dumps(report, indent=2) + "\n")
        for name in ("workbench-draft.mp4", "draft.json"):
            (staging / name).replace(destination / name)


def _draft(project_path, scene_id):
    started = time.perf_counter()
    project_digest = _hash(project_path)
    project = _project(project_path)
    identities = {str(path): _hash(path) for path in [project["storyboard"], project["assets"]]}
    identities[str(project_path)] = project_digest
    storyboard, index, scene, assets = _inputs(project, scene_id)
    args = SimpleNamespace(output_dir=project["output_dir"], profile="preview")
    with _render_lock(project["output_dir"]):
        clip, reused = render._scene_job(args, scene, index, len(storyboard["scenes"]), assets, _environment())
        _verify_clip(clip, scene["duration"])
        render._assert_sources_unchanged()
        if any(_hash(path) != digest for path, digest in identities.items()):
            raise ValueError("Project inputs changed during the draft; rerun after the edit")
        if any(_hash(asset["path"]) != asset["sha256"] for asset in assets.values()):
            raise ValueError("Source media changed during the draft; update its verified identity and rerun")
        report = {"kind": "muted-scene-draft", "scene": scene, "scene_reused": reused,
                  "audio": False, "sha256": _hash(clip), "input_sha256": identities,
                  "assets": assets, "renderer_sha256": render._LOADED_SOURCES,
                  "elapsed_seconds": round(time.perf_counter() - started, 3)}
        destination = project["output_dir"] / "draft" / scene_id
        _publish(clip, report, destination)
    print(json.dumps({"scene": scene_id, "reused": reused, "seconds": report["elapsed_seconds"],
                      "video": str(destination / "workbench-draft.mp4"), "audio": False}), flush=True)
    return destination / "workbench-draft.mp4"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--scene", required=True, help="Preview this scene; other narration and media may be absent.")
    parser.add_argument("--open", action="store_true", help="Open the completed muted scene.")
    return parser


def _main():
    parser = _parser()
    args = parser.parse_args()
    target = _draft(args.project.resolve(), args.scene)
    if args.open:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(target)], check=True)


if __name__ == "__main__":
    try:
        _main()
    except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Draft stopped: {error}") from None
