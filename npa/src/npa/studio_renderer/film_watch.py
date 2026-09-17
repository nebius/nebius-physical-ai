"""Rebuild a selected muted scene after local edits settle, preserving the last good draft."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from edit import _project

_ROOT = Path(__file__).parent


def _watch_paths(project_path, scene_id, previous=()):
    paths = {project_path, *_ROOT.glob("*.py"), *_ROOT.glob("fonts/*"), *_ROOT.glob("brand/*")}
    try:
        project = _project(project_path)
        paths.update([project["storyboard"], project["assets"]])
        story = json.loads(project["storyboard"].read_text())
        assets = json.loads(project["assets"].read_text())
        scene = next((scene for scene in story["scenes"] if scene["id"] == scene_id), None)
        if scene:
            for role in scene["assets"]:
                path = Path(assets[role]["path"]).expanduser()
                paths.add(path if path.is_absolute() else project["assets"].parent / path)
    except (ValueError, TypeError, KeyError, OSError):
        paths.update(previous)
    return paths


def _signature(paths):
    entries = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            identity = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
        except FileNotFoundError:
            identity = None
        entries.append((str(path), identity))
    return tuple(entries)


class _Changes:
    def __init__(self, debounce):
        self.debounce = debounce
        self.observed = None
        self.changed_at = 0.0
        self.pending = False

    def ready(self, signature, now):
        if signature != self.observed:
            self.observed, self.changed_at, self.pending = signature, now, True
        if self.pending and now - self.changed_at >= self.debounce:
            self.pending = False
            return True
        return False


def _watch(project_path, scene_id, interval=0.25, debounce=0.5):
    changes = _Changes(debounce)
    paths = set()
    command = [sys.executable, str(_ROOT / "film_draft.py"),
               "--project", str(project_path), "--scene", scene_id]
    print(f"Watching {scene_id}. Muted previews only; Ctrl+C stops watching.", flush=True)
    while True:
        paths = _watch_paths(project_path, scene_id, paths)
        if changes.ready(_signature(paths), time.monotonic()):
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                print("Draft failed; last good output retained. Waiting for the next edit.", flush=True)
        time.sleep(interval)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    return parser


def _main():
    parser = _parser()
    args = parser.parse_args()
    try:
        _watch(args.project.resolve(), args.scene)
    except KeyboardInterrupt:
        print("Stopped watching.")


if __name__ == "__main__":
    _main()
