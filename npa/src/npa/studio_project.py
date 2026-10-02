"""Create private, portable editing projects from a customer's selected local media."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEOS = {".mp4", ".mov", ".mkv", ".webm"}
_RESERVED = {"list", "init", "search", "create"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="New lowercase film name in studio.json.")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Local video or image to copy into the project.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Positive whole seconds; editable afterward.",
    )
    parser.add_argument(
        "--prompt", default="Explain this workflow and its observed result."
    )
    parser.add_argument("--audience", default="Your intended viewers")
    parser.add_argument(
        "--tool", help="Declared source tool; omitted attribution stays unknown."
    )
    return parser


def _arguments(arguments: list[str]) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(arguments)
    if (
        args.name in _RESERVED
        or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name) is None
    ):
        parser.error("Choose a lowercase film name; command names are reserved")
    if args.duration <= 0 or not args.prompt.strip() or not args.audience.strip():
        parser.error("Use a positive duration and nonempty prompt and audience")
    args.input_path = args.input_path.expanduser().resolve(strict=True)
    if (
        not args.input_path.is_file()
        or args.input_path.suffix.lower() not in _IMAGES | _VIDEOS
    ):
        parser.error("--input-path must be a supported local video or image")
    return args


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _source(directory: Path, args: argparse.Namespace) -> dict:
    media = directory / "media"
    media.mkdir()
    target = media / ("source" + args.input_path.suffix.lower())
    shutil.copyfile(args.input_path, target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {"status": "unattributed"}
    if args.tool:
        provenance = {"tool": args.tool, "status": "declared", "basis": "user_supplied"}
    return {
        "source": {
            "path": target.relative_to(directory).as_posix(),
            "kind": "image" if target.suffix in _IMAGES else "video",
            "sha256": digest.hexdigest(),
            "playback": "hold",
            "provenance": provenance,
        }
    }


def _storyboard(args: argparse.Namespace) -> dict:
    brief = {
        "prompt": args.prompt,
        "audience": args.audience,
        "goal": "Explain the workflow and its observed result",
        "tone": "Clear and concise",
        "call_to_action": "Choose your next step",
        "duration_seconds": args.duration,
    }
    scene = {
        "id": "opening",
        "duration": args.duration,
        "layout": "film",
        "title": [],
        "subtitle": "",
        "eyebrow": "",
        "assets": ["source"],
        "labels": [""],
        "narration": "Replace this text with your narration.",
        "transition": "cut",
    }
    return {
        "title": args.name,
        "duration": args.duration,
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "brief": brief,
        "scenes": [scene],
    }


def _scaffold(directory: Path, args: argparse.Namespace) -> None:
    _write_json(directory / "assets.json", _source(directory, args))
    _write_json(directory / "storyboard.json", _storyboard(args))
    _write_json(
        directory / "film-project.json",
        {
            "storyboard": "storyboard.json",
            "assets": "assets.json",
            "voice_dir": "narration",
            "output_dir": "renders",
        },
    )
    (directory / "README.md").write_text(
        "# Your film project\n\nEdit storyboard.json to author narration, shots and timing. "
        "The initial narration is a placeholder, not generated copy. assets.json binds your "
        "copied source to its hash; add source tool/model attribution and evidence there. "
        "A short source holds its last frame: adjust trims and scene durations as needed.\n\n"
        "Draft and watch are local and silent. narrate sends narration text to Microsoft's "
        "Edge speech service; use narrate --recorded to register your own MP3/SRT files offline. "
        "No cloud credentials or storage configuration belongs in this project.\n"
    )


def _publish(
    registry: Path, data: dict, target: Path, args: argparse.Namespace
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".studio-create-", dir=target.parent
    ) as temporary:
        staging = Path(temporary) / "project"
        staging.mkdir()
        _scaffold(staging, args)
        data["projects"][args.name] = (
            target.relative_to(registry.parent).as_posix() + "/film-project.json"
        )
        staged_registry = Path(temporary) / "registry.json"
        _write_json(staged_registry, data)
        staging.rename(target)
        try:
            staged_registry.replace(registry)
        except OSError:
            shutil.rmtree(target)
            raise


def create_project(registry: Path, arguments: list[str]) -> int:
    """Copy selected media and register a new portable editing project.

    Args:
        registry: Existing Studio registry path.
        arguments: Options following ``npa studio create``.
    Returns:
        Zero after the new project and registry are written.
    Raises:
        ValueError: Registry is invalid or the film already exists.
        OSError: Media or project files cannot be read or written.
        SystemExit: Arguments are invalid or help was requested.
    """
    args = _arguments(arguments)
    data = json.loads(registry.read_text())
    if not isinstance(data, dict) or set(data) - {"renderer", "projects"}:
        raise ValueError("Studio registry supports only renderer and projects")
    projects = data.get("projects")
    if not isinstance(projects, dict):
        raise ValueError("Studio projects must be a mapping")
    parent = registry.parent / "projects"
    target = parent / args.name
    if args.name in projects or target.exists():
        raise ValueError(f"Film {args.name} already exists; choose a new name")
    parent.mkdir(exist_ok=True)
    _publish(registry, data, target, args)
    print(
        f"Created {args.name}. Edit its storyboard, then run npa studio {args.name} draft --scene opening."
    )
    return 0
