"""Run a portable, project-owned film renderer without loading cloud configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

_HELP = """Usage: npa studio [--registry studio.json] <film> <command> [options]

Commands: init, create, list, brief, scenes, narrate, draft, watch, preview, final.
Example: npa studio inference draft --scene 01-opening --open

Search accessible object storage using external NPA configuration:
  npa studio search --all-projects --query cosmos --kind video

Create a portable studio using the installed renderer:
  npa studio init --directory ./my-studio
  cd my-studio
  npa studio create demo --input-path ./my-clip.mp4 --duration 30

The registry selects a local renderer directory (default: renderer beside it).
Add film aliases under projects in studio.json. Each alias names a film-project.json.
Rendering needs FFmpeg, Pillow and NumPy; narration additionally needs edge-tts.
Credentials belong in external NPA configuration, never in a studio project.
"""

_LAUNCHER = '''#!/bin/sh
set -eu
STUDIO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${NPA_STUDIO_PYTHON:-}" ]; then
  exec "$NPA_STUDIO_PYTHON" -m npa studio --registry "$STUDIO_DIR/studio.json" "$@"
fi
if [ -x "$STUDIO_DIR/.venv/bin/python" ]; then
  exec "$STUDIO_DIR/.venv/bin/python" -m npa studio --registry "$STUDIO_DIR/studio.json" "$@"
fi
exec npa studio --registry "$STUDIO_DIR/studio.json" "$@"
'''


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a portable local film studio.")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, default=Path(__file__).with_name("studio_renderer"),
                        help="Trusted local renderer override; defaults to the installed renderer.")
    return parser


def _initialize(arguments: list[str]) -> int:
    args = _init_parser().parse_args(arguments)
    source = args.renderer.resolve()
    _validate_renderer(source)
    target = args.directory.resolve()
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "renderer"
    destination.mkdir()
    for path in sorted(source.glob("*.py")):
        shutil.copy2(path, destination / path.name)
    for name in ("brand", "fonts"):
        shutil.copytree(source / name, destination / name, ignore=shutil.ignore_patterns(".*", "__pycache__"))
    if (source / "requirements.txt").is_file():
        shutil.copy2(source / "requirements.txt", destination / "requirements.txt")
    registry = {"renderer": "renderer", "projects": {}}
    (target / "studio.json").write_text(json.dumps(registry, indent=2) + "\n")
    (target / "projects").mkdir()
    (target / "README.md").write_text(_HELP)
    (target / ".gitignore").write_text(".venv/\nprojects/\nstudio.json\n__pycache__/\n")
    (target / "studio").write_text(_LAUNCHER)
    (target / "studio").chmod(0o755)
    print(f"Created {target}. Run npa studio create inside it to add your own media.")
    return 0


def _validate_renderer(directory: Path) -> None:
    required = ("studio.py", "edit.py", "render.py", "film_draft.py", "film_watch.py")
    missing = [name for name in required if not (directory / name).is_file()]
    missing += [name for name in ("brand", "fonts") if not (directory / name).is_dir()]
    if missing:
        raise ValueError(f"Incomplete studio renderer: {', '.join(missing)}")


def _dispatch(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--registry", type=Path, default=Path("studio.json"))
    args, options = parser.parse_known_args(arguments)
    if options and options[0] == "search":
        from npa.studio_artifacts import run_search

        return run_search(options[1:])
    if options and options[0] == "create":
        from npa.studio_project import create_project

        return create_project(args.registry.resolve(), options[1:])
    registry = args.registry.resolve()
    data = json.loads(registry.read_text())
    if not isinstance(data, dict) or set(data) - {"renderer", "projects"}:
        raise ValueError("Studio registry supports only renderer and projects")
    renderer = data.get("renderer", "renderer")
    if not isinstance(renderer, str) or not renderer.strip():
        raise ValueError("Studio renderer must name a local directory")
    source = (registry.parent / renderer).resolve()
    _validate_renderer(source)
    command = [sys.executable, str(source / "studio.py"), "--registry", str(registry), *options]
    return subprocess.run(command, check=False).returncode


def run(arguments: list[str]) -> int:
    """Dispatch local studio initialization or an installed project's film commands.

    Args:
        arguments: Arguments following ``npa studio``.
    Returns:
        The renderer's exit code, 1 for configuration errors, or 130 for interruption.
    Raises:
        SystemExit: Argument parsing requests help or rejects invalid syntax.
    """
    if not arguments or arguments in (["--help"], ["-h"]):
        print(_HELP)
        return 0
    try:
        if arguments[0] == "init":
            return _initialize(arguments[1:])
        return _dispatch(arguments)
    except (OSError, ValueError, TypeError) as error:
        print(f"Studio error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
