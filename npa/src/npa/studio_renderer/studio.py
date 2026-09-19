"""Select independent film projects and run the shared authoring and preview tools."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_COMMANDS = {"brief", "scenes", "narrate", "preview", "final", "draft", "watch"}


def _projects(path):
    projects = json.loads(path.read_text())["projects"]
    if not isinstance(projects, dict):
        raise ValueError("Studio projects must be a mapping of names to project files")
    resolved = {}
    for name, value in projects.items():
        if (
            name in {"list", "init", "search", "create"}
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None
        ):
            raise ValueError(
                "Film names must be lowercase names; list, init, search and create are reserved"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Film {name} requires a project file path")
        project = Path(value).expanduser()
        resolved[name] = (
            project.resolve()
            if project.is_absolute()
            else (path.parent / project).resolve()
        )
    return resolved


def _command(project, command, options):
    if command not in _COMMANDS:
        raise ValueError(f"Choose a studio command: {', '.join(sorted(_COMMANDS))}")
    if any(
        option == "--project" or option.startswith("--project=") for option in options
    ):
        raise ValueError(
            "Studio selects --project from the registry; choose the film name instead"
        )
    if command in {"draft", "watch"}:
        script = "film_draft.py" if command == "draft" else "film_watch.py"
        return [
            sys.executable,
            str(_ROOT / script),
            "--project",
            str(project),
            *options,
        ]
    return [
        sys.executable,
        str(_ROOT / "edit.py"),
        command,
        "--project",
        str(project),
        *options,
    ]


def _main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "film", nargs="?", help="Film name from the registry, or list to show projects."
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="brief, scenes, narrate, draft, watch, preview or final",
    )
    parser.add_argument("--registry", type=Path, default=Path("studio.json"))
    parser.add_argument(
        "--help",
        "-h",
        action="store_true",
        help="Show studio or selected command help.",
    )
    args, options = parser.parse_known_args()
    if args.help and not args.command:
        parser.print_help()
        return
    if args.help:
        options.append("--help")
    if not args.film:
        parser.error("Choose a film name, or list to show projects")
    projects = _projects(args.registry.resolve())
    if args.film == "list":
        if args.command or options:
            parser.error("list takes no film command or options")
        for name, path in projects.items():
            print(f"{name}\t{path}")
        return
    if args.film not in projects:
        parser.error(f"Unknown film {args.film}; choose {', '.join(projects)}")
    command = _command(projects[args.film], args.command, options)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    try:
        _main()
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise SystemExit(f"Studio error: {error}") from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
