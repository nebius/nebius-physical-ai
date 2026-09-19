"""Prepare creative briefs and run cached film edits from one private project."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from film_brief import _authoring_packet

_ROOT = Path(__file__).parent


def _project(path):
    project = json.loads(path.read_text())
    allowed = {"storyboard", "assets", "voice_dir", "output_dir", "voice", "music_path"}
    if not isinstance(project, dict) or set(project) - allowed:
        raise ValueError(
            "Film projects support only local editing paths and voice; keep cloud configuration external"
        )
    for field in [
        "storyboard",
        "assets",
        "voice_dir",
        "output_dir",
        *(["music_path"] if "music_path" in project else []),
    ]:
        value = Path(project[field]).expanduser()
        project[field] = (
            value.resolve() if value.is_absolute() else (path.parent / value).resolve()
        )
    return project


def _render_command(args, project):
    command = [
        sys.executable,
        str(_ROOT / "render.py"),
        "--profile",
        args.command,
        "--storyboard",
        str(project["storyboard"]),
        "--assets",
        str(project["assets"]),
        "--voice-dir",
        str(project["voice_dir"]),
        "--output-dir",
        str(project["output_dir"]),
        "--workers",
        str(args.workers),
    ]
    if args.scene:
        command += ["--scene", args.scene]
    if project.get("music_path"):
        command += ["--music-path", str(project["music_path"])]
    if args.plan:
        command.append("--plan")
    return command


def _narrate_command(args, project):
    command = [
        sys.executable,
        str(_ROOT / "narrate.py"),
        "--storyboard",
        str(project["storyboard"]),
        "--output-dir",
        str(project["voice_dir"]),
        "--voice",
        project.get("voice", "en-US-AndrewMultilingualNeural"),
    ]
    if args.recorded:
        command.append("--recorded")
    if args.force:
        command.append("--force")
    return command


def _scenes(project):
    elapsed = 0
    for scene in json.loads(project["storyboard"].read_text())["scenes"]:
        print(
            f"{elapsed // 60}:{elapsed % 60:02d}  {scene['id']:<18} {' '.join(scene['title'])}"
        )
        elapsed += scene["duration"]


def _open_video(args, project):
    destination = project["output_dir"] / args.command
    if args.scene:
        destination /= args.scene
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.run([opener, str(destination / "film.mp4")], check=True)


def _brief_command(args, project):
    storyboard = json.loads(project["storyboard"].read_text())
    assets = json.loads(project["assets"].read_text())
    overrides = {
        "prompt": args.prompt,
        "audience": args.audience,
        "goal": args.goal,
        "tone": args.tone,
        "call_to_action": args.cta,
        "duration_seconds": args.duration,
    }
    print(json.dumps(_authoring_packet(storyboard, assets, overrides), indent=2))


def _brief_arguments(parser):
    group = parser.add_argument_group("creative brief (with brief only)")
    group.add_argument(
        "--prompt", help="Free-form creative request for an editor or coding agent."
    )
    group.add_argument(
        "--audience", help="Intended viewers; no fixed audience presets."
    )
    group.add_argument("--goal", help="What viewers should understand or do.")
    group.add_argument("--tone", help="Voice, depth and pacing guidance.")
    group.add_argument("--cta", help="Requested closing action or destination.")
    group.add_argument(
        "--duration",
        type=int,
        help="Target whole seconds; defaults to the current timeline.",
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["brief", "preview", "final", "narrate", "scenes"]
    )
    parser.add_argument("--project", type=Path, default=Path("film-project.json"))
    parser.add_argument("--scene", help="Preview or render only this storyboard scene.")
    parser.add_argument(
        "--plan", action="store_true", help="List visual cache hits without rendering."
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the completed video in the default player.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--recorded",
        action="store_true",
        help="With narrate, register supplied recordings offline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With narrate, regenerate every speech clip.",
    )
    _brief_arguments(parser)
    return parser


def _main():
    parser = _parser()
    args = parser.parse_args()
    if args.command != "brief" and any(
        getattr(args, field) is not None
        for field in ("prompt", "audience", "goal", "tone", "cta", "duration")
    ):
        parser.error(
            "Creative options apply to brief; author its storyboard before rendering"
        )
    project = _project(args.project.resolve())
    if args.command == "brief":
        _brief_command(args, project)
        return
    if args.command == "scenes":
        _scenes(project)
        return
    command = (
        _narrate_command(args, project)
        if args.command == "narrate"
        else _render_command(args, project)
    )
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if args.open and not args.plan and args.command in {"preview", "final"}:
        _open_video(args, project)


if __name__ == "__main__":
    try:
        _main()
    except (ValueError, TypeError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"Film project error: {error}") from None
