"""Typer CLI for `npa workbench foxglove`.

Foxglove embedded-viewer tooling:

- ``convert-run``  pack a run's real frames/metrics/logs into an MCAP recording
- ``inspect``      read an MCAP back and report what the viewer will actually show
- ``install-sdk``  install the pinned, integrity-verified ``@foxglove/embed`` assets
- ``config``       show the resolved embed settings (what the agent will serve)
"""

from __future__ import annotations

import json
import os
import subprocess
import webbrowser
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from npa.workbench.foxglove import (
    DEFAULT_FOXGLOVE_EMBED_SRC,
    FOXGLOVE_EMBED_DEFAULT_REGISTRY,
    FOXGLOVE_EMBED_SDK_INTEGRITY,
    FOXGLOVE_EMBED_SDK_VERSION,
    FOXGLOVE_SERVICE_PORT,
    sdk_assets_present,
    sdk_tarball_url,
)

app = typer.Typer(
    name="foxglove",
    help="Foxglove embedded viewer: MCAP conversion, inspection, and SDK assets.",
    no_args_is_help=True,
)

INSTALL_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "docker"
    / "workbench"
    / "foxglove-embed"
    / "install-sdk.sh"
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _emit(payload: dict[str, Any], *, output: OutputFormat, text: str) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text)


@app.command("convert-run")
def convert_run_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="Directory of run artifacts (frames, metrics JSON, logs).",
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="Destination .mcap file."
    ),
    run_id: str = typer.Option(
        "", "--run-id", help="Run id recorded in MCAP metadata."
    ),
    fps: float = typer.Option(
        10.0,
        "--fps",
        help="Synthetic frame rate used for timestamps (run frames carry no capture time).",
    ),
    max_frames: int = typer.Option(
        0, "--max-frames", help="Cap the number of image frames (0 = all)."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Convert a run's artifacts into an MCAP recording the Foxglove viewer can open."""
    from npa.sdk.workbench.foxglove import convert_run

    try:
        summary = convert_run(
            input_path=input_path,
            output_path=output_path,
            fps=fps,
            max_frames=max_frames,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean operator message
        _fail(f"MCAP conversion failed: {exc}")
        return

    payload = summary.to_dict()
    lines = [
        f"wrote {payload['output']} ({payload['size_bytes']} bytes)",
        f"messages: {payload['message_count']} "
        f"(frames={payload['frames']}, metrics={payload['metrics']}, logs={payload['logs']})",
        f"timestamps: {payload['timestamps']} @ {payload['fps']} fps "
        "(run artifacts carry no capture time)",
    ]
    for topic in sorted(payload["channels"]):
        lines.append(f"  {topic}  x{payload['channels'][topic]}")
    if payload["skipped"]:
        lines.append(f"skipped {len(payload['skipped'])} non-convertible artifact(s)")
    _emit(payload, output=output, text="\n".join(lines))


@app.command("inspect")
def inspect_cmd(
    input_path: str = typer.Option(..., "--input-path", help="MCAP file to inspect."),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Report the channels, schemas, and message counts inside an MCAP recording."""
    from npa.sdk.workbench.foxglove import inspect_mcap

    try:
        info = inspect_mcap(input_path)
    except Exception as exc:  # noqa: BLE001
        _fail(f"MCAP inspect failed: {exc}")
        return

    from npa.workbench.foxglove.inspect import format_mcap_info

    _emit(info.to_dict(), output=output, text=format_mcap_info(info))


@app.command("export-run")
def export_run_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="Directory of run artifacts to pack into MCAP."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="Destination .mcap file."
    ),
    run_id: str = typer.Option(
        "", "--run-id", help="Run id recorded in MCAP metadata."
    ),
    fps: float = typer.Option(
        10.0, "--fps", help="Synthetic frame rate for timestamps."
    ),
    max_frames: int = typer.Option(0, "--max-frames", help="Frame cap (0 = all)."),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Export run artifacts to MCAP."""
    from npa.sdk.workbench.foxglove import export_run

    try:
        payload = export_run(
            input_path=input_path,
            output_path=output_path,
            fps=fps,
            max_frames=max_frames,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"MCAP export failed: {exc}")
        return
    summary = payload["summary"]
    text_lines = [f"wrote {summary['output']} ({summary['size_bytes']} bytes)"]
    _emit(payload, output=output, text="\n".join(text_lines))


@app.command("open")
def open_cmd(
    recording_id: str = typer.Option(
        ..., "--recording-id", help="Indexed Foxglove Cloud recording ID."
    ),
    launch: bool = typer.Option(
        False, "--launch/--no-launch", help="Open the link with the system URL handler."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Build (and optionally launch) an official Foxglove Web recording link."""
    from npa.sdk.workbench.foxglove import foxglove_recording_link

    payload = foxglove_recording_link(recording_id)
    if not payload["available"]:
        _fail(str(payload["reason"]))
    url = str(payload["web_url"])
    if launch and not webbrowser.open(url):
        _fail("Could not open the Foxglove Web link")
    _emit(payload, output=output, text=url)


@app.command("install-sdk")
def install_sdk_cmd(
    dest: str = typer.Option(
        ...,
        "--dest",
        help="Directory to install the @foxglove/embed browser assets into.",
    ),
    version: str = typer.Option(
        FOXGLOVE_EMBED_SDK_VERSION, "--version", help="Pinned @foxglove/embed release."
    ),
    integrity: str = typer.Option(
        FOXGLOVE_EMBED_SDK_INTEGRITY,
        "--integrity",
        help="npm dist.integrity digest verified after download.",
    ),
    registry: str = typer.Option(
        FOXGLOVE_EMBED_DEFAULT_REGISTRY,
        "--registry",
        help="npm registry (or mirror) base URL.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Install the pinned, sha512-verified Foxglove embed SDK assets."""
    if not INSTALL_SCRIPT.is_file():
        _fail(f"install script not found: {INSTALL_SCRIPT}")
    command = [
        "bash",
        str(INSTALL_SCRIPT),
        "--dest",
        dest,
        "--version",
        version,
        "--integrity",
        integrity,
        "--registry",
        registry,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    ready, reason = sdk_assets_present(dest)
    payload = {
        "ok": proc.returncode == 0 and ready,
        "dest": dest,
        "version": version,
        "tarball": sdk_tarball_url(version, registry=registry),
        "ready": ready,
        "reason": reason,
        "stderr": proc.stderr.strip()[-2000:],
    }
    if not payload["ok"]:
        typer.echo(proc.stdout.strip(), err=True)
        typer.echo(proc.stderr.strip(), err=True)
        _fail(f"Foxglove SDK install failed (exit={proc.returncode}). {reason}".strip())
    _emit(
        payload,
        output=output,
        text=f"installed @foxglove/embed@{version} into {dest}",
    )


@app.command("config")
def config_cmd(
    assets_dir: str = typer.Option(
        "", "--assets-dir", help="Installed SDK asset directory to probe (optional)."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Show the resolved Foxglove embed settings for this environment."""
    embed_src = (
        os.environ.get("NPA_FOXGLOVE_EMBED_SRC", "").strip()
        or DEFAULT_FOXGLOVE_EMBED_SRC
    )
    org_slug = os.environ.get("NPA_FOXGLOVE_ORG_SLUG", "").strip()
    live_url = os.environ.get("NPA_FOXGLOVE_LIVE_URL", "").strip()
    ready, reason = (
        sdk_assets_present(assets_dir)
        if assets_dir
        else (False, "no --assets-dir given")
    )
    payload = {
        "sdk_version": FOXGLOVE_EMBED_SDK_VERSION,
        "sdk_integrity": FOXGLOVE_EMBED_SDK_INTEGRITY,
        "embed_src": embed_src,
        "org_slug": org_slug,
        "live_url": live_url,
        "service_port": FOXGLOVE_SERVICE_PORT,
        "assets_dir": assets_dir,
        "assets_ready": ready,
        "assets_reason": reason,
        "note": (
            "NPA serves the MIT-licensed @foxglove/embed SDK and your recordings; "
            "the viewer application runs at embed_src (Foxglove-hosted or self-hosted)."
        ),
    }
    text = "\n".join(f"{key}: {value}" for key, value in payload.items())
    _emit(payload, output=output, text=text)
