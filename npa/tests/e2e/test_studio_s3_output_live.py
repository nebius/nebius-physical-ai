"""Render synthetic footage to an explicitly configured S3 prefix without GPUs.

Set NPA_INTEGRATION_E2E=1, NPA_STUDIO_S3_PREFIX and optionally
NPA_STUDIO_STORAGE_PROJECT. Each run retains three small MP4s under a new UUID.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from urllib.parse import urlsplit
import uuid

import pytest

from npa import studio
from npa.clients.project_credentials import s3_client_for_project

pytestmark = pytest.mark.e2e


def _command(arguments):
    completed = subprocess.run(arguments, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _studio_command(registry, *arguments):
    return _command(
        [
            sys.executable,
            "-m",
            "npa",
            "studio",
            "--registry",
            str(registry),
            "demo",
            *arguments,
        ]
    )


def _synthetic_video(source):
    _command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30",
            "-t",
            "2",
            "-c:v",
            "libx264",
            str(source),
        ]
    )


def _recorded_narration(registry, project):
    narration = project / "narration"
    narration.mkdir()
    _command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "1",
            str(narration / "opening.mp3"),
        ]
    )
    (narration / "opening.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nSynthetic output test.\n"
    )
    _studio_command(registry, "narrate", "--recorded")


def _recorded_project(tmp_path):
    source = tmp_path / "source.mp4"
    _synthetic_video(source)
    workspace = tmp_path / "studio"
    assert studio.run(["init", "--directory", str(workspace)]) == 0
    registry = workspace / "studio.json"
    arguments = ["create", "demo", "--input-path", str(source), "--duration", "2"]
    assert studio.run(["--registry", str(registry), *arguments]) == 0
    project = workspace / "projects/demo"
    _recorded_narration(registry, project)
    return registry, project


def _records(stdout):
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


def _verify_delivery(stdout, local, client):
    result = next(record for record in _records(stdout) if record.get("verified"))
    target = urlsplit(result["output_path"])
    with client.get_object(Bucket=target.netloc, Key=target.path[1:])["Body"] as body:
        remote = body.read()
    assert remote == local.read_bytes()
    assert result["sha256"] == hashlib.sha256(remote).hexdigest()
    assert result["bytes"] == len(remote)
    _command(["ffmpeg", "-v", "error", "-i", str(local), "-f", "null", "-"])


def _final_and_cache_reuse(registry, project, destination, options, client):
    first = _studio_command(
        registry, "final", "--output-path", destination + "/film.mp4", *options
    )
    final = project / "renders/final/film.mp4"
    _verify_delivery(first, final, client)
    second = _studio_command(
        registry, "final", "--output-path", destination + "/reused.mp4", *options
    )
    _verify_delivery(second, final, client)
    timing = next(record for record in _records(second) if "scenes_reused" in record)
    assert timing["scenes_reused"] == 1 and timing["scenes_rendered"] == 0
    assert timing["audio_reused"] and timing["assembly_reused"]


def _preview_and_plan(registry, project, destination, options, client):
    preview = _studio_command(
        registry,
        "preview",
        "--scene",
        "opening",
        "--output-path",
        destination + "/preview.mp4",
        *options,
    )
    _verify_delivery(preview, project / "renders/preview/opening/film.mp4", client)
    plan = _studio_command(
        registry,
        "final",
        "--plan",
        "--output-path",
        destination + "/not-uploaded.mp4",
        "--storage-project",
        "unconfigured-test-alias",
    )
    assert json.loads(plan) == [{"scene": "opening", "cached": True}]


def test_studio_renders_and_reuses_cache_with_s3_output(tmp_path):
    prefix = os.environ.get("NPA_STUDIO_S3_PREFIX", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not prefix:
        pytest.skip("Set NPA_INTEGRATION_E2E=1 and NPA_STUDIO_S3_PREFIX")
    assert shutil.which("ffmpeg") and shutil.which("ffprobe")
    destination = prefix.rstrip("/") + "/" + uuid.uuid4().hex
    project_alias = os.environ.get("NPA_STUDIO_STORAGE_PROJECT")
    options = ["--storage-project", project_alias] if project_alias else []
    client = s3_client_for_project(project_alias)
    registry, project = _recorded_project(tmp_path)
    _final_and_cache_reuse(registry, project, destination, options, client)
    _preview_and_plan(registry, project, destination, options, client)
    (tmp_path / "verified-output.json").write_text(
        json.dumps(
            {
                "prefix": destination,
                "videos_verified": 3,
                "cache_reused": True,
                "network_free_plan": True,
            },
            indent=2,
        )
        + "\n"
    )
