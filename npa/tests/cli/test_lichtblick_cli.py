"""CLI + module tests for the Lichtblick (Foxglove-compatible OSS) workbench viewer.

Infra-free: no Docker, S3, or network calls. Image resolution is pinned via
NPA_REGISTRY so the tests never touch the real registry.
"""

from __future__ import annotations

import json

import pytest
import yaml
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.main import app
from npa.deploy.images import CONTAINER_IMAGE_NAMES, SUPPORTED_TOOL_VERSIONS
from npa.workbench.lichtblick import (
    DEFAULT_PORT,
    LichtblickError,
    LichtblickLaunchPlan,
    build_launch_plan,
    build_mcap_from_frames,
    launch_viewer,
    serve_viewer,
    stage_input_to_mcap,
)

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE_PATH = REPO_ROOT / "npa" / "docker" / "workbench" / "lichtblick" / "Dockerfile"
PACKAGING_CONTRACT = REPO_ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"


@pytest.fixture(autouse=True)
def _pin_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "cr.example/reg")


def test_lichtblick_is_registered_everywhere() -> None:
    assert CONTAINER_IMAGE_NAMES["lichtblick"] == "npa-lichtblick"
    assert SUPPORTED_TOOL_VERSIONS["lichtblick"] == "1.26.0"
    # The old, incorrectly-named key must not exist.
    assert "foxglove" not in CONTAINER_IMAGE_NAMES


def test_dockerfile_exists_and_is_non_root_service() -> None:
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert 'LABEL npa.tool="lichtblick"' in text
    assert "EXPOSE 8080" in text
    assert "USER nobody" in text
    assert "HEALTHCHECK" in text
    # Digest-pinned bases (fiftyone header convention).
    assert "node:22-bookworm@sha256:" in text
    assert "caddy:2.11.4-alpine@sha256:" in text
    # OSS source is commit-pinned (immutable), not just tag-pinned.
    assert "ARG LICHTBLICK_COMMIT=" in text
    assert 'git fetch --depth 1 origin "${LICHTBLICK_COMMIT}"' in text
    assert 'if [ "${head}" != "${LICHTBLICK_COMMIT}" ]' in text
    # Caddy's XDG dirs must be writable by the non-root runtime user.
    assert "chown -R 65534:65534" in text


def test_packaging_contract_entry() -> None:
    contract = yaml.safe_load(PACKAGING_CONTRACT.read_text(encoding="utf-8"))
    assert "foxglove" not in contract["images"]
    entry = contract["images"]["lichtblick"]
    assert entry["tier"] == "service"
    assert entry["ports"] == [8080]
    assert entry["final_user"] == "nobody"


def test_lichtblick_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])
    assert result.exit_code == 0
    assert "lichtblick" in result.output


def test_lichtblick_command_help() -> None:
    for command in ("serve", "launch", "status", "list"):
        result = runner.invoke(app, ["workbench", "lichtblick", command, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output


def test_serve_plans_viewer_for_s3_mcap() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "lichtblick",
            "serve",
            "--input-path",
            "s3://bucket/run42/recording.mcap",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "planned"
    assert payload["artifact_name"] == "recording.mcap"
    assert payload["image"] == "cr.example/reg/npa-lichtblick:1.26.0"
    assert payload["port"] == DEFAULT_PORT
    assert payload["served_artifact_path"] == "/srv/data/recording.mcap"
    assert "ds=remote-file" in payload["viewer_url"]
    # The deep link targets the app root `/` with the data source as a query
    # string (never a client-routed sub-path), so caddy file-server needs no SPA
    # fallback: `GET /` always serves index.html. A wildcard bind (0.0.0.0) is
    # rewritten to a navigable loopback connect host in the URL.
    assert payload["viewer_url"].startswith("http://127.0.0.1:8080/?")
    # The co-served artifact URL shares the viewer origin -> no CORS / no
    # mixed-content / no signed URL needed for the browser to fetch the MCAP.
    assert "ds.url=http%3A%2F%2F127.0.0.1%3A8080%2Fdata%2Frecording.mcap" in payload["viewer_url"]


def test_serve_rejects_unsupported_artifact() -> None:
    result = runner.invoke(
        app,
        ["workbench", "lichtblick", "serve", "--input-path", "s3://bucket/run/notes.txt"],
    )
    assert result.exit_code == 1
    assert "unsupported artifact" in result.output.lower()


def test_serve_rejects_non_s3_scheme() -> None:
    result = runner.invoke(
        app,
        ["workbench", "lichtblick", "serve", "--input-path", "gs://bucket/x.mcap"],
    )
    assert result.exit_code == 1


def test_build_launch_plan_requires_input() -> None:
    with pytest.raises(LichtblickError):
        build_launch_plan(input_path="")


def test_build_launch_plan_local_path() -> None:
    plan = build_launch_plan(input_path="/data/local.mcap", image="npa-lichtblick:test", port=9099)
    assert isinstance(plan, LichtblickLaunchPlan)
    assert plan.artifact_name == "local.mcap"
    assert plan.image == "npa-lichtblick:test"
    assert plan.port == 9099


def test_launch_viewer_uses_injected_runner() -> None:
    plan = build_launch_plan(input_path="s3://b/k/x.mcap", image="npa-lichtblick:test")
    # Wildcard bind stays 0.0.0.0 for the container port mapping...
    assert plan.host == "0.0.0.0"
    # ...but the browser deep link uses a navigable loopback host.
    assert plan.viewer_url.startswith("http://127.0.0.1:8080/?")
    captured: list[list[str]] = []
    result = launch_viewer(plan, local_artifact="/tmp/x.mcap", runner=captured.append)
    assert result.status == "launched"
    assert captured, "runner was not invoked"
    assert "0.0.0.0:8080:8080" in captured[0]
    argv = captured[0]
    assert argv[0] == "docker"
    assert "/tmp/x.mcap:/srv/data/x.mcap:ro" in argv
    assert "npa-lichtblick:test" in argv


def test_launch_viewer_requires_local_artifact() -> None:
    plan = build_launch_plan(input_path="s3://b/k/x.mcap", image="npa-lichtblick:test")
    with pytest.raises(LichtblickError):
        launch_viewer(plan, local_artifact="", runner=lambda argv: None)


# --------------------------------------------------------------------------- #
# Tangible capability: robot camera frames -> MCAP, staging, and real launch.
# --------------------------------------------------------------------------- #
_PNG = b"\x89PNG\r\n\x1a\n"


class _FakeS3:
    """Minimal boto3 s3 stand-in for infra-free staging tests."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):  # noqa: N803
        contents = [{"Key": k} for k in sorted(self.blobs) if k.startswith(Prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, Bucket, Key, Filename):  # noqa: N803
        with open(Filename, "wb") as handle:
            handle.write(self.blobs[Key])


def test_build_mcap_from_frames_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    frames = []
    for i in range(3):
        p = tmp_path / f"frame-{i:04d}.png"
        p.write_bytes(_PNG + f"frame{i}".encode())
        frames.append(str(p))
    out = tmp_path / "camera.mcap"
    info = build_mcap_from_frames(frames, str(out), topic="/rollout/camera", fps=5.0)
    assert info["message_count"] == 3
    assert out.is_file()

    with open(out, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        topics = [c.topic for c in summary.channels.values()]
        schema_names = [s.name for s in summary.schemas.values()]
        messages = list(reader.iter_messages())
    assert topics == ["/rollout/camera"]
    assert "foxglove.CompressedImage" in schema_names
    assert len(messages) == 3
    # Deterministic 5 fps timeline: 0ns, 200ms, 400ms.
    log_times = [m[2].log_time for m in messages]
    assert log_times == [0, 200_000_000, 400_000_000]


def test_build_mcap_from_frames_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(LichtblickError):
        build_mcap_from_frames([], str(tmp_path / "x.mcap"))


def test_stage_frames_from_s3_builds_mcap(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    s3 = _FakeS3(
        {
            "run/augment/frames/frame-00000.png": _PNG + b"a",
            "run/augment/frames/frame-00001.png": _PNG + b"b",
            "run/augment/cosmos2-transfer-result.json": b"{}",  # non-image, ignored
        }
    )
    mcap_path, count = stage_input_to_mcap(
        "s3://bucket/run/augment/frames/",
        str(tmp_path / "work"),
        from_frames=True,
        s3_client=s3,
    )
    assert count == 2
    assert mcap_path.endswith("camera.mcap")
    assert Path(mcap_path).is_file()


def test_stage_local_mcap_copies_as_is(tmp_path: Path) -> None:
    src = tmp_path / "recording.mcap"
    src.write_bytes(b"\x89MCAP0\r\n")
    out, count = stage_input_to_mcap(str(src), str(tmp_path / "work"), from_frames=False)
    assert count is None
    assert Path(out).read_bytes() == b"\x89MCAP0\r\n"


def test_serve_viewer_execute_stages_and_launches(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    s3 = _FakeS3(
        {
            "run/rollouts/iter_01/rollout-0000/camera/000.png": _PNG + b"1",
            "run/rollouts/iter_01/rollout-0000/camera/001.png": _PNG + b"2",
        }
    )
    calls: list[list[str]] = []
    plan = serve_viewer(
        input_path="s3://bucket/run/rollouts/iter_01/rollout-0000/camera/",
        from_frames=True,
        execute=True,
        image="npa-lichtblick:test",
        s3_client=s3,
        runner=calls.append,
        workdir=str(tmp_path / "work"),
    )
    assert plan.status == "launched"
    assert plan.staged is True
    assert plan.message_count == 2
    assert plan.served_artifact_path == "/srv/data/camera.mcap"
    argv = calls[0]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "-d" in argv
    assert any(a.endswith(":/srv/data/camera.mcap:ro") for a in argv)
    assert "npa-lichtblick:test" in argv


def test_serve_from_frames_plan_only_does_not_touch_s3() -> None:
    # execute=False must not require S3 or build an MCAP; it just plans.
    plan = serve_viewer(
        input_path="s3://bucket/run/augment/frames/",
        from_frames=True,
        execute=False,
        image="npa-lichtblick:test",
    )
    assert plan.status == "planned"
    assert plan.artifact_name == "camera.mcap"
    assert "camera.mcap" in plan.viewer_url


def test_cli_serve_from_frames_plan() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "lichtblick",
            "serve",
            "--input-path",
            "s3://bucket/run/augment/frames/",
            "--from-frames",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["artifact_name"] == "camera.mcap"
    assert payload["image"] == "cr.example/reg/npa-lichtblick:1.26.0"
