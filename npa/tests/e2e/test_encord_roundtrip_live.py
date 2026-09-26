"""Run the shipped Encord workflow and decode its verified MP4 returns."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from npa.clients.storage import StorageClient
from npa.orchestration.npa_workflow import build_plan, load_spec, run_workflow
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.workbench.encord.schemas import RoundtripReport
from npa.workbench.encord.storage import (
    ConditionalArtifactStore,
    S3ObjectStorageGateway,
    TransferDigest,
)

ROOT = Path(__file__).resolve().parents[3]


def test_demo_download_decodes_verified_bytes_and_rejects_corruption(
    tmp_path, monkeypatch
):
    source = ROOT / "npa/tests/browser/cypress/fixtures/browser-compatible.mp4"
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    def download(_self, _uri, destination):
        destination.write_bytes(payload)
        return TransferDigest(size=len(payload), sha256=digest)

    monkeypatch.setattr(S3ObjectStorageGateway, "download_to_file", download)
    item = SimpleNamespace(
        destination_uri="s3://test-bucket/video.mp4",
        observed_checksum=digest,
        observed_size=len(payload),
    )
    report = SimpleNamespace(items=[item])
    videos = _download_verified_videos(report, None, tmp_path)
    assert videos[0]["frames"] == 2
    assert videos[0]["sha256"] == digest
    assert (tmp_path / "demo-1.mp4").read_bytes() == payload
    item.observed_checksum = "0" * 64
    with pytest.raises(ValueError, match="changed after"):
        _download_verified_videos(report, None, tmp_path)


def test_demo_requires_video_media(tmp_path):
    with pytest.raises(ValueError, match="at least one MP4"):
        _download_verified_videos(SimpleNamespace(items=[]), None, tmp_path)


def test_missing_live_targets_skip_without_loading_workflow(monkeypatch):
    monkeypatch.delenv("NPA_E2E_ENCORD_CONFIG", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _configured_spec()


def _configured_spec():
    path = os.environ.get("NPA_E2E_ENCORD_CONFIG", "")
    if not path:
        pytest.skip(
            "Set NPA_E2E_ENCORD_CONFIG to operator-selected Encord and S3 targets"
        )
    overrides = json.loads(Path(path).read_text())
    required = {"bucket", "prefix", "encord_media_uri", "encord_integration"}
    if not isinstance(overrides, dict) or not required <= overrides.keys():
        raise ValueError(
            "Live Encord config must explicitly select all storage targets"
        )
    spec = load_spec(ROOT / "workflows/partners/encord/encord-roundtrip-smoke.yaml")
    return merge_config_overrides(spec, overrides)


def _write_private(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        path.chmod(0o600)
        handle.write(text)


def _download_verified_videos(report: RoundtripReport, storage, evidence: Path):
    import av

    gateway = S3ObjectStorageGateway(storage)
    videos = []
    for item in report.items:
        if not item.destination_uri.lower().endswith(".mp4"):
            continue
        destination = evidence / f"demo-{len(videos) + 1}.mp4"
        digest = gateway.download_to_file(item.destination_uri, destination)
        destination.chmod(0o600)
        if digest.sha256 != item.observed_checksum or digest.size != item.observed_size:
            raise ValueError("MP4 bytes changed after roundtrip verification")
        with av.open(str(destination)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        if not frames:
            raise ValueError("Returned MP4 contains no decodable video frames")
        videos.append(
            {
                "file": destination.name,
                "bytes": digest.size,
                "sha256": digest.sha256,
                "frames": frames,
            }
        )
    if not videos:
        raise ValueError("Live demo requires at least one MP4 input")
    return videos


def _run_roundtrip(spec, run_id: str, evidence: Path) -> None:
    plan = build_plan(spec, run_id=run_id)
    if [step.state for step in plan.steps] != ["push", "pull", "verify"]:
        raise ValueError("Expected the shipped three-stage Encord roundtrip")
    result = run_workflow(spec, run_id=run_id, execute=True, require_inputs=True)
    _write_private(evidence / "workflow.json", json.dumps(result, indent=2))
    storage = StorageClient.from_environment()
    report = RoundtripReport.model_validate(
        ConditionalArtifactStore(storage).read_json(plan.steps[-1].outputs[0]["uri"])
    )
    if not report.passed or not report.matched:
        raise ValueError("Roundtrip did not verify any media")
    videos = _download_verified_videos(report, storage, evidence)
    _write_private(
        evidence / "demo.json",
        json.dumps({"matched": report.matched, "videos": videos}, indent=2),
    )


@pytest.mark.e2e
def test_live_encord_roundtrip_returns_decodable_verified_mp4(tmp_path, monkeypatch):
    run_id = "encord-live-" + uuid4().hex
    root = Path(os.environ.get("NPA_E2E_ENCORD_EVIDENCE_DIR", str(tmp_path)))
    evidence = root / run_id
    evidence.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv(
        "PATH",
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    try:
        _run_roundtrip(_configured_spec(), run_id, evidence)
    except Exception as exc:
        _write_private(evidence / "failure.txt", traceback.format_exc())
        pytest.fail(
            f"Encord live roundtrip failed ({type(exc).__name__}); inspect private evidence",
            pytrace=False,
        )
