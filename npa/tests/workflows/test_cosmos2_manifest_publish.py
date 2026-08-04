"""The cosmos2 transfer stage publishes its manifest, not just its clip.

Live job 287 ran the real Cosmos-Transfer2.5 model and left exactly one object in S3: a 3.9 MB
augmented MP4. The manifest — prompt, control spec, guidance, whether the run was conditioned on
an input clip — went to stdout and died with the pod. For a synthetic-data stage that provenance
is the product, and a spec has nothing durable to declare as its output without it.
"""

from __future__ import annotations

import json
from pathlib import Path

from npa.cli.workbench import cosmos2


class _RecordingClient:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, local: str, uri: str) -> str:
        self.uploads.append((local, uri))
        if uri.endswith("/"):
            return uri + Path(local).name
        return uri


def test_manifest_lands_next_to_the_clip() -> None:
    client = _RecordingClient()
    payload = {
        "schema": "npa.cosmos2.transfer.v1",
        "status": "executed",
        "prompt": "a rainy warehouse",
        "control_spec": "_npa_input_spec_test.json",
    }

    uri = cosmos2._publish_manifest(client, payload, "s3://bucket/run/augmented")

    assert uri == "s3://bucket/run/augmented/manifest.json"
    local, dest = client.uploads[0]
    assert dest == "s3://bucket/run/augmented/manifest.json"
    # The uploaded file is the manifest itself, written before the temp dir goes away.
    assert Path(local).name == cosmos2.MANIFEST_FILENAME


def test_manifest_content_is_the_payload() -> None:
    captured: dict[str, object] = {}

    class _Reader(_RecordingClient):
        def upload_file(self, local: str, uri: str) -> str:
            captured.update(json.loads(Path(local).read_text(encoding="utf-8")))
            return super().upload_file(local, uri)

    payload = {"status": "executed", "output_kind": "video", "video_bytes": 3907034}
    cosmos2._publish_manifest(_Reader(), payload, "s3://bucket/run/augmented/")

    assert captured == payload


def test_local_and_s3_manifest_bytes_are_identical(
    tmp_path: Path, monkeypatch
) -> None:
    payload = {
        "schema": "npa.cosmos2.transfer.v1",
        "status": "executed_reference",
        "mode": "reference_augment",
    }
    output = tmp_path / "augment"
    cosmos2._publish_output_manifest(payload, f"local://{output}")
    local_bytes = (output / "manifest.json").read_bytes()
    uploaded: dict[str, bytes] = {}

    class _CaptureStorage:
        def upload_file(self, local: str, uri: str) -> str:
            uploaded[uri] = Path(local).read_bytes()
            return uri

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        staticmethod(_CaptureStorage),
    )
    cosmos2._publish_output_manifest(payload, "s3://bucket/run/augment")

    assert uploaded["s3://bucket/run/augment/manifest.json"] == local_bytes
    assert local_bytes.endswith(b"\n")


def test_the_spec_declares_the_filename_the_tool_writes() -> None:
    """Guardrail in the same shape as `test_spec_declared_outputs.py`."""

    import yaml

    spec_path = (
        Path(__file__).resolve().parents[3]
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "cosmos2-transfer.yaml"
    )
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    declared = spec["states"]["transfer"]["outputs"][0]["uri"]
    manifest_uri = spec["config"]["augment_manifest_uri"]

    assert declared == "{{config.augment_manifest_uri}}"
    assert manifest_uri.endswith(cosmos2.MANIFEST_FILENAME)
