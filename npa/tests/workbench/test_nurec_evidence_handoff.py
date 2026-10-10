from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workbench.nurec.evidence_handoff import (
    NcoreEvidenceHandoffError,
    publish_qualification_evidence,
)


class _Storage:
    def __init__(self, *, existing=False, wrong_readback=False):
        self.existing = existing
        self.wrong_readback = wrong_readback
        self.payload = b""
        self.uri = ""
        self.etag = '"etag"'

    def put_bytes_conditional(self, payload, uri, *, if_none_match, content_type):
        assert if_none_match is True
        assert content_type == "application/json"
        if self.existing:
            raise StoragePreconditionFailed("exists")
        self.payload = payload
        self.uri = uri
        return self.etag

    def read_bytes_with_etag(self, uri):
        assert uri == self.uri
        return (
            b"wrong" if self.wrong_readback else self.payload,
            self.etag,
        )


def _runtime(path: Path, run_id: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "npa_nurec_runtime_attestation_v4",
                "status": "pass",
                "workflow_run_id_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
            }
        )
    )
    path.chmod(0o600)
    return path


def test_runtime_handoff_is_conditional_and_read_back(tmp_path: Path) -> None:
    run_id = "private-run"
    source = _runtime(tmp_path / "runtime.json", run_id)
    receipt_path = tmp_path / "handoff.json"
    receipt = publish_qualification_evidence(
        source,
        "s3://private/run/evidence/nre-runtime.json",
        kind="runtime-attestation",
        run_id=run_id,
        receipt_path=receipt_path,
        storage_client=_Storage(),
    )
    assert receipt["status"] == "pass"
    assert receipt["conditional_create"] is True
    assert receipt["readback_verified"] is True
    assert receipt_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("storage", "match"),
    [
        (_Storage(existing=True), "already exists"),
        (_Storage(wrong_readback=True), "read-back differs"),
    ],
)
def test_runtime_handoff_rejects_reuse_or_changed_readback(
    tmp_path: Path, storage, match: str
) -> None:
    run_id = "private-run"
    source = _runtime(tmp_path / "runtime.json", run_id)
    with pytest.raises(NcoreEvidenceHandoffError, match=match):
        publish_qualification_evidence(
            source,
            "s3://private/run/evidence/nre-runtime.json",
            kind="runtime-attestation",
            run_id=run_id,
            receipt_path=tmp_path / "handoff.json",
            storage_client=storage,
        )
    assert not (tmp_path / "handoff.json").exists()


def test_workflow_status_handoff_requires_terminal_success(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text(
        json.dumps({"run_id": "private-run", "status": "FAILED", "stages": {}})
    )
    path.chmod(0o600)
    with pytest.raises(NcoreEvidenceHandoffError, match="binding differs"):
        publish_qualification_evidence(
            path,
            "s3://private/run/evidence/workflow-status.json",
            kind="workflow-status",
            run_id="private-run",
            receipt_path=tmp_path / "handoff.json",
            storage_client=_Storage(),
        )
