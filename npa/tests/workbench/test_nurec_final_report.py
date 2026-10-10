"""Bind qualification to the actual finalize producer and reject false success."""

import hashlib
import json

import pytest

from npa.workbench.nurec.qualification_audit import (
    NcoreQualificationAuditError,
    _final_report,
)


RUN_URI = "s3://example-bucket/qualification-run/"
RUN_ID = "workflow-qualification-run"
PREFIX_SHA256 = hashlib.sha256(RUN_URI.encode()).hexdigest()


@pytest.fixture
def final_report(tmp_path, monkeypatch):
    from npa.cli import nurec
    from npa.workbench.nurec.nurec import NurecStatusResult

    status = NurecStatusResult(
        ok=True,
        run_uri=RUN_URI,
        object_count=9,
        stages={"reconstruction": {"objects": 4, "bytes": 300}},
        has_rrd=True,
        has_usdz=True,
        has_novel_views=True,
    )
    path = tmp_path / "final.json"

    def publish(source, target):
        assert target == RUN_URI + "reports/final.json"
        path.write_bytes(source.read_bytes())
        return target

    monkeypatch.setattr(nurec, "nurec_run_status", lambda uri: status)
    monkeypatch.setattr(nurec, "_publish", publish)
    nurec.finalize_cmd(
        input_uri=RUN_URI,
        output_uri="",
        run_id=RUN_ID,
        output=nurec.OutputFormat.json,
    )
    return path


def test_standard_finalize_report_passes_without_discarding_metadata(final_report):
    original = final_report.read_bytes()
    recording_id = _final_report(
        final_report, recording_id=RUN_ID, prefix_sha256=PREFIX_SHA256
    )
    assert recording_id == "qualification-run"
    assert recording_id != RUN_ID
    assert final_report.read_bytes() == original
    payload = json.loads(original)
    assert payload["artifact_count"] == 9
    assert payload["stages"] == {"reconstruction": {"objects": 4, "bytes": 300}}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "failed"),
        ("status", True),
        ("capability", "other-workload"),
        ("run_id", "different-run"),
        ("run_uri", "s3://example-bucket/different-run/"),
        ("run_uri", True),
        ("errors", ["producer failure"]),
        ("errors", None),
        ("errors", {}),
        *[
            (key, value)
            for key in ("has_usdz", "has_novel_views", "has_rrd")
            for value in (False, 1, "true", None)
        ],
    ],
)
def test_finalize_report_rejects_wrong_identity_or_nonliteral_success(
    final_report, field, value
):
    payload = json.loads(final_report.read_bytes())
    payload[field] = value
    final_report.write_text(json.dumps(payload))
    with pytest.raises(NcoreQualificationAuditError, match="terminal qualification"):
        _final_report(final_report, recording_id=RUN_ID, prefix_sha256=PREFIX_SHA256)


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "capability",
        "run_id",
        "run_uri",
        "errors",
        "has_usdz",
        "has_novel_views",
        "has_rrd",
    ],
)
def test_finalize_report_rejects_missing_producer_contract(final_report, field):
    payload = json.loads(final_report.read_bytes())
    payload.pop(field)
    final_report.write_text(json.dumps(payload))
    with pytest.raises(NcoreQualificationAuditError, match="terminal qualification"):
        _final_report(final_report, recording_id=RUN_ID, prefix_sha256=PREFIX_SHA256)
