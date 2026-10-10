"""Each consumed publication read must match the selected journal identity."""

import hashlib
import io
import json

import pytest
from fastapi import HTTPException

from npa.cli import agent_stage_runtime as runtime, agent_stages
from npa.workflows import artifacts
from npa.workflows.sim2real import publication


_BUCKET, _RUN, _PREFIX = "demo-bucket", "run-a", "runs"


def _report_inventory(good):
    root = f"s3://{_BUCKET}/{_PREFIX}/{_RUN}"
    generation = "a" * 64
    data, targets, inventory = {}, [], []
    for suffix, payload in (
        ("reports/sim2real-report.json", good),
        ("reports/sim2real.rrd", b"explicit transport fixture"),
        ("components/stage_14.json", b"{}"),
    ):
        uri = f"{root}/reports/generations/{generation}/{suffix.rsplit('/', 1)[1]}"
        if suffix.startswith("components/"):
            uri = f"{root}/components/history/stage_14/{hashlib.sha256(payload).hexdigest()}.json"
        key = uri.removeprefix(f"s3://{_BUCKET}/")
        data[key] = payload
        targets.append(
            {
                "uri": f"{root}/{suffix}",
                "state": "present",
                "immutable_uri": uri,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        inventory.append(
            artifacts.Artifact(
                run_id=_RUN,
                key=key,
                s3_uri=uri,
                size=len(payload),
                last_modified="",
                render="json",
                inline=True,
                relative_key=key.removeprefix(f"{_PREFIX}/{_RUN}/"),
            )
        )
    targets.append({"uri": f"{root}/reports/sim2real.mcap", "state": "absent"})
    return data, targets, inventory


def _report_store(good, forged, tamper):
    data, targets, inventory = _report_inventory(good)
    journal = publication._journal_bytes(
        transaction_id="a" * 64,
        attempt_id="b" * 32,
        state="committed",
        objects=targets,
    )

    class Store:
        report_reads = 0
        bodies = []

        def get_object(self, *, Bucket, Key):
            assert Bucket == _BUCKET
            payload = (
                journal if Key.endswith(".sim2real-publication.json") else data[Key]
            )
            if Key.endswith("sim2real-report.json"):
                self.report_reads += 1
                if tamper and self.report_reads == 2:
                    payload = forged
            body = io.BytesIO(payload)
            self.bodies.append(body)
            return {"Body": body}

        def head_object(self, *, Bucket, Key):
            assert Bucket == _BUCKET
            return {"ContentLength": len(data[Key])}

    return Store(), inventory


def _install_runtime_dependencies(monkeypatch, store, inventory):
    dependencies = {
        "_validated_resolved_prefix": lambda value: value,
        "_agent_artifact_s3_client": lambda: (
            store,
            {"bucket": _BUCKET, "prefix": _PREFIX},
        ),
        "_agent_access_report": lambda: {},
        "artifact_bucket_projects": lambda _: {},
        "_load_selected_run_artifacts": lambda **_: (
            _BUCKET,
            "fixture-project",
            _PREFIX,
            inventory,
        ),
        "_discovery_exclude_roots": lambda: [],
        "HTTPException": HTTPException,
        "ArtifactDiscoveryError": artifacts.ArtifactDiscoveryError,
        "select_preferred_artifact": artifacts.select_preferred_artifact,
        "parse_stage_evidence_documents": agent_stages.parse_stage_evidence_documents,
        "build_artifact_backed_stages": agent_stages.build_artifact_backed_stages,
        "summarize_stage_evidence": agent_stages.summarize_stage_evidence,
        "run_owns_workflow_stage_overlay": lambda *_: False,
        "_now_iso": lambda: "fixture-time",
    }
    for key, value in dependencies.items():
        monkeypatch.setattr(runtime, key, value, raising=False)
    monkeypatch.setattr(
        runtime, "_workflow_stage_defs_from_state", lambda _: [], raising=False
    )


def _run_details():
    return runtime._artifact_backed_run_details(
        {},
        _RUN,
        resource_bucket=_BUCKET,
        project_id="fixture-project",
        resolved_prefix=_PREFIX,
        source_selected=True,
    )


@pytest.mark.parametrize("tamper", [False, True])
def test_stage_authority_rejects_an_unverified_second_report_get(monkeypatch, tamper):
    good = json.dumps({"stages": {"diagnostic": {"status": "failed"}}}).encode()
    forged = json.dumps({"stages": {"diagnostic": {"status": "passed"}}}).encode()
    assert len(good) == len(forged)
    store, inventory = _report_store(good, forged, tamper)
    _install_runtime_dependencies(monkeypatch, store, inventory)
    if tamper:
        with pytest.raises(HTTPException) as rejected:
            _run_details()
        assert rejected.value.status_code == 409
        assert store.report_reads == 2
    else:
        response = _run_details()
        stage = next(row for row in response["stages"] if row["id"] == "diagnostic")
        assert stage["status"] == "failed"
        assert stage["authority"] == "authoritative"
    assert all(body.closed for body in store.bodies)
