"""Keep CPU and GPU execution authority separate across Sim2Real graphs."""

from __future__ import annotations

import hashlib
import json

import pytest

from npa.workflows.sim2real import workflow_io


def _provenance(*, gpu: bool = False) -> dict:
    digest = "sha256:" + "2" * 64
    proof = {
        "source_sha": "1" * 40,
        "image": "cr.example/npa/runtime@" + digest,
        "image_digest": digest,
        "workflow_job": "test-owned-stage-job",
        "execution_mode": "standard_npa_workflow_skypilot",
    }
    if gpu:
        proof.update(
            gpu_products=["NVIDIA RTX PRO 6000"],
            gpu_rows=["NVIDIA RTX PRO 6000, GPU-test-0001"],
        )
    return proof


def _record(stage: int, schema: str, *, gpu: bool = False) -> dict:
    return workflow_io.build_component_record(
        stage=stage,
        name="test_component",
        tier="WORKS",
        evidence="Completed the owned task.",
        artifacts={"result": "s3://example-bucket/run/result.json"},
        execution_provenance=_provenance(gpu=gpu),
        component_schema=schema,
    )


def _validate(record: dict, schema: str) -> None:
    workflow_io.validate_component_record(
        record,
        expected_stage=record["stage"],
        expected_name="test_component",
        expected_tier="WORKS",
        required_artifacts=("result",),
        expected_source_sha="1" * 40,
        expected_schema=schema,
    )


@pytest.mark.parametrize("stage", [1, 3, 4, 5, 7, 10, 13, 14])
def test_pi05_cpu_stages_require_job_and_image_but_no_gpu(stage):
    record = _record(stage, workflow_io.PI05_COMPONENT_SCHEMA)
    _validate(record, workflow_io.PI05_COMPONENT_SCHEMA)
    assert "gpu_products" not in record["artifacts"]


@pytest.mark.parametrize("stage", [2, 6, 8, 9, 11])
def test_pi05_gpu_stages_require_device_evidence_on_production_and_readback(stage):
    schema = workflow_io.PI05_COMPONENT_SCHEMA
    with pytest.raises(ValueError, match="execution provenance is invalid"):
        _record(stage, schema)
    record = _record(stage, schema, gpu=True)
    _validate(record, schema)
    del record["artifacts"]["gpu_rows"]
    material = {key: value for key, value in record.items() if key != "content_sha256"}
    record["content_sha256"] = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="execution provenance is invalid"):
        _validate(record, schema)


@pytest.mark.parametrize("stage", [3, 4, 7, 9, 10])
def test_canonical_gpu_requirements_remain_enforced(stage):
    with pytest.raises(ValueError, match="execution provenance is invalid"):
        _record(stage, workflow_io.CANONICAL_COMPONENT_SCHEMA)
    _validate(
        _record(stage, workflow_io.CANONICAL_COMPONENT_SCHEMA, gpu=True),
        workflow_io.CANONICAL_COMPONENT_SCHEMA,
    )


def test_default_canonical_reader_cannot_admit_a_pi05_cpu_stage_record():
    record = _record(4, workflow_io.PI05_COMPONENT_SCHEMA)
    with pytest.raises(ValueError, match="ComponentRecord is invalid"):
        workflow_io.validate_component_record(
            record,
            expected_stage=4,
            expected_name="test_component",
            expected_tier="WORKS",
            required_artifacts=("result",),
        )


def test_pi05_reader_cannot_admit_a_canonical_record():
    with pytest.raises(ValueError, match="ComponentRecord is invalid"):
        _validate(
            _record(1, workflow_io.CANONICAL_COMPONENT_SCHEMA),
            workflow_io.PI05_COMPONENT_SCHEMA,
        )


@pytest.mark.parametrize("schema", ["", "npa.sim2real.unknown.v1"])
def test_unknown_schema_cannot_select_cpu_execution_requirements(schema):
    with pytest.raises(ValueError, match="Unknown ComponentRecord workflow schema"):
        _record(4, schema)


@pytest.mark.parametrize("missing", ["workflow_job", "image", "source_sha"])
def test_pi05_cpu_records_still_reject_missing_execution_authority(missing):
    proof = _provenance()
    del proof[missing]
    with pytest.raises(ValueError, match="execution provenance is invalid"):
        workflow_io.build_component_record(
            stage=4,
            name="test_component",
            tier="WORKS",
            evidence="Completed the owned task.",
            artifacts={"result": "s3://example-bucket/run/result.json"},
            execution_provenance=proof,
            component_schema=workflow_io.PI05_COMPONENT_SCHEMA,
        )
