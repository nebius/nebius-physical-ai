from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from npa.workbench.nurec.qualification_cleanup import (
    NcoreQualificationCleanupError,
    cleanup_qualification,
)


SOURCE_SHA = "a" * 40


class _Paginator:
    def paginate(self, *, Bucket, Prefix):
        assert Bucket == "private"
        assert Prefix == "run/evidence/"
        yield {
            "Contents": [
                {
                    "Key": "run/evidence/report.json",
                    "Size": 10,
                    "ETag": '"etag"',
                }
            ]
        }


class _S3:
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator()


class _Storage:
    s3 = _S3()


def _build_receipt(tmp_path: Path) -> Path:
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(
            {
                "schema": "npa.ncore.committed-oci-build.v1",
                "source_sha": SOURCE_SHA,
                "argv": [
                    "bash",
                    "build.sh",
                    "--oci-output",
                    "/private/image.oci.tar",
                ],
            }
        )
    )
    path.chmod(0o600)
    return path


def _workflow_status(tmp_path: Path) -> Path:
    path = tmp_path / "workflow-status.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "private-run",
                "stages": {
                    "reconstruct": {
                        "workflow_state": "reconstruct",
                        "managed_job_id": "41",
                        "job_name": "private-run-01-reconstruct",
                        "job_attribution": "exact",
                    },
                    "render": {
                        "workflow_state": "render",
                        "managed_job_id": "42",
                        "job_name": "private-run-02-render",
                        "job_attribution": "exact",
                    },
                    "visualize": {
                        "workflow_state": "visualize",
                        "managed_job_id": "43",
                        "job_name": "private-run-03-visualize",
                        "job_attribution": "exact",
                    },
                    "finalize": {
                        "workflow_state": "finalize",
                        "managed_job_id": "44",
                        "job_name": "private-run-04-finalize",
                        "job_attribution": "exact",
                    },
                },
            }
        )
    )
    return path


def _runner(*, active=False):
    def run(command, **kwargs):
        if "kubectl" in command[0]:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "annotations": {
                                "skypilot-managed-job-name": "private-run-02-render",
                                "skypilot-managed-job-id": "42",
                            }
                        },
                        "status": {"phase": "Running" if active else "Succeeded"},
                    }
                ]
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if "inspect" in command:
            return subprocess.CompletedProcess(
                command, 1, "", "not found: no such image or builder"
            )
        return subprocess.CompletedProcess(command, 0, "removed", "")

    return run


def _cleaner(*_args, **_kwargs):
    return SimpleNamespace(
        errors=[],
        commands=[
            ["sky", "jobs", "cancel", "--yes", "41"],
            ["sky", "jobs", "cancel", "--yes", "42"],
            ["sky", "jobs", "cancel", "--yes", "43"],
            ["sky", "jobs", "cancel", "--yes", "44"],
            ["sky", "down", "--yes", "private-run"],
        ],
    )


def test_cleanup_receipt_derives_order_absence_and_retained_storage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cleanup.json"
    receipt = cleanup_qualification(
        run_id="private-run",
        workflow_status_path=_workflow_status(tmp_path),
        context="private-context",
        namespace="private-namespace",
        storage_prefix="s3://private/run/evidence/",
        local_image="local/ncore:candidate",
        builder="private-builder",
        build_receipt_path=_build_receipt(tmp_path),
        source_sha=SOURCE_SHA,
        output_path=path,
        storage_client=_Storage(),
        process_runner=_runner(),
        workflow_cleaner=_cleaner,
    )

    assert receipt["status"] == "pass"
    assert receipt["cancel_before_destroy"] is True
    assert receipt["active_job_pods"] == 0
    assert receipt["managed_jobs"] == 4
    assert receipt["registry_disposition"] == "not_created_local_oci_route"
    assert receipt["storage_disposition"] == "retained_declared"
    assert receipt["orphan_count"] == 0
    assert json.loads(path.read_text()) == receipt


def test_cleanup_refuses_active_exact_job_pod(tmp_path: Path) -> None:
    with pytest.raises(NcoreQualificationCleanupError, match="active Kubernetes"):
        cleanup_qualification(
            run_id="private-run",
            workflow_status_path=_workflow_status(tmp_path),
            context="private-context",
            namespace="private-namespace",
            storage_prefix="s3://private/run/evidence/",
            local_image="local/ncore:candidate",
            builder="private-builder",
            build_receipt_path=_build_receipt(tmp_path),
            source_sha=SOURCE_SHA,
            output_path=tmp_path / "cleanup.json",
            storage_client=_Storage(),
            process_runner=_runner(active=True),
            workflow_cleaner=_cleaner,
        )


def test_cleanup_refuses_destroy_before_cancel(tmp_path: Path) -> None:
    def wrong_order(*_args, **_kwargs):
        return SimpleNamespace(
            errors=[],
            commands=[
                ["sky", "down", "--yes", "private-run"],
                ["sky", "jobs", "cancel", "--yes", "41"],
                ["sky", "jobs", "cancel", "--yes", "42"],
                ["sky", "jobs", "cancel", "--yes", "43"],
                ["sky", "jobs", "cancel", "--yes", "44"],
            ],
        )

    with pytest.raises(NcoreQualificationCleanupError, match="cancel-before"):
        cleanup_qualification(
            run_id="private-run",
            workflow_status_path=_workflow_status(tmp_path),
            context="private-context",
            namespace="private-namespace",
            storage_prefix="s3://private/run/evidence/",
            local_image="local/ncore:candidate",
            builder="private-builder",
            build_receipt_path=_build_receipt(tmp_path),
            source_sha=SOURCE_SHA,
            output_path=tmp_path / "cleanup.json",
            storage_client=_Storage(),
            process_runner=_runner(),
            workflow_cleaner=wrong_order,
        )


def test_cleanup_refuses_incomplete_workflow_job_inventory(tmp_path: Path) -> None:
    status = _workflow_status(tmp_path)
    payload = json.loads(status.read_text())
    del payload["stages"]["finalize"]
    status.write_text(json.dumps(payload))

    with pytest.raises(NcoreQualificationCleanupError, match="complete"):
        cleanup_qualification(
            run_id="private-run",
            workflow_status_path=status,
            context="private-context",
            namespace="private-namespace",
            storage_prefix="s3://private/run/evidence/",
            local_image="local/ncore:candidate",
            builder="private-builder",
            build_receipt_path=_build_receipt(tmp_path),
            source_sha=SOURCE_SHA,
            output_path=tmp_path / "cleanup.json",
            storage_client=_Storage(),
            process_runner=_runner(),
            workflow_cleaner=_cleaner,
        )
