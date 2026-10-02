"""Credential-redaction contracts for durable SkyPilot workflow records."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from npa.orchestration.skypilot.launch_transaction import (
    CommandEvidence,
    EvidenceState,
    FailureCategory,
    LaunchState,
    LaunchTransactionResult,
    ProbeObservation,
    ReconciliationEvidence,
    ReconciliationState,
    StabilityResult,
)
from npa.orchestration.skypilot.workflow_state import cancel_workflow_job


def _provider_diagnostic() -> str:
    return (
        "provider rejected Bearer SYNTHETIC-BEARER "
        "at https://synthetic-user:synthetic-password@provider.invalid/path?"
        "X-Amz-Signature=synthetic-signature\n"
        "-----BEGIN PRIVATE KEY-----\n"
        "synthetic-key\n"
        "-----END PRIVATE KEY-----\n"
        "retry after fixing access"
    )


def _assert_redacted(payload: object) -> None:
    serialized = json.dumps(payload)
    assert "retry after fixing access" in serialized
    assert all(
        secret not in serialized
        for secret in (
            "SYNTHETIC-BEARER",
            "synthetic-user",
            "synthetic-password",
            "synthetic-signature",
            "synthetic-key",
        )
    )


def test_launch_evidence_records_redact_unknown_provider_credentials() -> None:
    message = _provider_diagnostic()
    payloads = [
        CommandEvidence(
            argv=("sky", "jobs", "queue"),
            returncode=1,
            stdout=message,
            stderr=message,
        ).to_dict(),
        ProbeObservation(EvidenceState.TERMINAL, message=message).to_dict(),
        StabilityResult(
            EvidenceState.TERMINAL,
            FailureCategory.UNKNOWN,
            error=message,
        ).to_dict(),
        ReconciliationEvidence(
            ReconciliationState.UNAVAILABLE,
            workload_evidence=message,
            error=message,
        ).to_dict(),
        LaunchTransactionResult(
            LaunchState.INDETERMINATE,
            "synthetic-launch",
            primary_error=message,
            reconciliation_error=message,
        ).to_dict(),
    ]

    _assert_redacted(payloads)


def test_cancel_record_redacts_unknown_provider_credentials(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(0o755)
    message = _provider_diagnostic()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout=message,
            stderr=message,
        ),
    )

    payload = cancel_workflow_job(
        sky_bin=str(sky_bin),
        job_id="42",
        run_id="synthetic-run",
        also_down_cluster=False,
        timeout=0,
    )

    _assert_redacted(payload)
