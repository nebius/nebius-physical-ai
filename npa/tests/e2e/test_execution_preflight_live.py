"""Opt-in exact-target checks; owner-only config supplies authorized destinations.

Set NPA_EXECUTION_PREFLIGHT_LIVE_CONFIG to a private JSON file with project,
context, output_uris, gpu_type, gpu_count, preset, and evidence_path. Workload
execution and reservation allocation evidence are separate acceptance records;
this test proves the gate against the same target and executing S3 principal.
"""

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from npa import execution_preflight

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or not os.environ.get("NPA_EXECUTION_PREFLIGHT_LIVE_CONFIG"),
    reason="requires an authorized owner-only execution target",
)]


def test_effective_target_live_roundtrip_and_scope_rejection():
    path = Path(os.environ["NPA_EXECUTION_PREFLIGHT_LIVE_CONFIG"])
    assert path.stat().st_mode & 0o077 == 0, "live configuration must be owner-only"
    config = json.loads(path.read_text())
    target = execution_preflight.resolve_execution_target(
        project=config["project"], context=config.get("context", ""),
        output_uris=config["output_uris"],
    )
    report = execution_preflight.verify_execution_target(
        target, gpu_check=lambda: execution_preflight.verify_serverless_gpu(
            project_id=target.project_id, gpu_type=config["gpu_type"],
            gpu_count=config["gpu_count"], preset=config["preset"],
        ),
    )
    # Purely scoped negative cases: the actual project's provider response is
    # compared with intentionally wrong expectations before any storage write.
    negatives = []
    for field in ("tenant_id", "region"):
        with pytest.raises(execution_preflight.ExecutionPreflightError):
            execution_preflight.verify_execution_target(replace(target, **{field: "intentionally-mismatched"}))
        negatives.append(field)
    module_path = Path(inspect.getfile(execution_preflight)).resolve()
    evidence = {"report": report, "scoped_negative_checks": negatives,
                "source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "module_path": str(module_path)}
    output = Path(config["evidence_path"])
    assert output.parent.stat().st_mode & 0o077 == 0
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(evidence, handle, indent=2)
    assert report["execution_readiness"] == "pass"
