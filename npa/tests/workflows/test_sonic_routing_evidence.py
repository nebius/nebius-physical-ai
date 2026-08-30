from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.workflows.sonic_routing_evidence import (
    SonicRoutingEvidenceError,
    generate_routing_evidence,
)

SPEC = (
    Path(__file__).parents[2]
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "sonic-b300-routing-evidence.yaml"
)


def _generate(tmp_path: Path, **overrides):
    kwargs = {
        "manifest_uri": str(tmp_path / "manifest.json"),
        "report_uri": str(tmp_path / "test-report.json"),
        "rrd_uri": str(tmp_path / "reports" / "sonic-b300-routing.rrd"),
        "run_id": "sonic-routing-test",
        "tested_commit_sha": "d2ecf538b79bda060e74841c565647b9541ab279",
    }
    kwargs.update(overrides)
    return generate_routing_evidence(**kwargs)


def test_generate_routing_evidence_is_fail_closed_and_time_structured(tmp_path: Path) -> None:
    result = _generate(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "test-report.json").read_text(encoding="utf-8"))
    rrd = tmp_path / "reports" / "sonic-b300-routing.rrd"

    assert result["status"] == "passed"
    assert report["failed"] == 0
    assert manifest["visualization"] == {
        "duration_seconds": 6.0,
        "samples_per_series": 7,
        "timeline": "evidence_time",
    }
    focused = manifest["focused_verification"]
    assert focused == {
        "expected": "B300:1",
        "resolved": "B300:1",
        "status": "passed",
        "target": "gpu-b300-sxm",
    }
    assert manifest["assertions"]["scheduling_placement"]["status"] == "unverified"
    assert rrd.stat().st_size > 4_096

    rerun = Path(sys.executable).with_name("rerun")
    verified = subprocess.run(
        [str(rerun), "rrd", "verify", str(rrd)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    printed = subprocess.run(
        [str(rerun), "rrd", "print", "-vv", str(rrd)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert printed.returncode == 0, printed.stderr
    assert "evidence_time" in printed.stdout
    assert "assertions/routing_resolution" in printed.stdout
    assert "routing_map/target_to_accelerator" in printed.stdout


def test_live_assertions_are_recorded_separately(tmp_path: Path) -> None:
    _generate(
        tmp_path,
        provider_accelerator="B300",
        allocated_count=1,
        provider_recognition_status="passed",
        scheduling_status="passed",
        workload_status="failed",
        terminal_status="FAILED",
        job_evidence_digest="a" * 64,
        image_digest="sha256:" + "b" * 64,
        workload_kind="train",
        output_kind="checkpoint",
        output_bytes=123,
        output_digest="c" * 64,
        semantic_verification="weights-only-load-passed",
        output_verification_status="passed",
        cleanup_status="passed",
        pool_type="preemptible",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["assertions"]["routing_resolution"]["status"] == "passed"
    assert manifest["assertions"]["provider_accelerator_recognition"]["status"] == "passed"
    assert manifest["assertions"]["scheduling_placement"]["status"] == "passed"
    assert manifest["assertions"]["workload_completion"]["status"] == "failed"
    assert manifest["assertions"]["output_verification"]["status"] == "passed"
    assert manifest["assertions"]["attempt_cleanup"]["status"] == "passed"
    assert manifest["provider"]["allocated_count"] == 1
    assert manifest["output"]["bytes"] == 123


def test_live_pass_claims_fail_closed_without_objective_output(tmp_path: Path) -> None:
    with pytest.raises(SonicRoutingEvidenceError, match="passed output verification"):
        _generate(
            tmp_path,
            output_verification_status="passed",
            output_kind="checkpoint",
            output_bytes=0,
            output_digest="c" * 64,
            semantic_verification="weights-only-load-passed",
        )


def test_wrong_checked_in_route_raises_after_writing_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.workbench.sonic import workflow

    original = workflow.default_accelerators
    monkeypatch.setattr(
        workflow,
        "default_accelerators",
        lambda target: "L40S:1" if target == "b300" else original(target),
    )
    with pytest.raises(SonicRoutingEvidenceError, match="failed closed"):
        _generate(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["assertions"]["routing_resolution"]["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider_accelerator", "private-cluster@customer"),
        ("job_evidence_digest", "not-a-digest"),
        ("image_digest", "sha256:short"),
    ],
)
def test_publication_fields_reject_unsanitized_values(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(SonicRoutingEvidenceError):
        _generate(tmp_path, **{field: value})


def test_workflow_uses_argv_without_shell_interpolation() -> None:
    hostile = "value'; touch /tmp/sonic-token-injection; #"
    spec = merge_config_overrides(
        load_spec(SPEC),
        {"semantic_verification": hostile},
    )
    step = build_plan(spec, run_id="sonic-argv-test").steps[0]

    assert step.shell == ""
    assert step.argv[:3] == ["python3", "-m", "npa.workflows.sonic_routing_evidence"]
    assert step.argv[step.argv.index("--semantic-verification") + 1] == hostile
    assert all("python3 -c" not in argument for argument in step.argv)
