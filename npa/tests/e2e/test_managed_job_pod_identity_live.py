"""Read-only live proof that pod diagnostics stay inside one managed-job owner.

Set ``NPA_INTEGRATION_E2E=1`` and
``NPA_LIVE_MANAGED_JOB_POD_DIAGNOSTICS=1``, then supply exact existing values in
``NPA_E2E_MANAGED_JOB_CONTROLLER``, ``NPA_E2E_MANAGED_JOB_ID``,
``NPA_E2E_MANAGED_JOB_TASK_NAME``, ``NPA_E2E_MANAGED_JOB_CONTEXT``,
``NPA_E2E_MANAGED_JOB_KUBECONFIG``, and ``NPA_E2E_MANAGED_JOB_SKY_BIN``.
The test only reads the controller queue and Kubernetes pods.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pytest

from npa.cli.workbench.workflow import (
    _isolated_controller_diagnostic_environment,
)
from npa.orchestration.skypilot.job_blockers import (
    JobBlockerReport,
    inspect_job_blockers,
)
from npa.orchestration.skypilot.workflow import workflow_task_statuses


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot]


@dataclass(frozen=True)
class _LivePodScope:
    controller: Path
    job_id: str
    task_name: str
    context: str
    kubeconfig: Path
    sky_bin: Path


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"set {name} to an exact operator-reviewed live selector")
    return value


def _live_scope() -> _LivePodScope:
    if os.environ.get("NPA_LIVE_MANAGED_JOB_POD_DIAGNOSTICS") != "1":
        pytest.skip(
            "set NPA_LIVE_MANAGED_JOB_POD_DIAGNOSTICS=1 for this read-only probe"
        )
    job_id = _required("NPA_E2E_MANAGED_JOB_ID")
    if not job_id.isdigit():
        pytest.fail("NPA_E2E_MANAGED_JOB_ID must be one exact numeric job id")
    return _LivePodScope(
        controller=Path(_required("NPA_E2E_MANAGED_JOB_CONTROLLER")).resolve(
            strict=True
        ),
        job_id=job_id,
        task_name=_required("NPA_E2E_MANAGED_JOB_TASK_NAME"),
        context=_required("NPA_E2E_MANAGED_JOB_CONTEXT"),
        kubeconfig=Path(_required("NPA_E2E_MANAGED_JOB_KUBECONFIG")).resolve(
            strict=True
        ),
        sky_bin=Path(_required("NPA_E2E_MANAGED_JOB_SKY_BIN")).resolve(strict=True),
    )


def _inspect(
    scope: _LivePodScope, *, job_id: str, task: str, owner: str
) -> JobBlockerReport:
    environment, _ = _isolated_controller_diagnostic_environment(
        scope.controller, scope.kubeconfig
    )
    return inspect_job_blockers(
        job_id=job_id,
        context=scope.context,
        kubeconfig=scope.kubeconfig,
        environment=environment,
        expected_task_names=[task],
        controller_user_id=owner,
    )


def _assert_rejected(
    scope: _LivePodScope, *, job_id: str, task: str, owner: str
) -> None:
    report = _inspect(scope, job_id=job_id, task=task, owner=owner)
    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND", report.render()


def test_exact_managed_job_annotations_bind_live_pod_to_isolated_controller() -> None:
    """Query one existing job and reject adjacent identities without mutation."""

    scope = _live_scope()
    rows = workflow_task_statuses(
        scope.job_id,
        isolated_config_dir=scope.controller,
        sky_bin=scope.sky_bin,
        timeout=60,
        raise_on_error=True,
    )
    assert scope.task_name in {str(row.get("task_name") or "") for row in rows}
    environment, owner = _isolated_controller_diagnostic_environment(
        scope.controller, scope.kubeconfig
    )
    assert environment["KUBECONFIG"] == str(scope.kubeconfig)

    report = _inspect(scope, job_id=scope.job_id, task=scope.task_name, owner=owner)
    assert report.error == "", report.render()
    _assert_rejected(
        scope, job_id=f"{scope.job_id}0", task=scope.task_name, owner=owner
    )
    _assert_rejected(
        scope, job_id=scope.job_id, task=f"{scope.task_name}-other", owner=owner
    )
    _assert_rejected(
        scope, job_id=scope.job_id, task=scope.task_name, owner=f"{owner}-other"
    )
