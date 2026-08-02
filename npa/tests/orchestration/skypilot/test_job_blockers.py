"""A managed job whose pod cannot start must explain itself.

Kubernetes retries image pulls and scheduling forever, so SkyPilot keeps
reporting such a job as PENDING and it never becomes FAILED. The reported case
sat that way for ~14 hours before anyone cancelled it.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from npa.orchestration.skypilot.job_blockers import (
    CLUSTER_LABEL,
    inspect_job_blockers,
)


def _pods(*items: dict) -> str:
    return json.dumps({"items": list(items)})


def _waiting_pod(name: str, reason: str, message: str = "") -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "phase": "Pending",
            "containerStatuses": [
                {"state": {"waiting": {"reason": reason, "message": message}}}
            ],
        },
    }


def _runner(stdout: str, *, returncode: int = 0, stderr: str = ""):
    seen: dict[str, list[str]] = {}

    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        seen["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_image_pull_backoff_is_reported_with_the_pull_permission_remedy() -> None:
    runner = _runner(
        _pods(
            _waiting_pod(
                "sky-abc-worker-0",
                "ImagePullBackOff",
                'Back-off pulling image "cr.example/npa-cosmos2-transfer:2.5.1"',
            )
        )
    )

    report = inspect_job_blockers(job_id="2", cluster_name="sky-abc", runner=runner)

    assert report.blocked is True
    assert report.blockers[0].reason == "ImagePullBackOff"
    assert "retries this forever" in report.remedy()
    assert "different permission from pulling" in report.remedy()
    assert "stays PENDING instead of failing" in report.remedy()


def test_the_pods_are_found_by_skypilots_own_cluster_label() -> None:
    runner = _runner(_pods())

    inspect_job_blockers(cluster_name="sky-abc", context="npa-cluster", runner=runner)

    cmd = runner.seen["cmd"]  # type: ignore[attr-defined]
    assert f"{CLUSTER_LABEL}=sky-abc" in cmd
    assert "--context" in cmd and "npa-cluster" in cmd


def test_an_unschedulable_pod_points_at_the_accelerator_request() -> None:
    runner = _runner(
        _pods(
            {
                "metadata": {"name": "sky-abc-worker-0"},
                "status": {
                    "phase": "Pending",
                    "conditions": [
                        {
                            "type": "PodScheduled",
                            "status": "False",
                            "reason": "Unschedulable",
                            "message": "0/3 nodes are available: insufficient nvidia.com/gpu",
                        }
                    ],
                },
            }
        )
    )

    report = inspect_job_blockers(cluster_name="sky-abc", runner=runner)

    assert report.blockers[0].reason == "Unschedulable"
    assert "single node" in report.remedy()


def test_a_running_pod_is_not_a_blocker() -> None:
    runner = _runner(
        _pods({"metadata": {"name": "sky-abc-worker-0"}, "status": {"phase": "Running"}})
    )

    report = inspect_job_blockers(cluster_name="sky-abc", runner=runner)

    assert report.blocked is False
    assert report.render() == "blockers: none found"


def test_container_creating_is_progress_not_a_blocker() -> None:
    runner = _runner(_pods(_waiting_pod("sky-abc-worker-0", "ContainerCreating")))

    report = inspect_job_blockers(cluster_name="sky-abc", runner=runner)

    assert report.blocked is False


def test_an_unreachable_cluster_is_an_error_not_a_clean_bill_of_health() -> None:
    runner = _runner("", returncode=1, stderr="Unable to connect to the server")

    report = inspect_job_blockers(cluster_name="sky-abc", runner=runner)

    assert report.blocked is False
    assert "Unable to connect" in report.error
    assert "unavailable" in report.render()


def test_a_job_with_no_cluster_yet_says_so() -> None:
    report = inspect_job_blockers(job_id="2", cluster_name="")

    assert "nothing has been scheduled" in report.error


def test_missing_kubectl_is_reported() -> None:
    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        raise OSError("No such file or directory: 'kubectl'")

    report = inspect_job_blockers(cluster_name="sky-abc", runner=run)

    assert "could not run kubectl" in report.error


@pytest.mark.parametrize(
    "reason", ["ErrImagePull", "InvalidImageName", "CreateContainerConfigError", "CrashLoopBackOff"]
)
def test_every_retry_forever_reason_carries_a_remedy(reason: str) -> None:
    runner = _runner(_pods(_waiting_pod("sky-abc-worker-0", reason)))

    report = inspect_job_blockers(cluster_name="sky-abc", runner=runner)

    assert report.blockers[0].reason == reason
    assert report.remedy()


def test_render_lists_each_blocked_pod() -> None:
    runner = _runner(
        _pods(
            _waiting_pod("worker-0", "ImagePullBackOff"),
            _waiting_pod("worker-1", "ImagePullBackOff"),
        )
    )

    rendered = inspect_job_blockers(cluster_name="sky-abc", runner=runner).render()

    assert "blockers (2)" in rendered
    assert "worker-0" in rendered and "worker-1" in rendered
    assert "Suggested action:" in rendered
