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


def test_a_job_with_no_pods_yet_says_so() -> None:
    runner = _runner(_pods())

    report = inspect_job_blockers(job_id="2", cluster_name="", runner=runner)

    assert "nothing has been scheduled yet" in report.error


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


# --- lookup by job id ---------------------------------------------------------
#
# `sky jobs queue` reports cluster_name_on_cloud as null for a job that never
# provisioned -- which is exactly the job worth diagnosing. SkyPilot labels its
# pods `<task>-<job_id>-<user_hash>`, so the job id is enough to find them.


def _labelled_pod(label: str, reason: str) -> dict:
    pod = _waiting_pod(f"{label}-head", reason)
    pod["metadata"]["labels"] = {CLUSTER_LABEL: label}
    return pod


def test_pods_are_found_by_job_id_when_the_queue_reports_no_cluster() -> None:
    runner = _runner(
        _pods(
            _labelled_pod("train-333-64ce57a0", "ImagePullBackOff"),
            _labelled_pod("cosmos-curate-332-64ce57a0", "ImagePullBackOff"),
        )
    )

    report = inspect_job_blockers(job_id="333", runner=runner)

    assert [blocker.pod for blocker in report.blockers] == ["train-333-64ce57a0-head"]
    # A bare label selector, filtered client-side by the job id component.
    assert f"{CLUSTER_LABEL}" in runner.seen["cmd"]  # type: ignore[attr-defined]


def test_a_job_id_must_match_a_whole_label_component() -> None:
    # Job 3 must not match `train-333-abc`.
    runner = _runner(_pods(_labelled_pod("train-333-64ce57a0", "ImagePullBackOff")))

    report = inspect_job_blockers(job_id="3", runner=runner)

    assert report.blockers == []
    assert "nothing has been scheduled yet" in report.error


def test_no_cluster_and_no_job_id_is_an_error() -> None:
    report = inspect_job_blockers()

    assert "no cluster name or job id" in report.error


def test_the_lookup_is_not_limited_to_the_context_default_namespace() -> None:
    # SkyPilot's namespace is configurable, so a default-namespace-only query
    # would silently report a healthy job.
    runner = _runner(_pods())

    inspect_job_blockers(job_id="333", runner=runner)

    assert "--all-namespaces" in runner.seen["cmd"]  # type: ignore[attr-defined]


def test_an_explicit_namespace_is_honored() -> None:
    runner = _runner(_pods())

    inspect_job_blockers(cluster_name="sky-abc", namespace="sky", runner=runner)

    cmd = runner.seen["cmd"]  # type: ignore[attr-defined]
    assert "-n" in cmd and "sky" in cmd
    assert "--all-namespaces" not in cmd
