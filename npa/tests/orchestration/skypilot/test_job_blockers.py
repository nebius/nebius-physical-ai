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
    MANAGED_JOB_ID_ANNOTATION,
    MANAGED_JOB_NAME_ANNOTATION,
    classify_pending_reason,
    inspect_job_blockers,
    persistent_volume_claim_names,
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
    """A kubectl stub that records every call.

    An empty pod list now triggers a second `kubectl get nodes` (to tell "no pods
    yet" from "the nodes are gone"), so tests must assert on the call they mean.
    """

    seen: dict[str, list[str]] = {}
    calls: list[list[str]] = []

    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        calls.append(list(cmd))
        seen["cmd"] = list(cmd)
        if "nodes" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"items": []}', stderr=""
            )
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr=stderr
        )

    run.seen = seen  # type: ignore[attr-defined]
    run.calls = calls  # type: ignore[attr-defined]
    return run


def _pod_call(runner) -> list[str]:  # noqa: ANN001 - test helper
    return next(cmd for cmd in runner.calls if "pods" in cmd)


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

    cmd = _pod_call(runner)
    assert f"{CLUSTER_LABEL}=sky-abc" in cmd
    assert "--context" in cmd and "npa-cluster" in cmd


def test_explicit_controller_kubeconfig_and_environment_reach_kubectl(
    tmp_path,
) -> None:
    kubeconfig = tmp_path / "controller" / "home" / ".kube" / "config"
    environment = {"HOME": str(kubeconfig.parents[2]), "KUBECONFIG": str(kubeconfig)}
    calls = []

    def runner(cmd, **kwargs):  # noqa: ANN001, ANN202 - subprocess test double
        calls.append((list(cmd), kwargs.get("env")))
        return subprocess.CompletedProcess(cmd, 0, stdout=_pods(), stderr="")

    inspect_job_blockers(
        cluster_name="sky-abc",
        kubeconfig=kubeconfig,
        environment=environment,
        runner=runner,
    )

    command, actual_environment = calls[0]
    assert command[:3] == ["kubectl", "--kubeconfig", str(kubeconfig)]
    assert actual_environment == environment


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
        _pods(
            {"metadata": {"name": "sky-abc-worker-0"}, "status": {"phase": "Running"}}
        )
    )

    report = inspect_job_blockers(cluster_name="sky-abc", runner=runner)

    assert report.blocked is False
    assert report.render() == "blockers: none found"


def test_container_creating_is_progress_not_a_blocker() -> None:
    runner = _runner(_pods(_waiting_pod("sky-abc-worker-0", "ContainerCreating")))

    report = inspect_job_blockers(cluster_name="sky-abc", runner=runner)

    assert report.blocked is False


def test_pod_initializing_is_progress_not_an_init_container_failure() -> None:
    # A main container waiting with reason PodInitializing means init containers
    # completed and the main container is starting -- normal progress, not the
    # fatal INIT_CONTAINER_FAILED a substring match used to manufacture.
    runner = _runner(_pods(_waiting_pod("sky-abc-worker-0", "PodInitializing")))

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


def _pvc_runner(*, claims: list[dict], events: list[dict]):
    calls: list[list[str]] = []

    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        calls.append(list(cmd))
        if "pods" in cmd:
            payload = {"items": []}
        elif "pvc" in cmd:
            payload = {"items": claims}
        elif "events" in cmd:
            payload = {"items": events}
        else:
            payload = {"items": []}
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        )

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _claim(
    name: str,
    *,
    namespace: str = "jobs",
    uid: str = "claim-uid",
    phase: str = "Pending",
    deleting: bool = False,
) -> dict:
    metadata = {"name": name, "namespace": namespace, "uid": uid}
    if deleting:
        metadata["deletionTimestamp"] = "2026-09-22T12:00:00Z"
    return {"metadata": metadata, "status": {"phase": phase}}


def _claim_event(
    name: str,
    *,
    namespace: str = "jobs",
    uid: str = "claim-uid",
    timestamp: str = "2026-09-22T12:01:00Z",
    message: str = "failed to provision volume: disk quota exceeded",
) -> dict:
    return {
        "metadata": {"namespace": namespace, "creationTimestamp": timestamp},
        "involvedObject": {
            "kind": "PersistentVolumeClaim",
            "name": name,
            "namespace": namespace,
            "uid": uid,
        },
        "type": "Warning",
        "reason": "ProvisioningFailed",
        "message": message,
        "eventTime": timestamp,
    }


def test_no_pod_reports_current_exact_claim_quota_event() -> None:
    runner = _pvc_runner(
        claims=[_claim("run-workspace")],
        events=[_claim_event("run-workspace")],
    )

    report = inspect_job_blockers(
        job_id="2",
        claim_names=("run-workspace",),
        event_not_before="2026-09-22T12:00:00Z",
        runner=runner,
    )

    assert report.error == ""
    assert len(report.blockers) == 1
    blocker = report.blockers[0]
    assert blocker.pod == "pvc/run-workspace"
    assert blocker.reason_code == "STORAGE_QUOTA_EXCEEDED"
    assert blocker.source == "kubernetes_pvc_event"
    assert blocker.namespace == "jobs"
    assert blocker.resource_uid == "claim-uid"
    assert blocker.temporally_bound is True
    assert "automatic retry is disabled" in report.remedy()
    pvc_call = next(cmd for cmd in runner.calls if "pvc" in cmd)
    assert "metadata.name=run-workspace" in pvc_call
    event_call = next(cmd for cmd in runner.calls if "events" in cmd)
    assert "involvedObject.uid=claim-uid" in event_call
    assert "-n" in event_call and "jobs" in event_call
    assert "--all-namespaces" not in event_call


@pytest.mark.parametrize(
    "message",
    [
        "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
        "rpc error: code = Unavailable desc = storage service unavailable",
        "request rate limit reached; retry after backoff",
        "failed to query storage quota: rpc error: code = DeadlineExceeded; timed out",
    ],
)
def test_retryable_provisioning_warning_does_not_admit_cancellation(
    message: str,
) -> None:
    runner = _pvc_runner(
        claims=[_claim("run-workspace")],
        events=[_claim_event("run-workspace", message=message)],
    )

    report = inspect_job_blockers(
        job_id="2",
        claim_names=("run-workspace",),
        event_not_before="2026-09-22T12:00:00Z",
        runner=runner,
    )

    assert classify_pending_reason("ProvisioningFailed", message) == "STORAGE_PENDING"
    assert report.blockers == []
    assert "no pods found" in report.error


def test_no_pod_claim_diagnostic_does_not_require_a_pod_task_annotation() -> None:
    runner = _pvc_runner(
        claims=[_claim("run-workspace")],
        events=[_claim_event("run-workspace")],
    )

    report = inspect_job_blockers(
        job_id="2",
        controller_user_id="isolated-owner",
        expected_task_names=(),
        claim_names=("run-workspace",),
        runner=runner,
    )

    assert report.blockers[0].reason_code == "STORAGE_QUOTA_EXCEEDED"


def test_stale_claim_event_does_not_replace_no_pod_unknown() -> None:
    runner = _pvc_runner(
        claims=[_claim("run-workspace")],
        events=[_claim_event("run-workspace", timestamp="2026-09-22T11:59:59Z")],
    )

    report = inspect_job_blockers(
        job_id="2",
        claim_names=("run-workspace",),
        event_not_before="2026-09-22T12:00:00Z",
        runner=runner,
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND"


def test_repeated_claim_event_uses_latest_observation_for_attempt_binding() -> None:
    event = _claim_event("run-workspace", timestamp="2026-09-22T11:00:00Z")
    event["series"] = {"lastObservedTime": "2026-09-22T12:01:00Z"}
    runner = _pvc_runner(claims=[_claim("run-workspace")], events=[event])

    report = inspect_job_blockers(
        job_id="2",
        claim_names=("run-workspace",),
        event_not_before="2026-09-22T12:00:00Z",
        runner=runner,
    )

    assert report.blockers[0].event_timestamp == "2026-09-22T12:01:00Z"
    assert report.blockers[0].temporally_bound is True


@pytest.mark.parametrize(
    "reason", ["ExternalProvisioning", "Provisioning", "WaitForFirstConsumer"]
)
def test_normal_claim_wait_events_do_not_cancel(reason: str) -> None:
    event = _claim_event("run-workspace")
    event["reason"] = reason
    event["message"] = "waiting for a volume to be created"
    runner = _pvc_runner(claims=[_claim("run-workspace")], events=[event])

    report = inspect_job_blockers(
        job_id="2", claim_names=("run-workspace",), runner=runner
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND"


@pytest.mark.parametrize(
    "claim",
    [
        _claim("run-workspace", phase="Bound"),
        _claim("run-workspace", deleting=True),
    ],
)
def test_historical_failure_is_ignored_for_nonpending_claim(claim: dict) -> None:
    runner = _pvc_runner(
        claims=[claim],
        events=[_claim_event("run-workspace")],
    )

    report = inspect_job_blockers(
        job_id="2", claim_names=("run-workspace",), runner=runner
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND"
    assert not any("events" in cmd for cmd in runner.calls)


def test_duplicate_claim_name_across_namespaces_is_ambiguous() -> None:
    runner = _pvc_runner(
        claims=[
            _claim("run-workspace", namespace="jobs-a", uid="a"),
            _claim("run-workspace", namespace="jobs-b", uid="b"),
        ],
        events=[_claim_event("run-workspace")],
    )

    report = inspect_job_blockers(
        job_id="2", claim_names=("run-workspace",), runner=runner
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND"
    assert not any("events" in cmd for cmd in runner.calls)


def test_current_claim_uid_must_match_event_uid() -> None:
    runner = _pvc_runner(
        claims=[_claim("run-workspace", uid="current")],
        events=[_claim_event("run-workspace", uid="replaced")],
    )

    report = inspect_job_blockers(
        job_id="2", claim_names=("run-workspace",), runner=runner
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND"


def test_known_namespace_never_lists_claims_across_namespaces() -> None:
    runner = _pvc_runner(
        claims=[_claim("run-workspace")],
        events=[_claim_event("run-workspace")],
    )

    report = inspect_job_blockers(
        job_id="2",
        namespace="jobs",
        claim_names=("run-workspace",),
        runner=runner,
    )

    assert report.blockers[0].reason_code == "STORAGE_QUOTA_EXCEEDED"
    pvc_call = next(cmd for cmd in runner.calls if "pvc" in cmd)
    assert "-n" in pvc_call and "jobs" in pvc_call
    assert "--all-namespaces" not in pvc_call


def test_rendered_claim_extraction_is_narrow_and_validated() -> None:
    profile = {
        "kubernetes": {
            "pod_config": {
                "spec": {
                    "volumes": [
                        {
                            "name": "workspace",
                            "persistentVolumeClaim": {"claimName": "run-workspace"},
                        },
                        {
                            "name": "duplicate",
                            "persistentVolumeClaim": {"claimName": "run-workspace"},
                        },
                        {
                            "name": "unsafe",
                            "persistentVolumeClaim": {"claimName": "../other"},
                        },
                        {
                            "name": "empty-label",
                            "persistentVolumeClaim": {"claimName": "a..b"},
                        },
                        {
                            "name": "whitespace",
                            "persistentVolumeClaim": {"claimName": " other-claim "},
                        },
                        {
                            "name": "leading-hyphen",
                            "persistentVolumeClaim": {"claimName": "-other"},
                        },
                        {"name": "memory", "emptyDir": {}},
                    ]
                }
            }
        }
    }

    assert persistent_volume_claim_names(profile) == ("run-workspace",)


def test_missing_kubectl_is_reported() -> None:
    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        raise OSError("No such file or directory: 'kubectl'")

    report = inspect_job_blockers(cluster_name="sky-abc", runner=run)

    assert "could not run kubectl" in report.error


@pytest.mark.parametrize(
    "reason",
    [
        "ErrImagePull",
        "InvalidImageName",
        "CreateContainerConfigError",
        "CrashLoopBackOff",
    ],
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
# provisioned -- which is exactly the job worth diagnosing. Legacy callers use
# the job id embedded in the label. Isolated workflow callers use SkyPilot's
# exact managed-job annotations plus the controller owner.


def _labelled_pod(
    label: str,
    reason: str,
    *,
    managed_job_id: str = "",
    managed_job_name: str = "",
) -> dict:
    pod = _waiting_pod(f"{label}-head", reason)
    pod["metadata"]["labels"] = {CLUSTER_LABEL: label}
    if managed_job_id or managed_job_name:
        pod["metadata"]["annotations"] = {
            MANAGED_JOB_ID_ANNOTATION: managed_job_id,
            MANAGED_JOB_NAME_ANNOTATION: managed_job_name,
        }
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
    assert f"{CLUSTER_LABEL}" in _pod_call(runner)


def test_workflow_task_and_isolated_controller_find_cpu_stage_pod() -> None:
    task = "focused-tokenizer-discovery-cpu-r1-01-discover"
    owner = "npa-fixture001"
    runner = _runner(
        _pods(
            _labelled_pod(
                f"focused-tokenizer-discov-27-{owner}",
                "ImagePullBackOff",
                managed_job_id="1",
                managed_job_name=task,
            )
        )
    )

    report = inspect_job_blockers(
        job_id="1",
        expected_task_names=[task],
        controller_user_id=owner,
        runner=runner,
    )

    assert report.error == ""
    assert [row.pod for row in report.blockers] == [
        f"focused-tokenizer-discov-27-{owner}-head"
    ]


def test_same_owner_and_truncated_prefix_need_exact_task_annotation() -> None:
    task = "focused-tokenizer-discovery-cpu-r1-01-discover"
    owner = "npa-fixture001"
    runner = _runner(
        _pods(
            _labelled_pod(
                f"focused-tokenizer-discov-27-{owner}",
                "ImagePullBackOff",
                managed_job_id="1",
                managed_job_name="focused-tokenizer-discovery-other-stage",
            )
        )
    )

    report = inspect_job_blockers(
        job_id="1",
        expected_task_names=[task],
        controller_user_id=owner,
        runner=runner,
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND"


def test_exact_task_annotation_from_another_controller_is_rejected() -> None:
    task = "focused-tokenizer-discovery-cpu-r1-01-discover"
    runner = _runner(
        _pods(
            _labelled_pod(
                "focused-tokenizer-discov-27-npa-fixture002",
                "ImagePullBackOff",
                managed_job_id="1",
                managed_job_name=task,
            )
        )
    )

    report = inspect_job_blockers(
        job_id="1",
        expected_task_names=[task],
        controller_user_id="npa-fixture001",
        runner=runner,
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_PODS_NOT_FOUND"


def test_isolated_controller_without_queue_task_identity_fails_closed() -> None:
    task = "focused-tokenizer-discovery-cpu-r1-01-discover"
    owner = "npa-fixture001"
    runner = _runner(
        _pods(
            _labelled_pod(
                f"focused-tokenizer-discov-27-{owner}",
                "ImagePullBackOff",
                managed_job_id="1",
                managed_job_name=task,
            )
        )
    )

    report = inspect_job_blockers(
        job_id="1",
        controller_user_id=owner,
        runner=runner,
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_POD_IDENTITY_INCOMPLETE"


def test_ambiguous_owned_workflow_pods_fail_closed() -> None:
    task = "focused-tokenizer-discovery-cpu-r1-01-discover"
    owner = "npa-fixture001"
    runner = _runner(
        _pods(
            _labelled_pod(
                f"focused-tokenizer-discov-27-{owner}",
                "ErrImagePull",
                managed_job_id="1",
                managed_job_name=task,
            ),
            _labelled_pod(
                f"focused-tokenizer-discov-28-{owner}",
                "ErrImagePull",
                managed_job_id="1",
                managed_job_name=task,
            ),
        )
    )

    report = inspect_job_blockers(
        job_id="1",
        expected_task_names=[task],
        controller_user_id=owner,
        runner=runner,
    )

    assert report.blockers == []
    assert report.error_code == "KUBERNETES_POD_IDENTITY_AMBIGUOUS"


def test_multiple_pods_in_one_exact_owned_cluster_are_not_ambiguous() -> None:
    task = "focused-tokenizer-discovery-cpu-r1-01-discover"
    owner = "npa-fixture001"
    cluster = f"focused-tokenizer-discov-27-{owner}"
    head = _labelled_pod(
        cluster,
        "ErrImagePull",
        managed_job_id="1",
        managed_job_name=task,
    )
    worker = _labelled_pod(
        cluster,
        "ImagePullBackOff",
        managed_job_id="1",
        managed_job_name=task,
    )
    worker["metadata"]["name"] = f"{cluster}-worker"

    report = inspect_job_blockers(
        job_id="1",
        expected_task_names=[task],
        controller_user_id=owner,
        runner=_runner(_pods(head, worker)),
    )

    assert report.error == ""
    assert {blocker.pod for blocker in report.blockers} == {
        f"{cluster}-head",
        f"{cluster}-worker",
    }


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

    assert "--all-namespaces" in _pod_call(runner)


def test_an_explicit_namespace_is_honored() -> None:
    runner = _runner(_pods())

    inspect_job_blockers(cluster_name="sky-abc", namespace="sky", runner=runner)

    cmd = _pod_call(runner)
    assert "-n" in cmd and "sky" in cmd
    assert "--all-namespaces" not in cmd


# --- the nodes went away, which is not the same as "still starting" -----------


def _node(name: str, ready: str, reason: str = "") -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "conditions": [{"type": "Ready", "status": ready, "reason": reason}]
        },
    }


def _pods_then_nodes(nodes: dict, *, assigned_nodes: tuple[str, ...] = ()):
    def run(cmd, **kwargs):  # noqa: ANN001 - test stub
        if "nodes" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(nodes), stderr=""
            )
        pods = _pods(
            *(
                {
                    "metadata": {"name": f"worker-{index}"},
                    "spec": {"nodeName": name},
                    "status": {"phase": "Pending"},
                }
                for index, name in enumerate(assigned_nodes)
            )
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=pods, stderr="")

    return run


def test_a_pending_job_whose_nodes_were_reclaimed_says_so() -> None:
    """Preempted GPU nodes leave no pod-level reason at all.

    Both RTX6000 instances were reclaimed mid-run; the job sat PENDING and
    `sky jobs queue` reported nothing an operator could act on.
    """

    runner = _pods_then_nodes(
        {
            "items": [
                _node("gpu-0", "Unknown", "NodeStatusUnknown"),
                _node("gpu-1", "Unknown", "NodeStatusUnknown"),
                _node("cpu-0", "True"),
            ]
        },
        assigned_nodes=("gpu-0", "gpu-1"),
    )

    report = inspect_job_blockers(job_id="1", cluster_name="sky-abc", runner=runner)

    assert report.blocked is True
    assert report.unready_nodes == [
        "gpu-0 (NodeStatusUnknown)",
        "gpu-1 (NodeStatusUnknown)",
    ]
    assert "reclaimed without warning" in report.remedy()
    assert "--on-demand" in report.remedy()
    rendered = report.render()
    assert "2 node(s) not Ready" in rendered


def test_healthy_nodes_and_no_blocked_pods_stays_quiet() -> None:
    runner = _pods_then_nodes(
        {"items": [_node("cpu-0", "True"), _node("gpu-0", "True")]}
    )

    report = inspect_job_blockers(job_id="1", cluster_name="sky-abc", runner=runner)

    assert report.blocked is False
    assert report.render() == "blockers: none found"


def test_unready_node_unrelated_to_the_exact_job_is_ignored() -> None:
    runner = _pods_then_nodes(
        {
            "items": [
                _node("assigned-ready", "True"),
                _node("unrelated-lost", "Unknown", "NodeStatusUnknown"),
            ]
        },
        assigned_nodes=("assigned-ready",),
    )

    report = inspect_job_blockers(job_id="1", cluster_name="sky-abc", runner=runner)

    assert report.blocked is False
    assert report.unready_nodes == []


def test_a_pod_level_reason_still_wins_over_the_node_check() -> None:
    # A pod that cannot pull is a better answer than "a node is down elsewhere".
    runner = _runner(_pods(_waiting_pod("worker-0", "ImagePullBackOff")))

    report = inspect_job_blockers(job_id="1", cluster_name="sky-abc", runner=runner)

    assert report.unready_nodes == []
    assert "retries this forever" in report.remedy()


@pytest.mark.parametrize(
    ("reason", "message", "source", "code"),
    [
        (
            "Unschedulable",
            "0/3 nodes: insufficient nvidia.com/gpu",
            "scheduler",
            "CAPACITY_OR_QUOTA",
        ),
        (
            "Unschedulable",
            "cloud capacity quota exhausted",
            "scheduler",
            "CAPACITY_OR_QUOTA",
        ),
        ("Unschedulable", "node selector did not match", "scheduler", "UNSCHEDULABLE"),
        ("ImagePullBackOff", "401 unauthorized", "container", "IMAGE_PULL_AUTH"),
        ("ErrImagePull", "manifest unknown: not found", "container", "IMAGE_NOT_FOUND"),
        ("CrashLoopBackOff", "init setup failed", "init", "INIT_CONTAINER_FAILED"),
        (
            "PodInitializing",
            "",
            "container",
            "PENDING_UNKNOWN",
        ),
        ("CrashLoopBackOff", "worker exited", "container", "CONTAINER_CRASH"),
        ("BackOff", "controller retry backoff", "event", "CONTROLLER_BACKOFF"),
        ("FailedMount", "persistentvolumeclaim is pending", "event", "STORAGE_PENDING"),
        (
            "ProvisioningFailed",
            "failed to provision volume: disk quota exceeded",
            "pvc_event",
            "STORAGE_QUOTA_EXCEEDED",
        ),
        (
            "ProvisioningFailed",
            "insufficient storage capacity",
            "pvc_event",
            "STORAGE_CAPACITY_UNAVAILABLE",
        ),
        (
            "ProvisioningFailed",
            "storage class configuration rejected the request",
            "pvc_event",
            "STORAGE_PROVISIONING_FAILED",
        ),
        (
            "ExternalProvisioning",
            "waiting for a volume to be created",
            "pvc_event",
            "PENDING_UNKNOWN",
        ),
    ],
)
def test_pending_reason_codes_are_stable(
    reason: str, message: str, source: str, code: str
) -> None:
    assert classify_pending_reason(reason, message, source=source) == code


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            "dial tcp: lookup kubernetes.example.invalid: no such host",
            "KUBERNETES_DNS",
        ),
        ("pods is forbidden: RBAC denied", "KUBERNETES_RBAC"),
        ("401 Unauthorized", "KUBERNETES_AUTHENTICATION"),
    ],
)
def test_kubernetes_diagnostic_failures_are_typed_and_sanitized(
    error: str, code: str
) -> None:
    runner = _runner(
        "",
        returncode=1,
        stderr=f"{error}; authorization=synthetic-secret",
    )

    report = inspect_job_blockers(cluster_name="sky-synthetic", runner=runner)

    assert report.error_code == code
    assert "synthetic-secret" not in report.error
    assert report.observed_at


@pytest.mark.parametrize("reason", ["Unschedulable", "FailedScheduling"])
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "0/4 nodes: 1 Insufficient cpu, 2 Insufficient nvidia.com/gpu, "
            "2 node(s) did not match Pod's node affinity/selector",
            "CAPACITY_OR_QUOTA",
        ),
        ("Insufficient nvidia.com/gpu", "CAPACITY_OR_QUOTA"),
        ("Insufficient cpu", "CAPACITY_OR_QUOTA"),
        ("GPU capacity temporarily unavailable", "CAPACITY_OR_QUOTA"),
        ("GPU quota exceeded", "CAPACITY_OR_QUOTA"),
        ("no nodes match requested GPU accelerator", "ACCELERATOR_MISMATCH"),
        ("GPU accelerator label did not match", "ACCELERATOR_MISMATCH"),
        ("persistentvolumeclaim has volume node affinity conflict", "STORAGE_PENDING"),
        ("no nodes are available", "CAPACITY_OR_QUOTA"),
        ("node selector did not match", "UNSCHEDULABLE"),
    ],
)
def test_scheduler_shortage_is_distinct_from_accelerator_mismatch(
    reason: str,
    message: str,
    expected: str,
) -> None:
    assert classify_pending_reason(reason, message, source="scheduler") == expected
