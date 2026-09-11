from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa"))
live = importlib.import_module("tests.e2e.test_byof_onboarding_live_e2e")


DIGEST = "sha256:" + "1" * 64
IMAGE = f"registry.example/operator-private/gymnasium-robotics@{DIGEST}"
RUN_ID = "gymnasium-robotics-unit-receipt"
NAMESPACE = "operator-private"
NODE_NAME = "reserved-rtx-node"
NODE_UID = "reserved-rtx-node-uid"
PROVIDER_GROUP = "reserved-provider-group"
CONTEXT = "child-context"


class _RunningProcess:
    def poll(self) -> None:
        return None


def _pod(*, image_id: str) -> dict[str, object]:
    return {
        "metadata": {
            "name": "sky-gymnasium-unit",
            "namespace": NAMESPACE,
            "uid": "unit-pod-uid",
            "labels": {"parent": "skypilot"},
            "annotations": {"skypilot-cluster-name": RUN_ID},
        },
        "spec": {
            "nodeName": NODE_NAME,
            "containers": [
                {
                    "name": "task",
                    "image": IMAGE,
                    "resources": {
                        "requests": {"nvidia.com/gpu": "1"},
                        "limits": {"nvidia.com/gpu": "1"},
                    },
                }
            ]
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "task",
                    "imageID": image_id,
                    "state": {"running": {"startedAt": "2026-09-11T00:00:00Z"}},
                }
            ],
        },
    }


def _receipt_env(evidence: Path) -> dict[str, str]:
    return {
        "NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR": str(evidence),
        "NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_NAME": NODE_NAME,
    }


def test_owner_receipt_binds_exact_running_pod_and_writes_private_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = _receipt_env(evidence)
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, stdin))
        if args[:2] == ("get", "pods"):
            pod = _pod(image_id=f"containerd://{DIGEST}")
            pod["spec"]["containers"].append(
                {"name": "same-image-helper", "image": IMAGE, "resources": {}}
            )
            payload = {"items": [pod]}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        assert args[0] == "exec"
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    receipt = live._gymnasium_pod_image_receipt(
        _RunningProcess(),
        env=env,
        namespace=NAMESPACE,
        run_id=RUN_ID,
        image=IMAGE,
    )

    assert receipt["expected_digest"] == DIGEST
    assert receipt["observed_digest"] == DIGEST
    assert receipt["node_name"] == NODE_NAME
    assert calls[1][0][0] == "exec"
    assert json.loads(calls[1][1] or "{}") == receipt
    receipt_path = evidence / f"{RUN_ID}-pod-image-receipt.json"
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert receipt_path.stat().st_mode & 0o077 == 0


def test_owner_receipt_refuses_a_different_runtime_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = _receipt_env(evidence)

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del stdin
        assert args[:2] == ("get", "pods")
        other = "sha256:" + "2" * 64
        payload = {"items": [_pod(image_id=f"containerd://{other}")]}
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    with pytest.raises(AssertionError, match="imageID differs"):
        live._gymnasium_pod_image_receipt(
            _RunningProcess(),
            env=env,
            namespace=NAMESPACE,
            run_id=RUN_ID,
            image=IMAGE,
        )
    assert not list(evidence.iterdir())


@pytest.mark.parametrize("observed_node", ["", "another-rtx-node"])
def test_owner_receipt_refuses_missing_or_mismatched_node_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, observed_node: str
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = _receipt_env(evidence)
    pod = _pod(image_id=f"containerd://{DIGEST}")
    pod["spec"]["nodeName"] = observed_node

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del stdin
        assert args[:2] == ("get", "pods")
        return subprocess.CompletedProcess(args, 0, json.dumps({"items": [pod]}), "")

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    with pytest.raises(AssertionError, match="manager-assigned node"):
        live._gymnasium_pod_image_receipt(
            _RunningProcess(),
            env=env,
            namespace=NAMESPACE,
            run_id=RUN_ID,
            image=IMAGE,
        )
    assert not list(evidence.iterdir())


def test_owner_permission_preflight_matches_list_and_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del stdin
        observed.append(args)
        return subprocess.CompletedProcess(args, 0, "yes\n", "")

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    live._require_gymnasium_owner_receipt_access({}, namespace=NAMESPACE)
    assert observed == [
        ("auth", "can-i", "list", "pods"),
        ("auth", "can-i", "create", "pods/exec"),
    ]


def test_scheduling_contract_requires_one_exact_named_ready_rtx_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "sky.yaml"
    config.write_text(
        "kubernetes:\n"
        "  allowed_contexts:\n"
        "    - child-context\n"
        "  allowed_nodes:\n"
        "    names:\n"
        "      - reserved-rtx-node\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    kubeconfig = tmp_path / "kubeconfig.yaml"
    kubeconfig.write_text(
        "current-context: child-context\n"
        "contexts:\n"
        "  - name: child-context\n"
        "    context:\n"
        "      namespace: operator-private\n",
        encoding="utf-8",
    )
    kubeconfig.chmod(0o600)
    env = {
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": CONTEXT,
        "NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_NAME": NODE_NAME,
        "NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_UID": NODE_UID,
        "NPA_BYOF_GYMNASIUM_ROBOTICS_PROVIDER_NODE_GROUP_ID": PROVIDER_GROUP,
    }
    node = {
        "metadata": {
            "name": NODE_NAME,
            "uid": NODE_UID,
            "labels": {
                "provider.example/node-group-id": PROVIDER_GROUP,
                "nvidia.com/gpu.product": (
                    "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
                ),
            },
        },
        "status": {
            "allocatable": {"nvidia.com/gpu": "1"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }

    def fake_kubectl(
        _env: dict[str, str],
        namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert stdin is None
        assert namespace == NAMESPACE
        assert args == ("get", "node", NODE_NAME, "--output", "json")
        return subprocess.CompletedProcess(args, 0, json.dumps(node), "")

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    live._require_gymnasium_scheduling_contract(
        env, namespace=NAMESPACE, config_path=str(config)
    )


def test_scheduling_contract_rejects_legacy_allowed_nodes_list(
    tmp_path: Path,
) -> None:
    config = tmp_path / "sky.yaml"
    config.write_text(
        "kubernetes:\n  allowed_nodes:\n    - reserved-rtx-node\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    with pytest.raises(AssertionError, match="supported names mapping"):
        live._require_gymnasium_scheduling_contract(
            {}, namespace=NAMESPACE, config_path=str(config)
        )


def test_owner_receipt_scopes_gpu_request_to_matching_task_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = _receipt_env(evidence)
    pod = _pod(image_id=f"containerd://{DIGEST}")
    pod["spec"]["containers"][0]["resources"] = {
        "requests": {"nvidia.com/gpu": "0"},
        "limits": {"nvidia.com/gpu": "0"},
    }
    pod["spec"]["containers"].append(
        {
            "name": "sidecar",
            "image": "registry.example/operator-private/sidecar:latest",
            "resources": {
                "requests": {"nvidia.com/gpu": "1"},
                "limits": {"nvidia.com/gpu": "1"},
            },
        }
    )

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del stdin
        assert args[:2] == ("get", "pods")
        return subprocess.CompletedProcess(args, 0, json.dumps({"items": [pod]}), "")

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    with pytest.raises(AssertionError, match="one immutable one-GPU task container"):
        live._gymnasium_pod_image_receipt(
            _RunningProcess(),
            env=env,
            namespace=NAMESPACE,
            run_id=RUN_ID,
            image=IMAGE,
        )


def test_owner_receipt_rejects_gpu_request_from_an_extra_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = _receipt_env(evidence)
    pod = _pod(image_id=f"containerd://{DIGEST}")
    pod["spec"]["containers"].append(
        {
            "name": "sidecar",
            "image": "registry.example/operator-private/sidecar:latest",
            "resources": {
                "requests": {"nvidia.com/gpu": "1"},
                "limits": {"nvidia.com/gpu": "1"},
            },
        }
    )

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del stdin
        assert args[:2] == ("get", "pods")
        return subprocess.CompletedProcess(args, 0, json.dumps({"items": [pod]}), "")

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    with pytest.raises(AssertionError, match="only the exact task container"):
        live._gymnasium_pod_image_receipt(
            _RunningProcess(),
            env=env,
            namespace=NAMESPACE,
            run_id=RUN_ID,
            image=IMAGE,
        )


def test_owner_receipt_poll_has_an_overall_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = _receipt_env(evidence)
    monkeypatch.setattr(live, "GYMNASIUM_RECEIPT_TIMEOUT_SECONDS", 0)
    with pytest.raises(AssertionError, match="timed out waiting"):
        live._gymnasium_pod_image_receipt(
            _RunningProcess(),
            env=env,
            namespace=NAMESPACE,
            run_id=RUN_ID,
            image=IMAGE,
        )


def test_kubectl_calls_have_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(live.subprocess, "run", fake_run)
    live._gymnasium_kubectl(
        {
            "NPA_BYOF_KUBECONFIG": "/operator/kubeconfig",
            "NPA_BYOF_K8S_CONTEXT": "child-context",
        },
        NAMESPACE,
        "get",
        "pods",
    )
    assert observed["timeout"] == live.GYMNASIUM_KUBECTL_TIMEOUT_SECONDS


def test_cleanup_selection_keeps_exact_terminating_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminating = _pod(image_id=f"containerd://{DIGEST}")
    terminating["metadata"]["deletionTimestamp"] = "2026-09-11T00:01:00Z"

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del stdin
        assert args[:2] == ("get", "pods")
        return subprocess.CompletedProcess(
            args, 0, json.dumps({"items": [terminating]}), ""
        )

    monkeypatch.setattr(live, "_gymnasium_kubectl", fake_kubectl)
    assert not live._exact_gymnasium_run_pods({}, namespace=NAMESPACE, run_id=RUN_ID)
    assert live._exact_gymnasium_run_pods(
        {}, namespace=NAMESPACE, run_id=RUN_ID, include_terminating=True
    ) == [terminating]


def test_failed_sky_down_is_not_accepted_while_exact_pod_remains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = {"NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR": str(evidence)}
    commands: list[list[str]] = []

    observed: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 1, "", "cleanup failed")

    monkeypatch.setattr(live, "resolve_skypilot_bin", lambda: "/usr/bin/sky")
    monkeypatch.setattr(live.subprocess, "run", fake_run)
    monkeypatch.setattr(live, "_exact_gymnasium_run_pods", lambda *args, **kwargs: [{}])
    monkeypatch.setattr(live, "GYMNASIUM_CLEANUP_TIMEOUT_SECONDS", 0)

    with pytest.raises(AssertionError, match="Pod remains"):
        live._cleanup_gymnasium_run(
            env,
            namespace=NAMESPACE,
            run_id=RUN_ID,
            config_path="/operator/config.yaml",
            issue_down=True,
        )
    assert commands == [
        [
            "/usr/bin/sky",
            "down",
            "--config",
            "/operator/config.yaml",
            "--yes",
            RUN_ID,
        ]
    ]
    assert observed["timeout"] == live.GYMNASIUM_SKY_DOWN_TIMEOUT_SECONDS
    assert all(path.stat().st_mode & 0o077 == 0 for path in evidence.iterdir())


def test_failed_sky_down_polls_until_terminating_pod_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = {"NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR": str(evidence)}
    observed_pods = iter([[{}], []])

    monkeypatch.setattr(live, "resolve_skypilot_bin", lambda: "/usr/bin/sky")
    monkeypatch.setattr(
        live.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", "cleanup raced deletion"
        ),
    )
    monkeypatch.setattr(
        live,
        "_exact_gymnasium_run_pods",
        lambda *args, **kwargs: next(observed_pods),
    )
    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)

    live._cleanup_gymnasium_run(
        env,
        namespace=NAMESPACE,
        run_id=RUN_ID,
        config_path=None,
        issue_down=True,
    )


def test_runner_termination_escalates_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[signal.Signals] = []

    class StuckProcess:
        pid = 123

        def __init__(self) -> None:
            self.waits = 0

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: int) -> int:
            assert timeout == live.GYMNASIUM_RUNNER_TERM_GRACE_SECONDS
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("runner", timeout)
            return -int(signal.SIGKILL)

    monkeypatch.setattr(live.os, "killpg", lambda _pid, sent: signals.append(sent))
    process = StuckProcess()
    live._terminate_gymnasium_runner(process)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.waits == 2


def test_success_cleanup_escalates_to_active_sky_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[bool] = []

    def fake_cleanup(*args: object, issue_down: bool, **kwargs: object) -> None:
        del args, kwargs
        attempts.append(issue_down)
        if not issue_down:
            raise AssertionError("passive cleanup timed out")

    monkeypatch.setattr(live, "_cleanup_gymnasium_run", fake_cleanup)
    live._cleanup_gymnasium_run_after_success(
        {}, namespace=NAMESPACE, run_id=RUN_ID, config_path="/operator/sky.yaml"
    )
    assert attempts == [False, True]


def test_success_cleanup_precedes_runner_log_decoding() -> None:
    source = inspect.getsource(live.test_live_gymnasium_robotics_exact_digest_capability)
    cleanup = source.index("_cleanup_gymnasium_run_after_success(")
    stdout_decode = source.index("stdout_path.read_text(")
    stderr_decode = source.index("stderr_path.read_text(")
    assert cleanup < stdout_decode < stderr_decode
    assert 'errors="replace"' in source[stdout_decode:]
