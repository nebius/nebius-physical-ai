from __future__ import annotations

import importlib
import json
from pathlib import Path
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


def test_owner_receipt_binds_exact_running_pod_and_writes_private_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = {"NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR": str(evidence)}
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def fake_kubectl(
        _env: dict[str, str],
        _namespace: str,
        *args: str,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, stdin))
        if args[:2] == ("get", "pods"):
            payload = {"items": [_pod(image_id=f"containerd://{DIGEST}")]}
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
    env = {"NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR": str(evidence)}

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


def test_failed_sky_down_is_not_accepted_while_exact_pod_remains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    env = {"NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR": str(evidence)}
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, "", "cleanup failed")

    monkeypatch.setattr(live, "resolve_skypilot_bin", lambda: "/usr/bin/sky")
    monkeypatch.setattr(live.subprocess, "run", fake_run)
    monkeypatch.setattr(live, "_exact_gymnasium_run_pods", lambda *args, **kwargs: [{}])

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
    assert all(path.stat().st_mode & 0o077 == 0 for path in evidence.iterdir())
