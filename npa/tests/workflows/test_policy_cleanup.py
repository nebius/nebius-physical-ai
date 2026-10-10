"""Check artifact verification, namespace ownership and non-masking cleanup failures."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from npa.workflows.policy_training import (
    cli,
    public_vla_cleanup as cleanup,
    public_vla_collect as collect,
)
from npa.workflows.policy_training.public_vla_launch import IMAGE


def _collection(root):
    values = {
        "completed.json": {"training_executed": True, "evaluation_executed": True},
        "selection.json": {"checkpoint_sha256": hashlib.sha256(b"weights").hexdigest()},
        "exported-policy/manifest.json": {},
        "report/evidence.json": {},
    }
    for relative, data in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    (root / "report/index.html").write_text("report")
    weights = root / "exported-policy/policy/model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")


def test_cleanup_requires_verified_collected_weights(tmp_path):
    _collection(tmp_path)
    cleanup._verify_collection(tmp_path)
    (tmp_path / "exported-policy/policy/model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        cleanup._verify_collection(tmp_path)


def _job(*, active=None, condition="Complete", name="pipeline"):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        status=SimpleNamespace(
            active=active, conditions=[SimpleNamespace(type=condition, status="True")]
        ),
        spec=SimpleNamespace(
            template=SimpleNamespace(
                spec=SimpleNamespace(containers=[SimpleNamespace(image=IMAGE)])
            )
        ),
    )


@pytest.mark.parametrize(
    "job", [_job(active=1), _job(condition="Failed"), _job(name="other")]
)
def test_cleanup_refuses_active_failed_or_unrelated_workloads(job):
    api = SimpleNamespace(
        list_namespaced_job=lambda namespace: SimpleNamespace(items=[job])
    )
    with pytest.raises(ValueError):
        cleanup._require_completed_job(api, {"namespace": "npa-vla-0123456789ab"})


def test_cleanup_uses_namespace_uid_precondition_and_waits_for_absence(
    tmp_path, monkeypatch
):
    from kubernetes import client
    from kubernetes.client.exceptions import ApiException

    receipt = {"namespace": "npa-vla-0123456789ab", "namespace_uid": "test-uid"}
    namespace = SimpleNamespace(
        metadata=SimpleNamespace(
            uid="test-uid",
            labels={"npa.nebius.ai/public-vla-run": receipt["namespace"]},
        )
    )
    observations = iter([namespace, ApiException(status=404)])
    deletes = []

    def read(name):
        value = next(observations)
        if isinstance(value, Exception):
            raise value
        return value

    core = SimpleNamespace(
        read_namespace=read, delete_namespace=lambda name, body: deletes.append(body)
    )
    monkeypatch.setattr(client, "CoreV1Api", lambda api: core)
    monkeypatch.setattr(
        client,
        "BatchV1Api",
        lambda api: SimpleNamespace(
            list_namespaced_job=lambda name: SimpleNamespace(items=[_job()])
        ),
    )
    cleanup._delete_owned_namespace(object(), receipt, tmp_path)
    assert deletes[0].preconditions.uid == "test-uid"
    assert (
        json.loads((tmp_path / "cleanup.json").read_text())["namespace_deleted"] is True
    )


def test_cleanup_rejects_namespace_name_reuse(tmp_path, monkeypatch):
    from kubernetes import client

    core = SimpleNamespace(
        read_namespace=lambda name: SimpleNamespace(
            metadata=SimpleNamespace(uid="replacement", labels={})
        )
    )
    monkeypatch.setattr(client, "CoreV1Api", lambda api: core)
    with pytest.raises(ValueError, match="ownership changed"):
        cleanup._delete_owned_namespace(
            object(), {"namespace": "example", "namespace_uid": "original"}, tmp_path
        )


def test_reader_cleanup_failure_does_not_mask_download_failure(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "wait" in command:
            raise subprocess.CalledProcessError(8, command)
        if "delete" in command:
            raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(collect.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        collect._download(["kubectl"], {"namespace": "example"}, tmp_path)
    assert error.value.returncode == 8
    assert "delete" in calls[-1]
    assert "Cleanup failed" in capsys.readouterr().err


def test_stage_failure_retains_private_traceback(tmp_path, monkeypatch, capsys):
    output = str(tmp_path / "result.json")
    monkeypatch.setattr(
        "sys.argv",
        [
            "policy",
            "split",
            "--input-uri",
            "input",
            "--output-uri",
            output,
            "--seed",
            "test",
        ],
    )

    def fail(**kwargs):
        raise ValueError("PRIVATE_SENTINEL")

    with pytest.raises(SystemExit) as error:
        cli._run({"split": fail})
    assert "PRIVATE_SENTINEL" not in str(error.value) + capsys.readouterr().err
    assert "PRIVATE_SENTINEL" in next(tmp_path.rglob("diagnostics/*.json")).read_text()
