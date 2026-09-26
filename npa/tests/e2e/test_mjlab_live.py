"""Real GPU and S3 qualification for MJLab, never a synthetic score test."""

import hashlib
import os
from pathlib import Path
import uuid

import pytest

from npa.clients.storage import StorageClient
from npa.workbench.mjlab import evaluate, export, train
from npa.workbench.mjlab.schemas import EvalRequest, ExportRequest, TrainRequest

pytestmark = pytest.mark.e2e


@pytest.mark.gpu
def test_native_train_eval_export_roundtrip(tmp_path: Path):
    prefix = os.environ.get("NPA_MJLAB_E2E_OUTPUT_PATH", "")
    if not prefix:
        pytest.skip(
            "Set NPA_MJLAB_E2E_OUTPUT_PATH to an authorized disposable S3 prefix"
        )
    root = prefix.rstrip("/") + "/" + uuid.uuid4().hex
    task = "Mjlab-Cartpole-Balance"
    training = train(
        TrainRequest(task=task, output_path=root + "/train", iterations=1, num_envs=16)
    )
    checkpoint = training["artifacts"]["checkpoint.pt"]["uri"]
    evaluated = evaluate(
        EvalRequest(
            task=task, checkpoint=checkpoint, output_path=root + "/eval", episodes=2
        )
    )
    exported = export(
        ExportRequest(task=task, checkpoint=checkpoint, output_path=root + "/export")
    )
    assert evaluated["episodes_completed"] == 2
    assert (
        evaluated["input_sha256"]["checkpoint"]
        == training["artifacts"]["checkpoint.pt"]["sha256"]
    )
    client = StorageClient.from_environment()
    for report in (training, evaluated, exported):
        for index, artifact in enumerate(report["artifacts"].values()):
            path = tmp_path / f"{report['operation']}-{index}"
            client.download_file(artifact["uri"], str(path))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    import onnx

    path = tmp_path / "policy.onnx"
    client.download_file(exported["artifacts"]["policy.onnx"]["uri"], str(path))
    onnx.checker.check_model(str(path))


def test_native_cpu_evaluation_and_export(tmp_path: Path, monkeypatch):
    """Prove the real adapter on CPU using an explicitly untrained native fixture."""
    from dataclasses import asdict

    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from npa.workbench.mjlab.worker import _configs, _evaluate, _export

    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    common = {"task": "Mjlab-Cartpole-Balance", "output_path": "s3://fixture/mjlab"}
    env_cfg, agent_cfg = _configs(TrainRequest(**common, num_envs=1), {})
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu"))
    checkpoint = tmp_path / "checkpoint.pt"
    try:
        runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cpu")
        runner.save(str(checkpoint))
    finally:
        env.close()
    inputs = {"checkpoint": {"path": str(checkpoint)}}
    evaluation, exported = tmp_path / "eval", tmp_path / "export"
    evaluation.mkdir()
    exported.mkdir()
    report = _evaluate(
        EvalRequest(
            **common, checkpoint="s3://fixture/checkpoint.pt", episodes=2, device="cpu"
        ),
        inputs,
        evaluation,
    )
    assert report["episodes_completed"] == 2
    assert all(row["length"] > 0 for row in report["episodes"])
    assert report["mean_return"] == pytest.approx(
        sum(row["return"] for row in report["episodes"]) / 2
    )
    _export(
        ExportRequest(**common, checkpoint="s3://fixture/checkpoint.pt", device="cpu"),
        inputs,
        exported,
    )
    assert (exported / "policy.onnx").stat().st_size > 0
