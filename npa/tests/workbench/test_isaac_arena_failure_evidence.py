"""Keep actual failed-run evidence after simulator or publication failures."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from npa.workbench.isaac_arena import runtime


def test_actual_simulator_child_does_not_inherit_admission_secrets(
    tmp_path, monkeypatch
):
    names = (
        "NEBIUS_TOKEN_FACTORY_KEY",
        "NPA_AGENT_ARTIFACT_S3_ACCESS_KEY_ID",
        "NPA_AGENT_ARTIFACT_S3_SECRET_ACCESS_KEY",
        "NPA_AGENT_LANGFUSE_SECRET_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HF_TOKEN",
        "NGC_API_KEY",
        "NEBIUS_IAM_TOKEN",
        "NPA_AGENT_AUTH_PASSWORD",
    )
    secrets = {
        name: f"synthetic-admission-secret-{index}" for index, name in enumerate(names)
    }
    allowed = {"CUDA_VISIBLE_DEVICES": "0", "ACCEPT_EULA": "Y", "LANG": "C.UTF-8"}
    monkeypatch.setattr(runtime.os, "environ", {**secrets, **allowed})
    observed = {}

    def inspect_child(argv, **kwargs):
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import json,os;print(json.dumps(dict(os.environ)))",
            ],
            env=kwargs["env"],
            capture_output=True,
            text=True,
            check=True,
        )
        observed.update(json.loads(child.stdout))
        return _failed_process(argv, **kwargs)

    destination = tmp_path / "out"
    with pytest.raises(runtime.IsaacArenaError, match="policy_runner failed"):
        runtime.evaluate(
            runtime.IsaacArenaRequest(output_path=str(destination)),
            runner=inspect_child,
        )
    assert not set(secrets).intersection(observed)
    assert all(observed[name] == value for name, value in allowed.items())
    assert dict(runtime.os.environ) == {**secrets, **allowed}
    for path in (destination / "result.json", destination / "evaluation.log"):
        assert all(value not in path.read_text() for value in secrets.values())


def _failed_process(argv, **kwargs):
    output = Path(argv[argv.index("--output_base_dir") + 1]) / "failed-run"
    output.mkdir()
    (output / "diagnostic.txt").write_text("retained simulator diagnostic")
    return subprocess.CompletedProcess(
        argv, 23, stdout="simulator failed after writing diagnostics"
    )


def _assert_failed_evidence(directory):
    manifest = json.loads((directory / "result.json").read_text())
    assert manifest["status"] == "failed"
    assert "policy_runner failed (23)" in manifest["error"]
    assert (directory / "evaluation.log").read_text().startswith("simulator failed")
    diagnostic = directory / "upstream/failed-run/diagnostic.txt"
    entry = next(
        item
        for item in manifest["artifacts"]
        if item["path"].endswith("diagnostic.txt")
    )
    assert hashlib.sha256(diagnostic.read_bytes()).hexdigest() == entry["sha256"]


def test_failed_simulator_publishes_hashed_diagnostics_before_raising(tmp_path):
    destination = tmp_path / "published"
    with pytest.raises(runtime.IsaacArenaError, match="policy_runner failed"):
        runtime.evaluate(
            runtime.IsaacArenaRequest(output_path=str(destination)),
            runner=_failed_process,
        )
    _assert_failed_evidence(destination)


def test_pre_rollout_failure_reports_prepared_replay_without_execution_claim(tmp_path):
    import h5py
    import numpy as np

    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("actions", data=np.ones((8, 3), dtype=np.float32))
        episode.create_dataset(
            "initial_state/articulation/robot/joint_position",
            data=np.zeros((1, 3), dtype=np.float32),
        )
        episode.attrs["success"] = True
    destination = tmp_path / "failed"
    observed = {}

    def failed_before_rollout(argv, **kwargs):
        prepared = Path(argv[argv.index("--replay_file_path") + 1])
        observed["prepared_sha256"] = hashlib.sha256(prepared.read_bytes()).hexdigest()
        return _failed_process(argv, **kwargs)

    request = runtime.IsaacArenaRequest(
        output_path=str(destination),
        policy_type="replay",
        input_path=str(source),
    )
    with pytest.raises(runtime.IsaacArenaError, match="policy_runner failed"):
        runtime.evaluate(request, runner=failed_before_rollout)
    result = json.loads((destination / "result.json").read_text())
    prepared = result["input"]["execution"]
    assert prepared["source_steps"] == prepared["prepared_steps"] == 8
    assert prepared["prepared_sha256"] == observed["prepared_sha256"]
    assert prepared["runtime_outcome_claim"] is False
    assert not {"executed_steps", "executed_sha256"}.intersection(prepared)
    assert result["input"]["trajectory"]["source_recorded_success"] is True
    assert result["status"] == "failed" and "summary" not in result


def test_publication_failure_preserves_private_local_evidence_after_scratch_cleanup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime.tempfile, "tempdir", str(tmp_path))

    def unavailable(*args):
        raise OSError("artifact destination unavailable")

    monkeypatch.setattr(runtime, "_publish", unavailable)
    with pytest.raises(runtime.IsaacArenaError, match="evidence retained privately"):
        runtime.evaluate(
            runtime.IsaacArenaRequest(output_path=str(tmp_path / "out")),
            runner=_failed_process,
        )
    (retained,) = tmp_path.glob("npa-isaac-arena-unpublished-*")
    assert retained.stat().st_mode & 0o777 == 0o700
    _assert_failed_evidence(retained / "artifacts")
    assert list(tmp_path.iterdir()) == [retained]


def test_private_checkpoint_location_is_used_but_never_published(tmp_path):
    checkpoint = tmp_path / "operator-private-checkpoint" / "model_1.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"operator model")
    (checkpoint.parent / "params").mkdir()
    (checkpoint.parent / "params/agent.yaml").write_text("policy: fixture\n")
    destination = tmp_path / "out"

    def failed_policy(argv, **kwargs):
        assert argv[argv.index("--checkpoint_path") + 1] == str(checkpoint)
        result = _failed_process(argv, **kwargs)
        result.stdout += (
            f"\nCheckpoint: {checkpoint}\nConfig: {checkpoint.parent}/params/agent.yaml"
        )
        return result

    request = runtime.IsaacArenaRequest(
        output_path=str(destination), policy_type="rsl_rl", input_path=str(checkpoint)
    )
    with pytest.raises(
        runtime.IsaacArenaError, match="policy_runner failed"
    ) as failure:
        runtime.evaluate(request, runner=failed_policy)
    for retained in (destination / "result.json", destination / "evaluation.log"):
        assert "operator-private-checkpoint" not in retained.read_text()
        assert str(checkpoint) not in retained.read_text()
    assert str(checkpoint) not in str(failure.value)
    manifest = json.loads((destination / "result.json").read_text())
    assert (
        manifest["argv"][manifest["argv"].index("--checkpoint_path") + 1]
        == "<operator-input>"
    )
    assert (
        manifest["input"]["sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
