"""Keep actual failed-run evidence after simulator or publication failures."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from npa.workbench.isaac_arena import runtime


def _failed_process(argv, **kwargs):
    output = Path(argv[argv.index("--output_base_dir") + 1]) / "failed-run"
    output.mkdir()
    (output / "diagnostic.txt").write_text("retained simulator diagnostic")
    return subprocess.CompletedProcess(argv, 23, stdout="simulator failed after writing diagnostics")


def _assert_failed_evidence(directory):
    manifest = json.loads((directory / "result.json").read_text())
    assert manifest["status"] == "failed"
    assert "policy_runner failed (23)" in manifest["error"]
    assert (directory / "evaluation.log").read_text().startswith("simulator failed")
    diagnostic = directory / "upstream/failed-run/diagnostic.txt"
    entry = next(item for item in manifest["artifacts"] if item["path"].endswith("diagnostic.txt"))
    assert hashlib.sha256(diagnostic.read_bytes()).hexdigest() == entry["sha256"]


def test_failed_simulator_publishes_hashed_diagnostics_before_raising(tmp_path):
    destination = tmp_path / "published"
    with pytest.raises(runtime.IsaacArenaError, match="policy_runner failed"):
        runtime.evaluate(runtime.IsaacArenaRequest(output_path=str(destination)), runner=_failed_process)
    _assert_failed_evidence(destination)


def test_publication_failure_preserves_private_local_evidence_after_scratch_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.tempfile, "tempdir", str(tmp_path))

    def unavailable(*args):
        raise OSError("artifact destination unavailable")

    monkeypatch.setattr(runtime, "_publish", unavailable)
    with pytest.raises(runtime.IsaacArenaError, match="evidence retained privately"):
        runtime.evaluate(runtime.IsaacArenaRequest(output_path=str(tmp_path / "out")), runner=_failed_process)
    retained, = tmp_path.glob("npa-isaac-arena-unpublished-*")
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
        result.stdout += f"\nCheckpoint: {checkpoint}\nConfig: {checkpoint.parent}/params/agent.yaml"
        return result

    request = runtime.IsaacArenaRequest(output_path=str(destination), policy_type="rsl_rl", input_path=str(checkpoint))
    with pytest.raises(runtime.IsaacArenaError, match="policy_runner failed") as failure:
        runtime.evaluate(request, runner=failed_policy)
    for retained in (destination / "result.json", destination / "evaluation.log"):
        assert "operator-private-checkpoint" not in retained.read_text()
        assert str(checkpoint) not in retained.read_text()
    assert str(checkpoint) not in str(failure.value)
    manifest = json.loads((destination / "result.json").read_text())
    assert manifest["argv"][manifest["argv"].index("--checkpoint_path") + 1] == "<operator-input>"
    assert manifest["input"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
