"""Exercise the rendered trainer's native physics-failure admission gate."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from npa.cli.isaac_lab import TrainingConfig, _build_rsl_rl_train_shell
from npa.workbench.isaac_lab.training_validity import inspect_training_log


@pytest.fixture
def trainer_checkout(tmp_path: Path) -> Path:
    """Provide a trainer that writes a diagnostic checkpoint and controlled logs."""
    checkout = tmp_path / "checkout"
    trainer = checkout / "scripts/reinforcement_learning/rsl_rl/train.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if os.environ.get('STUB_CHECKPOINT', '1') == '1':\n"
        "    folder = Path('logs/rsl_rl/test'); folder.mkdir(parents=True)\n"
        "    (folder / 'model_9.pt').write_bytes(b'diagnostic-checkpoint')\n"
        "Path('received-args.json').write_text(json.dumps(sys.argv[1:]))\n"
        "print(os.environ['STUB_LOG'], flush=True)\n"
        "raise SystemExit(int(os.environ['STUB_EXIT']))\n"
    )
    return checkout


def _run_trainer(checkout, output, log, exit_code, checkpoint=True, overrides=()):
    output.mkdir()
    (output / "npa_isaac_lab_checkpoint.pt").write_bytes(b"stale-checkpoint")
    script = _build_rsl_rl_train_shell(
        "Isaac-Cartpole-v0",
        64,
        10,
        str(output),
        run_name="test",
        python_bin=sys.executable,
        training_config=TrainingConfig(overrides=list(overrides)),
    )
    environment = dict(
        os.environ,
        ISAACLAB_PATH=str(checkout),
        ISAACLAB_PKG="",
        STUB_LOG=log,
        STUB_EXIT=str(exit_code),
        STUB_CHECKPOINT=str(int(checkpoint)),
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    summary = json.loads((output / "npa_isaac_lab_train_summary.json").read_text())
    manifest = json.loads(
        (output / "npa_isaac_lab_checkpoint_manifest.json").read_text()
    )
    return result, summary, manifest


@pytest.mark.parametrize(
    "log,child_code,checkpoint,expected",
    [
        ("PPO completed", 0, True, 0),
        ("PhysX error: GPU buffer too small", 0, True, 78),
        ("the simulation will miss interactions", 0, True, 78),
        ("PhysX error: dropped contacts", 7, True, 7),
        ("ordinary trainer failure", 7, True, 7),
        ("PPO completed without checkpoint", 0, False, 3),
    ],
)
def test_native_errors_reject_zero_exit_with_checkpoint(
    tmp_path, trainer_checkout, log, child_code, checkpoint, expected
):
    output = tmp_path / "run"
    result, summary, manifest = _run_trainer(
        trainer_checkout,
        output,
        log,
        child_code,
        checkpoint,
    )
    assert result.returncode == expected, result.stderr
    assert summary["status"] == ("success" if expected == 0 else "failed")
    physics_valid = not ("PhysX error" in log or "miss interactions" in log)
    assert summary["physics_valid"] is physics_valid
    assert manifest["physics_valid"] is physics_valid
    if not physics_valid:
        assert summary["physics_error_count"] == 1
        assert summary["stable_checkpoint_path"] == ""
        assert not (output / "npa_isaac_lab_checkpoint.pt").exists()
        assert (
            output / "logs/rsl_rl/test/model_9.pt"
        ).read_bytes() == b"diagnostic-checkpoint"
    if expected == 0:
        assert (
            output / "npa_isaac_lab_checkpoint.pt"
        ).read_bytes() == b"diagnostic-checkpoint"


def test_overrides_are_quoted_and_never_reexpanded(tmp_path, trainer_checkout):
    output = tmp_path / "run"
    marker = tmp_path / "unwanted"
    overrides = ["env.label=@VALIDITY_SOURCE@", f"env.label=$(touch {marker})"]
    result, _, _ = _run_trainer(
        trainer_checkout, output, "completed", 0, overrides=overrides
    )
    assert result.returncode == 0, result.stderr
    arguments = json.loads((output / "received-args.json").read_text())
    assert all(value in arguments for value in overrides)
    assert not marker.exists()


def test_training_log_is_required_and_decoded_conservatively(tmp_path):
    log = tmp_path / "training.log"
    with pytest.raises(FileNotFoundError):
        inspect_training_log(log)
    log.write_bytes(b"\xffPHYSX ERROR: native failure\nwill miss interactions\n")
    assert inspect_training_log(log) == {
        "physics_valid": False,
        "physics_error_count": 1,
    }
