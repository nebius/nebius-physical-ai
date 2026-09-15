"""Exercise real checkpoint decoding and architecture selection boundaries."""

from __future__ import annotations

import ast
import pickle
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from functools import partial

import pytest

from npa.workbench import sonic


class ExecutableCheckpoint:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (eval, (f"__import__('pathlib').Path({str(self.marker)!r}).touch()",))


def test_sonic_restricted_rejection_never_retries_full_pickle(tmp_path):
    torch = pytest.importorskip("torch")
    marker, checkpoint = tmp_path / "executed", tmp_path / "checkpoint.pt"
    torch.save(ExecutableCheckpoint(marker), checkpoint)

    with pytest.raises(sonic.SonicExportError, match="loaded safely"):
        sonic._load_checkpoint_payload(checkpoint, torch)

    assert not marker.exists()


def test_plain_checkpoint_metadata_cannot_invoke_an_arbitrary_callable(tmp_path):
    torch = pytest.importorskip("torch")
    marker, checkpoint = tmp_path / "executed", tmp_path / "checkpoint.pt"
    torch.save({"policy": {"class": "subprocess.check_call", "kwargs": {
        "args": ["touch", str(marker)]
    }}, "policy_state_dict": {}}, checkpoint)

    with pytest.raises(sonic.SonicExportError, match="supported built-in architecture"):
        sonic._load_policy_from_checkpoint(str(checkpoint), torch, {})

    assert not marker.exists()


def test_supported_checkpoint_metadata_restores_real_policy(tmp_path):
    torch = pytest.importorskip("torch")
    from npa.workbench.sonic.reference_policy import ReferenceLocomotionPolicy

    original = ReferenceLocomotionPolicy().eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "policy": {"class": "npa.workbench.sonic.reference_policy.ReferenceLocomotionPolicy"},
        "policy_state_dict": original.state_dict(),
    }, checkpoint)
    restored = sonic._load_policy_from_checkpoint(str(checkpoint), torch, {})
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


@pytest.mark.parametrize("relative,function_name", [
    ("src/npa/workflows/sim2real/policy_export.py", "export_policy_onnx"),
    ("docker/workbench/sonic/mujoco_eval.py", "_load_checkpoint"),
])
def test_production_checkpoint_loaders_reject_executable_pickle(tmp_path, relative, function_name):
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    source = ast.parse((root / relative).read_text())
    # Run the actual loader function with heavyweight simulator dependencies
    # excluded. The malicious pickle must be rejected before policy construction.
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    namespace = {
        "torch": torch, "Path": Path, "_import_torch": lambda: torch,
        "DEFAULT_ACTIVATION": "elu", "DEFAULT_OPSET": 17,
        "DEFAULT_INPUT_NAME": "obs", "DEFAULT_OUTPUT_NAME": "action",
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(root / relative), "exec"), namespace)
    marker, checkpoint = tmp_path / "executed", tmp_path / "checkpoint.pt"
    torch.save(ExecutableCheckpoint(marker), checkpoint)
    kwargs = {"out_dir": str(tmp_path / "export")} if function_name == "export_policy_onnx" else {}

    with pytest.raises(pickle.UnpicklingError):
        namespace[function_name](checkpoint, **kwargs)

    assert not marker.exists()


def test_genesis_teacher_loader_rejects_executable_pickle(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    source_path = Path(__file__).resolve().parents[2] / "src/npa/genesis/generate_demos.py"
    function = next(node for node in ast.parse(source_path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "_load_teacher_policy")
    actor = SimpleNamespace(to=lambda _device: object())
    monkeypatch.setitem(sys.modules, "rsl_rl.modules", SimpleNamespace(ActorCritic=lambda **kwargs: actor))
    namespace = {"torch": torch, "Path": Path, "Any": object, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    marker, checkpoint = tmp_path / "executed", tmp_path / "checkpoint.pt"
    torch.save(ExecutableCheckpoint(marker), checkpoint)

    with pytest.raises(pickle.UnpicklingError):
        namespace["_load_teacher_policy"](checkpoint, SimpleNamespace(obs_dim=4, act_dim=2))

    assert not marker.exists()


@pytest.mark.parametrize("loader", ["detection", "lerobot"])
def test_binary_checkpoint_validation_uses_restricted_loading(tmp_path, monkeypatch, loader):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "executed"
    checkpoint = tmp_path / "pytorch_model.bin"
    torch.save(ExecutableCheckpoint(marker), checkpoint)
    if loader == "detection":
        from npa.workbench.detection_training import evaluation

        monkeypatch.setattr(evaluation, "read_bytes_uri", lambda _: checkpoint.read_bytes())
        call = partial(evaluation._load_checkpoint, "checkpoint")
    else:
        from npa.workbench.lerobot.policy_container import validate_lerobot_checkpoint

        call = partial(validate_lerobot_checkpoint, str(tmp_path))
    with pytest.raises(pickle.UnpicklingError):
        call()
    assert not marker.exists()
