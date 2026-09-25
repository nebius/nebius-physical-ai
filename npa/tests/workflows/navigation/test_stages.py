"""Verify stage artifact/provenance failures without launching Isaac or infrastructure."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from npa.workflows.navigation import artifacts, stages
from npa.workflows.navigation.native import task_adapter


@pytest.fixture
def prepared(raw_bundle, recipe, tmp_path):
    output = tmp_path / "prepared"
    stages.prepare(str(raw_bundle), str(output), recipe.image)
    return output


def test_no_cpu_interpreter_fallback(prepared, recipe, tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_TASK_IMAGE", recipe.image)
    monkeypatch.setenv("ISAAC_LAB_PYTHON", str(tmp_path / "missing-isaac"))
    output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError, match="native Isaac interpreter missing"):
        stages.run_stage("train", str(prepared), str(output))
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    assert not (output / "training.json").exists()


def test_runtime_image_must_match_recipe(prepared, tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_TASK_IMAGE", "image:latest")
    monkeypatch.setattr(
        stages.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("native execution forbidden"),
    )
    with pytest.raises(ValueError, match="NPA_TASK_IMAGE"):
        stages.run_stage("train", str(prepared), str(tmp_path / "failed"))


def test_native_argv_preserves_spaces_without_shell(tmp_path, monkeypatch):
    interpreter = tmp_path / "isaac python"
    interpreter.touch()
    monkeypatch.setenv("ISAAC_LAB_PYTHON", str(interpreter))
    output = tmp_path / "output"
    output.mkdir()
    observed = []

    def run(argv, **kwargs):
        observed.extend(argv)
        assert kwargs["check"] is True and "shell" not in kwargs
        (output / "policy.pt").write_bytes(b"fixture-checkpoint")
        artifacts.write_json(
            output / "training.json",
            {
                "schema": "npa.navigation.training.v1",
                "checkpoint_sha256": artifacts.file_sha256(output / "policy.pt"),
            },
        )

    monkeypatch.setattr(stages.subprocess, "run", run)
    stages._native("train", tmp_path / "input ; literal", output)
    assert observed[0] == str(interpreter)
    assert observed[-2:] == ["--visualizer", "none"]
    assert str(tmp_path / "input ; literal") in observed


@pytest.mark.parametrize("failure", ["missing", "wrong-hash", "physics", "not-loaded"])
def test_runtime_missing_or_invalid_evidence(tmp_path, monkeypatch, failure):
    interpreter = tmp_path / "native"
    interpreter.touch()
    monkeypatch.setenv("ISAAC_LAB_PYTHON", str(interpreter))
    output = tmp_path / "output"
    output.mkdir()

    def run(argv, **kwargs):
        if failure == "physics":
            kwargs["stdout"].write("PhysX error: test fixture")
        (output / "policy.pt").write_bytes(b"fixture")
        if failure != "missing":
            artifacts.write_json(
                output / "training.json",
                {"schema": "npa.navigation.training.v1", "checkpoint_sha256": "0" * 64},
            )
            artifacts.write_json(
                output / "evaluation.json",
                {"schema": "npa.navigation.evaluation.v1", "policy_loaded": False},
            )

    monkeypatch.setattr(stages.subprocess, "run", run)
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        stages._native(
            "evaluate" if failure == "not-loaded" else "train", tmp_path, output
        )


def test_adapter_hash_and_required_methods(recipe, tmp_path, monkeypatch):
    source = tmp_path / "adapter.py"
    source.write_text("# contract fixture\n")
    module = SimpleNamespace(__file__=str(source))
    monkeypatch.setattr(
        "npa.workflows.navigation.native._module_source", lambda _: source
    )
    monkeypatch.setattr(
        "npa.workflows.navigation.native.importlib.import_module", lambda _: module
    )
    with pytest.raises(ValueError, match="SHA-256"):
        task_adapter(recipe)
    recipe.adapter_sha256 = artifacts.file_sha256(source)
    with pytest.raises(ValueError, match="callable configure"):
        task_adapter(recipe)


def test_workflow_graph_routes_native_stages_to_rtx():
    root = Path(__file__).resolve().parents[4]
    spec = yaml.safe_load(
        (root / "workflows/testing/shared-scene-navigation.yaml").read_text()
    )
    assert spec["initial"] == "prepare"
    assert spec["states"]["prepare"]["next"] == "train"
    assert spec["states"]["train"]["next"] == "evaluate"
    assert spec["states"]["evaluate"]["terminal"] is True
    assert spec["resources"]["isaac"]["accelerators"] == "RTXPRO6000:1"
    assert spec["config"]["input_uri"] == ""
    from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

    case = next(
        c for c in SUBMIT_LIVE_MATRIX if c.spec == "shared-scene-navigation.yaml"
    )
    assert case.runtime and case.rotation_skip and not case.plan_only


def test_adapter_digest_is_checked_before_module_or_package_executes(
    recipe, tmp_path, monkeypatch
):
    package = tmp_path / "navigation_import_marker"
    package.mkdir()
    marker = tmp_path / "executed"
    statement = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    (package / "__init__.py").write_text(statement)
    source = package / "adapter.py"
    source.write_text(
        statement
        + "def configure(**kwargs): return None\n"
        + "def reset(*args): return None\n"
        + "def measure(*args): return {}\n"
        + "def probe_mode(*args): return None\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    recipe.adapter_module = "navigation_import_marker.adapter"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        task_adapter(recipe)
    assert not marker.exists()
    recipe.adapter_sha256 = artifacts.file_sha256(source)
    module = task_adapter(recipe)
    assert marker.is_file() and callable(module.configure)
    import sys

    monkeypatch.delitem(sys.modules, "navigation_import_marker.adapter")
    monkeypatch.delitem(sys.modules, "navigation_import_marker")
