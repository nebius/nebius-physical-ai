"""Check selected-step loading and rejection before upstream model construction."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from npa.workflows.byof import openpi_selected_policy as selected
from npa.workflows.byof import openpi_service_lifecycle as lifecycle
from npa.workflows.byof.openpi_pipeline import OpenPIPipelineError


@pytest.fixture
def selection(tmp_path, monkeypatch):
    root, manifest = _checkpoint_root(tmp_path)
    training = {
        "status": "passed",
        "base_checkpoint": {"uri": "gs://example/base"},
        "checkpoint": {
            "uri": "s3://example/checkpoint/",
            "saved_step": 37,
            "reload_passed": True,
            **manifest,
        },
        "dataset": {
            "uri": "s3://example/data.npz",
            "manifest_uri": "s3://example/data.json",
            "archive_sha256": "b" * 64,
            "split_disjoint": True,
        },
    }
    evaluation = _selection_evaluation()
    _selection_environment(monkeypatch)
    monkeypatch.setattr(
        selected.pipeline,
        "_read_json_uri",
        lambda uri: training if uri.endswith("training.json") else evaluation,
    )
    return root, training, evaluation


def _checkpoint_root(tmp_path):
    root = tmp_path / "checkpoint"
    (root / "37" / "params").mkdir(parents=True)
    (root / "37" / "assets").mkdir()
    manifest = {"content_manifest_sha256": "a" * 64}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def _selection_evaluation():
    return {
        "status": "passed",
        "lineage": {
            "training_artifact_uri": "s3://example/training.json",
            "trained_checkpoint_uri": "s3://example/checkpoint/",
            "trained_checkpoint_manifest_sha256": "a" * 64,
            "dataset_archive_sha256": "b" * 64,
            "saved_step": 37,
            "exact_training_checkpoint_consumed": True,
        },
    }


def _selection_environment(monkeypatch):
    for key, value in {
        "OPENPI_CHECKPOINT_URI": "/workspace/openpi-server-cache/adapted-checkpoint",
        "NPA_OPENPI_CHECKPOINT_SOURCE_URI": "s3://example/checkpoint",
        "NPA_OPENPI_TRAINING_URI": "s3://example/training.json",
        "NPA_OPENPI_EVALUATION_URI": "s3://example/evaluation.json",
        "NPA_OPENPI_DATASET_URI": "s3://example/data.npz",
        "NPA_OPENPI_DATASET_MANIFEST_URI": "s3://example/data.json",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("change", ["step", "hash", "uri", "selection", "training"])
def test_selection_rejects_mismatched_lineage(selection, change):
    root, training, evaluation = selection
    if change == "step":
        evaluation["lineage"]["saved_step"] = 38
    elif change == "hash":
        evaluation["lineage"]["trained_checkpoint_manifest_sha256"] = "c" * 64
    elif change == "uri":
        evaluation["lineage"]["trained_checkpoint_uri"] = "s3://other/checkpoint"
    elif change == "selection":
        evaluation["lineage"]["exact_training_checkpoint_consumed"] = False
    else:
        training["status"] = "failed"
    with pytest.raises(OpenPIPipelineError, match="lineage mismatch"):
        selected._selected_lineage(root)


def test_model_loader_receives_selected_step_and_training_configuration(
    selection, monkeypatch
):
    root, training, evaluation = selection
    config = SimpleNamespace(name="npa_pi05_franka_mini_lora")
    calls = []
    policy_config = ModuleType("openpi.policies.policy_config")
    policy_config.create_trained_policy = lambda cfg, path: (
        calls.append((cfg, path)) or "policy"
    )
    policies = ModuleType("openpi.policies")
    policies.policy_config = policy_config
    monkeypatch.setitem(sys.modules, "openpi", ModuleType("openpi"))
    monkeypatch.setitem(sys.modules, "openpi.policies", policies)
    monkeypatch.setattr(
        selected, "_selected_configuration", lambda *_: (config, "data", {})
    )
    monkeypatch.setattr(
        selected, "_verify_saved_normalization", lambda path, _: calls.append(path)
    )
    monkeypatch.setattr(
        selected, "_record_loaded_policy", lambda *args: calls.append(args[-1])
    )
    assert selected.load_selected_policy(str(root)) == ("policy", config)
    assert calls == [root / "37", (config, root / "37"), 37]
    (root / "37" / "params").rmdir()
    with pytest.raises(OpenPIPipelineError, match="lacks params or assets"):
        selected.load_selected_policy(str(root))
    assert len(calls) == 3


def test_configuration_rejects_a_different_dataset_before_gpu_imports(
    selection, monkeypatch
):
    root, training, evaluation = selection
    monkeypatch.setattr(
        selected.pipeline,
        "_load_dataset",
        lambda *_: ({}, {"archive_sha256": "c" * 64}),
    )
    with pytest.raises(OpenPIPipelineError, match="dataset lineage mismatch"):
        selected._selected_configuration(root, training, evaluation, 37)


def test_saved_assets_are_compared_to_train_only_statistics(tmp_path, monkeypatch):
    stats = SimpleNamespace(
        **{key: np.array([0.25]) for key in ("mean", "std", "q01", "q99")}
    )
    saved = SimpleNamespace(**vars(stats))
    normalize = ModuleType("openpi.shared.normalize")
    paths = []
    normalize.load = lambda path: paths.append(path) or {"state": saved}
    shared = ModuleType("openpi.shared")
    shared.normalize = normalize
    monkeypatch.setitem(sys.modules, "openpi", ModuleType("openpi"))
    monkeypatch.setitem(sys.modules, "openpi.shared", shared)
    config = SimpleNamespace(asset_id="droid", norm_stats={"state": stats})
    selected._verify_saved_normalization(tmp_path, config)
    assert paths == [tmp_path / "assets" / "droid"]
    saved.q99 = np.array([0.26])
    with pytest.raises(OpenPIPipelineError, match="differs from training"):
        selected._verify_saved_normalization(tmp_path, config)


def test_adapted_manifest_requires_lineage_but_cleanup_does_not():
    from npa.workflows.byof.openpi_service import OpenPIServiceError

    args = SimpleNamespace(checkpoint_uri="s3://example/checkpoint")
    with pytest.raises(OpenPIServiceError, match="requires dataset"):
        lifecycle._bind_selected_policy({}, args)
