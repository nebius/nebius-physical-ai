"""Test sealed resume and independent evaluation boundaries with real tensor files."""

import json
from types import SimpleNamespace

import pytest

from npa.workflows.navigation import initialization, runtime, stages
from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.contract import InitialCheckpoint, read_recipe


def test_checkpoint_digest_is_checked_before_decoding(recipe, tmp_path, monkeypatch):
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"changed")
    recipe.initial_checkpoint = InitialCheckpoint(file="baseline.pt", sha256="0" * 64)
    monkeypatch.setattr(
        initialization,
        "load_native_checkpoint",
        lambda *_: pytest.fail("must not decode"),
    )
    with pytest.raises(ValueError, match="changed before"):
        initialization.initialize_runner(None, recipe, tmp_path)


@pytest.mark.parametrize("complete", [True, False])
def test_tensor_checkpoint_requires_complete_native_state(
    tmp_path, monkeypatch, complete
):
    torch = pytest.importorskip("torch")
    path = tmp_path / "baseline.pt"
    torch.save({"iter": 83, "model": torch.tensor([2.0])}, path)
    original = torch.load

    def decode(path, *, map_location, weights_only):
        assert map_location == "cuda:0" and weights_only is True
        return original(path, map_location="cpu", weights_only=weights_only)

    def restore(payload, *, load_cfg, strict):
        assert payload["model"].item() == 2.0 and strict is True and load_cfg is None
        return complete

    monkeypatch.setattr(torch, "load", decode)
    runner = SimpleNamespace(
        alg=SimpleNamespace(load=restore), current_learning_iteration=0
    )
    if complete:
        initialization.load_native_checkpoint(runner, path)
        assert runner.current_learning_iteration == 83
    else:
        with pytest.raises(ValueError, match="complete native PPO"):
            initialization.load_native_checkpoint(runner, path)
        assert runner.current_learning_iteration == 0


def test_initial_checkpoint_is_contained_and_carried(raw_bundle, tmp_path):
    payload = json.loads((raw_bundle / "recipe.json").read_text())
    baseline = raw_bundle / "baseline.pt"
    baseline.write_bytes(b"fixture-checkpoint")
    payload["initial_checkpoint"] = {
        "file": "baseline.pt",
        "sha256": file_sha256(baseline),
    }
    (raw_bundle / "recipe.json").write_text(json.dumps(payload))
    recipe = read_recipe(raw_bundle)
    output = tmp_path / "output"
    output.mkdir()
    stages._carry_inputs(raw_bundle, output, recipe, "evaluate-checkpoint")
    assert (output / "baseline.pt").read_bytes() == baseline.read_bytes()
    assert not (output / "training.json").exists()
    baseline.unlink()
    baseline.symlink_to(output / "baseline.pt")
    with pytest.raises(ValueError, match="escapes"):
        read_recipe(raw_bundle)


@pytest.mark.parametrize(
    "name", ["../baseline.pt", "/baseline.pt", "policy.pt", "baseline.pickle"]
)
def test_checkpoint_paths_cannot_escape_or_clobber_native_output(name):
    with pytest.raises(ValueError):
        InitialCheckpoint(file=name, sha256="a" * 64)


def test_independent_evaluation_needs_no_training_receipt(
    recipe, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"native-fixture")
    recipe.initial_checkpoint = InitialCheckpoint(
        file="baseline.pt", sha256=file_sha256(checkpoint)
    )
    seen = []
    monkeypatch.setattr(
        initialization,
        "initialize_runner",
        lambda *args: {"mode": "resume_native_checkpoint"},
    )
    monkeypatch.setattr(
        runtime,
        "_score_checkpoint",
        lambda *args: seen.append(args[-2]) or {"passed": False},
    )
    result = runtime._evaluate_initial(
        None, None, None, None, recipe, tmp_path, tmp_path
    )
    assert seen == [checkpoint] and result["passed"] is False
    assert result["initialization"]["mode"] == "resume_native_checkpoint"


def test_paired_evaluation_publishes_valid_losing_policy(
    raw_bundle, recipe, tmp_path, monkeypatch
):
    prepared = tmp_path / "prepared"
    stages.prepare(str(raw_bundle), str(prepared), recipe.image)
    monkeypatch.setenv("NPA_TASK_IMAGE", recipe.image)
    monkeypatch.setattr(stages, "_native", lambda *args: {"passed": False})
    output = tmp_path / "evaluation"
    result = stages.run_stage("evaluate-checkpoint", str(prepared), str(output))
    assert result["passed"] is False and (output / "checksums.json").is_file()
