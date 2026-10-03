"""Keep anchor intent explicit, checkpoint-bound and absent from inference recipes."""

import json

import pytest

from npa.workflows.navigation.anchor_config import (
    anchor_enabled,
    configure_training_anchor,
    inference_settings,
    validate_anchor_coefficient,
)
from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.contract import Recipe, read_recipe
from npa.workflows.field_failure import native_artifacts, native_policy


@pytest.mark.parametrize(
    "value", [True, False, None, "10", 0, -1, float("nan"), float("inf")]
)
def test_unbound_coefficient_is_strict(value):
    with pytest.raises(ValueError, match="finite and positive"):
        validate_anchor_coefficient(value)


def test_default_recipe_serialization_remains_unchanged(recipe_dict):
    recipe = Recipe(**recipe_dict)
    before = recipe.model_dump()
    assert "baseline_anchor" not in before
    recipe_dict["baseline_anchor"] = None
    assert Recipe(**recipe_dict).model_dump() == before
    assert not anchor_enabled(recipe)


@pytest.mark.parametrize(
    "coefficient,training,enabled",
    [(None, True, False), (0.0, True, False), (10.0, False, False), (10.0, True, True)],
)
def test_only_training_selects_the_extension(
    recipe_dict, coefficient, training, enabled
):
    from copy import deepcopy

    identity = {"file": "original.pt", "sha256": "a" * 64}
    if coefficient is not None:
        recipe_dict.update(
            initial_checkpoint=identity,
            baseline_anchor={"coefficient": coefficient, "checkpoint": identity},
        )
    recipe = Recipe(**recipe_dict)
    settings = {"algorithm": {"class_name": "PPO", "learning_rate": 0.001}}
    original = deepcopy(settings)
    configure_training_anchor(settings, recipe, training=training)
    if enabled:
        assert settings["algorithm"]["class_name"].endswith(":BaselineAnchoredPPO")
        assert (
            settings["algorithm"]["baseline_anchor"]
            == recipe.baseline_anchor.model_dump()
        )
    else:
        assert settings == original


def test_training_configuration_roundtrips_to_native_inference(recipe_dict):
    from copy import deepcopy

    identity = {"file": "original.pt", "sha256": "a" * 64}
    recipe = Recipe(
        **{
            **recipe_dict,
            "initial_checkpoint": identity,
            "baseline_anchor": {"coefficient": 10.0, "checkpoint": identity},
        }
    )
    original = {"algorithm": {"class_name": "PPO", "learning_rate": 0.001}}
    settings = deepcopy(original)
    configure_training_anchor(settings, recipe, training=True)
    assert inference_settings(settings, recipe) == original
    settings["algorithm"]["class_name"] = "unexpected:Algorithm"
    with pytest.raises(ValueError, match="unexpected anchor class"):
        inference_settings(settings, recipe)


def _protocol(recipe):
    return {
        "schema_version": "npa.field-failure.native-protocol.v1",
        "navigation_image": recipe["image"],
        **{
            name: recipe.get(name)
            for name in (
                "task",
                "adapter_module",
                "adapter_sha256",
                "source_bundle_sha256",
            )
        },
    }


@pytest.mark.parametrize("extra", [{}, {"baseline_anchor_coefficient": 10.0}])
def test_legacy_and_enabled_protocol_are_explicit(
    recipe_dict, tmp_path, monkeypatch, extra
):
    protocol = {**_protocol(recipe_dict), **extra}
    monkeypatch.setattr(
        native_artifacts,
        "_download",
        lambda artifact, path: path.write_text(json.dumps(protocol)),
    )
    assert native_artifacts._protocol({"protocol": {}}, tmp_path) == protocol


@pytest.mark.parametrize(
    "extra",
    [
        {"baseline_anchor_coefficient": None},
        {"baseline_anchor_coefficient": True},
        {"baseline_anchor_coefficient": 0},
        {"unknown": 10.0},
    ],
)
def test_protocol_never_silently_disables_invalid_anchor(
    recipe_dict, tmp_path, monkeypatch, extra
):
    protocol = {**_protocol(recipe_dict), **extra}
    monkeypatch.setattr(
        native_artifacts,
        "_download",
        lambda artifact, path: path.write_text(json.dumps(protocol)),
    )
    with pytest.raises(ValueError):
        native_artifacts._protocol({"protocol": {}}, tmp_path)


def test_training_binds_original_teacher_separately_from_continuation(
    raw_bundle, recipe_dict, tmp_path
):
    original, continuation = tmp_path / "original.pt", tmp_path / "continued.pt"
    original.write_bytes(b"synthetic original native checkpoint")
    continuation.write_bytes(b"synthetic continued native checkpoint")
    protocol = {**_protocol(recipe_dict), "baseline_anchor_coefficient": 10.0}
    native_policy._initialize(
        raw_bundle, protocol, continuation, original_baseline=original
    )
    recipe = read_recipe(raw_bundle)
    assert recipe.initial_checkpoint.sha256 == file_sha256(continuation)
    assert recipe.baseline_anchor.checkpoint.sha256 == file_sha256(original)
    assert (raw_bundle / "anchor-reference.pt").read_bytes() == original.read_bytes()
    assert recipe.baseline_anchor.coefficient == 10.0


def test_inference_protocol_intent_does_not_bind_teacher(
    raw_bundle, recipe_dict, tmp_path
):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"synthetic candidate native checkpoint")
    protocol = {**_protocol(recipe_dict), "baseline_anchor_coefficient": 10.0}
    native_policy._initialize(raw_bundle, protocol, candidate)
    recipe = read_recipe(raw_bundle)
    assert recipe.baseline_anchor is None
    assert not (raw_bundle / "anchor-reference.pt").exists()
    assert recipe.initial_checkpoint.sha256 == file_sha256(candidate)


def test_scene_cannot_choose_baseline_teacher(raw_bundle, recipe_dict):
    recipe_dict["baseline_anchor"] = {"coefficient": 10.0}
    (raw_bundle / "recipe.json").write_text(json.dumps(recipe_dict))
    with pytest.raises(ValueError, match="cannot choose the baseline teacher"):
        native_artifacts._recipe(raw_bundle, _protocol(recipe_dict))


def test_training_anchor_requires_initialization_and_authenticated_teacher(
    raw_bundle, recipe_dict, tmp_path
):
    identity = {"file": "original.pt", "sha256": "a" * 64}
    recipe_dict["baseline_anchor"] = {"coefficient": 10.0, "checkpoint": identity}
    with pytest.raises(ValueError, match="requires a native initial checkpoint"):
        Recipe(**recipe_dict)
    recipe_dict["initial_checkpoint"] = identity
    (raw_bundle / "recipe.json").write_text(json.dumps(recipe_dict))
    (raw_bundle / "original.pt").write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_recipe(raw_bundle)


def test_inference_configuration_removes_only_the_exact_anchor(recipe_dict):
    identity = {"file": "original.pt", "sha256": "a" * 64}
    anchor = {"coefficient": 10.0, "checkpoint": identity}
    recipe = Recipe(
        **{**recipe_dict, "initial_checkpoint": identity, "baseline_anchor": anchor}
    )
    settings = {"algorithm": {"learning_rate": 0.001, "baseline_anchor": anchor}}
    assert inference_settings(settings, recipe) == {
        "algorithm": {"learning_rate": 0.001}
    }
    assert settings["algorithm"]["baseline_anchor"] == anchor
    settings["algorithm"]["baseline_anchor"] = {**anchor, "coefficient": 1.0}
    with pytest.raises(ValueError, match="differs from the sealed recipe"):
        inference_settings(settings, recipe)


def test_diagnostic_replay_strips_training_teacher_even_when_file_absent(
    raw_bundle, recipe_dict, tmp_path, monkeypatch
):
    original, candidate = tmp_path / "original.pt", tmp_path / "candidate.pt"
    original.write_bytes(b"synthetic original")
    candidate.write_bytes(b"synthetic candidate")
    protocol = {**_protocol(recipe_dict), "baseline_anchor_coefficient": 10.0}
    native_policy._initialize(
        raw_bundle, protocol, candidate, original_baseline=original
    )
    recipe = json.loads((raw_bundle / "recipe.json").read_text())
    (raw_bundle / "anchor-reference.pt").unlink()
    original.unlink()
    observed = []

    def execute(request, stage, source, output, protocol):
        observed.append(read_recipe(source))
        return {
            "checkpoint_sha256": file_sha256(candidate),
            "success_rate": 0.5,
            "episodes": [],
        }

    monkeypatch.setattr(native_policy, "_execute", execute)
    monkeypatch.setattr(native_policy, "_archive", lambda *args: None)
    monkeypatch.setattr(native_policy, "_upload", lambda *args: {"fixture": True})
    native_policy._replay_batch(
        {"output_prefix": "fixture/"},
        raw_bundle,
        protocol,
        candidate,
        tmp_path,
        "replay",
        recipe,
        recipe["train_cases"],
    )
    assert observed[0].baseline_anchor is None
    assert observed[0].initial_checkpoint.sha256 == file_sha256(candidate)
