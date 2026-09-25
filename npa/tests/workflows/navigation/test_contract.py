"""Reject malformed navigation inputs before native execution or artifact writes."""

import json

import pytest

from npa.workflows.navigation.artifacts import materialize, publish
from npa.workflows.navigation.contract import Recipe, finite_array, read_recipe
from npa.workflows.navigation.stages import prepare


@pytest.mark.parametrize(
    "key,value",
    [
        ("scene_file", "../scene.usdz"),
        ("scene_file", "/scene.usdz"),
        ("scene_file", "a/../scene.usdz"),
        ("scene_file", "a//scene.usdz"),
        ("scene_file", "scene.usd"),
        ("adapter_module", "bad;import os"),
        ("num_envs", True),
        ("num_envs", 1),
        ("iterations", 0),
        ("minimum_success_rate", float("nan")),
        ("goal_tolerance_m", float("inf")),
        ("image", "image:latest"),
        ("sensor_mode", "camera"),
        ("sensor_mode", "rgbd"),
    ],
)
def test_invalid_recipe(recipe_dict, key, value):
    recipe_dict[key] = value
    with pytest.raises(ValueError):
        Recipe.model_validate(recipe_dict)


@pytest.mark.parametrize("field", ["id", "seed", "position_m"])
def test_heldout_overlap_rejected(recipe_dict, field):
    recipe_dict["eval_cases"][0][field] = recipe_dict["train_cases"][0][field]
    if field == "position_m":
        recipe_dict["eval_cases"][0]["goal_m"] = recipe_dict["train_cases"][0]["goal_m"]
    with pytest.raises(ValueError, match="disjoint"):
        Recipe.model_validate(recipe_dict)


def test_camera_mode_is_explicit(recipe_dict, camera_config):
    recipe_dict.update(sensor_mode="rgbd", camera=camera_config.model_dump())
    assert (
        Recipe.model_validate(recipe_dict).camera.camera_visibility
        == "all_robot_geometry_hidden"
    )
    recipe_dict["camera"]["camera_visibility"] = "only_peers_hidden"
    with pytest.raises(ValueError):
        Recipe.model_validate(recipe_dict)


def test_missing_operator_input_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="operator input bundle is required"):
        prepare("", str(tmp_path / "output"), "tool://isaac-lab")
    assert not (tmp_path / "output").exists()


def test_prepare_seals_bytes_but_never_runtime_readiness(raw_bundle, recipe, tmp_path):
    output = tmp_path / "prepared"
    report = prepare(str(raw_bundle), str(output), recipe.image)
    assert report["native_runtime_verified"] is False
    assert read_recipe(output) == recipe
    assert materialize(str(output), tmp_path / "readback").is_dir()
    (output / "warehouse.usdz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        materialize(str(output), tmp_path / "corrupt")


def test_scene_digest_and_image_binding(raw_bundle, recipe, tmp_path):
    with pytest.raises(ValueError, match="image"):
        prepare(str(raw_bundle), str(tmp_path / "bad-image"), "tool://isaac-lab")
    (raw_bundle / "warehouse.usdz").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        read_recipe(raw_bundle)


def test_symlink_artifact_and_scene_rejected(raw_bundle, tmp_path):
    (raw_bundle / "warehouse.usdz").unlink()
    private = tmp_path / "outside.usdz"
    private.write_bytes(b"fixture-only")
    (raw_bundle / "warehouse.usdz").symlink_to(private)
    with pytest.raises(ValueError, match="symlinks"):
        publish(raw_bundle, str(tmp_path / "output"))
    with pytest.raises(ValueError, match="escapes"):
        read_recipe(raw_bundle)


@pytest.mark.parametrize(
    "source",
    [
        "https://example.invalid/input",
        "s3://bucket/../escape",
        "s3://bucket",
        "s3://bucket/run?token=x",
    ],
)
def test_hostile_input_uri_never_calls_storage(source, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "npa.workflows.navigation.artifacts.StorageClient.from_environment",
        lambda: pytest.fail("must not contact storage"),
    )
    with pytest.raises(ValueError):
        materialize(source, tmp_path / "input")


@pytest.mark.parametrize("value", [[[float("nan"), 1]], [[float("inf"), 1]], [[1]], []])
def test_bad_measurement_shape_or_nan(value):
    with pytest.raises(ValueError, match="finite shape"):
        finite_array(value, (1, 2), "positions")


def test_missing_data_and_unknown_fields(raw_bundle, recipe_dict):
    recipe_dict["task_internal_guess"] = "unsupported"
    (raw_bundle / "recipe.json").write_text(json.dumps(recipe_dict))
    with pytest.raises(ValueError):
        read_recipe(raw_bundle)
    (raw_bundle / "recipe.json").unlink()
    with pytest.raises(FileNotFoundError):
        read_recipe(raw_bundle)
