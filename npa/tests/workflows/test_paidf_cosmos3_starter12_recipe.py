"""Verify the twelve-variant starter recipe changes only its declared settings."""

import json
from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.workflows.data_factory_stages import generate_configs


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "docs/workbench/examples/paidf-cosmos3-starter-12.yaml"
WORKFLOW = ROOT / "workflows/main/paidf-cosmos3.yaml"
RECIPE_KEYS = {
    "variant_count",
    "transfer_first_chunk_conditional_frames",
    "appearance_profiles_json",
}


def _recipe_config():
    return yaml.safe_load(RECIPE.read_text())["config"]


def _argument(step, flag):
    return step.argv[step.argv.index(flag) + 1]


def test_recipe_overrides_only_count_first_window_and_profiles():
    config = _recipe_config()
    defaults = yaml.safe_load(WORKFLOW.read_text())["config"]
    assert set(config) == RECIPE_KEYS
    assert config["variant_count"] == "12"
    assert config["transfer_first_chunk_conditional_frames"] == "0"
    assert defaults["transfer_first_chunk_conditional_frames"] == "1"


def test_recipe_requests_twelve_unique_profiles_in_one_cycle(tmp_path):
    config = _recipe_config()
    defaults = yaml.safe_load(WORKFLOW.read_text())["config"]
    profiles = json.loads(config["appearance_profiles_json"])
    manifest = generate_configs(
        str(tmp_path / "manifest.json"),
        n_augmentations=config["variant_count"],
        augmentation_seed=defaults["augmentation_seed"],
        appearance_profiles_json=config["appearance_profiles_json"],
        augment_subject=defaults["augment_subject"],
    )
    requested = {tuple(sorted(profile.items())) for profile in profiles}
    generated = {
        tuple(sorted((key, item[key]) for key in profiles[0]))
        for item in manifest["augmentations"]
    }
    assert len(requested) == len(manifest["augmentations"]) == 12
    assert generated == requested
    for item in manifest["augmentations"]:
        assert all(item[key] in item["prompt"] for key in profiles[0])


def test_recipe_plan_keeps_default_sampling_and_quality(tmp_path):
    document = yaml.safe_load(WORKFLOW.read_text())
    document["config"].update(_recipe_config())
    private_spec = tmp_path / "paidf-cosmos3.yaml"
    private_spec.write_text(yaml.safe_dump(document))
    plan = build_plan(
        load_spec(private_spec),
        run_id="recipe-test",
        assume_decision="promote_checkpoint",
    )
    generation = next(step for step in plan.steps if step.state == "generate-variants")
    assert _argument(generation, "--variant-count") == "12"
    assert _argument(generation, "--transfer-first-chunk-conditional-frames") == "0"
    assert _argument(generation, "--guidance") == "5.0"
    assert _argument(generation, "--steps") == "24"
    assert _argument(generation, "--variant-parallelism") == "1"
    assert _argument(generation, "--transfer-rgb-weight") == "0.0"
    assert document["config"]["grade_threshold"] == "0.2"
    assert document["config"]["attribute_threshold"] == "0.25"
    assert document["config"]["alignment_mode"] == "required"
