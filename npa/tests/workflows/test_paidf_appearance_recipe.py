"""Verify that the shipped appearance recipe reaches all twelve generation prompts."""

import json
from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.workflows.data_factory_stages import generate_configs


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "docs/workbench/examples/paidf-appearance-12.yaml"
WORKFLOW = ROOT / "workflows/main/paidf-cosmos3.yaml"


def test_recipe_reaches_every_profile_in_one_generation_cycle(tmp_path):
    config = yaml.safe_load(RECIPE.read_text())["config"]
    profiles = json.loads(config["appearance_profiles_json"])
    manifest = generate_configs(
        str(tmp_path / "manifest.json"),
        n_augmentations=config["variant_count"],
        augmentation_seed=config["augmentation_seed"],
        appearance_profiles_json=config["appearance_profiles_json"],
        augment_subject=config["augment_subject"],
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


def test_recipe_preserves_workflow_and_forwards_generation_values(tmp_path):
    document = yaml.safe_load(WORKFLOW.read_text())
    original_states = yaml.safe_load(WORKFLOW.read_text())["states"]
    config = yaml.safe_load(RECIPE.read_text())["config"]
    document["config"].update(config)
    private_spec = tmp_path / "paidf-cosmos3.yaml"
    private_spec.write_text(yaml.safe_dump(document))
    spec = load_spec(private_spec)
    plan = build_plan(spec, run_id="recipe-test", assume_decision="promote_checkpoint")
    assert document["states"] == original_states
    generation = next(step for step in plan.steps if step.state == "generate-variants")
    assert generation.argv[generation.argv.index("--variant-count") + 1] == "12"
    configuration = next(
        step for step in plan.steps if step.state == "generate-configs"
    )
    assert config["appearance_profiles_json"] in configuration.argv
    assert document["config"]["source_overlay"] is True
    assert document["config"]["alignment_mode"] == "required"
