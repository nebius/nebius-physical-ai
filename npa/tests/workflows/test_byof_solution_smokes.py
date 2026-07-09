from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SKILL_PATH = ROOT / "skills" / "workflows" / "oss-solution-registry-onboard" / "SKILL.md"
CATALOG_PATH = ROOT / "docs" / "workbench" / "oss-solution-catalog.md"
SOLUTION_SPECS = sorted(
    path
    for path in WORKFLOW_DIR.glob("byof-*.yaml")
    if path.name != "byof.yaml"
)

# Primary capability contracts for onboarded solutions (solution-specific ids).
# Keep in sync with skills/workflows/oss-solution-registry-onboard/SKILL.md
# and docs/workbench/oss-solution-catalog.md.
ACCEPTED_CAPABILITIES = {
    "maniskill": {
        "capability_name": "gymnasium_pickcube_registration",
        "smoke_artifact_name": "maniskill_pickcube_step.json",
        "spec": "byof-maniskill.yaml",
        "must_exercise": [
            "gymnasium_pickcube_registration",
            "pickcube_cpu_step",
            "pickcube_parallel_envs",
        ],
    },
    "mujoco-playground": {
        "capability_name": "mjx_cartpole_step",
        "smoke_artifact_name": "mujoco_playground_cartpole_step.json",
        "spec": "byof-mujoco-playground.yaml",
        "must_exercise": [
            "mjx_cartpole_step",
            "mjx_cheetah_run_step",
            "train_jax_ppo_cartpole_smoke",  # attempted; may remain deferred
        ],
    },
    "robocasa": {
        "capability_name": "kitchen_task_registration",
        "smoke_artifact_name": "robocasa_kitchen_env_reset.json",
        "spec": "byof-robocasa.yaml",
        "must_exercise": [
            "kitchen_task_registration",
            "download_kitchen_assets_lw",
            "kitchen_egl_env_reset",
        ],
    },
    "openpi": {
        "capability_name": "policy_config_materialization",
        "smoke_artifact_name": "openpi_pi05_droid_config.json",
        "spec": "byof-openpi.yaml",
        "must_exercise": [
            "policy_config_materialization",
            "pi05_droid_checkpoint_download",
            "pi05_droid_checkpoint_infer",
        ],
    },
    "droid-policy-learning": {
        "capability_name": "rlds_config_generator_contract",
        "smoke_artifact_name": "droid_rlds_config_generator.json",
        "spec": "byof-droid-policy-learning.yaml",
        "must_exercise": [
            "rlds_config_generator_contract",
            "droid_100_download",
            "droid_100_config_gen",
        ],
    },
}


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), path
    config = payload.get("config")
    assert isinstance(config, dict), path
    return config


def test_byof_solution_specs_have_capability_smokes() -> None:
    assert SOLUTION_SPECS, "expected BYOF solution candidate specs"
    for path in SOLUTION_SPECS:
        config = _load_config(path)
        assert config.get("workload") == "solution-smoke", path.name
        assert str(config.get("solution_name") or "").strip(), path.name
        assert str(config.get("capability_name") or "").strip(), path.name
        artifact = str(config.get("smoke_artifact_name") or "").strip()
        smoke = str(config.get("smoke_command") or "")
        assert artifact.endswith(".json"), path.name
        assert "NPA_SMOKE_OUTPUT_DIR" in smoke, path.name
        assert artifact in smoke, path.name


def test_byof_solution_smokes_are_not_import_only() -> None:
    for path in SOLUTION_SPECS:
        smoke = str(_load_config(path).get("smoke_command") or "")
        assert ".write_text(" in smoke, path.name
        assert "json.dumps(" in smoke, path.name
        assert '"capability"' in smoke or "'capability'" in smoke, path.name
        assert '"solution"' in smoke or "'solution'" in smoke, path.name
        assert "capabilities_exercised" in smoke, path.name


def test_accepted_capability_contracts_match_specs() -> None:
    by_solution = {
        str(_load_config(path).get("solution_name")): path for path in SOLUTION_SPECS
    }
    assert set(by_solution) == set(ACCEPTED_CAPABILITIES)
    for solution, expected in ACCEPTED_CAPABILITIES.items():
        path = by_solution[solution]
        config = _load_config(path)
        assert path.name == expected["spec"]
        assert config.get("capability_name") == expected["capability_name"]
        assert config.get("smoke_artifact_name") == expected["smoke_artifact_name"]
        smoke = str(config.get("smoke_command") or "")
        assert expected["capability_name"] in smoke
        assert expected["smoke_artifact_name"] in smoke
        for capability in expected["must_exercise"]:
            assert capability in smoke, (solution, capability)


def test_registry_skill_is_solution_specific_not_taxonomy() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "Do **not** force capabilities into a shared taxonomy" in text
    assert "Capability Testing Built Into Onboarding" in text
    assert "Capability Families (required taxonomy)" not in text
    for solution, expected in ACCEPTED_CAPABILITIES.items():
        assert expected["capability_name"] in text or expected["spec"] in text, solution
        assert f"byof-{solution}.yaml" in text or expected["spec"] in text


def test_oss_catalog_lists_solution_specific_capabilities() -> None:
    text = CATALOG_PATH.read_text(encoding="utf-8")
    assert "Native Capabilities Per Container" in text
    assert "shared taxonomy" in text.lower() or "solution-specific" in text.lower()
    for solution, expected in ACCEPTED_CAPABILITIES.items():
        assert expected["capability_name"] in text, solution
        assert expected["smoke_artifact_name"] in text, solution
