"""Guardrail: BYOF profiles stay *resource profiles*, not workflow templates.

They were moved out of the retiring raw SkyPilot workflow catalog because they are reached *through* the
``npa.workflow`` surface — ``byof.yaml``'s ``workbench.byof.repo`` toolRef passes one
through ``--yaml {{config.resource_profile_yaml}}``. This keeps them from quietly
becoming a second workflow catalog in a new location.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILES = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "byof" / "profiles"

#: The only file here that is not a SkyPilot task: it is a SkyPilot *global config*
#: passed with ``--config`` to set ``imagePullSecrets``.
GLOBAL_CONFIG = "skypilot-kubernetes-rtxpro.yaml"

EXPECTED_PROFILES = frozenset(
    {
        "byof-container-smoke-rtxpro.yaml",
        "byof-datagen-rtxpro-smoke.yaml",
        "byof-solution-smoke-openpi-b200-gpu.yaml",
        "byof-solution-smoke-wan22-rtxpro-gpu.yaml",
        "byof-solution-smoke-wan22-b200-4gpu.yaml",
        "byof-solution-smoke-ltx2-rtxpro-gpu.yaml",
        "byof-solution-smoke-rtxpro-2gpu.yaml",
        "byof-solution-smoke-rtxpro-gpu.yaml",
        "isaac-lab-rl-train.yaml",
        "isaac-lab-rl-train-rtxpro.yaml",
        "isaac-lab-rl-train-rtxpro-smoke.yaml",
        GLOBAL_CONFIG,
    }
)


def _task_profiles() -> list[Path]:
    return sorted(p for p in PROFILES.glob("*.yaml") if p.name != GLOBAL_CONFIG)


def test_profile_set_is_pinned() -> None:
    """A new file here needs a deliberate edit — and probably wants to be a spec."""

    on_disk = {path.name for path in PROFILES.glob("*.yaml")}

    assert on_disk == EXPECTED_PROFILES, (
        "BYOF profiles changed. A multi-stage pipeline is a workflow: author an "
        "npa.workflow/v0.0.1 spec under npa/workflows/workbench/npa-workflows/ instead. "
        f"expected {sorted(EXPECTED_PROFILES)}, found {sorted(on_disk)}"
    )


def test_global_config_contains_only_skypilot_config_fields() -> None:
    config = yaml.safe_load((PROFILES / GLOBAL_CONFIG).read_text(encoding="utf-8"))

    assert set(config) == {"kubernetes"}
    assert config["kubernetes"] == {}


@pytest.mark.parametrize("path", _task_profiles(), ids=lambda p: p.name)
def test_profile_is_a_single_task(path: Path) -> None:
    """One pod shape per profile. Chaining stages means it should be a spec."""

    docs = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]

    assert docs, path
    # Either a bare single task, or a 2-document `execution: serial` pair whose second
    # document is the only task.
    tasks = [doc for doc in docs if "run" in doc]
    assert len(tasks) == 1, (
        f"{path.name} declares {len(tasks)} tasks; a resource profile describes ONE pod"
    )
    assert "resources" in tasks[0], f"{path.name} must declare a resources block"


def test_the_readme_explains_the_boundary() -> None:
    text = (PROFILES / "README.md").read_text(encoding="utf-8")

    for token in (
        "resource profiles",
        "byof.yaml",
        "workbench.byof.repo",
        "npa.workflow",
    ):
        assert token in text, f"profiles README should mention {token!r}"


def test_live_module_resolves_every_profile_it_names() -> None:
    """`live.py`'s constants must point at files that exist after the relocation."""

    from npa.workflows.byof import live

    named = [
        live.DEFAULT_TRAIN_YAML,
        live.RTXPRO_TRAIN_YAML,
        live.RTXPRO_SMOKE_TRAIN_YAML,
        live.BYOF_DATAGEN_SMOKE_YAML,
        live.BYOF_CONTAINER_SMOKE_YAML,
        live.RTXPRO_SKYPILOT_CONFIG,
    ]
    missing = [str(path) for path in named if not path.is_file()]

    assert not missing, f"byof/live.py names missing profiles: {missing}"
    assert all(path.parent == PROFILES for path in named), (
        "every profile constant should resolve inside byof/profiles/"
    )


def test_runner_defaults_point_at_the_profiles_directory() -> None:
    """The three runner scripts' DEFAULT_YAML must have moved with the files."""

    scripts = {
        "run_isaac_lab_rl.py": "isaac-lab-rl-train.yaml",
        "run_byof_datagen.py": "byof-datagen-rtxpro-smoke.yaml",
        "run_byof_container_verify.py": "byof-container-smoke-rtxpro.yaml",
    }
    for name, profile in scripts.items():
        text = (REPO_ROOT / "npa" / "scripts" / name).read_text(encoding="utf-8")
        assert '"profiles"' in text, (
            f"{name} should resolve DEFAULT_YAML under profiles/"
        )
        assert profile in text, f"{name} should still name {profile}"
        assert '"skypilot"' not in text, (
            f"{name} still resolves a path under the retiring skypilot catalog"
        )
