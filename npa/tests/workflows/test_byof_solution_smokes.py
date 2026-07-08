from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SOLUTION_SPECS = sorted(
    path
    for path in WORKFLOW_DIR.glob("byof-*.yaml")
    if path.name != "byof.yaml"
)


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
