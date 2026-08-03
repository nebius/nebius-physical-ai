"""Guardrail: the retired raw SkyPilot workflow catalog must not return.

`npa.workflow/v0.0.1` specs are the only shipped workflow authoring surface. The
old raw SkyPilot task catalog under ``npa/src/npa/workflows/skypilot/`` has been
retired or relocated, so this guardrail is now inverted from "these may remain"
to "the directory must not exist".

The submit wrapper still accepts customer-provided raw SkyPilot YAML. Tests for
that wrapper must use fixtures under ``npa/tests/fixtures/skypilot/`` or guarded
tool-specific examples, not files from a shipped catalog.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_DIR = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
NPA_WORKFLOW_ROOT = REPO_ROOT / "npa" / "workflows"


def _npa_workflow_documents() -> list[tuple[Path, dict]]:
    docs: list[tuple[Path, dict]] = []
    for path in sorted(NPA_WORKFLOW_ROOT.rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if isinstance(doc, dict) and str(doc.get("apiVersion", "")).startswith(
                "npa.workflow/"
            ):
                docs.append((path, doc))
    return docs


def test_retired_skypilot_catalog_directory_does_not_exist() -> None:
    assert not SKYPILOT_DIR.exists(), (
        "npa/src/npa/workflows/skypilot/ is retired. Author workflow YAML under "
        "npa/workflows/workbench/npa-workflows/ as npa.workflow/v0.0.1 specs; use "
        "npa/tests/fixtures/skypilot/ for raw SkyPilot submit-wrapper tests."
    )


def test_npa_workflow_specs_do_not_carry_skypilot_twin_metadata() -> None:
    offenders: list[str] = []
    for path, doc in _npa_workflow_documents():
        metadata = doc.get("metadata") or {}
        for key in ("skypilotTwin", "skypilotTwins"):
            if key in metadata:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: metadata.{key}")

    assert not offenders, (
        "skypilotTwin metadata is retired with the raw catalog; keep lineage in "
        "EVIDENCE.md instead: " + ", ".join(offenders)
    )


@pytest.mark.parametrize("field", ["skypilotTwin", "skypilotTwins"])
def test_schema_validator_rejects_retired_twin_metadata(tmp_path: Path, field: str) -> None:
    from npa.orchestration.npa_workflow import NpaWorkflowError, load_spec

    value = "npa/src/npa/workflows/skypilot/example.yaml"
    if field == "skypilotTwins":
        value = f"[{value}]"
    spec = tmp_path / "bad-twin.yaml"
    spec.write_text(
        f"""
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: bad-twin
  {field}: {value}
states:
  done:
    run:
      shell: echo ok
    terminal: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(NpaWorkflowError, match=field):
        load_spec(spec)


# Workbench CLI modules that advertise workflow files through module constants.
# These are printed by `<tool> workflow` / `<tool> status`, so a retired template
# must not silently turn an advertised path into a 404 for an operator.
CLI_WORKFLOW_PATH_MODULES = (
    "npa.cli.workbench.mjlab",
    "npa.cli.workbench.retargeting",
    "npa.cli.workbench.token_factory",
    "npa.cli.workbench.vlm_eval",
)


def test_cli_advertised_workflow_paths_exist() -> None:
    """Every `*_WORKFLOW_PATH` a CLI prints must be a real file."""

    from importlib import import_module
    from pathlib import Path as _Path

    missing: list[str] = []
    checked = 0
    for module_name in CLI_WORKFLOW_PATH_MODULES:
        module = import_module(module_name)
        for attr in dir(module):
            if not attr.endswith("WORKFLOW_PATH"):
                continue
            value = getattr(module, attr)
            if not isinstance(value, _Path):
                continue
            checked += 1
            if not (REPO_ROOT / value).is_file():
                missing.append(f"{module_name}.{attr} -> {value}")
    assert checked >= 8, f"expected to check several CLI workflow paths, saw {checked}"
    assert not missing, "CLI modules advertise workflow files that do not exist: " + ", ".join(
        missing
    )


def test_reference_skill_does_not_advertise_the_retired_catalog() -> None:
    skill = REPO_ROOT / "skills" / "workflows" / "workbench-reference-workflows" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")

    assert "npa/src/npa/workflows/skypilot/" not in text
    assert "No raw SkyPilot templates remain" in text
