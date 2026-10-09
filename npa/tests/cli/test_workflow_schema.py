"""Behavior tests for the read-only workflow schema discovery surface."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench import workflow

runner = CliRunner()
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "npa"
    / "orchestration"
    / "npa_workflow"
    / "schema"
    / "npa.workflow.v0.0.1.schema.json"
)


def test_schema_command_exports_bundled_editor_schema() -> None:
    """The CLI emits the exact bundled schema consumable by JSON Schema editors."""
    result = runner.invoke(app, ["workbench", "workflow", "schema"])

    assert result.exit_code == 0, result.output
    exported = json.loads(result.stdout)
    Draft202012Validator.check_schema(exported)
    assert exported == json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_sdk_matches_cli_export() -> None:
    """The SDK returns the same document exposed by the registered CLI command."""
    result = runner.invoke(app, ["workbench", "workflow", "schema"])

    assert result.exit_code == 0, result.output
    assert workflow.schema() == json.loads(result.stdout)


def test_validate_spec_rejects_executable_yaml_tags(tmp_path: Path) -> None:
    """Normal validation refuses executable YAML tags without evaluating them."""
    marker = tmp_path / "yaml-tag-was-evaluated"
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        "\n".join(
            [
                "apiVersion: npa.workflow/v0.0.1",
                "kind: Workflow",
                "metadata: {name: unsafe}",
                "states:",
                "  reject:",
                "    run:",
                f"      shell: !!python/object/apply:os.system ['touch {marker}']",
                "    terminal: true",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["workbench", "workflow", "validate-spec", str(path)]
    )

    assert result.exit_code == 1, result.output
    assert not marker.exists()
