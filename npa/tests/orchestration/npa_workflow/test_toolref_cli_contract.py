"""Enforce the toolRef <-> CLI flag contract.

Backs the `workbench-tool` and `author-npa-workflow` skill rule: a toolRef argv
template must use the tool's ACTUAL CLI option names (a mismatch validates/plans
fine but crashes on real submit). Regression guard for the cosmos2.transfer bug
(`--input-path` vs `--input-uri`).
"""

from __future__ import annotations

import importlib

import pytest
import typer

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG


def _cli_option_names(module_path: str, command_name: str) -> set[str]:
    module = importlib.import_module(module_path)
    click_cmd = typer.main.get_command(module.app)
    # Multi-command apps are click Groups; single-command apps collapse to the
    # command itself.
    commands = getattr(click_cmd, "commands", None)
    command = commands[command_name] if commands else click_cmd
    opts: set[str] = set()
    for param in command.params:
        opts.update(getattr(param, "opts", []) or [])
        opts.update(getattr(param, "secondary_opts", []) or [])
    return opts


def _toolref_flags(tool_ref: str) -> list[str]:
    return [a for a in TOOL_CATALOG[tool_ref].argv_template if a.startswith("--")]


@pytest.mark.parametrize(
    ("tool_ref", "module_path", "command_name"),
    [
        ("workbench.cosmos2.transfer", "npa.cli.workbench.cosmos2", "transfer"),
        ("workbench.cosmos3.reason", "npa.cli.workbench.cosmos3", "reason"),
        ("workbench.token_factory.caption", "npa.cli.workbench.token_factory", "caption"),
        ("workbench.token_factory.generate", "npa.cli.workbench.token_factory", "generate"),
        ("workbench.token_factory.reason", "npa.cli.workbench.token_factory", "reason"),
        ("workbench.vlm_eval.run", "npa.cli.workbench.vlm_eval", "run"),
    ],
)
def test_toolref_flags_are_real_cli_options(tool_ref: str, module_path: str, command_name: str) -> None:
    cli_opts = _cli_option_names(module_path, command_name)
    for flag in _toolref_flags(tool_ref):
        assert flag in cli_opts, (
            f"{tool_ref} argv uses {flag}, which is not an option of "
            f"`{command_name}` ({sorted(cli_opts)}). Fix catalog.py to match the CLI."
        )


def test_cosmos2_transfer_uses_uri_flags_not_path() -> None:
    flags = _toolref_flags("workbench.cosmos2.transfer")
    assert "--input-uri" in flags and "--output-uri" in flags
    assert "--input-path" not in flags and "--output-path" not in flags
    assert "--run-id" in flags
