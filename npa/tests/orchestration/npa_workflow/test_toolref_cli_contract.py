"""Enforce the toolRef <-> CLI flag contract.

Backs the `workbench-tool` and `author-npa-workflow` skill rule: a toolRef argv
template must use the tool's ACTUAL CLI option names (a mismatch validates/plans
fine but crashes on real submit). Regression guard for the cosmos2.transfer bug
(`--input-path` vs `--input-uri`).
"""

from __future__ import annotations

import importlib
from pathlib import Path

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


CHECKED_TOOLREFS = [
    ("workbench.groot.finetune", "npa.cli.groot", "finetune"),
    ("workbench.cosmos2.transfer", "npa.cli.workbench.cosmos2", "transfer"),
    ("workbench.cosmos2.transfer_execute", "npa.cli.workbench.cosmos2", "transfer"),
    (
        "workbench.cosmos2.transfer_conditioned_execute",
        "npa.cli.workbench.cosmos2",
        "transfer",
    ),
    (
        "workbench.sim2real_envgen.raw_shard",
        "npa.cli.workbench.sim2real_envgen",
        "raw-shard",
    ),
    ("workbench.cosmos3.generate", "npa.cli.workbench.cosmos3", "generate"),
    ("workbench.cosmos3.reason", "npa.cli.workbench.cosmos3", "reason"),
    (
        "workbench.cosmos_curate.curate",
        "npa.cli.workbench.cosmos_curate",
        "curate-augmented",
    ),
    (
        "workbench.cosmos_evaluator.evaluate",
        "npa.cli.workbench.cosmos_evaluator",
        "evaluate",
    ),
    (
        "workbench.fiftyone.curate_augmented",
        "npa.cli.fiftyone",
        "curate-augmented",
    ),
    (
        "workbench.fiftyone.review_augmented",
        "npa.cli.fiftyone",
        "review-augmented",
    ),
    ("workbench.token_factory.caption", "npa.cli.workbench.token_factory", "caption"),
    ("workbench.token_factory.generate", "npa.cli.workbench.token_factory", "generate"),
    ("workbench.token_factory.reason", "npa.cli.workbench.token_factory", "reason"),
    ("workbench.vlm_eval.run", "npa.cli.workbench.vlm_eval", "run"),
    ("workbench.nurec.visualize", "npa.cli.nurec", "visualize"),
]


@pytest.mark.parametrize(("tool_ref", "module_path", "command_name"), CHECKED_TOOLREFS)
def test_toolref_flags_are_real_cli_options(
    tool_ref: str, module_path: str, command_name: str
) -> None:
    cli_opts = _cli_option_names(module_path, command_name)
    for flag in _toolref_flags(tool_ref):
        assert flag in cli_opts, (
            f"{tool_ref} argv uses {flag}, which is not an option of "
            f"`{command_name}` ({sorted(cli_opts)}). Fix catalog.py to match the CLI."
        )


def test_every_toolref_the_data_factory_submits_is_checked() -> None:
    """The list above is an allowlist, so a new toolRef would silently go unchecked.

    A flag mismatch validates and plans fine and only crashes on a real submit, which
    for this blueprint means after a GPU stage has already run.
    """

    import yaml

    from npa.orchestration.npa_workflow.spec import load_spec

    blueprint = (
        Path(__file__).resolve().parents[3]
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "physical-ai-data-factory.yaml"
    )
    states = yaml.safe_load(blueprint.read_text(encoding="utf-8"))["states"]
    submitted = {
        state["toolRef"]
        for state in states.values()
        if isinstance(state, dict) and state.get("toolRef")
    }
    assert load_spec(blueprint).name  # the blueprint still parses as a spec

    unchecked = sorted(submitted - {ref for ref, _, _ in CHECKED_TOOLREFS})
    assert not unchecked, (
        f"the data factory submits {unchecked} but their flags are never checked "
        "against the real CLI; add them to CHECKED_TOOLREFS"
    )


def test_cosmos2_transfer_uses_uri_flags_not_path() -> None:
    flags = _toolref_flags("workbench.cosmos2.transfer")
    assert "--input-uri" in flags and "--output-uri" in flags
    assert "--input-path" not in flags and "--output-path" not in flags
    assert "--run-id" in flags


def test_visualize_stage_uses_prebuilt_rerun_image_without_runtime_install() -> None:
    import yaml

    from npa.orchestration.npa_workflow.skypilot_render import tool_image_key

    repo = Path(__file__).resolve().parents[3]
    blueprint = yaml.safe_load(
        (
            repo
            / "workflows"
            / "workbench"
            / "npa-workflows"
            / "physical-ai-data-factory.yaml"
        ).read_text(encoding="utf-8")
    )
    state = blueprint["states"]["visualize"]
    assert state["toolRef"] == "workbench.nurec.visualize"
    assert tool_image_key(state["toolRef"]) == "rerun-viewer"
    assert "pip install" not in str(state)
