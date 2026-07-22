"""Enforce the real-components skill for the Physical AI Data Factory blueprint.

Fails if the blueprint uses a known-stub toolRef, if a run.shell stage isn't a
real command/module call, or if the augment stage isn't the real Cosmos execute.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer
import yaml

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

REPO_ROOT = Path(__file__).resolve().parents[4]
BLUEPRINT = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "physical-ai-data-factory.yaml"

# toolRefs that only echo or write a contract/manifest — never advertise as real.
KNOWN_STUB_TOOLREFS = {
    "workbench.cosmos2.transfer",  # manifest only; use transfer_execute
    "workbench.fiftyone.launch_app",  # echo hook
    "workbench.sim2real.finalize",  # echo
    "workbench.sim2real.write_decision",  # demo stub
    "workbench.sim2real.policy_rollouts",
    "workbench.sim2real.heldout_eval",
}
REAL_RUN_MARKERS = ("npa workbench", "data_factory_stages", "data_factory_viz")


def _states() -> dict:
    spec = yaml.safe_load(BLUEPRINT.read_text(encoding="utf-8"))
    return spec["states"]


def test_blueprint_uses_no_stub_toolrefs() -> None:
    for name, state in _states().items():
        tool_ref = state.get("toolRef")
        if tool_ref:
            assert tool_ref not in KNOWN_STUB_TOOLREFS, (
                f"stage '{name}' uses stub toolRef '{tool_ref}'; wire the real component"
            )


def test_blueprint_run_shell_stages_are_real() -> None:
    for name, state in _states().items():
        run = state.get("run")
        if not run:
            continue
        shell = str(run.get("shell", ""))
        assert any(m in shell for m in REAL_RUN_MARKERS), (
            f"stage '{name}' run.shell is not a real command/module call: {shell[:100]}"
        )


def test_augment_runs_real_cosmos_transfer() -> None:
    states = _states()
    assert states["augment"].get("toolRef") == "workbench.cosmos2.transfer_execute", (
        "augment must run the real Cosmos Transfer 2.5 execute path"
    )
    argv = TOOL_CATALOG["workbench.cosmos2.transfer_execute"].argv_template
    assert "--execute" in argv, "transfer_execute must pass --execute to run the real model"
    assert "--input-uri" in argv and "--output-uri" in argv


def test_blueprint_toolrefs_exist_in_catalog() -> None:
    for name, state in _states().items():
        tool_ref = state.get("toolRef")
        if tool_ref:
            assert tool_ref in TOOL_CATALOG, f"stage '{name}' toolRef '{tool_ref}' not in catalog"


def _cli_options_for(path_parts: list[str]) -> set[str]:
    """Return the real CLI option names for `npa <path_parts...>` (e.g.
    ['workbench','token-factory','caption'])."""
    from npa.cli.main import app as main_app

    node = typer.main.get_command(main_app)
    for part in path_parts:
        commands = getattr(node, "commands", None)
        assert commands and part in commands, (
            f"CLI path npa {' '.join(path_parts)} is invalid at '{part}'"
        )
        node = commands[part]
    opts: set[str] = set()
    for param in node.params:
        opts.update(getattr(param, "opts", []) or [])
        opts.update(getattr(param, "secondary_opts", []) or [])
    return opts


def test_blueprint_run_shell_cli_flags_match_real_cli() -> None:
    """Any `npa workbench <group> <cmd> --flags` in a run.shell stage must use the
    tool's ACTUAL CLI options. Guards raw run.shell CLI calls (e.g.
    annotate-augmented's token-factory caption) against flag drift the same way
    the toolRef contract test guards catalog argv."""
    checked = 0
    for name, state in _states().items():
        run = state.get("run") or {}
        shell = str(run.get("shell", ""))
        # Whitespace-collapse the folded YAML scalar, then find npa workbench calls.
        for match in re.finditer(r"npa\s+workbench\s+(\S+)\s+(\S+)((?:\s+--?\S+|\s+\"[^\"]*\"|\s+\S+)*)", shell):
            group, cmd, rest = match.group(1), match.group(2), match.group(3)
            flags = re.findall(r"(--[A-Za-z0-9][A-Za-z0-9-]*)", rest)
            if not flags:
                continue
            cli_opts = _cli_options_for(["workbench", group, cmd])
            for flag in flags:
                assert flag in cli_opts, (
                    f"stage '{name}' run.shell uses `{flag}` for `npa workbench {group} {cmd}`, "
                    f"which is not a real CLI option ({sorted(cli_opts)}). Fix the run.shell."
                )
            checked += 1
    assert checked >= 1, "expected at least one npa-workbench run.shell call to validate"
