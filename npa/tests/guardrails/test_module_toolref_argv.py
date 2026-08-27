"""Guardrail: a `python -m <module>` toolRef argv must parse against its own CLI.

The catalog-wide flag audit (`test_tool_catalog_argv.py`) understands Typer commands invoked
as ``npa …``, and now also the ``npa …`` calls inside a ``bash -c`` script. It cannot check a
toolRef that runs a module directly — and one of those was broken:

Historically ``workbench.sim2real_envgen.raw_shard`` invoked its module directly
and omitted required ``--run-id``. It now routes through the public Typer command,
so the catalog-wide CLI audit covers it; this file retains the general direct-module
guard for the remaining entries.

This test asks the module's real ``argparse`` parser, which is the only source of truth for a
module CLI: placeholders are replaced with dummy values and the remainder is parsed. A missing
required option, an unknown flag, or a value the parser's ``type=`` rejects all fail here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from importlib import import_module
from typing import Sequence

import pytest

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

#: `{{...}}` placeholder, replaced with something every `type=` in the catalog accepts.
PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")

#: Modules whose CLI parser is reachable. A module without one cannot be checked, and is
#: listed here so adding a module toolRef without an entry point is a visible choice.
PARSER_FACTORIES = {
    "npa.workflows.sim2real_envgen": "build_parser",
    "npa.workflows.token_factory_triage": "build_parser",
    "npa.workbench.lerobot.policy_container": "build_parser",
    "npa.workflows.isaac_capture": "build_parser",
    "npa.workflows.groot_visualization": "build_parser",
    "npa.workflows.groot_learning": "build_parser",
    "npa.workflows.groot_task_performance": "build_parser",
    "npa.workflows.byof.openpi_pipeline": "build_parser",
    "npa.workflows.byof.openpi_service": "build_parser",
    "npa.workflows.content_agents": "build_parser",
}


def _module_tool_refs() -> list[tuple[str, tuple[str, ...]]]:
    """Return (toolRef, argv) for every `python[3] -m <module> …` catalog entry."""

    out: list[tuple[str, tuple[str, ...]]] = []
    for tool_ref, entry in TOOL_CATALOG.items():
        argv = tuple(str(part) for part in entry.argv_template)
        if entry.stub or len(argv) < 4:
            continue
        if not Path(argv[0]).name.startswith("python") or argv[1] != "-m":
            continue
        out.append((tool_ref, argv))
    return out


MODULE_TOOL_REFS = _module_tool_refs()


def _dummy(value: str, *, action: argparse.Action | None) -> str:
    """Replace a placeholder with a value the parser's own `type=` will accept.

    Derived from the parser rather than a hand-kept flag list: a list drifts silently the moment
    a toolRef gains a numeric option nobody remembered to add (which is how this guardrail first
    failed on `--max-steps`), and a drifted list reports a fake parse error instead of a real one.
    """

    if not PLACEHOLDER.search(value):
        return value
    caster = getattr(action, "type", None) if action is not None else None
    if caster is None:
        return "dummy"
    for candidate in ("dummy", "1", "0.8"):
        try:
            caster(candidate)
        except (TypeError, ValueError):
            continue
        return candidate
    return "dummy"


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _actions_by_flag(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> dict[str, argparse.Action]:
    """Options visible to this argv, following the subcommand it names.

    These CLIs are subcommand-based (`sim2real_envgen actions …`), and a subcommand's options do
    not appear on the top-level parser — reading only the top level made every numeric option
    look untyped.
    """

    parsers = [parser]
    choices = _subparsers(parser)
    for token in argv:
        if token in choices:
            parsers.append(choices[token])
            break
    actions: dict[str, argparse.Action] = {}
    for candidate in parsers:
        for action in candidate._actions:
            for option in action.option_strings:
                actions.setdefault(option, action)
    return actions


def _resolve(argv: Sequence[str], parser: argparse.ArgumentParser) -> list[str]:
    actions = _actions_by_flag(parser, argv)
    resolved: list[str] = []
    action: argparse.Action | None = None
    for token in argv:
        if token.startswith("--"):
            action = actions.get(token)
            resolved.append(token)
            continue
        resolved.append(_dummy(token, action=action))
    return resolved


def test_there_is_at_least_one_module_tool_ref_to_check() -> None:
    assert MODULE_TOOL_REFS, "expected at least one `python -m` toolRef in the catalog"


@pytest.mark.parametrize(
    ("tool_ref", "argv"), MODULE_TOOL_REFS, ids=[ref for ref, _ in MODULE_TOOL_REFS]
)
def test_module_tool_ref_argv_parses(tool_ref: str, argv: tuple[str, ...]) -> None:
    module_name = argv[2]
    factory = PARSER_FACTORIES.get(module_name)
    assert factory, (
        f"{tool_ref} runs `python -m {module_name}`, which exposes no parser factory. Add one "
        f"(like `build_parser`) and register it in PARSER_FACTORIES so its argv can be checked."
    )
    parser: argparse.ArgumentParser = getattr(import_module(module_name), factory)()

    try:
        parser.parse_args(_resolve(argv[3:], parser))
    except SystemExit as exc:  # argparse exits 2 on a usage error
        pytest.fail(
            f"{tool_ref} argv does not parse against {module_name}: exit {exc.code}"
        )


def test_raw_shard_routes_through_the_public_cli_with_fan_out_options() -> None:
    """The catalog and documented public command must be the same surface."""

    argv = tuple(
        str(part)
        for part in TOOL_CATALOG["workbench.sim2real_envgen.raw_shard"].argv_template
    )

    assert argv[:4] == ("npa", "workbench", "sim2real-envgen", "raw-shard")
    assert {
        "--run-id",
        "--output-uri",
        "--env-count",
        "--shard-index",
        "--shard-count",
        "--seed",
        "--augmented-frames-uri",
    } <= set(argv), argv
    assert "--train-fraction" not in argv, (
        "raw shard generation does not split the dataset"
    )


def test_the_guardrail_would_have_caught_the_missing_run_id() -> None:
    """Negative control: the argv that shipped must be rejected."""

    from npa.workflows.sim2real_envgen import build_parser

    shipped = ["raw-shard", "--output-uri", "s3://bucket/envs/", "--env-count", "10"]

    with pytest.raises(SystemExit):
        build_parser().parse_args(shipped)


def test_no_tool_ref_invokes_bare_python() -> None:
    """`python` is not guaranteed to exist; `python3` is, and it is what the shim provides.

    Live job 242: `workbench.lerobot.policy_train` ran `python -m …` inside the LeRobot vendor
    image and died with ``bash: python: command not found`` before training started. The same
    argv had *worked* two runs earlier on SkyPilot's default image, where miniconda provides
    `python` — so this is image-dependent breakage that only one image exposes, which is
    precisely what a rule should catch instead of a live run.

    The renderer's interpreter shim records a `python3`, so `python3` is also the interpreter
    that can import `npa`.
    """

    offenders = sorted(
        tool_ref
        for tool_ref, entry in TOOL_CATALOG.items()
        if entry.argv_template and str(entry.argv_template[0]) == "python"
    )

    assert not offenders, (
        "these toolRefs invoke bare `python`, which some images do not provide; use `python3`: "
        f"{offenders}"
    )
