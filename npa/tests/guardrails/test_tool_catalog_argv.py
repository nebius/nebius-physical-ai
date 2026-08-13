"""Guardrail: every ``toolRef`` argv template must name real CLI options.

An ``npa.workflow`` spec's only way to invoke a workbench tool is a ``toolRef``,
which the engine expands into an argv list from ``TOOL_CATALOG`` and runs inside the
task pod. Nothing in validate → plan → render checks those flags against the CLI
command they will be handed to, so a template can be perfectly valid and still die
with ``No such option`` on real infrastructure. ``DESIGN.md`` §7 records exactly
that for ``workbench.rl.policy_train``.

Retiring the raw SkyPilot catalog makes this the load-bearing check: while a
SkyPilot YAML shipped per tool, the three-tier contract's "the YAML declares an
``envs`` key for each CLI flag" was a proxy for "the documented way to run this tool
at scale exposes its parameters". With the spec as the only workflow surface, the
sharper question is whether the toolRef argv can run at all.
"""

from __future__ import annotations

import pytest

from npa.guardrails.tool_catalog_argv import (
    ArgvResolutionError,
    argv_flag_drift,
    argv_literal_value_mismatches,
    argv_template_flags,
    catalog_argv_drift,
    catalog_argv_literal_mismatches,
    resolve_argv_command,
)
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

#: Catalog entries whose argv is not an ``npa ...`` invocation (an inline
#: ``python3 -c`` snippet or a ``bash -c`` wrapper). Their flags cannot be checked
#: against a Typer signature. Pinned so the set can shrink but not silently grow.
NON_CLI_ARGV = frozenset(
    {
        "workbench.data_transform.improvement_summary",
        "workbench.data_transform.rollout_contract",
        "workbench.dataset.report_rejection",
        "workbench.dataset.write_quality_decision",
        "workbench.lancedb.backfill_cpu_bundle",
        "workbench.lancedb.create_failure_views",
        "workbench.rl.publish_policy",
        "workbench.rl.report_failure",
        "workbench.rl.write_success_decision",
        "workbench.scenario_gen.write_hardening_decision",
        "workbench.sim2real.write_decision",
        "workbench.lerobot.policy_rollout",
        "workbench.lerobot.policy_train",
        "workbench.token_factory.triage",
        "workbench.sim2real_envgen.split",
        "workbench.sim2real_envgen.actions",
        "workbench.isaac_lab.capture_frames",
        "workflow.groot.prepare_split",
        "workflow.groot.preflight_rigor",
        "workflow.groot.resolve_trained_checkpoint",
        "workbench.groot.baseline_eval",
        "workbench.groot.posttrain_eval",
        "workflow.groot.compare_learning",
        "workflow.groot.emit_learning_mcap",
        "workflow.groot.emit_learning_rrd",
        "workflow.groot.publish_learning",
        "workflow.groot.verify_agent_ui",
    }
)

#: The `bash -c` and `python -m` members of NON_CLI_ARGV are no longer unchecked: the
#: former have their embedded `npa …` calls audited here (see
#: test_bash_wrapped_tool_refs_are_audited_too), and the latter are parsed against their
#: module's own argparse parser by tests/guardrails/test_module_toolref_argv.py. What
#: remains genuinely exempt is inline `python -c` source.
AUDITED_ELSEWHERE = frozenset(
    {
        "workbench.lancedb.backfill_cpu_bundle",
        "workbench.lancedb.create_failure_views",
        "workbench.sim2real_envgen.split",
        "workbench.sim2real_envgen.actions",
        "workbench.isaac_lab.capture_frames",
        "workbench.lerobot.policy_rollout",
        "workbench.lerobot.policy_train",
        "workbench.token_factory.triage",
        "workflow.groot.prepare_split",
        "workflow.groot.preflight_rigor",
        "workflow.groot.resolve_trained_checkpoint",
        "workbench.groot.baseline_eval",
        "workbench.groot.posttrain_eval",
        "workflow.groot.compare_learning",
        "workflow.groot.emit_learning_mcap",
        "workflow.groot.emit_learning_rrd",
        "workflow.groot.publish_learning",
        "workflow.groot.verify_agent_ui",
    }
)


def test_the_exemption_list_shrinks_as_coverage_grows() -> None:
    """Every entry claimed as audited elsewhere must still be exempt here."""

    assert AUDITED_ELSEWHERE <= NON_CLI_ARGV
    # And each really is covered: bash wrappers by the embedded extractor, module
    # entries by the argparse guardrail.
    from npa.guardrails.tool_catalog_argv import embedded_npa_commands

    for tool_ref in sorted(AUDITED_ELSEWHERE):
        argv = TOOL_CATALOG[tool_ref].argv_template
        head = str(argv[0])
        if head == "bash":
            assert embedded_npa_commands(argv), tool_ref
        else:
            assert head in {"python", "python3"} and str(argv[1]) == "-m", tool_ref


def _cli_backed_tool_refs() -> list[str]:
    return sorted(
        tool_ref
        for tool_ref, entry in TOOL_CATALOG.items()
        if not entry.stub
        and entry.argv_template
        and str(entry.argv_template[0]) == "npa"
    )


def test_no_tool_ref_argv_passes_a_flag_its_cli_rejects() -> None:
    """The whole catalog must be runnable, not merely renderable."""

    drift = catalog_argv_drift()
    assert not drift, (
        "toolRef argv templates pass options their CLI command does not accept; "
        "these specs would crash in the pod after a successful render:\n"
        + "\n".join(f"  {ref}: {flags}" for ref, flags in sorted(drift.items()))
    )


#: Top-level CLI groups a toolRef may invoke. `workbench` is the tool layer;
#: `soperator` and `fleet` are the infra-provisioning groups behind the
#: `infra.*` toolRefs.
TOOL_REF_CLI_GROUPS = ("workbench", "soperator", "fleet")


@pytest.mark.parametrize("tool_ref", _cli_backed_tool_refs())
def test_tool_ref_argv_resolves_to_a_real_command(tool_ref: str) -> None:
    command = resolve_argv_command(TOOL_CATALOG[tool_ref].argv_template)
    assert command.path.startswith(TOOL_REF_CLI_GROUPS)
    assert command.flags, f"{tool_ref}: {command.path} declares no options"


def test_non_cli_argv_entries_are_pinned() -> None:
    """Inline python/bash toolRefs are exempt; the exemption list may only shrink."""

    actual = {
        tool_ref
        for tool_ref, entry in TOOL_CATALOG.items()
        if not entry.stub
        and entry.argv_template
        and str(entry.argv_template[0]) != "npa"
    }
    unexpected = actual - NON_CLI_ARGV
    assert not unexpected, (
        "new non-CLI toolRef argv templates are unchecked by this guardrail; "
        f"prefer an `npa ...` invocation, or pin them explicitly: {sorted(unexpected)}"
    )
    stale = NON_CLI_ARGV - actual
    assert not stale, f"NON_CLI_ARGV lists entries that no longer exist: {sorted(stale)}"


#: Options typed as a plain ``str`` whose value genuinely IS a format word. Verified by
#: reading the command: `npa soperator deploy --output json` selects the output format
#: (the parameter is a str, not an Enum). Pinned so the set can only shrink.
FORMAT_STYLE_STR_OPTIONS = frozenset({"infra.soperator.deploy"})


def test_no_tool_ref_argv_passes_a_format_word_to_a_path_option() -> None:
    """`--output json` on a path option silently writes the artifact somewhere else.

    Live: `workbench.sonic.eval` passed "json" to `--output`, which is
    ``output_path: str`` (``--output-format`` is the format). The stage SUCCEEDED and
    the spec's declared eval.json never appeared — runs
    ``npa-wf-gpu-sonic-eval-87a704ad`` / ``npa-wf-multi-sonic-export-eval-744b9c1e``.
    """

    mismatches = {
        ref: problems
        for ref, problems in catalog_argv_literal_mismatches().items()
        if ref not in FORMAT_STYLE_STR_OPTIONS
    }
    assert not mismatches, (
        "toolRef argv passes a literal value its CLI option cannot mean:\n"
        + "\n".join(f"  {ref}: {problems}" for ref, problems in sorted(mismatches.items()))
    )


def test_sonic_eval_argv_separates_the_result_path_from_the_format() -> None:
    argv = [str(token) for token in TOOL_CATALOG["workbench.sonic.eval"].argv_template]

    assert "--output" in argv and argv[argv.index("--output") + 1] == "{{config.eval_uri}}"
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "json"


def test_format_style_str_options_are_still_str_typed() -> None:
    """Pin the exemption's premise: if the option becomes an Enum, drop it here."""

    from npa.guardrails.tool_catalog_argv import (
        _cli_parameters,
        _is_enum_annotation,
        _parameter_for_flag,
        _resolved_annotations,
        resolve_argv_command,
    )

    for ref in FORMAT_STYLE_STR_OPTIONS:
        command = resolve_argv_command(TOOL_CATALOG[ref].argv_template)
        param = _parameter_for_flag(_cli_parameters(command.callback_ref), "--output")
        assert param is not None, ref
        resolved = _resolved_annotations(command.callback_ref).get(param.name)
        assert not _is_enum_annotation(param, resolved), (
            f"{ref}: --output is now an Enum, so it no longer needs an exemption"
        )


def test_guardrail_detects_a_format_word_on_a_path_option() -> None:
    """Negative control for the literal-value check."""

    problems = argv_literal_value_mismatches(
        ["npa", "workbench", "sonic", "eval", "--onnx", "/p.onnx", "--output", "json"]
    )

    assert problems and "takes a path/URI" in problems[0]


def test_guardrail_detects_a_bad_enum_value() -> None:
    problems = argv_literal_value_mismatches(
        ["npa", "workbench", "mjlab", "eval", "--output", "totally-not-a-format"]
    )

    assert problems and "is not one of" in problems[0]


def test_guardrail_ignores_unresolved_templates() -> None:
    assert (
        argv_literal_value_mismatches(
            ["npa", "workbench", "mjlab", "eval", "--output", "{{config.fmt}}"]
        )
        == ()
    )


def test_guardrail_detects_an_invented_flag() -> None:
    """Negative control: the check must fail on a flag the command does not have."""

    argv = ["npa", "workbench", "mjlab", "eval", "--totally-invented-flag", "x"]

    assert argv_flag_drift("fixture.invented", argv) == ("--totally-invented-flag",)


def test_guardrail_detects_a_nonexistent_command() -> None:
    """Negative control: a bad CLI path is a harder failure than flag drift."""

    with pytest.raises(ArgvResolutionError, match="not a registered command"):
        resolve_argv_command(["npa", "workbench", "no-such-tool", "run", "--x", "1"])


def test_guardrail_rejects_a_group_without_a_subcommand() -> None:
    with pytest.raises(ArgvResolutionError, match="missing a subcommand"):
        resolve_argv_command(["npa", "workbench", "mjlab"])


def test_argv_template_flags_ignores_values() -> None:
    argv = ["npa", "workbench", "mjlab", "eval", "--input-path", "--not-a-flag-value"]

    # A value that merely *looks* like a flag is still reported; that is deliberate,
    # since the CLI would also read it as one.
    assert argv_template_flags(argv) == ("--input-path", "--not-a-flag-value")


def test_isaac_lab_rl_tool_refs_use_the_real_flag_names() -> None:
    """Pin the two fixes this guardrail found (DESIGN §7 drift).

    ``policy_train`` used ``--learning-rate`` / ``--batch-size`` / ``--input-path``
    and ``evaluate_policy`` used ``--episodes``; none of those exist on
    ``npa workbench isaac-lab {train,eval}``.
    """

    train = argv_template_flags(TOOL_CATALOG["workbench.rl.policy_train"].argv_template)
    assert "--learning-rate" not in train and "--batch-size" not in train
    assert "--input-path" not in train
    assert {"--task", "--steps", "--num-envs", "--override", "--data-path", "--output-path"} <= set(
        train
    )

    evaluate = argv_template_flags(
        TOOL_CATALOG["workbench.rl.evaluate_policy"].argv_template
    )
    assert "--episodes" not in evaluate
    assert "--num-episodes" in evaluate


def test_bash_wrapped_tool_refs_are_audited_too() -> None:
    """The blind spot that shipped a defect: `bash -c` toolRefs were never checked.

    ``create_failure_views`` passed ``--table`` to ``lancedb create-mv``, whose option is
    ``--source-table``. The audit skipped it because ``argv_template[0] != "npa"``, and the
    stage could only fail live with "No such option '--table'".
    """

    from npa.guardrails.tool_catalog_argv import embedded_npa_commands

    entry = TOOL_CATALOG["workbench.lancedb.create_failure_views"]

    assert entry.argv_template[0] == "bash"
    commands = embedded_npa_commands(entry.argv_template)
    # One `npa workbench lancedb create-mv` per failure-mode view.
    assert len(commands) == 3, commands
    for command in commands:
        assert command[:4] == ("npa", "workbench", "lancedb", "create-mv")
        assert "--source-table" in command
        assert "--table" not in command
        assert not argv_flag_drift("workbench.lancedb.create_failure_views", command)


def test_embedded_extraction_handles_a_loop_and_shell_keywords() -> None:
    """`backfill_cpu_bundle` wraps its call in a `for … do … done` loop."""

    from npa.guardrails.tool_catalog_argv import embedded_npa_commands

    commands = embedded_npa_commands(
        TOOL_CATALOG["workbench.lancedb.backfill_cpu_bundle"].argv_template
    )

    assert len(commands) == 1, commands
    command = commands[0]
    assert command[:4] == ("npa", "workbench", "lancedb", "backfill")
    assert "done" not in command
    assert not argv_flag_drift("workbench.lancedb.backfill_cpu_bundle", command)


def test_embedded_extraction_ignores_non_bash_and_bare_scripts() -> None:
    from npa.guardrails.tool_catalog_argv import embedded_npa_commands

    assert embedded_npa_commands(["npa", "workbench", "mjlab", "eval"]) == ()
    assert embedded_npa_commands(["python", "-c", "print('npa')"]) == ()
    assert embedded_npa_commands(["bash", "-c", "echo hello"]) == ()


def test_the_audit_would_catch_the_original_bash_defect() -> None:
    """Negative control: the exact broken command must be reported."""

    broken = (
        "npa",
        "workbench",
        "lancedb",
        "create-mv",
        "--name",
        "view",
        "--table",
        "bdd100k",
    )

    assert argv_flag_drift("workbench.lancedb.create_failure_views", broken) == ("--table",)
