"""Enforce the real-components skill for the Physical AI Data Factory blueprint.

Fails if the blueprint uses a known-stub toolRef, if a run.shell stage isn't a
real command/module call, or if the augment stage isn't the real Cosmos execute.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import typer
import yaml

from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.cli.agent_workflow import generate_data_factory_yaml, generate_sim2real_staged_yaml

BLUEPRINT = resolve_npa_workflow_spec("physical-ai-data-factory.yaml")
assert BLUEPRINT is not None, "physical-ai-data-factory.yaml not found in any spec root"

NUREC_BLUEPRINT = resolve_npa_workflow_spec("nurec-reconstruct.yaml")
assert NUREC_BLUEPRINT is not None, "nurec-reconstruct.yaml not found in any spec root"

NUREC_SKYPILOT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "npa"
    / "workbench"
    / "nurec"
    / "examples"
    / "nurec-reconstruct.yaml"
)

# toolRefs that only echo or write a contract/manifest — never advertise as real.
KNOWN_STUB_TOOLREFS = {
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


def _spec() -> dict:
    return yaml.safe_load(BLUEPRINT.read_text(encoding="utf-8"))


def _memory_gi(value: object) -> int:
    match = re.fullmatch(r"(\d+)Gi", str(value))
    assert match is not None, f"expected Gi memory value, got {value!r}"
    return int(match.group(1))


def test_blueprint_uses_no_stub_toolrefs() -> None:
    for name, state in _states().items():
        tool_ref = state.get("toolRef")
        if tool_ref:
            assert tool_ref not in KNOWN_STUB_TOOLREFS, (
                f"stage '{name}' uses stub toolRef '{tool_ref}'; wire the real component"
            )


@pytest.mark.parametrize(
    "yaml_text",
    [
        generate_data_factory_yaml(user_text="fan out 4 variants on 4 GPUs"),
        generate_sim2real_staged_yaml(user_text="Isaac sim2real with 2 outer iterations"),
    ],
)
def test_agent_generated_blueprints_use_only_real_toolrefs(yaml_text: str) -> None:
    spec = yaml.safe_load(yaml_text)
    for name, state in spec["states"].items():
        tool_ref = state.get("toolRef")
        if not tool_ref:
            continue
        assert tool_ref not in KNOWN_STUB_TOOLREFS, (name, tool_ref)
        assert TOOL_CATALOG[tool_ref].stub is False, (name, tool_ref)


def test_agent_generated_paidf_runs_named_real_components() -> None:
    spec = yaml.safe_load(generate_data_factory_yaml(user_text="fan out 2 variants on 2 GPUs"))
    states = spec["states"]
    assert states["grade"]["sequence"] == ["augment", "evaluate", "quality-gate"]
    assert states["evaluate"]["toolRef"] == "workbench.cosmos_evaluator.evaluate"
    assert states["cosmos-curate"]["toolRef"] == "workbench.cosmos_curate.curate"
    assert states["curate"]["toolRef"] == "workbench.fiftyone.curate_augmented"
    assert "--curator-report-uri" in TOOL_CATALOG[
        "workbench.fiftyone.curate_augmented"
    ].argv_template
    assert states["visualize"]["toolRef"] == "workbench.nurec.visualize"
    assert "pip install" not in str(states["visualize"])


def test_blueprint_run_shell_stages_are_real() -> None:
    for name, state in _states().items():
        run = state.get("run")
        if not run:
            continue
        command = str(run.get("shell", "")) or " ".join(
            str(item) for item in run.get("argv", [])
        )
        assert any(m in command for m in REAL_RUN_MARKERS), (
            f"stage '{name}' run is not a real command/module call: {command[:100]}"
        )


def test_augment_runs_real_cosmos_transfer() -> None:
    spec = _spec()
    states = _states()
    assert states["augment"].get("toolRef") == "workbench.cosmos2.transfer_execute", (
        "augment must run the real Cosmos Transfer 2.5 execute path"
    )
    argv = TOOL_CATALOG["workbench.cosmos2.transfer_execute"].argv_template
    assert "--execute" in argv, (
        "transfer_execute must pass --execute to run the real model"
    )
    assert "--condition-on-input" in argv
    assert "--input-uri" in argv and "--output-uri" in argv
    assert spec["config"]["trigger_uri"] == spec["config"]["input_uri"]
    description = states["augment"]["description"].lower()
    assert "input/conditioning.mp4" in description
    assert "no bundled or geometric fallback" in description


def test_input_conditioned_cosmos_toolref_fails_closed_without_input() -> None:
    argv = TOOL_CATALOG["workbench.cosmos2.transfer_conditioned_execute"].argv_template

    assert "--execute" in argv
    assert "--condition-on-input" in argv


def test_evaluate_runs_the_real_cosmos_evaluator() -> None:
    """The grade loop must grade with Cosmos Evaluator, not a generic VLM scorer."""

    states = _states()
    assert states["evaluate"].get("toolRef") == "workbench.cosmos_evaluator.evaluate", (
        "evaluate must run the real NVIDIA Cosmos Evaluator checks"
    )
    argv = TOOL_CATALOG["workbench.cosmos_evaluator.evaluate"].argv_template
    assert argv[:4] == ["npa", "workbench", "cosmos-evaluator", "evaluate"]
    # The hallucination check needs the run's source clip and attribute
    # verification needs the sampled option table, so both must be passed.
    assert "--input-uri" in argv and "--configs-uri" in argv
    for option in (
        "--temporal-mode",
        "--temporal-threshold",
        "--temporal-noise-floor",
        "--temporal-blur-ksize",
        "--temporal-regions-json",
        "--appearance-mode",
        "--appearance-threshold",
        "--appearance-regions-json",
        "--appearance-luminance-tolerance",
        "--appearance-global-chroma-tolerance",
        "--appearance-local-chroma-tolerance",
        "--appearance-chroma-instability-tolerance",
        "--appearance-blur-ksize",
        "--appearance-max-dimension",
    ):
        assert option in argv

    loop = states["grade"]["loop"]
    assert loop["until"] == "promote_checkpoint"
    assert states["grade"]["sequence"] == ["augment", "evaluate", "quality-gate"]
    assert states["grade"]["next"] == "quality-disposition"
    assert states["annotate-augmented"]["needs"] == ["quality-disposition"]
    assert (
        "enforce_quality_disposition" in states["quality-disposition"]["run"]["shell"]
    )
    assert float(_spec()["config"]["grade_threshold"]) >= 0.75
    assert float(_spec()["config"]["temporal_consistency_threshold"]) >= 0.8
    assert _spec()["config"]["temporal_consistency_mode"] == "advisory"
    assert float(_spec()["config"]["appearance_fidelity_threshold"]) >= 0.8
    assert _spec()["config"]["appearance_fidelity_mode"] == "advisory"


def test_curation_runs_the_real_cosmos_curator_before_review() -> None:
    """Curation must run Cosmos Curator, with FiftyOne reviewing its output."""

    states = _states()
    assert states["cosmos-curate"].get("toolRef") == "workbench.cosmos_curate.curate", (
        "cosmos-curate must run the real NVIDIA Cosmos Curator stages"
    )
    argv = TOOL_CATALOG["workbench.cosmos_curate.curate"].argv_template
    assert argv[:4] == ["npa", "workbench", "cosmos-curate", "curate-augmented"]
    assert "--curated-uri" in argv and "--report-uri" in argv

    assert states["cosmos-curate"]["next"] == "curate"
    assert states["curate"]["needs"] == ["cosmos-curate"]
    assert states["curate"]["toolRef"] == "workbench.fiftyone.curate_augmented"
    fiftyone_argv = TOOL_CATALOG["workbench.fiftyone.curate_augmented"].argv_template
    assert fiftyone_argv[:4] == ["npa", "workbench", "fiftyone", "curate-augmented"]
    assert "--curator-report-uri" in fiftyone_argv
    assert "--require-fiftyone" in fiftyone_argv


def test_quality_gate_reads_the_evaluator_report() -> None:
    from npa.workbench.cosmos_evaluator import RESULT_FILENAME

    states = _states()
    assert states["quality-gate"]["needs"] == ["evaluate"]
    outputs = [output["uri"] for output in states["evaluate"]["outputs"]]
    assert any(uri.endswith(RESULT_FILENAME) for uri in outputs), (
        f"evaluate must publish {RESULT_FILENAME}, which grade_gate reads"
    )


def test_gpu_resource_has_headroom_for_multi_variant_fanout() -> None:
    gpu = _spec()["resources"]["gpu"]
    assert int(gpu["cpus"]) >= 16, "4-way Cosmos fan-out needs CPU headroom"
    assert _memory_gi(gpu["memory"]) >= 128, (
        "4-way Cosmos fan-out OOMs with the old 16Gi profile"
    )


def test_blueprint_toolrefs_exist_in_catalog() -> None:
    for name, state in _states().items():
        tool_ref = state.get("toolRef")
        if tool_ref:
            assert tool_ref in TOOL_CATALOG, (
                f"stage '{name}' toolRef '{tool_ref}' not in catalog"
            )


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
        for match in re.finditer(
            r"npa\s+workbench\s+(\S+)\s+(\S+)((?:\s+--?\S+|\s+\"[^\"]*\"|\s+\S+)*)",
            shell,
        ):
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
    assert checked >= 1, (
        "expected at least one npa-workbench run.shell call to validate"
    )


# ---------------------------------------------------------------------------------
# NuRec / NRE neural reconstruction
# ---------------------------------------------------------------------------------
def _nurec_states() -> dict:
    spec = yaml.safe_load(NUREC_BLUEPRINT.read_text(encoding="utf-8"))
    return spec["states"]


def test_nurec_blueprint_uses_no_stub_toolrefs() -> None:
    for name, state in _nurec_states().items():
        tool_ref = state.get("toolRef")
        if tool_ref:
            assert tool_ref not in KNOWN_STUB_TOOLREFS, (
                f"stage '{name}' uses stub toolRef '{tool_ref}'; wire the real component"
            )


def test_nurec_blueprint_toolrefs_exist_in_catalog() -> None:
    for name, state in _nurec_states().items():
        tool_ref = state.get("toolRef")
        if tool_ref:
            assert tool_ref in TOOL_CATALOG, (
                f"stage '{name}' toolRef '{tool_ref}' not in catalog"
            )


def test_nurec_blueprint_run_shell_stages_are_real() -> None:
    for name, state in _nurec_states().items():
        run = state.get("run")
        if not run:
            continue
        shell = str(run.get("shell", ""))
        assert any(marker in shell for marker in REAL_RUN_MARKERS), (
            f"stage '{name}' run.shell is not a real command/module call: {shell[:100]}"
        )


def test_nurec_every_stage_runs_a_real_component() -> None:
    """No NuRec stage may be a bare description: each one runs a tool or a command."""
    for name, state in _nurec_states().items():
        assert state.get("toolRef") or state.get("run"), (
            f"stage '{name}' advertises work but invokes nothing"
        )


def test_nurec_reconstruct_stage_runs_the_real_nre_training_path() -> None:
    states = _nurec_states()

    assert states["reconstruct"].get("toolRef") == "workbench.nurec.reconstruct"
    argv = TOOL_CATALOG["workbench.nurec.reconstruct"].argv_template
    # Without the recipe and the artifact export there is no renderable scene.
    assert "--config-name" in argv
    assert "--export-gt" in argv
    assert "--output-uri" in argv


def test_nurec_render_stage_produces_novel_views_not_training_views() -> None:
    argv = TOOL_CATALOG["workbench.nurec.render"].argv_template

    assert "--rig-translation-offset" in argv
    assert "--no-replicate-training-views" in argv


def test_nurec_visualize_stage_builds_the_real_rerun_recording() -> None:
    states = _nurec_states()

    assert states["visualize"].get("toolRef") == "workbench.nurec.visualize"
    argv = TOOL_CATALOG["workbench.nurec.visualize"].argv_template
    assert "--output-uri" in argv


def test_nurec_skypilot_task_has_no_echo_or_manifest_stub_stage() -> None:
    """The submitted SkyPilot task must invoke the real tool for every stage."""
    doc = next(
        d for d in yaml.safe_load_all(NUREC_SKYPILOT.read_text(encoding="utf-8")) if d
    )
    run = doc["run"]

    for verb in ("check", "fetch", "reconstruct", "render", "visualize", "finalize"):
        assert f"npa workbench nurec {verb}" in run, verb
    assert "contract_ready" not in run
    # An `echo` that merely announces a stage is fine; one that stands IN for a
    # stage is not, so assert the tool call count matches the advertised stages.
    assert run.count("npa workbench nurec") >= 6
