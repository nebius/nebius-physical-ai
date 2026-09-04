from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import argv_for_tool
from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
PUSH = WORKFLOWS / "encord-push.yaml"
PULL = WORKFLOWS / "encord-pull.yaml"
ROUNDTRIP = WORKFLOWS / "encord-roundtrip-smoke.yaml"


def test_push_spec_is_a_single_terminal_state() -> None:
    spec = load_spec(PUSH)
    validate_spec(spec)
    assert spec.name == "encord-push"
    steps = [step.state for step in build_plan(spec, run_id="t").steps]
    assert steps == ["push"]
    outputs = spec.states["push"].outputs
    assert outputs and outputs[0].schema == "npa.encord.push_receipt.v1"
    assert outputs[0].uri.endswith("push_receipt.json")
    # Human curation happens between the workflows: push declares no inputs.
    assert not spec.states["push"].inputs


def test_pull_spec_is_a_single_terminal_state() -> None:
    spec = load_spec(PULL)
    validate_spec(spec)
    assert spec.name == "encord-pull"
    steps = [step.state for step in build_plan(spec, run_id="t").steps]
    assert steps == ["pull"]
    outputs = spec.states["pull"].outputs
    assert outputs and outputs[0].schema == "npa.encord.pull_manifest.v1"
    assert outputs[0].uri.endswith("manifest.json")


def test_roundtrip_smoke_chains_push_curate_then_both_pulls() -> None:
    spec = load_spec(ROUNDTRIP)
    validate_spec(spec)
    assert spec.name == "encord-roundtrip-smoke"
    plan = build_plan(spec, run_id="t")
    steps = [step.state for step in plan.steps]
    assert steps == ["push", "curate", "pull", "pull-curated", "verify"]
    assert spec.states["curate"].needs == ["push"]
    assert spec.states["pull"].needs == ["curate"]
    # needs states the true data dependency (the curate receipt), not the
    # linear next-chain position.
    assert spec.states["pull-curated"].needs == ["curate"]
    # The terminal verifier joins the push receipt to the dataset pull.
    assert spec.states["verify"].needs == ["pull"]
    assert spec.states["verify"].outputs[0].schema == "npa.encord.roundtrip_report.v1"
    # Schema chain: receipt -> curate receipt -> both pull manifests.
    assert spec.states["push"].outputs[0].schema == "npa.encord.push_receipt.v1"
    assert spec.states["curate"].inputs[0].schema == "npa.encord.push_receipt.v1"
    assert spec.states["curate"].outputs[0].schema == "npa.encord.curate_receipt.v1"
    assert spec.states["pull"].inputs[0].schema == "npa.encord.push_receipt.v1"
    assert spec.states["pull-curated"].inputs[0].schema == "npa.encord.curate_receipt.v1"
    # The e2e pulls the dataset push just created, resolved by run-scoped title.
    assert spec.config["encord_source"] == "dataset"
    assert spec.config["encord_source_id"] == spec.config["encord_dataset"]
    # The curate stage uses an intrinsic metric so it evaluates on any folder
    # without app-side quality-metric computation.
    assert spec.config["encord_curate_filters"].startswith("width:")
    # pull-curated pulls the headlessly curated Collection by run-scoped title.
    curated_argv = next(s.argv for s in plan.steps if s.state == "pull-curated")
    assert curated_argv[curated_argv.index("--source") + 1] == "collection"
    assert curated_argv[curated_argv.index("--source-id") + 1] == "npa-e2e-t"


def test_curate_argv_renders_every_flag() -> None:
    argv = argv_for_tool("workbench.encord.curate")
    assert argv[:4] == ["npa", "workbench", "encord", "curate"]
    for flag in (
        "--folder",
        "--filter",
        "--collection",
        "--poll-seconds",
        "--output-path",
        "--workflow-run",
        "--output",
    ):
        assert flag in argv, flag
    assert "{{run.id}}" in argv


def test_verify_stage_needs_no_encord_secret() -> None:
    """verify compares two S3 artifacts; only the Encord-facing verbs need the key."""

    step = lambda ref: SimpleNamespace(tool_ref=ref, argv=[])  # noqa: E731
    assert secret_env_hints_for_plan([step("workbench.encord.verify")]) == ()
    for ref in ("workbench.encord.push", "workbench.encord.curate", "workbench.encord.pull"):
        assert secret_env_hints_for_plan([step(ref)]) == ("ENCORD_SSH_KEY_B64",), ref


def test_verify_argv_renders_every_flag() -> None:
    argv = argv_for_tool("workbench.encord.verify")
    assert argv[:4] == ["npa", "workbench", "encord", "verify"]
    for flag in ("--receipt-uri", "--manifest-uri", "--output-path", "--workflow-run", "--output"):
        assert flag in argv, flag
    assert "{{run.id}}" in argv


def test_specs_declare_cpu_resource_blocks() -> None:
    for path in (PUSH, PULL, ROUNDTRIP):
        spec = load_spec(path)
        assert "cpu" in spec.resources, path.name
        profile = spec.resources["cpu"]
        assert profile.get("cloud") == "kubernetes", path.name
        assert "accelerators" not in profile, f"{path.name}: encord stages are CPU-only"
        for state in spec.states.values():
            assert state.resources == "cpu", f"{path.name}:{state.name}"


def test_push_argv_renders_every_flag() -> None:
    argv = argv_for_tool("workbench.encord.push")
    assert argv[:4] == ["npa", "workbench", "encord", "push"]
    for flag in (
        "--input-path",
        "--integration",
        "--folder",
        "--dataset",
        "--media",
        "--poll-timeout-seconds",
        "--output-path",
        "--workflow-run",
        "--output",
    ):
        assert flag in argv, flag
    assert "{{run.id}}" in argv


def test_pull_argv_renders_every_flag() -> None:
    argv = argv_for_tool("workbench.encord.pull")
    assert argv[:4] == ["npa", "workbench", "encord", "pull"]
    for flag in ("--source", "--source-id", "--output-path", "--workflow-run", "--output"):
        assert flag in argv, flag


def test_push_plan_omits_dataset_flag_when_empty() -> None:
    spec = load_spec(PUSH)
    assert "--dataset" in build_plan(spec, run_id="t").steps[0].argv
    spec.config["encord_dataset"] = ""
    argv = build_plan(spec, run_id="t").steps[0].argv
    assert "--dataset" not in argv
