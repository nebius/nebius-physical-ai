"""Unit coverage for the live npa.workflow submit matrix (no cluster)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.blueprints import (
    iter_npa_workflow_specs,
    resolve_npa_workflow_spec,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import (
    SUBMIT_LIVE_MATRIX,
    gpu_submit_cases,
    one_shot_submit_cases,
    runtime_submit_cases,
    selected_submit_cases,
)


def _load_live_helpers():
    path = Path(__file__).resolve().parents[2] / "e2e" / "npa_workflow_live_helpers.py"
    spec = importlib.util.spec_from_file_location("npa_workflow_live_helpers", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_force_accelerators_on_cpu_profiles() -> None:
    helpers = _load_live_helpers()
    src = """resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi
  gpu:
    cloud: kubernetes
    accelerators: H100:1
"""
    out = helpers._force_accelerators_on_cpu_profiles(src, "L40S:1")
    assert "accelerators: L40S:1" in out
    assert out.count("accelerators: L40S:1") == 1
    assert "accelerators: H100:1" in out
    assert "cloud: kubernetes" in out
    assert "cpus: 4+" in out
    assert "memory: 16+" in out
    assert "cpus: 4\n" not in out


def test_submit_live_matrix_specs_exist() -> None:
    missing = [
        case.spec
        for case in SUBMIT_LIVE_MATRIX
        if resolve_npa_workflow_spec(case.spec) is None
    ]
    assert not missing, f"matrix references missing specs: {missing}"


def test_every_shipped_catalog_spec_has_one_live_matrix_case() -> None:
    """A shipped spec without a matrix case is unobserved, not passing."""

    shipped = {path.name for path in iter_npa_workflow_specs()}
    registered = [case.spec for case in SUBMIT_LIVE_MATRIX]

    assert shipped, "expected a non-empty workflow catalog"
    assert not (shipped - set(registered)), (
        "shipped specs missing from SUBMIT_LIVE_MATRIX: "
        f"{sorted(shipped - set(registered))}"
    )
    assert not (set(registered) - shipped), (
        "SUBMIT_LIVE_MATRIX references specs outside the shipped catalog: "
        f"{sorted(set(registered) - shipped)}"
    )
    duplicates = sorted(name for name in set(registered) if registered.count(name) > 1)
    assert not duplicates, f"duplicate SUBMIT_LIVE_MATRIX cases: {duplicates}"


def test_plan_only_cases_have_machine_checked_justifications() -> None:
    for case in SUBMIT_LIVE_MATRIX:
        if case.plan_only:
            assert case.plan_only_justification.strip(), (
                f"{case.spec} is plan-only without an explicit justification"
            )
            assert len(case.plan_only_justification.split()) >= 5
        else:
            assert not case.plan_only_justification, (
                f"{case.spec} executes live and must not carry a plan-only exemption"
            )


def test_coverage_backfill_cases_are_honestly_plan_only() -> None:
    plan_only = {
        "adversarial-scenario-hardening.yaml",
        "av-night-scene-hardening.yaml",
        "byof-droid-policy-learning.yaml",
        "byof-maniskill.yaml",
        "byof-mujoco-playground.yaml",
        "byof-open-dreamer.yaml",
        "byof-openpi.yaml",
        "byof-robocasa.yaml",
        "cosmos-synth-fanout-curation.yaml",
        "hardening-with-insights.yaml",
        "sim2real-gpu-cross-region-agent.yaml",
    }

    for name in plan_only:
        case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == name)
        assert case.plan_only, f"{name} must retain its reviewed plan-only classification"


def test_reviewed_matrix_cases_have_honest_gpu_eligibility() -> None:
    incomplete = {
        "adversarial-scenario-hardening.yaml",
        "hardening-with-insights.yaml",
    }
    real_conditioned = {
        "sim2real-two-step-agent.yaml",
        "sim2real-two-step.yaml",
    }
    gpu_specs = {case.spec for case in gpu_submit_cases()}

    assert incomplete.isdisjoint(gpu_specs)
    assert real_conditioned <= gpu_specs


def test_reviewed_matrix_notes_disclose_the_execution_contracts() -> None:
    expected_fragments = {
        "adversarial-scenario-hardening.yaml": ("VM config", "does not consume", "/tmp"),
        "hardening-with-insights.yaml": ("VM config", "ignores evaluation", "/tmp"),
        "sim2real-two-step.yaml": (
            "dedicated toolRef",
            "seeded MP4",
            "frames list",
        ),
        "sim2real-two-step-agent.yaml": (
            "conditioned execute",
            "seeded MP4",
            "manifest",
        ),
    }

    for name, fragments in expected_fragments.items():
        case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == name)
        assert all(fragment in case.notes for fragment in fragments), case.notes


def test_standalone_cosmos_case_exercises_conditioned_real_toolref() -> None:
    from npa.orchestration.npa_workflow.interpreter import build_plan

    path = resolve_npa_workflow_spec("cosmos2-transfer.yaml")
    assert path is not None
    spec = load_spec(path)
    assert spec.states["transfer"].tool_ref == "workbench.cosmos2.transfer_execute"

    plan = build_plan(spec, run_id="generic-transfer-check")
    transfer = next(step for step in plan.steps if step.state == "transfer")

    assert "--execute" in transfer.argv
    assert "--condition-on-input" in transfer.argv


def test_groot_case_truthfully_describes_offline_configurable_training() -> None:
    case = next(
        case for case in SUBMIT_LIVE_MATRIX if case.spec == "groot-1-7-finetune.yaml"
    )

    assert case.tier == "multi"
    assert not case.plan_only
    assert case.image_tool == "groot"
    assert not case.config_vars
    assert "offline held-out baseline inference" in case.notes
    assert "one-to-many-GPU" in case.notes
    assert "learning outcome separately from pipeline status" in case.notes
    assert "not closed-loop or physical-robot task evidence" in case.notes


@pytest.mark.parametrize(
    "name", ["sim2real-two-step.yaml", "sim2real-two-step-agent.yaml"]
)
def test_two_step_real_augment_flows_into_envgen(name: str) -> None:
    from npa.orchestration.npa_workflow.interpreter import build_plan

    path = resolve_npa_workflow_spec(name)
    assert path is not None
    spec = load_spec(path)
    assert (
        spec.states["augment"].tool_ref
        == "workbench.cosmos2.transfer_conditioned_execute"
    )
    assert len(spec.states["envgen"].inputs) == 1
    assert spec.states["envgen"].inputs[0].uri == "{{config.augment_manifest_uri}}"
    assert spec.states["envgen"].inputs[0].schema == "npa.cosmos2.transfer.v1"

    plan = build_plan(spec, run_id="data-flow-check")
    augment = next(step for step in plan.steps if step.state == "augment")
    envgen = next(step for step in plan.steps if step.state == "envgen")
    augment_uri = augment.argv[augment.argv.index("--output-uri") + 1]
    consumed_uri = envgen.argv[envgen.argv.index("--augmented-frames-uri") + 1]

    assert "--execute" in augment.argv
    assert "--condition-on-input" in augment.argv
    assert consumed_uri == f"{augment_uri.rstrip('/')}/manifest.json"
    assert envgen.inputs == [
        {"uri": consumed_uri, "schema": "npa.cosmos2.transfer.v1"}
    ]


@pytest.mark.parametrize(
    ("name", "expected_key"),
    [
        (
            "sim2real-two-step.yaml",
            "sim2real-triggers/seed-run/lerobot-pusht/input.mp4",
        ),
        (
            "sim2real-two-step-agent.yaml",
            "sim2real-triggers/seed-run/lerobot-pusht/input.mp4",
        ),
        (
            "tokenfactory-cosmos-gate.yaml",
            "npa-workflow-e2e/seed-run/tokenfactory-cosmos-gate/scene/input.mp4",
        ),
    ],
)
def test_real_cosmos_cases_seed_an_actual_input_video(
    monkeypatch, name: str, expected_key: str
) -> None:
    helpers = _load_live_helpers()
    writes: list[dict[str, object]] = []

    class _S3:
        def put_object(self, **kwargs) -> None:
            writes.append(kwargs)

    monkeypatch.setattr(
        "npa.clients.project_credentials.s3_client_for_project",
        lambda *_args, **_kwargs: _S3(),
    )

    helpers.seed_live_workflow_inputs(
        spec_name=name,
        bucket="unit-bucket",
        run_id="seed-run",
    )

    videos = [item for item in writes if item.get("ContentType") == "video/mp4"]
    assert len(videos) == 1
    assert videos[0]["Key"] == expected_key
    body = bytes(videos[0]["Body"])
    assert len(body) > 1_000
    assert body[4:8] == b"ftyp"


def test_matrix_cases_declare_every_secret_the_renderer_hints_at() -> None:
    """A missing secret_env makes the CLI print an advisory line before its JSON.

    That line broke ``json.loads(result.output)`` in the harness and reported a
    *successful* sonic-export submit (SkyPilot job 189) as a test failure. The parser is
    now tolerant, and this keeps the matrix honest about what each spec needs.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan

    helpers = _load_live_helpers()
    missing: list[str] = []
    for case in SUBMIT_LIVE_MATRIX:
        path = resolve_npa_workflow_spec(case.spec)
        assert path is not None, case.spec
        spec = load_spec(path)
        plan = build_plan(
            spec,
            run_id="matrix-check",
            assume_decision=helpers.assume_decision_for(case.spec) or None,
        )
        hints = set(secret_env_hints_for_plan(plan.steps))
        gap = sorted(hints - set(case.secret_envs))
        if gap:
            missing.append(f"{case.spec}: {gap}")
    assert not missing, (
        "live-matrix cases must declare the secret envs the renderer hints at:\n"
        + "\n".join(missing)
    )


def test_selected_submit_cases_tier_filter(monkeypatch) -> None:
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu")
    monkeypatch.delenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS", raising=False)
    cases = selected_submit_cases()
    assert cases
    assert all(case.tier == "cpu" for case in cases)


def test_selected_submit_cases_spec_filter(monkeypatch) -> None:
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu,gpu,multi")
    monkeypatch.setenv(
        "NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS",
        "vlm-eval-single.yaml,token-factory-caption.yaml",
    )
    cases = selected_submit_cases()
    assert {case.spec for case in cases} == {
        "vlm-eval-single.yaml",
        "token-factory-caption.yaml",
    }


def test_selected_submit_cases_explicit_empty_filter_fails(monkeypatch) -> None:
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu,gpu,multi")
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS", "typo.yaml")
    with pytest.raises(ValueError, match="selected no npa\\.workflow cases"):
        selected_submit_cases()


def test_live_submit_wrapper_preserves_explicit_operator_environment() -> None:
    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "npa-workflow-submit-live-e2e.sh"
    ).read_text(encoding="utf-8")
    snapshot = script.index("declare -A _npa_explicit_env_values=()")
    cloud_defaults = script.index("source /home/ubuntu/bin/npa-cloud-env.sh")
    user_defaults = script.index('. "${HOME}/.npa/live-e2e.env"')
    restore = script.index('for _npa_env_name in "${!_npa_explicit_env_values[@]}"')
    python_select = script.index('PY="${NPA_LIVE_E2E_PYTHON_BIN:')

    assert snapshot < cloud_defaults < user_defaults < restore < python_select
    assert "NPA_*|SKYPILOT_DOCKER_*" in script
    assert 'printf -v "$_npa_env_name"' in script
    assert 'export "$_npa_env_name"' in script


def test_submit_live_matrix_has_cpu_gpu_and_multi() -> None:
    tiers = {case.tier for case in SUBMIT_LIVE_MATRIX}
    assert tiers == {"cpu", "gpu", "multi"}
    assert any(case.plan_only for case in SUBMIT_LIVE_MATRIX)
    assert any(not case.plan_only and case.tier == "gpu" for case in SUBMIT_LIVE_MATRIX)


def test_physical_ai_data_factory_registered_for_live_infra() -> None:
    """Backs the author-npa-workflow / testing-conventions rule: a new spec with a
    dynamic gate must be in SUBMIT_LIVE_MATRIX and DYNAMIC_SPECS."""
    spec = "physical-ai-data-factory.yaml"
    matrix_case = next((c for c in SUBMIT_LIVE_MATRIX if c.spec == spec), None)
    assert matrix_case is not None, "physical-ai-data-factory.yaml missing from SUBMIT_LIVE_MATRIX"
    assert matrix_case.requires_token_factory
    assert matrix_case.tier == "multi"
    assert matrix_case.runtime
    assert not matrix_case.plan_only
    assert dict(matrix_case.config_vars)["n_augmentations"] == "1"
    assert dict(matrix_case.image_overrides) == {
        "workbench.cosmos2.transfer_execute": "cosmos2-transfer",
        "workbench.cosmos_evaluator.evaluate": "cosmos-evaluator",
        "workbench.cosmos_curate.curate": "cosmos-curate",
        "workbench.fiftyone.curate_augmented": "fiftyone",
    }
    helpers = _load_live_helpers()
    assert spec in helpers.DYNAMIC_SPECS, "dynamic-gate spec must be in DYNAMIC_SPECS"
    assert helpers.assume_decision_for(spec) == "promote_checkpoint"


def test_cosmos3_generate_registered_for_live_infra() -> None:
    """The Cosmos 3 generate twin is the live proof for retiring its raw template."""

    case = next((c for c in SUBMIT_LIVE_MATRIX if c.spec == "cosmos3-generate.yaml"), None)

    assert case is not None, "cosmos3-generate.yaml missing from SUBMIT_LIVE_MATRIX"
    assert case.tier == "gpu"
    assert case.image_tool == "cosmos3"
    assert {"HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"} <= set(case.secret_envs)
    assert not case.plan_only
    assert not case.runtime


@pytest.mark.parametrize(
    "spec",
    ["byof-wan2.2.yaml", "byof-wan2.2-multigpu.yaml"],
)
def test_wan_submit_cases_forward_runtime_acceptance_as_secret(spec: str) -> None:
    case = next(
        (candidate for candidate in SUBMIT_LIVE_MATRIX if candidate.spec == spec), None
    )

    assert case is not None
    assert {
        "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS",
        "HF_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    } <= set(case.secret_envs)


# --------------------------------------------------------------- runtime cases


def _case(spec: str):
    return next((c for c in SUBMIT_LIVE_MATRIX if c.spec == spec), None)


def test_runtime_specs_are_registered_with_the_right_tiers() -> None:
    """Parallel / runtime-gate specs must be in the matrix so live CI covers them."""

    expected = {
        "token-factory-parallel-fanout.yaml": "cpu",
        "token-factory-gate-loop.yaml": "cpu",
        "isaac-lab-rl-sweep.yaml": "multi",
    }
    for spec, tier in expected.items():
        case = _case(spec)
        assert case is not None, f"{spec} missing from SUBMIT_LIVE_MATRIX"
        assert case.runtime, f"{spec} needs runtime=True (it cannot run one-shot)"
        assert case.tier == tier
        assert not case.plan_only, f"{spec} is the live proof; it must not be plan-only"


def test_runtime_cases_declare_their_secrets_and_are_not_plan_only() -> None:
    for case in (c for c in SUBMIT_LIVE_MATRIX if c.runtime):
        assert case.secret_envs, f"{case.spec} must declare the secrets its tasks need"
        assert "AWS_ACCESS_KEY_ID" in case.secret_envs
        assert not case.plan_only


def test_every_live_case_declares_the_object_store_credentials_setup_needs() -> None:
    """`setup:` syncs the npa source from S3 with boto3, so EVERY live case needs the keys.

    Learned the expensive way: `cosmos-fetch` declared only `HF_TOKEN`, because nothing in its
    *plan* touches object storage — and the run died in **setup** with
    ``botocore.exceptions.NoCredentialsError`` before either stage started.
    ``test_matrix_cases_declare_every_secret_the_renderer_hints_at`` cannot see this: the need
    comes from the source-staging setup, not from a toolRef's argv.
    """

    required = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
    missing = sorted(
        f"{case.spec}: {sorted(required - set(case.secret_envs))}"
        for case in SUBMIT_LIVE_MATRIX
        if not case.plan_only and not required <= set(case.secret_envs)
    )

    assert not missing, (
        "every live case's setup stages npa from NPA_SRC_S3_URI with boto3, so each must "
        "declare the object-store credentials:\n  " + "\n  ".join(missing)
    )


def test_expected_parallel_tasks_matches_the_spec_fan_out() -> None:
    """Matrix metadata must not drift from the spec it describes.

    The live test asserts that exactly ``expected_parallel_tasks`` tasks ran
    concurrently, so this number has to equal the spec's largest parallel group.
    """

    for case in (c for c in SUBMIT_LIVE_MATRIX if c.expected_parallel_tasks):
        path = resolve_npa_workflow_spec(case.spec)
        assert path is not None, case.spec
        spec = load_spec(path)
        groups = [state for state in spec.states.values() if state.parallel]
        assert groups, f"{case.spec} declares expected_parallel_tasks but has no parallel group"
        assert max(len(state.parallel) for state in groups) == case.expected_parallel_tasks


def test_specs_with_a_parallel_group_are_registered_as_runtime_cases() -> None:
    """A `parallel:` spec submitted one-shot would silently serialize; forbid it."""

    for case in SUBMIT_LIVE_MATRIX:
        path = resolve_npa_workflow_spec(case.spec)
        assert path is not None, case.spec
        spec = load_spec(path)
        if any(state.parallel for state in spec.states.values()):
            assert case.runtime, (
                f"{case.spec} has a parallel group, so it must be marked runtime=True"
            )
            assert case.expected_parallel_tasks >= 2


def test_gate_loop_case_drives_the_gate_through_config_vars() -> None:
    case = _case("token-factory-gate-loop.yaml")
    assert case is not None
    assert dict(case.config_vars).get("grade_threshold") == "0.0", (
        "the default live run must pass the gate on iteration 1 (early-exit proof)"
    )
    helpers = _load_live_helpers()
    assert "token-factory-gate-loop.yaml" in helpers.DYNAMIC_SPECS
    assert helpers.assume_decision_for("token-factory-gate-loop.yaml") == "promote_checkpoint"


def test_image_tool_is_a_known_container_image(monkeypatch) -> None:
    from npa.deploy.images import container_image_for_tool

    for case in (c for c in SUBMIT_LIVE_MATRIX if c.image_tool):
        image = container_image_for_tool(case.image_tool, registry="cr.example.invalid/reg")
        assert image.startswith("cr.example.invalid/reg/"), case.spec
    for case in (c for c in SUBMIT_LIVE_MATRIX if c.image_overrides):
        for _tool_ref, tool in case.image_overrides:
            image = container_image_for_tool(tool, registry="cr.example.invalid/reg")
            assert image.startswith("cr.example.invalid/reg/"), case.spec


def test_runtime_and_one_shot_selections_are_disjoint(monkeypatch) -> None:
    """The two live tests must never submit the same case twice."""

    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu,gpu,multi")
    monkeypatch.delenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS", raising=False)
    runtime = {case.spec for case in runtime_submit_cases()}
    one_shot = {case.spec for case in one_shot_submit_cases()}
    assert runtime and one_shot
    assert not (runtime & one_shot)
    assert runtime | one_shot == {
        case.spec for case in selected_submit_cases() if not (case.runtime and case.plan_only)
    }


def test_runtime_selection_honours_the_tier_filter(monkeypatch) -> None:
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu")
    monkeypatch.delenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS", raising=False)
    specs = {case.spec for case in runtime_submit_cases()}
    assert specs == {
        "token-factory-parallel-fanout.yaml",
        "token-factory-gate-loop.yaml",
        "token-factory-trigger-watch.yaml",
    }
    assert "isaac-lab-rl-sweep.yaml" not in specs  # multi tier


def test_slow_cases_carry_their_own_deadline() -> None:
    """A slow case must not force the whole runtime tier onto its deadline."""

    sweep = _case("isaac-lab-rl-sweep.yaml")
    assert sweep is not None
    assert sweep.max_wait_seconds >= 3600, (
        "the GPU sweep pulls an ~8 GB image and trains; it needs a longer per-wave "
        "deadline than the CPU cases"
    )
    for case in (
        c
        for c in SUBMIT_LIVE_MATRIX
        if c.runtime and not c.image_tool and not c.image_overrides
    ):
        assert case.max_wait_seconds == 0, (
            f"{case.spec} should inherit the env deadline instead of pinning one"
        )


def test_image_tool_override_env_name_is_derived_from_the_tool() -> None:
    """The operator hook name must match what the runner exports."""

    sweep = _case("isaac-lab-rl-sweep.yaml")
    assert sweep is not None and sweep.image_tool == "isaac-lab"
    expected = "NPA_E2E_IMAGE_OVERRIDE_ISAAC_LAB"
    derived = f"NPA_E2E_IMAGE_OVERRIDE_{sweep.image_tool.upper().replace('-', '_')}"
    assert derived == expected


def test_gpu_sweep_live_case_caps_concurrency_for_cost() -> None:
    """The live sweep must not hold four GPUs at once, and must batch."""

    sweep = _case("isaac-lab-rl-sweep.yaml")
    assert sweep is not None
    vars_ = dict(sweep.config_vars)
    assert vars_.get("max_concurrency") == "2"
    # 4 members with maxConcurrency 2 means the runtime submits two JobGroups, which
    # is also the only live coverage of the multi-batch path.
    assert sweep.expected_parallel_tasks == 4
