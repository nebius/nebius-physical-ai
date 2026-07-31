"""Unit coverage for the live npa.workflow submit matrix (no cluster)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import (
    SUBMIT_LIVE_MATRIX,
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
    helpers = _load_live_helpers()
    assert spec in helpers.DYNAMIC_SPECS, "dynamic-gate spec must be in DYNAMIC_SPECS"
    assert helpers.assume_decision_for(spec) == "promote_checkpoint"


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
    for case in (c for c in SUBMIT_LIVE_MATRIX if c.runtime and not c.image_tool):
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
