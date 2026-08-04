"""Guardrail: CLI, SDK and the npa.workflow surface stay coherent per capability.

The third tier used to be a raw SkyPilot task YAML ("the YAML declares an ``envs``
key per CLI flag"). As that catalog is retired, the third tier moves onto the
surface that survives: the shipped ``npa.workflow`` spec plus the ``toolRef`` argv
template the engine expands. See ``npa/src/npa/guardrails/three_tier.py`` for why
that is sharper in one direction and narrower in another, and
``test_tool_catalog_argv.py`` for the catalog-wide flag check that the narrowing
buys us.

Each contract pins ``spec_gap``: the CLI parameters a spec author *cannot* set
today. That is a real capability regression against the SkyPilot YAML, so it is
recorded per contract rather than quietly dropped, and
``test_spec_gaps_are_categorised`` forces every entry to have a stated reason.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from npa.guardrails.three_tier import (
    CapabilityContract,
    ParameterContract,
    registered_workbench_tools,
    validate_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = Path("npa/workflows/workbench/npa-workflows")


def _p(cli_param: str, sdk_param: str, cli_flag: str, yaml_env: str = "") -> ParameterContract:
    return ParameterContract(
        cli_param=cli_param, sdk_param=sdk_param, cli_flag=cli_flag, yaml_env=yaml_env
    )


# --------------------------------------------------------------------------- gaps
#
# Why each parameter is unreachable from a spec today. Categories:
#
#   boolean   - the CLI option is a paired/flag-only boolean
#               (`--headless/--no-headless`, `--verify/--no-verify`,
#               `--individual/--combined`, bare `--sample-data`). A v0.0.1 argv
#               template is a fixed list with no conditional rendering, so a
#               boolean cannot be expressed as `--flag {{config.x}}`. Closing these
#               needs a spec-level conditional-argv feature.
#   infra     - the option selects infrastructure (`--image`, `--gpu-type`,
#               `--image-variant`). The engine already owns image and accelerator
#               selection through `resources.<profile>`; passing it again inside the
#               pod would nest infrastructure choices (the same trap DESIGN §7
#               records for `workbench.rl.policy_train`).
#   knob      - a plain value the argv template simply does not pass yet. These are
#               the ones worth closing, tool by tool, with a live run each.
#
SPEC_GAP_REASONS: dict[str, dict[str, str]] = {
    "workflow/trigger/run": {
        # Where to watch and what has already been seen: driver state, not a stage input.
        "s3_endpoint": "infra",
        "s3_bucket": "infra",
        "s3_prefix": "infra",
        "watermark_uri": "infra",
        # Which spec to submit, and how — the driver's own launch settings.
        "pipeline_yaml": "infra",
        "pipeline_bucket": "infra",
        "pipeline_s3_prefix": "infra",
        "pipeline_input_data_uri": "infra",
        "pipeline_render_only": "boolean",
        "task_cloud": "infra",
        "controller_backend": "infra",
        "gpu": "infra",
        "gpu_failover": "infra",
        "submit_timeout": "knob",
    },
    "workflow/trigger/watch": {
        "s3_endpoint": "infra",
        "s3_bucket": "infra",
        "s3_prefix": "infra",
        # Loop shape: how often to look and when to stop. A stage runs once.
        "poll_interval": "knob",
        "max_polls": "knob",
        "max_launches": "knob",
    },
    "sonic/train": {
        "sample_data": "boolean",
        "headless": "boolean",
        "image": "infra",
        "gpu_type": "infra",
        "image_variant": "infra",
        "embodiment": "knob",
        "num_envs": "knob",
    },
    "sonic/export": {
        "normalize": "knob",
        "verify": "boolean",
        "opset": "knob",
        "axes": "knob",
        "metadata": "knob",
        "obs_spec": "knob",
        "action_spec": "knob",
        "config": "knob",
        "parity_atol": "knob",
    },
    "sonic/retargeting/run": {
        "individual": "boolean",
        "retarget_map": "knob",
        "frame_rate": "knob",
        "source_frame_rate": "knob",
        "max_frames": "knob",
        "num_workers": "knob",
    },
    "vlm-eval/run": {
        "task": "knob",
        "model": "knob",
        "endpoint_url": "knob",
        "frame_selection": "knob",
        "max_frames": "knob",
        "success_threshold": "knob",
    },
    "cosmos2/transfer": {
        "assets_uri": "knob",
        "scene_spec_uri": "knob",
        "image": "infra",
    },
    "cosmos3/reason": {
        "image": "infra",
        "prompt": "knob",
    },
}

VALID_GAP_CATEGORIES = frozenset({"boolean", "infra", "knob"})


CONTRACTS: tuple[CapabilityContract, ...] = (
    CapabilityContract(
        name="sonic/train",
        cli_module="npa.cli.workbench.sonic.train",
        cli_callback="train_cmd",
        sdk_module="npa.sdk.workbench.sonic",
        sdk_attr="train",
        spec_path=SPECS / "sonic-train.yaml",
        tool_ref="workbench.sonic.train",
        spec_gap=(
            "sample_data",
            "embodiment",
            "num_envs",
            "headless",
            "image",
            "gpu_type",
            "image_variant",
        ),
        params=(
            _p("checkpoint", "checkpoint", "--checkpoint"),
            _p("data_path", "data_path", "--data-path"),
            _p("sample_data", "sample_data", "--sample-data"),
            _p("embodiment", "embodiment", "--embodiment"),
            _p("num_envs", "num_envs", "--num-envs"),
            _p("headless", "headless", "--headless"),
            _p("max_iterations", "max_iterations", "--max-iterations"),
            _p("output_path", "output_path", "--output-path"),
            _p("image", "image", "--image"),
            _p("gpu_type", "gpu_type", "--gpu-type"),
            _p("image_variant", "image_variant", "--image-variant"),
        ),
    ),
    CapabilityContract(
        name="sonic/export",
        cli_module="npa.cli.workbench.sonic.export",
        cli_callback="export_cmd",
        sdk_module="npa.sdk.workbench.sonic",
        sdk_attr="export_onnx",
        spec_path=SPECS / "sonic-export.yaml",
        tool_ref="workbench.sonic.export",
        spec_gap=(
            "opset",
            "axes",
            "normalize",
            "metadata",
            "obs_spec",
            "action_spec",
            "config",
            "verify",
            "parity_atol",
        ),
        params=(
            _p("checkpoint", "checkpoint", "--checkpoint"),
            _p("output_path", "output", "--output"),
            _p("opset", "opset", "--opset"),
            _p("axes", "axes", "--axes"),
            _p("normalize", "normalize", "--normalize"),
            _p("metadata", "metadata", "--metadata"),
            _p("obs_spec", "obs_spec", "--obs-spec"),
            _p("action_spec", "action_spec", "--action-spec"),
            _p("config", "config", "--config"),
            _p("verify", "verify", "--verify"),
            _p("parity_atol", "parity_atol", "--parity-atol"),
        ),
    ),
    CapabilityContract(
        name="sonic/retargeting/run",
        cli_module="npa.cli.workbench.retargeting",
        cli_callback="run_cmd",
        sdk_module="npa.sdk.workbench.retargeting",
        sdk_attr="run",
        spec_path=SPECS / "retargeting.yaml",
        tool_ref="workbench.retargeting.run",
        spec_gap=(
            "retarget_map",
            "frame_rate",
            "source_frame_rate",
            "max_frames",
            "individual",
            "num_workers",
        ),
        params=(
            _p("input_path", "input_path", "--input-path"),
            _p("output_path", "output_path", "--output-path"),
            _p("source_format", "source_format", "--source-format"),
            _p("embodiment", "embodiment", "--embodiment"),
            _p("retarget_map", "retarget_map", "--retarget-map"),
            _p("frame_rate", "frame_rate", "--frame-rate"),
            _p("source_frame_rate", "source_frame_rate", "--source-frame-rate"),
            _p("max_frames", "max_frames", "--max-frames"),
            _p("individual", "individual", "--individual"),
            _p("num_workers", "num_workers", "--num-workers"),
        ),
    ),
    CapabilityContract(
        name="vlm-eval/run",
        cli_module="npa.cli.workbench.vlm_eval",
        cli_callback="run_cmd",
        sdk_module="npa.sdk.workbench.vlm_eval",
        sdk_attr="run",
        spec_path=SPECS / "vlm-eval-single.yaml",
        tool_ref="workbench.vlm_eval.run",
        spec_gap=(
            "task",
            "model",
            "endpoint_url",
            "frame_selection",
            "max_frames",
            "success_threshold",
        ),
        params=(
            _p("input_path", "input_path", "--input-path"),
            _p("output_path", "output_path", "--output-path"),
            _p("task", "task", "--task"),
            _p("backend", "backend", "--backend"),
            _p("model", "model", "--model"),
            _p("endpoint_url", "endpoint_url", "--endpoint-url"),
            _p("frame_selection", "frame_selection", "--frame-selection"),
            _p("max_frames", "max_frames", "--max-frames"),
            _p("success_threshold", "success_threshold", "--success-threshold"),
        ),
    ),
    CapabilityContract(
        name="cosmos2/transfer",
        cli_module="npa.cli.workbench.cosmos2",
        cli_callback="transfer_cmd",
        sdk_module="npa.sdk.workbench.cosmos2",
        sdk_attr="transfer",
        spec_path=SPECS / "cosmos-synth-fanout-curation.yaml",
        tool_ref="workbench.cosmos2.transfer_conditioned_execute",
        spec_gap=("assets_uri", "scene_spec_uri", "image"),
        params=(
            _p("input_uri", "input_uri", "--input-uri"),
            _p("output_uri", "output_uri", "--output-uri"),
            _p("assets_uri", "assets_uri", "--assets-uri"),
            _p("scene_spec_uri", "scene_spec_uri", "--scene-spec-uri"),
            _p("image", "image", "--image"),
            _p("run_id", "run_id", "--run-id"),
        ),
    ),
    CapabilityContract(
        name="cosmos3/reason",
        cli_module="npa.cli.workbench.cosmos3",
        cli_callback="reason_cmd",
        sdk_module="npa.sdk.workbench.cosmos3",
        sdk_attr="reason",
        spec_path=SPECS / "cosmos3-reason.yaml",
        tool_ref="workbench.cosmos3.reason",
        spec_gap=("image", "prompt"),
        params=(
            _p("input_uri", "input_uri", "--input-uri"),
            _p("output_uri", "output_uri", "--output-uri"),
            _p("model", "model", "--model"),
            _p("image", "image", "--image"),
            _p("prompt", "prompt", "--prompt"),
            _p("run_id", "run_id", "--run-id"),
        ),
    ),
    CapabilityContract(
        name="detection-training/train",
        cli_module="npa.cli.workbench.detection_training",
        cli_callback="train_cmd",
        sdk_module="npa.sdk.workbench.detection_training",
        sdk_attr="train",
        spec_path=SPECS / "bdd100k-pipeline.yaml",
        tool_ref="workbench.detection_training.train_nighttime",
        params=(
            _p("view", "view", "--view"),
            _p("output_uri", "output_uri", "--output-uri"),
            _p("lance_uri", "lance_uri", "--lance-uri"),
            _p("epochs", "epochs", "--epochs"),
            _p("batch_size", "batch_size", "--batch-size"),
            _p("learning_rate", "learning_rate", "--learning-rate"),
        ),
    ),
    CapabilityContract(
        name="detection-training/eval",
        cli_module="npa.cli.workbench.detection_training",
        cli_callback="eval_cmd",
        sdk_module="npa.sdk.workbench.detection_training",
        sdk_attr="eval",
        spec_path=SPECS / "bdd100k-pipeline.yaml",
        tool_ref="workbench.detection_training.eval_nighttime",
        params=(
            _p("eval_view", "eval_view", "--eval-view"),
            _p("output_uri", "output_uri", "--output-uri"),
            _p("lance_uri", "lance_uri", "--lance-uri"),
        ),
    ),
    # --- the watcher: a DRIVER, so its third tier is the spec it submits --------
    # `sim-to-real-trigger.yaml` is retired. Its stage ran the watch loop, and the loop's
    # third tier was that YAML's `envs:`. The loop now submits `sim2real-vlm-rl.yaml`, so the
    # spec is the third tier — but only for the parameters that describe the RUN. The watch
    # parameters (where to look, how often, what it has already seen) are driver state with no
    # stage analogue, which is exactly what `spec_gap` is for.
    CapabilityContract(
        name="workflow/trigger/run",
        cli_module="npa.cli.workbench.trigger",
        cli_callback="run_cmd",
        sdk_module="npa.sdk.workbench.trigger",
        sdk_attr="run_once",
        spec_path=SPECS / "sim2real-vlm-rl.yaml",
        tool_ref="workbench.cosmos2.transfer_conditioned_execute",
        spec_gap=(
            "s3_endpoint",
            "s3_bucket",
            "s3_prefix",
            "watermark_uri",
            "pipeline_yaml",
            "pipeline_bucket",
            "pipeline_s3_prefix",
            "pipeline_input_data_uri",
            "pipeline_render_only",
            "task_cloud",
            "controller_backend",
            "gpu",
            "gpu_failover",
            "submit_timeout",
        ),
        params=(
            _p("s3_endpoint", "s3_endpoint", "--s3-endpoint", "NPA_TRIGGER_S3_ENDPOINT"),
            _p("s3_bucket", "s3_bucket", "--s3-bucket", "NPA_TRIGGER_S3_BUCKET"),
            _p("s3_prefix", "s3_prefix", "--s3-prefix", "NPA_TRIGGER_S3_PREFIX"),
            _p("watermark_uri", "watermark_uri", "--watermark-uri", "NPA_TRIGGER_WATERMARK_URI"),
            _p("pipeline_yaml", "pipeline_yaml", "--pipeline-yaml", "NPA_TRIGGER_PIPELINE_YAML"),
            _p(
                "pipeline_bucket",
                "pipeline_bucket",
                "--pipeline-bucket",
                "NPA_TRIGGER_PIPELINE_BUCKET",
            ),
            _p(
                "pipeline_s3_prefix",
                "pipeline_s3_prefix",
                "--pipeline-s3-prefix",
                "NPA_TRIGGER_PIPELINE_S3_PREFIX",
            ),
            _p(
                "pipeline_input_data_uri",
                "pipeline_input_data_uri",
                "--pipeline-input-data-uri",
                "NPA_TRIGGER_PIPELINE_INPUT_DATA_URI",
            ),
            _p(
                "pipeline_render_only",
                "pipeline_render_only",
                "--pipeline-render-only",
                "NPA_TRIGGER_PIPELINE_RENDER_ONLY",
            ),
            _p("task_cloud", "task_cloud", "--task-cloud", "NPA_TRIGGER_TASK_CLOUD"),
            _p(
                "controller_backend",
                "controller_backend",
                "--controller-backend",
                "NPA_TRIGGER_CONTROLLER_BACKEND",
            ),
            _p("gpu", "gpu", "--gpu", "NPA_GPU_TYPE"),
            _p("gpu_failover", "gpu_failover", "--gpu-failover", "NPA_GPU_FAILOVER"),
            _p("submit_timeout", "submit_timeout", "--submit-timeout", "NPA_TRIGGER_SUBMIT_TIMEOUT"),
        ),
    ),
    CapabilityContract(
        name="workflow/trigger/watch",
        cli_module="npa.cli.workbench.trigger",
        cli_callback="watch_cmd",
        sdk_module="npa.sdk.workbench.trigger",
        sdk_attr="watch",
        spec_path=SPECS / "sim2real-vlm-rl.yaml",
        tool_ref="workbench.cosmos2.transfer_conditioned_execute",
        spec_gap=(
            "s3_endpoint",
            "s3_bucket",
            "s3_prefix",
            "poll_interval",
            "max_polls",
            "max_launches",
        ),
        params=(
            _p("s3_endpoint", "s3_endpoint", "--s3-endpoint", "NPA_TRIGGER_S3_ENDPOINT"),
            _p("s3_bucket", "s3_bucket", "--s3-bucket", "NPA_TRIGGER_S3_BUCKET"),
            _p("s3_prefix", "s3_prefix", "--s3-prefix", "NPA_TRIGGER_S3_PREFIX"),
            _p("poll_interval", "poll_interval", "--poll-interval", "NPA_TRIGGER_POLL_INTERVAL"),
            _p("max_polls", "max_polls", "--max-polls", "NPA_TRIGGER_MAX_POLLS"),
            _p("max_launches", "max_launches", "--max-launches", "NPA_TRIGGER_MAX_LAUNCHES"),
        ),
    ),
)

#: Capabilities whose third tier is still a raw SkyPilot YAML. This set may only
#: shrink: every entry is a capability whose npa.workflow twin does not exist yet.
#: Capabilities whose third tier is still a raw SkyPilot YAML's `envs:`. Empty, and it must
#: stay that way: the last two (the sim-to-real watcher's run/watch pair) moved onto the spec
#: they submit when their template retired.
LEGACY_YAML_TIER: frozenset[str] = frozenset()


def test_current_three_tier_contracts_are_coherent() -> None:
    failures: list[str] = []
    for contract in CONTRACTS:
        failures.extend(validate_contract(contract, repo_root=REPO_ROOT))
    assert not failures, "\n".join(failures)


def test_no_contract_remains_on_the_legacy_yaml_tier() -> None:
    """The tier is empty, so the rule flips from "only shrinks" to "must stay empty".

    `LEGACY_YAML_TIER` was the shrinking list of capabilities whose third tier was a raw
    SkyPilot YAML's `envs:`. It is now empty: every contract points at an npa.workflow spec and
    a toolRef. Re-adding a `yaml_path` would put a capability back on a surface this work
    removed, so it fails here rather than being pinned as a straggler.
    """

    actual = {contract.name for contract in CONTRACTS if contract.yaml_path is not None}
    assert actual == LEGACY_YAML_TIER, (
        "a capability moved onto (or off) the legacy SkyPilot-YAML third tier; "
        f"expected {sorted(LEGACY_YAML_TIER)}, found {sorted(actual)}"
    )
    for contract in CONTRACTS:
        if contract.name in LEGACY_YAML_TIER:
            continue
        assert contract.spec_path is not None and contract.tool_ref, (
            f"{contract.name}: migrated contracts need spec_path + tool_ref"
        )


def test_spec_gaps_are_categorised() -> None:
    """A pinned capability regression must carry a stated reason, per parameter."""

    for contract in CONTRACTS:
        reasons = SPEC_GAP_REASONS.get(contract.name, {})
        assert set(reasons) == set(contract.spec_gap), (
            f"{contract.name}: SPEC_GAP_REASONS must explain exactly the pinned "
            f"spec_gap; reasons={sorted(reasons)} gap={sorted(contract.spec_gap)}"
        )
        for param, category in reasons.items():
            assert category in VALID_GAP_CATEGORIES, (
                f"{contract.name}.{param}: unknown gap category {category!r}"
            )


def test_sim2real_headline_workflow_is_three_tier_coherent() -> None:
    # The sim2real headline workflow uses a **overrides SDK surface, so it cannot
    # use the inspect-based CapabilityContract. Its coherence is enforced through
    # the doctor seam table instead (CLI flag <-> config/SDK field <-> YAML env).
    from npa.workflows.sim2real_health import coherence_failures

    failures = coherence_failures(REPO_ROOT)
    assert not failures, "\n".join(failures)


def test_new_workbench_tools_require_contract_or_explicit_seam() -> None:
    contracted = {contract.name.split("/", 1)[0] for contract in CONTRACTS}
    seam = {
        # Tier-0 BYOF onboarding CLI (script-backed; not a FastAPI service).
        "byof",
        "cosmos",
        # NVIDIA Cosmos OSS wrappers: they drive upstream code in-process (curator
        # stages) or hosted Token Factory calls (evaluator), so there is no service
        # tier to keep coherent with a YAML env block.
        "cosmos-curate",
        "cosmos-evaluator",
        "data",
        "dataset",
        "fiftyone",
        # Foxglove embed assets + MCAP convert/inspect: CLI + SDK tool, no
        # SkyPilot task surface (the viewer runs in the browser / static image).
        "foxglove",
        "genesis",
        "golden-eval",
        "groot",
        "health",
        "insights",
        "isaac-lab",
        "lancedb",
        "lerobot",
        # Static web viewer (caddy static server; no skypilot three-tier YAML).
        "lichtblick",
        "mjlab",
        # NuRec verbs take repeatable options (--camera-id, --override) and Hydra
        # passthrough, so the inspect-based CapabilityContract cannot express them.
        # CLI <-> SDK <-> YAML coherence is enforced instead by
        # npa/tests/workbench/test_nurec_access.py::
        # test_catalog_entries_call_the_real_cli_flags, which checks every catalog
        # argv flag against the real Typer options.
        "nurec",
        "scenario-gen",
        "sim2real",
        # Internal typed stage helper used by npa.workflow toolRefs. It delegates
        # to npa.workflows.sim2real_envgen rather than exposing a service SDK;
        # CLI/toolRef coherence and manifest behavior are covered separately.
        "sim2real-envgen",
        "token-factory",
        "workflow",
    }
    discovered = registered_workbench_tools()
    assert discovered == contracted | seam


# ------------------------------------------------------------------ negative controls


def _contract(name: str) -> CapabilityContract:
    return next(contract for contract in CONTRACTS if contract.name == name)


def test_contract_catches_an_understated_spec_gap() -> None:
    """Claiming full spec reachability when the argv does not pass a flag must fail."""

    broken = replace(_contract("cosmos3/reason"), spec_gap=())

    failures = validate_contract(broken, repo_root=REPO_ROOT)

    assert any("spec_gap drifted" in failure for failure in failures), failures


def test_contract_catches_a_spec_that_does_not_invoke_the_tool_ref() -> None:
    """The named spec must actually use the toolRef the contract describes."""

    broken = replace(
        _contract("cosmos3/reason"),
        tool_ref="workbench.sonic.eval",
        spec_gap=(
            "input_uri",
            "output_uri",
            "model",
            "image",
            "prompt",
            "run_id",
        ),
    )

    failures = validate_contract(broken, repo_root=REPO_ROOT)

    assert any("does not invoke" in failure for failure in failures), failures


def test_contract_catches_an_unknown_tool_ref() -> None:
    broken = replace(_contract("cosmos3/reason"), tool_ref="workbench.nope.nope")

    failures = validate_contract(broken, repo_root=REPO_ROOT)

    assert any("unknown toolRef" in failure for failure in failures), failures


def test_contract_catches_a_missing_third_tier() -> None:
    broken = replace(_contract("cosmos3/reason"), spec_path=None, tool_ref="")

    failures = validate_contract(broken, repo_root=REPO_ROOT)

    assert any("declares no third tier" in failure for failure in failures), failures


def test_the_legacy_yaml_tier_still_fails_loudly_if_anything_returns_to_it(
    tmp_path: Path,
) -> None:
    """No contract uses the legacy tier any more, so its check is exercised synthetically.

    Deleting the last user of a validation path is how that path quietly rots. This keeps the
    env-based check honest for the day someone adds a `yaml_path` back.
    """

    import yaml

    contract = _contract("workflow/trigger/run")
    legacy_yaml = tmp_path / "legacy.yaml"
    legacy_yaml.write_text(
        yaml.safe_dump({"name": "legacy", "envs": {"NPA_TRIGGER_S3_ENDPOINT": "x"}}),
        encoding="utf-8",
    )

    failures = validate_contract(
        replace(contract, spec_path=None, tool_ref="", spec_gap=(), yaml_path=legacy_yaml),
        repo_root=Path("/"),
    )

    assert any("YAML env missing: NPA_TRIGGER_S3_BUCKET" in f for f in failures), failures


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_every_contract_names_a_real_file(contract: CapabilityContract) -> None:
    target = contract.spec_path or contract.yaml_path
    assert target is not None
    assert (REPO_ROOT / target).is_file(), target
