"""Unit coverage for ``parallel:`` fan-out, ``params:`` overlays and wave planning.

No infrastructure is touched: specs are written to ``tmp_path`` and rendered in
process.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.skypilot_render import (
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    render_skypilot_job_group_yaml,
    render_skypilot_steps_yaml,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.waves import (
    WAVE_PARALLEL,
    WAVE_SERIAL,
    build_wave_plan,
)

PARALLEL_SPEC = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: fanout-demo

config:
  bucket: example-bucket
  prefix: "fanout/{{run.id}}"
  max_concurrency: "2"
  caption_model: model-a
  max_images: "4"
  max_tokens: "64"
  images_uri: "s3://{{config.bucket}}/{{config.prefix}}/images/"
  captions_uri: "s3://{{config.bucket}}/{{config.prefix}}/captions/"
  insights_store_uri: "s3://{{config.bucket}}/{{config.prefix}}/insights/"
  run_prefix_uri: "s3://{{config.bucket}}/{{config.prefix}}/"
  workflow_name: fanout-demo

resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi

initial: shards

states:
  shards:
    description: Three concurrent caption shards.
    parallel: [shard-a, shard-b, shard-c]
    maxConcurrency: "{{config.max_concurrency}}"
    next: join

  shard-a:
    description: Shard A.
    toolRef: workbench.token_factory.caption
    resources: cpu
    params:
      images_uri: "s3://{{config.bucket}}/{{config.prefix}}/images/a/"
      captions_uri: "s3://{{config.bucket}}/{{config.prefix}}/captions/a/"

  shard-b:
    description: Shard B.
    toolRef: workbench.token_factory.caption
    resources: cpu
    params:
      images_uri: "s3://{{config.bucket}}/{{config.prefix}}/images/b/"
      captions_uri: "s3://{{config.bucket}}/{{config.prefix}}/captions/b/"

  shard-c:
    description: Shard C.
    toolRef: workbench.token_factory.caption
    resources: cpu
    params:
      images_uri: "s3://{{config.bucket}}/{{config.prefix}}/images/c/"
      captions_uri: "s3://{{config.bucket}}/{{config.prefix}}/captions/c/"

  join:
    description: Barrier — aggregate every shard.
    needs: [shards]
    toolRef: workbench.insights.ingest_run
    resources: cpu
    terminal: true
"""


def _write(tmp_path: Path, text: str, name: str = "spec.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


@pytest.fixture()
def parallel_spec(tmp_path: Path):
    return load_spec(_write(tmp_path, PARALLEL_SPEC))


# --------------------------------------------------------------------------- spec


def test_parallel_group_parses(parallel_spec) -> None:
    group = parallel_spec.states["shards"]
    assert group.parallel == ["shard-a", "shard-b", "shard-c"]
    assert group.max_concurrency == "{{config.max_concurrency}}"
    assert parallel_spec.states["shard-a"].params["images_uri"].endswith("/images/a/")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("parallel: [shard-a, missing]", "unknown parallel member"),
        ("parallel: [shard-a, shard-a]", "duplicate parallel member"),
        ("parallel: [shard-a, join]", "cannot be terminal"),
    ],
)
def test_parallel_validation_errors(tmp_path: Path, mutation: str, message: str) -> None:
    text = PARALLEL_SPEC.replace("parallel: [shard-a, shard-b, shard-c]", mutation)
    with pytest.raises(NpaWorkflowError, match=message):
        load_spec(_write(tmp_path, text))


def test_parallel_member_may_not_declare_next(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "  shard-c:\n    description: Shard C.",
        "  shard-c:\n    next: join\n    description: Shard C.",
    )
    with pytest.raises(NpaWorkflowError, match="must not declare next/transitions"):
        load_spec(_write(tmp_path, text))


def test_parallel_and_sequence_are_exclusive(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "    parallel: [shard-a, shard-b, shard-c]",
        "    parallel: [shard-a, shard-b, shard-c]\n    sequence: [shard-a]",
    )
    with pytest.raises(NpaWorkflowError, match="not both"):
        load_spec(_write(tmp_path, text))


def test_loop_on_parallel_group_rejected(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "    maxConcurrency: \"{{config.max_concurrency}}\"",
        "    loop:\n      max: 2",
    )
    with pytest.raises(NpaWorkflowError, match="loop is not supported directly"):
        load_spec(_write(tmp_path, text))


def test_max_concurrency_requires_parallel(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "  join:\n    description: Barrier",
        "  join:\n    maxConcurrency: 2\n    description: Barrier",
    )
    with pytest.raises(NpaWorkflowError, match="maxConcurrency requires a parallel group"):
        load_spec(_write(tmp_path, text))


def test_bad_params_token_fails_validation(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        '      images_uri: "s3://{{config.bucket}}/{{config.prefix}}/images/a/"',
        '      images_uri: "s3://{{config.nope}}/images/a/"',
    )
    with pytest.raises(NpaWorkflowError, match="params.images_uri"):
        load_spec(_write(tmp_path, text))


def test_trigger_requires_work_on_the_same_state(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "  join:\n    description: Barrier — aggregate every shard.",
        "  join:\n    trigger:\n      uri: \"s3://{{config.bucket}}/inbox/\"\n"
        "    description: Barrier — aggregate every shard.",
    ).replace("    toolRef: workbench.insights.ingest_run\n    resources: cpu\n    terminal: true", "    terminal: true")
    with pytest.raises(NpaWorkflowError, match="trigger requires run or toolRef"):
        load_spec(_write(tmp_path, text))


def test_trigger_parses_with_defaults(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "  join:\n    description: Barrier — aggregate every shard.",
        "  join:\n    trigger:\n      uri: \"s3://{{config.bucket}}/inbox/\"\n"
        "      pollSeconds: 5\n      maxPolls: 3\n"
        "    description: Barrier — aggregate every shard.",
    )
    spec = load_spec(_write(tmp_path, text))
    trigger = spec.states["join"].trigger
    assert trigger is not None
    assert trigger.poll_seconds == 5
    assert trigger.max_polls == 3
    assert trigger.min_objects == 1


# ------------------------------------------------------------------------- plan


def test_params_overlay_reaches_argv(parallel_spec) -> None:
    plan = build_plan(parallel_spec, run_id="p1")
    by_state = {step.state: step for step in plan.steps}
    assert "/images/a/" in " ".join(by_state["shard-a"].argv)
    assert "/images/b/" in " ".join(by_state["shard-b"].argv)
    # The base config value is untouched for states without params.
    assert "/images/" in " ".join(by_state["join"].argv) or by_state["join"].argv


def test_plan_flattens_parallel_group_in_declared_order(parallel_spec) -> None:
    plan = build_plan(parallel_spec, run_id="p1")
    assert [step.state for step in plan.steps] == [
        "shard-a",
        "shard-b",
        "shard-c",
        "join",
    ]
    assert [step.group for step in plan.steps] == ["shards", "shards", "shards", ""]


def test_plan_only_render_stays_serial(parallel_spec) -> None:
    plan = build_plan(parallel_spec, run_id="p1")
    text = render_skypilot_yaml(
        parallel_spec,
        plan,
        run_id="p1",
        options=SkypilotRenderOptions(image_overrides={"*": "cr.example/x:1"}),
    )
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert docs[0]["execution"] == "serial"
    assert len(docs) == 5


# ------------------------------------------------------------------------ waves


def test_wave_plan_groups_parallel_members(parallel_spec) -> None:
    wave_plan = build_wave_plan(parallel_spec, run_id="p1")
    kinds = [wave.kind for wave in wave_plan.waves]
    assert kinds == [WAVE_PARALLEL, WAVE_SERIAL]
    fanout = wave_plan.waves[0]
    assert fanout.group == "shards"
    assert [step.state for step in fanout.steps] == ["shard-a", "shard-b", "shard-c"]
    # maxConcurrency: 2 over three members -> two batches (2 + 1).
    assert fanout.max_concurrency == 2
    assert [len(batch) for batch in fanout.batches()] == [2, 1]
    assert wave_plan.to_dict()["parallel_waves"] == 1


def test_wave_plan_defaults_concurrency_to_group_size(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace('    maxConcurrency: "{{config.max_concurrency}}"\n', "")
    spec = load_spec(_write(tmp_path, text))
    wave = build_wave_plan(spec, run_id="p1").waves[0]
    assert wave.max_concurrency == 3
    assert [len(batch) for batch in wave.batches()] == [3]


def test_wave_plan_of_serial_spec_is_all_serial() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec = load_spec(
        repo_root / "npa" / "workflows" / "workbench" / "npa-workflows" / "bdd100k-pipeline.yaml"
    )
    wave_plan = build_wave_plan(spec, run_id="serial-1")
    assert {wave.kind for wave in wave_plan.waves} == {WAVE_SERIAL}
    assert all(len(wave.steps) == 1 for wave in wave_plan.waves)


# ----------------------------------------------------------------------- render


def _render_options() -> SkypilotRenderOptions:
    return SkypilotRenderOptions(image_overrides={"*": "cr.example/x:1"})


def test_job_group_render_emits_parallel_header(parallel_spec) -> None:
    wave = build_wave_plan(parallel_spec, run_id="p1").waves[0]
    text = render_skypilot_job_group_yaml(
        parallel_spec,
        wave.steps,
        run_id="p1",
        options=_render_options(),
        name="fanout-demo-shards",
    )
    assert_no_unresolved_placeholders(text)
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert docs[0] == {"name": "fanout-demo-shards", "execution": "parallel"}
    assert [doc["name"] for doc in docs[1:]] == ["shard-a", "shard-b", "shard-c"]
    # primary_tasks is intentionally omitted: every task is primary -> barrier.
    assert "primary_tasks" not in docs[0]
    assert "/images/a/" in docs[1]["run"]
    assert "/images/c/" in docs[3]["run"]


def test_job_group_render_rejects_single_task(parallel_spec) -> None:
    wave = build_wave_plan(parallel_spec, run_id="p1").waves[0]
    with pytest.raises(NpaWorkflowRenderError, match="at least two tasks"):
        render_skypilot_job_group_yaml(
            parallel_spec, wave.steps[:1], run_id="p1", options=_render_options()
        )


def test_render_steps_yaml_dispatches(parallel_spec) -> None:
    waves = build_wave_plan(parallel_spec, run_id="p1").waves
    parallel_text = render_skypilot_steps_yaml(
        parallel_spec,
        waves[0].steps,
        run_id="p1",
        options=_render_options(),
        execution="parallel",
    )
    serial_text = render_skypilot_steps_yaml(
        parallel_spec,
        waves[1].steps,
        run_id="p1",
        options=_render_options(),
        execution="serial",
    )
    assert "execution: parallel" in parallel_text
    assert "execution: serial" in serial_text


def test_serial_renderer_still_rejects_parallel_option(parallel_spec) -> None:
    """The historic serial-only guard is untouched — parallel has its own entry point."""

    plan = build_plan(parallel_spec, run_id="p1")
    with pytest.raises(NpaWorkflowRenderError, match="execution=serial"):
        render_skypilot_yaml(
            parallel_spec,
            plan,
            run_id="p1",
            options=SkypilotRenderOptions(execution="parallel"),
        )


# --------------------------------------------------------- field type failures
#
# NOTE: the shipped JSON Schema is only enforced at the document level — the
# hand-rolled walker in schema_validation.py does not resolve `$ref`/`$defs`, so
# `states.<name>.*` bodies have never been schema-checked. Type errors for the new
# fields therefore have to raise (with an actionable message) from the Python
# parser/validator, which is what these tests pin.


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("    parallel: shard-a", "parallel must be a list of state names"),
        ("    parallel:\n      - 1\n      - 2", "parallel member must be a state name"),
    ],
)
def test_bad_parallel_types_raise_actionable_errors(
    tmp_path: Path, mutation: str, message: str
) -> None:
    text = PARALLEL_SPEC.replace("    parallel: [shard-a, shard-b, shard-c]", mutation)
    with pytest.raises(NpaWorkflowError, match=message):
        load_spec(_write(tmp_path, text))


def test_non_mapping_params_is_rejected(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        '    params:\n      images_uri: "s3://{{config.bucket}}/{{config.prefix}}/images/a/"\n'
        '      captions_uri: "s3://{{config.bucket}}/{{config.prefix}}/captions/a/"',
        "    params: not-a-mapping",
        1,
    )
    with pytest.raises(NpaWorkflowError, match="params must be a mapping"):
        load_spec(_write(tmp_path, text))


def test_trigger_uri_is_required(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "  join:\n    description: Barrier — aggregate every shard.",
        "  join:\n    trigger:\n      pollSeconds: 5\n"
        "    description: Barrier — aggregate every shard.",
    )
    with pytest.raises(NpaWorkflowError, match="trigger.uri is required"):
        load_spec(_write(tmp_path, text))


def test_non_mapping_trigger_is_rejected(tmp_path: Path) -> None:
    text = PARALLEL_SPEC.replace(
        "  join:\n    description: Barrier — aggregate every shard.",
        "  join:\n    trigger: soon\n    description: Barrier — aggregate every shard.",
    )
    with pytest.raises(NpaWorkflowError, match="trigger must be a mapping"):
        load_spec(_write(tmp_path, text))


@pytest.mark.parametrize(
    "value",
    ["pollSeconds: zero", "maxPolls: -1", "minObjects: 0"],
)
def test_trigger_numeric_fields_are_validated(tmp_path: Path, value: str) -> None:
    text = PARALLEL_SPEC.replace(
        "  join:\n    description: Barrier — aggregate every shard.",
        "  join:\n    trigger:\n      uri: \"s3://{{config.bucket}}/inbox/\"\n"
        f"      {value}\n"
        "    description: Barrier — aggregate every shard.",
    )
    with pytest.raises(NpaWorkflowError, match="trigger."):
        load_spec(_write(tmp_path, text))


# --------------------------------------------------- the shipped catalog specs


SHIPPED = Path(__file__).resolve().parents[4] / "npa" / "workflows" / "workbench" / "npa-workflows"


def test_shipped_fanout_spec_wave_shape() -> None:
    spec = load_spec(SHIPPED / "token-factory-parallel-fanout.yaml")
    waves = build_wave_plan(spec, run_id="shape-1").waves
    assert [(wave.kind, wave.name, len(wave.steps)) for wave in waves] == [
        (WAVE_PARALLEL, "caption-shards", 3),
        (WAVE_SERIAL, "aggregate", 1),
    ]
    assert waves[0].max_concurrency == 3
    # Each shard captions its own prefix (params overlay reached the argv).
    argvs = [" ".join(step.argv) for step in waves[0].steps]
    assert all(f"/images/shard-{letter}/" in argv for letter, argv in zip("abc", argvs))


def test_shipped_sweep_spec_wave_shape() -> None:
    spec = load_spec(SHIPPED / "isaac-lab-rl-sweep.yaml")
    waves = build_wave_plan(spec, run_id="shape-2").waves
    assert [(wave.kind, wave.name, len(wave.steps)) for wave in waves] == [
        (WAVE_PARALLEL, "sweep", 4),
        (WAVE_SERIAL, "select-best", 1),
    ]
    assert waves[0].max_concurrency == 4
    # Every variant trains with its own Hydra overrides and output prefix.
    shells = [step.shell for step in waves[0].steps]
    assert sum("learning_rate=1.0e-3" in shell for shell in shells) == 1
    assert sum("entropy_coef=0.01" in shell for shell in shells) == 1
    assert len({shell for shell in shells}) == 4


@pytest.mark.parametrize(
    ("assume", "expected_states"),
    [
        (
            "promote_checkpoint",
            ["caption-batch", "score-batch", "quality-gate", "route", "publish"],
        ),
        (
            "loop_back",
            [
                "caption-batch",
                "score-batch",
                "quality-gate",
                "caption-batch",
                "score-batch",
                "quality-gate",
                "caption-batch",
                "score-batch",
                "quality-gate",
                "route",
                "escalate",
            ],
        ),
    ],
)
def test_shipped_gate_loop_plan_matches_the_assumed_decision(
    assume: str, expected_states: list[str]
) -> None:
    """The plan-time contract behind the two live runs (early exit vs full budget)."""

    spec = load_spec(SHIPPED / "token-factory-gate-loop.yaml")
    plan = build_plan(spec, run_id="shape-3", assume_decision=assume)
    assert [step.state for step in plan.steps] == expected_states
    assert all(step.group == "" for step in plan.steps)


@pytest.mark.parametrize(
    "name",
    [
        "token-factory-parallel-fanout.yaml",
        "token-factory-gate-loop.yaml",
        "isaac-lab-rl-sweep.yaml",
    ],
)
def test_shipped_specs_render_without_placeholders(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline render check for the new specs (the live matrix does this too)."""

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    spec = load_spec(SHIPPED / name)
    plan = build_plan(spec, run_id="render-1", assume_decision="promote_checkpoint")
    text = render_skypilot_yaml(spec, plan, run_id="render-1", options=_render_options())
    assert_no_unresolved_placeholders(text)
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert docs[0]["execution"] == "serial"
    assert len(docs) - 1 == len(plan.steps)
    assert all(doc["run"].strip() for doc in docs[1:])

    # The parallel waves of the same plan render as a JobGroup.
    for wave in build_wave_plan(spec, run_id="render-1", assume_decision="promote_checkpoint").waves:
        if wave.kind != WAVE_PARALLEL:
            continue
        group_text = render_skypilot_job_group_yaml(
            spec, wave.steps, run_id="render-1", options=_render_options(), name=wave.name
        )
        assert_no_unresolved_placeholders(group_text)
        group_docs = [doc for doc in yaml.safe_load_all(group_text) if doc is not None]
        assert group_docs[0]["execution"] == "parallel"
        assert len(group_docs) - 1 == len(wave.steps)


def test_stage_shell_gets_the_right_interpreter(parallel_spec) -> None:
    """A stage body must run with an interpreter that has npa AND its dependencies.

    Regression guard for three live failures on real GPUs, all caused by stage
    commands running in a LOGIN shell that re-resolved python3:
      * SkyPilot GPU default image: login python3 had no npa (ModuleNotFoundError: npa)
      * same image after a PYTHONPATH patch: no dependencies (numpy)
      * and no pip at all, so installing into it was impossible
      * Isaac Lab image: PATH python3 is Isaac's kit interpreter, leaving the npa
        console script outside PATH
    Fix: stage commands use `bash -c` (inherit the task env), the run script sources
    /etc/profile.d/*.sh for images that activate that way, and a PATH shim points
    python3 at the interpreter setup recorded.
    """

    from npa.orchestration.npa_workflow.scheduler import build_scheduler_task
    from npa.orchestration.npa_workflow.skypilot_render import (
        default_npa_setup,
        render_task_run_script,
    )

    # The seam no longer wraps stage shells in a login shell.
    shipped = load_spec(SHIPPED / "token-factory-parallel-fanout.yaml")
    plan = build_plan(shipped, run_id="shell-1")
    join_step = [step for step in plan.steps if step.shell.strip()][0]
    task = build_scheduler_task(shipped, join_step, run_id="shell-1")
    assert task["command"][:2] == ["bash", "-c"]
    assert "join_shards" in task["command"][2]

    setup = default_npa_setup()
    assert "/tmp/npa-python" in setup  # records the good interpreter
    assert "npa is not importable after setup" in setup  # fails loudly
    assert "/usr/local/bin/npa" in setup  # console script reachable for toolRefs

    run_script = render_task_run_script(["python3", "-c", "import npa"])
    assert "/etc/profile.d" in run_script  # activation preserved without a login shell
    assert "/tmp/npa-shim" in run_script  # interpreter shim
    assert "export PATH=" in run_script
    # The shim must be UNCONDITIONAL: the stage command runs in its own `bash -c`,
    # which can resolve python3 differently than this script's shell (live: the run
    # shell expanded the Isaac image's python3 alias and imported npa, while the
    # stage's shell got the raw kit python and failed).
    assert "if [ -s /tmp/npa-python ]; then" in run_script
    assert "! python3 -c 'import npa'" not in run_script
    assert "${" not in run_script  # rendered YAML must stay placeholder-clean


def test_staged_source_is_not_published_as_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """PYTHONPATH must NOT be pre-set for staged source.

    Doing so was actively harmful on real GPUs: it let an interpreter without npa's
    dependencies import npa from the source tree, so the run script's shim condition
    ("python3 cannot import npa") was false and the stage died later on
    `import numpy`. The staged path is only used as a last-resort fallback inside the
    setup script.
    """

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    spec = load_spec(SHIPPED / "token-factory-parallel-fanout.yaml")
    plan = build_plan(spec, run_id="env-1")
    text = render_skypilot_yaml(
        spec, plan, run_id="env-1", options=SkypilotRenderOptions(image_overrides={"*": ""})
    )
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    for task in docs[1:]:
        assert task["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/npa-src/npa"
        assert "PYTHONPATH" not in task["envs"]


def test_setup_survives_pep668_managed_interpreters() -> None:
    """Installs must work on images whose system python is externally managed.

    Live: once the Isaac Lab image had a system python3 first on PATH (needed so
    SkyPilot can host the task at all), `pip install` failed with
    "error: externally-managed-environment" on Ubuntu 24.04. A task container is
    disposable, so the install retries with --break-system-packages and then --user.
    """

    from npa.orchestration.npa_workflow.skypilot_render import default_npa_setup

    setup = default_npa_setup()
    assert "npa_pip_install()" in setup
    assert "--break-system-packages" in setup
    assert "--user" in setup
    # Every install goes through the helper (no bare `pip install -e` left behind).
    assert "python3 -m pip install -q -e /tmp/npa-src\n" not in setup
    assert "npa_pip_install -e /tmp/npa-src" in setup
    assert "npa_pip_install -e /opt/nebius-physical-ai/npa" in setup


def test_shipped_trigger_spec_reads_its_knobs_from_config() -> None:
    """The trigger/watch reference must be config-driven, not hardcoded."""

    spec = load_spec(SHIPPED / "token-factory-trigger-watch.yaml")
    trigger = spec.states["caption-inbox"].trigger
    assert trigger is not None
    assert trigger.poll_seconds == int(spec.config["inbox_poll_seconds"])
    assert trigger.max_polls == int(spec.config["inbox_max_polls"])
    assert trigger.min_objects == int(spec.config["inbox_min_objects"])
    # A trigger gates real work on the same state.
    assert spec.states["caption-inbox"].tool_ref == "workbench.token_factory.caption"
    # Plans and renders like an ordinary serial stage (trigger is runtime-only).
    plan = build_plan(spec, run_id="trig-1")
    assert [step.state for step in plan.steps] == ["caption-inbox"]
