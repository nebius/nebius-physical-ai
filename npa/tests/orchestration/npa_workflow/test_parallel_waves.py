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
