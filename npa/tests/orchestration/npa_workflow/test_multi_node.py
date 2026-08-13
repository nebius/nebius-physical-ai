"""Multi-node stages: ``resources.<profile>.num_nodes``.

SkyPilot gang-schedules ``num_nodes`` identical pods for one task and exports
``SKYPILOT_NODE_RANK`` / ``SKYPILOT_NODE_IPS`` into each. The field is **task level** in
SkyPilot's schema (a sibling of ``resources``, see ``sky/utils/schemas.py`` and
``npa.burst.core.build_task_spec``), so it lives on the resource profile in a spec —
where per-stage shape belongs — and the renderer lifts it back out.

Before this, multi-node was reachable only through ``npa burst submit --nodes``, i.e.
outside the workflow surface.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.scheduler import build_scheduler_task, num_nodes_for_step
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    normalize_resources,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import MAX_PROFILE_NODES, load_spec

SPEC_TEMPLATE = """\
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: multi-node-probe
config:
  bucket: example-bucket
  prefix: "runs/{{run.id}}/probe"
resources:
  single:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi
  gang:
    cloud: kubernetes
    accelerators: L40S:1
    cpus: 8
    memory: 32Gi
    num_nodes: {nodes}
initial: solo
states:
  solo:
    resources: single
    run:
      shell: "echo solo"
    next: gang-stage
  gang-stage:
    resources: gang
    run:
      shell: "echo rank $SKYPILOT_NODE_RANK"
    terminal: true
"""


def _write(tmp_path: Path, nodes: object) -> Path:
    path = tmp_path / "multi-node.yaml"
    path.write_text(SPEC_TEMPLATE.format(nodes=nodes), encoding="utf-8")
    return path


def _render(tmp_path: Path, nodes: object, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(_write(tmp_path, nodes))
    plan = build_plan(spec, run_id="probe")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="probe",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )
    return [doc for doc in yaml.safe_load_all(text) if doc]


def test_num_nodes_is_emitted_at_task_level_not_inside_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = _render(tmp_path, 2, monkeypatch)
    tasks = {doc["name"]: doc for doc in docs[1:]}

    gang = tasks["gang-stage"]
    assert gang["num_nodes"] == 2
    # A `num_nodes` inside `resources` would be an invalid SkyPilot resources block.
    assert "num_nodes" not in gang["resources"]
    assert gang["resources"]["accelerators"] == "L40S:1"


def test_single_node_stages_render_exactly_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field is additive: a 1-node profile must not gain a `num_nodes` key."""

    docs = _render(tmp_path, 1, monkeypatch)
    for doc in docs[1:]:
        assert "num_nodes" not in doc, doc["name"]


def test_stages_on_other_profiles_are_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = _render(tmp_path, 4, monkeypatch)
    tasks = {doc["name"]: doc for doc in docs[1:]}

    assert "num_nodes" not in tasks["solo"]
    assert tasks["gang-stage"]["num_nodes"] == 4


def test_normalize_resources_never_passes_num_nodes_through() -> None:
    out = normalize_resources(
        {"cloud": "kubernetes", "accelerators": "L40S:1", "num_nodes": 3}
    )

    assert "num_nodes" not in out
    assert out["accelerators"] == "L40S:1"


def test_scheduler_task_carries_the_node_count(tmp_path: Path) -> None:
    """The portable seam must expose it, so a non-SkyPilot backend can honour it."""

    spec = load_spec(_write(tmp_path, 3))
    plan = build_plan(spec, run_id="probe")
    gang_step = next(step for step in plan.steps if step.state == "gang-stage")
    solo_step = next(step for step in plan.steps if step.state == "solo")

    assert num_nodes_for_step(spec, gang_step) == 3
    assert num_nodes_for_step(spec, solo_step) == 1
    assert build_scheduler_task(spec, gang_step, run_id="probe")["num_nodes"] == 3
    assert build_scheduler_task(spec, solo_step, run_id="probe")["num_nodes"] == 1


@pytest.mark.parametrize(
    ("nodes", "match"),
    [
        (0, "must be >= 1"),
        (-2, "must be >= 1"),
        ("two", "must be an integer"),
        (MAX_PROFILE_NODES + 1, f"must be <= {MAX_PROFILE_NODES}"),
        ("true", "must be an integer"),
    ],
)
def test_bad_num_nodes_fails_at_validate_time(
    tmp_path: Path, nodes: object, match: str
) -> None:
    with pytest.raises(NpaWorkflowError, match=match):
        load_spec(_write(tmp_path, nodes))


def test_a_bool_is_rejected_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "bool.yaml"
    path.write_text(
        SPEC_TEMPLATE.format(nodes="true").replace("num_nodes: true", "num_nodes: yes"),
        encoding="utf-8",
    )

    with pytest.raises(NpaWorkflowError, match="not a bool"):
        load_spec(path)


def test_a_non_mapping_resource_profile_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-profile.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            apiVersion: npa.workflow/v0.0.1
            kind: Workflow
            metadata:
              name: bad-profile
            resources:
              broken: "kubernetes"
            initial: only
            states:
              only:
                resources: broken
                run:
                  shell: "echo hi"
                terminal: true
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(NpaWorkflowError, match="must be a mapping"):
        load_spec(path)


TOKEN_SPEC = """\
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: token-nodes
config:
  bucket: example-bucket
  prefix: "runs/{{run.id}}/probe"
  augment_nodes: "1"
resources:
  gpu:
    cloud: kubernetes
    accelerators: RTXPRO6000:1
    num_nodes: "{{config.augment_nodes}}"
    deployIfAbsent: true
initial: augment
states:
  augment:
    resources: gpu
    run:
      shell: "echo rank $SKYPILOT_NODE_RANK"
    terminal: true
"""


def _token_spec(tmp_path: Path):
    path = tmp_path / "token-nodes.yaml"
    path.write_text(TOKEN_SPEC, encoding="utf-8")
    return load_spec(path)


def test_a_config_token_lets_submit_choose_the_block_size(tmp_path: Path) -> None:
    """`--var augment_nodes=4` must scale a shipped blueprint without editing it."""

    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    spec = _token_spec(tmp_path)
    plan = build_plan(spec, run_id="probe")
    step = next(s for s in plan.steps if s.state == "augment")
    assert num_nodes_for_step(spec, step) == 1

    scaled = merge_config_overrides(spec, {"augment_nodes": "4"})
    assert num_nodes_for_step(scaled, step) == 4


def test_a_config_token_resolving_to_nonsense_still_fails(tmp_path: Path) -> None:
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    spec = merge_config_overrides(_token_spec(tmp_path), {"augment_nodes": "lots"})
    step = next(s for s in build_plan(spec, run_id="probe").steps if s.state == "augment")

    with pytest.raises(NpaWorkflowError, match="must be an integer"):
        num_nodes_for_step(spec, step)


def test_a_gang_stage_provisions_a_cluster_that_can_hold_it(tmp_path: Path) -> None:
    """`num_nodes: 4` against a one-GPU-node cluster does not fail; it sits PENDING."""

    from npa.orchestration.npa_workflow.deploy import parse_deploy_targets
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    spec = _token_spec(tmp_path)
    assert [t.gpu_nodes for t in parse_deploy_targets(spec)] == [1]

    scaled = merge_config_overrides(spec, {"augment_nodes": "4"})
    assert [t.gpu_nodes for t in parse_deploy_targets(scaled)] == [4]


def test_paidf_augment_scales_from_one_pod_to_a_gang(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    blueprint = (
        Path(__file__).resolve().parents[3] / "workflows" / "physical-ai-data-factory.yaml"
    )
    spec = load_spec(blueprint)

    def _augment_task(loaded):
        plan = build_plan(loaded, run_id="paidf", assume_decision="promote_checkpoint")
        text = render_skypilot_yaml(
            loaded,
            plan,
            run_id="paidf",
            options=SkypilotRenderOptions(image_overrides={"*": ""}),
        )
        docs = [doc for doc in yaml.safe_load_all(text) if doc]
        return next(doc for doc in docs[1:] if doc["name"].startswith("augment"))

    # Shipped default: one augment pod, rendered exactly as before.
    assert "num_nodes" not in _augment_task(spec)

    gang = _augment_task(merge_config_overrides(spec, {"augment_nodes": "4"}))
    assert gang["num_nodes"] == 4
    assert "num_nodes" not in gang["resources"]


def test_every_shipped_spec_still_validates() -> None:
    """The new profile validation must not reject anything already in the catalog."""

    from npa.orchestration.npa_workflow.blueprints import iter_npa_workflow_specs

    for path in iter_npa_workflow_specs():
        load_spec(path)
