"""Structural tests for the DROID forward-dynamics workflow contract."""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan
from npa.orchestration.npa_workflow.spec import load_spec


def test_droid_forward_dynamics_has_five_connected_substantive_stages() -> None:
    path = (
        Path(__file__).parents[3]
        / "workflows/testing/cosmos3-droid-forward-dynamics.yaml"
    )
    spec = yaml.safe_load(path.read_text())
    states = spec["states"]
    assert list(states) == [
        "prepare",
        "predict_true",
        "predict_controls",
        "evaluate",
        "visualize",
    ]
    assert [states[name].get("next") for name in list(states)[:-1]] == [
        "predict_true",
        "predict_controls",
        "evaluate",
        "visualize",
    ]
    assert states["visualize"]["terminal"] is True
    refs = [state["toolRef"] for state in states.values()]
    assert refs == [
        "workbench.cosmos3.droid_fd_prepare",
        "workbench.cosmos3.droid_fd_predict",
        "workbench.cosmos3.droid_fd_controls",
        "workbench.cosmos3.droid_fd_evaluate",
        "workbench.cosmos3.droid_fd_visualize",
    ]
    assert all(TOOL_CATALOG[ref].stub is False for ref in refs)
    # This derivative's checkpoint and the two card-pinned runtime auxiliaries
    # are anonymously accessible at their immutable revisions. It does not run
    # the guarded generic Cosmos generation path, so inheriting its gated
    # ``cosmos3`` access capability would add a false HF_TOKEN prerequisite.
    assert TOOL_CATALOG["workbench.cosmos3.droid_fd_predict"].access_capabilities == ()
    assert TOOL_CATALOG["workbench.cosmos3.droid_fd_controls"].access_capabilities == ()
    assert "prepared.json" in str(states["predict_true"]["inputs"])
    predict_argv = TOOL_CATALOG["workbench.cosmos3.droid_fd_predict"].argv_template
    controls_argv = TOOL_CATALOG["workbench.cosmos3.droid_fd_controls"].argv_template
    assert predict_argv[predict_argv.index("--input-path") + 1].endswith(
        "prepared.json"
    )
    assert controls_argv[controls_argv.index("--input-path") + 1].endswith(
        "prepared.json"
    )
    assert "prediction.json" in str(states["evaluate"]["inputs"])
    assert "controls.json" in str(states["evaluate"]["inputs"])
    assert "evaluation.json" in str(states["visualize"]["inputs"])


def test_droid_forward_dynamics_has_no_generic_cosmos_hf_token_hint() -> None:
    """Keep the public DROID checkpoint closure distinct from guarded Cosmos3."""

    path = (
        Path(__file__).parents[3]
        / "workflows/testing/cosmos3-droid-forward-dynamics.yaml"
    )
    plan = build_plan(load_spec(path), run_id="droid-public-closure")

    assert secret_env_hints_for_plan(plan.steps) == ()


def test_droid_forward_dynamics_keeps_the_immutable_checkpoint_and_rerun_output() -> (
    None
):
    path = (
        Path(__file__).parents[3]
        / "workflows/testing/cosmos3-droid-forward-dynamics.yaml"
    )
    spec = yaml.safe_load(path.read_text())
    assert (
        spec["config"]["checkpoint_revision"]
        == "1dfff3cc3b86548b208341bb123d1c4f71043114"
    )
    outputs = spec["states"]["visualize"]["outputs"]
    assert any(
        output["uri"].endswith("droid_forward_dynamics.rrd") for output in outputs
    )
