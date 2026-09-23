"""Ensure image preflight covers real linear and decision-based workflow plans."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from typer import Exit
import yaml

from npa.cli.workbench import workflow as workflow_cli
from npa.orchestration.npa_workflow import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    plan_images,
)
from npa.orchestration.npa_workflow.submit import load_spec_for_submit
from npa.orchestration.skypilot.registry_preflight import ImagePullCheck


PRIMARY_IMAGE = "registry.example.invalid/workflow/primary:release"
SECONDARY_IMAGE = "registry.example.invalid/workflow/secondary@sha256:" + "b" * 64
PINS = {
    PRIMARY_IMAGE: "registry.example.invalid/workflow/primary@sha256:" + "a" * 64,
    SECONDARY_IMAGE: SECONDARY_IMAGE,
}


def _states(branched: bool):
    states = {
        "primary": {
            "resources": "primary",
            "run": {"shell": "true"},
            "next": "secondary",
        },
        "secondary": {
            "resources": "secondary",
            "run": {"shell": "true"},
            "terminal": True,
        },
    }
    if branched:
        states["primary"].pop("next")
        states["primary"]["terminal"] = True
        states["gate"] = {
            "transitions": [
                {"when": "promote_checkpoint", "goto": "primary"},
                {"when": "loop_back", "goto": "secondary"},
            ]
        }
    return states


def _spec_path(tmp_path: Path, *, branched: bool = False) -> Path:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "npa.workflow/v0.0.1",
                "kind": "Workflow",
                "metadata": {"name": "image-plan-coverage"},
                "config": {"primary_image": PRIMARY_IMAGE},
                "resources": {
                    "primary": {
                        "cloud": "kubernetes",
                        "image": "{{config.primary_image}}",
                    },
                    "secondary": {"cloud": "kubernetes", "image": SECONDARY_IMAGE},
                },
                "initial": "gate" if branched else "primary",
                "states": _states(branched),
            }
        )
    )
    return path


@pytest.fixture
def external_checks(monkeypatch):
    observed = {"pulls": [], "bootstrap": []}

    def pull(images, **_kwargs):
        observed["pulls"].append(list(images))
        return [ImagePullCheck(image=image, status="ok") for image in images]

    def bootstrap(*, images, **_kwargs):
        observed["bootstrap"].append(list(images))
        return [{"image": PINS[image], "state": "compatible"} for image in images]

    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        pull,
    )
    monkeypatch.setattr(workflow_cli, "_preflight_image_bootstrap_contracts", bootstrap)
    return observed


def test_linear_plan_checks_all_images_and_uses_returned_pins(
    tmp_path, external_checks
):
    path = _spec_path(tmp_path)
    options = SkypilotRenderOptions(materialize_registry_secrets=False)
    pins = workflow_cli._preflight_submit_images(
        path,
        options=options,
        assume_decision="",
        enabled=True,
    )

    assert external_checks == {
        "pulls": [[PRIMARY_IMAGE, SECONDARY_IMAGE]],
        "bootstrap": [[PRIMARY_IMAGE, SECONDARY_IMAGE]],
    }
    assert pins == PINS
    spec = load_spec_for_submit(path)
    plan = build_plan(spec, run_id="image-plan-proof")
    assert plan_images(
        spec,
        plan.steps,
        run_id="image-plan-proof",
        options=replace(options, image_digest_pins=pins),
    ) == list(PINS.values())


@pytest.mark.parametrize("assume_decision", ["", "promote_checkpoint", "loop_back"])
def test_all_decision_branches_keep_image_preflight_coverage(
    tmp_path,
    external_checks,
    assume_decision,
):
    pins = workflow_cli._preflight_submit_images(
        _spec_path(tmp_path, branched=True),
        options=SkypilotRenderOptions(),
        assume_decision=assume_decision,
        enabled=True,
    )

    assert pins == PINS
    assert len(external_checks["pulls"]) == 1
    assert set(external_checks["pulls"][0]) == set(PINS)
    assert external_checks["bootstrap"] == external_checks["pulls"]


def test_linear_image_pull_failure_blocks_before_bootstrap(
    tmp_path,
    external_checks,
    monkeypatch,
):
    def forbidden(images, **_kwargs):
        return [
            ImagePullCheck(image=image, status="forbidden", http_status=403)
            for image in images
        ]

    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        forbidden,
    )
    with pytest.raises(Exit):
        workflow_cli._preflight_submit_images(
            _spec_path(tmp_path),
            options=SkypilotRenderOptions(),
            assume_decision="",
            enabled=True,
        )
    assert external_checks["bootstrap"] == []


def test_disabled_linear_preflight_performs_no_external_checks(
    tmp_path, external_checks
):
    assert (
        workflow_cli._preflight_submit_images(
            _spec_path(tmp_path),
            options=SkypilotRenderOptions(),
            assume_decision="",
            enabled=False,
        )
        == {}
    )
    assert external_checks == {"pulls": [], "bootstrap": []}
