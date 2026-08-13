"""PAIDF wiring for segmentation-conditioned augmentation and region masks.

The capability is only usable if an operator can select it at submit time, so
these tests drive the shipped blueprint through the same path a submit takes:
config overrides -> plan -> rendered SkyPilot task command.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.submit import merge_config_overrides

BLUEPRINT = (
    Path(__file__).resolve().parents[3] / "workflows" / "physical-ai-data-factory.yaml"
)


@pytest.fixture(autouse=True)
def _src_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")


def _augment_flags(spec) -> dict[str, str]:
    """Parse the rendered augment command back into ``{flag: value}``.

    Parsing rather than substring matching is what makes an empty value
    meaningful: an unquoted ``--control-prompt`` followed by ``--mask-asset``
    would read as the prompt *being* ``--mask-asset``.
    """

    plan = build_plan(spec, run_id="paidf", assume_decision="promote_checkpoint")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="paidf",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )
    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    task = next(doc for doc in docs[1:] if doc["name"].startswith("augment"))
    line = next(
        statement
        for statement in str(task["run"]).splitlines()
        if "cosmos2 transfer" in statement
    )
    words = shlex.split(line)
    flags: dict[str, str] = {}
    for index, word in enumerate(words):
        if not word.startswith("--"):
            continue
        following = words[index + 1] if index + 1 < len(words) else ""
        flags[word] = "" if following.startswith("--") else following
    return flags


def test_the_shipped_default_still_conditions_on_edge() -> None:
    spec = load_spec(BLUEPRINT)
    validate_spec(spec)

    flags = _augment_flags(spec)

    assert flags["--control"] == "edge"
    assert flags["--control-weight"] == "1.0"
    # Unset asset/prompt flags render as empty strings and the CLI reads empty as
    # "not set", so the default run conditions exactly the way it did before.
    assert flags["--control-asset"] == ""
    assert flags["--control-prompt"] == ""
    assert flags["--mask-asset"] == ""
    assert flags["--mask-prompt"] == ""


def test_submit_can_switch_the_run_to_segmentation_conditioning() -> None:
    spec = merge_config_overrides(
        load_spec(BLUEPRINT),
        {
            "augment_control": "seg",
            "augment_control_weight": "0.8",
            "augment_control_prompt": "robot arm, conveyor, bin",
            "augment_mask_prompt": "robot arm",
        },
    )

    flags = _augment_flags(spec)

    assert flags["--control"] == "seg"
    assert flags["--control-weight"] == "0.8"
    assert flags["--control-prompt"] == "robot arm, conveyor, bin"
    assert flags["--mask-prompt"] == "robot arm"
    # On-the-fly segmentation needs no asset, which is the point.
    assert flags["--control-asset"] == ""


def test_submit_can_supply_a_precomputed_segmentation_map() -> None:
    spec = merge_config_overrides(
        load_spec(BLUEPRINT),
        {
            "augment_control": "seg",
            "augment_control_asset_uri": "s3://example-bucket/seg/robot_seg.mp4",
        },
    )

    assert _augment_flags(spec)["--control-asset"] == (
        "s3://example-bucket/seg/robot_seg.mp4"
    )


def test_the_control_prefix_is_a_sibling_of_the_augmented_clips() -> None:
    """Nesting it would make the evaluator read a control map as a variant."""

    spec = load_spec(BLUEPRINT)
    augment = str(spec.config["augment_uri"])
    control = str(spec.config["augment_control_uri"])

    assert control != augment
    assert not control.startswith(augment)
    assert not augment.startswith(control)
    published = _augment_flags(spec)["--control-output-uri"]
    assert published.endswith("/cosmos_control/")
    assert not published.startswith(
        "s3://example-bucket/physical-ai-data-factory/paidf/cosmos_augmented/"
    )
