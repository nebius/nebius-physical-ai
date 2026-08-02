"""Eval must read labels the way training wrote them.

Live: three BDD100K training stages SUCCEEDED and then `eval-rider` failed with
`mAP evaluation failed: invalid literal for int() with base 10: 'train'` (EVIDENCE §R46).
BDD100K stores string categories and one of them is literally `train` — the vehicle. Training
took `--label-map`; eval did not, so its loader fell through to `int(raw)`.

`EvalRequest.label_map` had existed all along with no CLI flag to fill it, which is the shape of
this bug: a field nobody could set.
"""

from __future__ import annotations

import inspect

import pytest

from npa.cli.workbench.detection_training import eval_cmd, train_cmd
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workbench.detection_training.schemas import EvalRequest


def test_eval_accepts_a_label_map_like_train_does() -> None:
    for command in (train_cmd, eval_cmd):
        assert "label_map" in inspect.signature(command).parameters, command.__name__


def test_the_request_carries_it_through() -> None:
    request = EvalRequest(
        checkpoint_uri="s3://b/ck.pt",
        eval_view="rider",
        output_uri="s3://b/out/",
        label_map={"train": 6, "car": 2},
    )

    assert request.label_map == {"train": 6, "car": 2}


@pytest.mark.parametrize(
    "tool_ref",
    [
        "workbench.detection_training.eval_rider",
        "workbench.detection_training.eval_nighttime",
        "workbench.detection_training.eval_distant",
    ],
)
def test_every_eval_toolref_passes_the_same_map_the_training_stages_use(tool_ref: str) -> None:
    argv = [str(part) for part in TOOL_CATALOG[tool_ref].argv_template]

    assert "--label-map" in argv, tool_ref
    assert argv[argv.index("--label-map") + 1] == "{{config.detection_label_map}}"


def test_the_bdd100k_spec_defines_that_config_key() -> None:
    from pathlib import Path

    import yaml

    spec = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3]
            / "npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml"
        ).read_text(encoding="utf-8")
    )
    label_map = spec["config"]["detection_label_map"]

    # The category that broke it must be in there.
    assert "train" in str(label_map)
