"""Recheck hosted visual evaluation on an actual retained Franka parts capture."""

import json
import os
from pathlib import Path

import pytest

from npa.clients.token_factory import TokenFactoryClient, resolve_config
from npa.workbench.vlm_eval.temporal import judge_manipulation

pytestmark = pytest.mark.token_factory_e2e


def test_token_factory_judges_actual_franka_parts_capture(tmp_path):
    source = os.environ.get("NPA_FRANKA_CAPTURE_PATH", "")
    if not source or not resolve_config(require_api_key=False).api_key:
        pytest.skip(
            "Requires an actual parts trajectory directory and the operator's Token Factory key"
        )
    root = Path(source)
    metadata = json.loads((root / "meta.json").read_text())
    assert metadata["genuine_simulator_pixels"] and metadata["policy_loaded"]
    assert metadata["assets"]["target"] in {"spool", "hex_nut", "bottle"}
    recipe = json.loads((root.parent / "recipe.json").read_text())
    result = judge_manipulation(
        rgb_path=root / "episode_000000/rgb.npy",
        output=tmp_path / "judgment",
        task=metadata["task_description"],
        fps=metadata["fps"],
        frame_count=recipe["visual_eval"]["frame_count"],
        model=recipe["visual_eval"]["model"],
        client=TokenFactoryClient(),
    )
    assert result["backend"] == "token_factory" and result["finish_reason"] == "stop"
    assert result["request_id"] and result["frames"] and result["verdict"]["rationale"]
