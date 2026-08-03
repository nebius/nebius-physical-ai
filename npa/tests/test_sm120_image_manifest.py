from __future__ import annotations

import json
from pathlib import Path


def test_sm120_image_manifest_has_required_images() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "docker" / "workbench" / "sm120-images.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format"] == "npa_sm120_image_manifest_v1"
    assert manifest["target"]["compute_capability"] == "sm_120"
    assert "cluster_context" not in manifest["target"]

    images = {(item["name"], item["tag"]): item for item in manifest["images"]}
    expected = {
        ("npa-base", "cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest"),
        (
            "npa-genesis",
            "cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
        ),
        ("npa-envgen", "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"),
        ("npa-reference-policy", "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"),
        ("npa-loop-eval", "cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"),
        ("npa-lerobot-vlm-rl", "cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"),
        ("npa-cosmos3-reason", "cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"),
        ("npa-sonic", "cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"),
    }

    assert set(images) == expected
    for image in images.values():
        assert image["digest"].startswith("sha256:")
        assert image["purpose"]
