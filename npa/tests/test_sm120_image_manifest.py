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
        ("npa-base", "cuda13-b300-sm80-sm90-sm120-latest"),
        ("npa-genesis", "0.4.6-sm80-sm90-sm120-latest"),
        ("npa-envgen", "0.1.1"),
        ("npa-reference-policy", "0.1.1"),
        ("npa-loop-eval", "0.1.0"),
        ("npa-lerobot-vlm-rl", "0.1.0"),
        ("npa-cosmos3-reason", "3.0.0"),
        ("npa-sonic", "0.1.2-k8s-runtime"),
    }

    assert set(images) == expected
    for image in images.values():
        assert image["digest"].startswith("sha256:")
        assert image["purpose"]
