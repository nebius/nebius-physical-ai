"""Unit tests for Physical AI Data Factory run provenance."""

from __future__ import annotations

from npa.workflows.data_factory_provenance import build_run_origin, build_run_provenance

RUN = "paidf-1"
PFX = f"checkpoints/physical-ai-data-factory/{RUN}"
KEYS = [
    f"{PFX}/configs/manifest.json",
    f"{PFX}/labeled_original/captions.json",
    f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00000.png",
    f"{PFX}/cosmos_augmented/manifest.json",
    f"{PFX}/grade/vlm_eval_stub.json",
    f"{PFX}/grade/decision.json",
    f"{PFX}/labeled_augmented/captions.json",
    f"{PFX}/curation/report.json",
    f"{PFX}/reports/sim2real.rrd",
]


def _read_gpu(key: str):
    if key.endswith("cosmos_augmented/manifest.json"):
        return {"mode": "cosmos_transfer2.5_gpu"}
    if key.endswith("captions.json"):
        return {"model": "Qwen/Qwen2.5-VL-72B-Instruct"}
    if key.endswith("vlm_eval_stub.json"):
        return {"model": "Qwen/Qwen2.5-VL-72B-Instruct", "backend": "api"}
    return {}


def test_provenance_lists_components_per_stage() -> None:
    prov = build_run_provenance(KEYS, run_id=RUN, read_json=_read_gpu)
    stages = {c["stage"]: c for c in prov["components"]}
    # Augment is the real Cosmos Transfer 2.5 GPU component.
    aug = stages["Augment"]
    assert aug["component"] == "Cosmos Transfer 2.5"
    assert "GPU" in aug["runtime"]
    assert aug["model"] == "nvidia/Cosmos-Transfer2.5-2B"
    assert aug.get("engine") == "cosmos_transfer_2.5_gpu"
    # VLM stages attribute to Token Factory + the real model.
    assert stages["Annotate originals"]["component"] == "Token Factory VLM"
    assert stages["Pseudo-label augmented"]["model"] == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert "Token Factory" in stages["Attribute verify + quality gate"]["runtime"]
    # Summary names where the data comes from + the components.
    assert "Cosmos Transfer 2.5" in prov["summary"]
    assert "Token Factory VLM" in prov["summary"]


def test_provenance_distinguishes_cpu_standin_from_gpu() -> None:
    def read_standin(key: str):
        if key.endswith("cosmos_augmented/manifest.json"):
            return {"mode": "cpu_appearance_transform_stand_in"}
        return {}

    prov = build_run_provenance(KEYS, run_id=RUN, read_json=read_standin)
    aug = next(c for c in prov["components"] if c["stage"] == "Augment")
    assert "stand-in" in aug["detail"].lower()
    assert aug.get("engine") == "cpu_appearance_transform_stand_in"


def test_provenance_detects_standin_from_per_clip_metadata() -> None:
    # Some runs record the augment mode only in a per-clip metadata.json (no
    # run-level manifest.json). The engine must still be classified as a stand-in
    # so the UI can flag it honestly instead of showing an unknown engine.
    keys = [
        f"{PFX}/cosmos_augmented/video_0_aug0/augmented_video.mp4",
        f"{PFX}/cosmos_augmented/video_0_aug0/metadata.json",
    ]

    def read_per_clip(key: str):
        if key.endswith("video_0_aug0/metadata.json"):
            return {"mode": "cpu_appearance_transform_stand_in"}
        return {}

    prov = build_run_provenance(keys, run_id=RUN, read_json=read_per_clip)
    aug = next(c for c in prov["components"] if c["stage"] == "Augment")
    assert aug.get("engine") == "cpu_appearance_transform_stand_in"
    assert "stand-in" in aug["detail"].lower()
    origin = build_run_origin(keys, run_id=RUN, read_json=read_per_clip)
    assert origin["augment"]["engine"] == "cpu_appearance_transform_stand_in"


def test_provenance_only_reports_present_stages() -> None:
    # A run with only augment artifacts must not claim curation/reports happened.
    keys = [f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00000.png"]
    prov = build_run_provenance(keys, run_id=RUN)
    stages = {c["stage"] for c in prov["components"]}
    assert stages == {"Augment"}


def test_provenance_carries_origin() -> None:
    prov = build_run_provenance(KEYS, run_id=RUN, read_json=_read_gpu)
    assert "origin" in prov
    assert prov["origin"]["run_id"] == RUN


# The real paidf GPU run had NO source-frames / annotate-originals stage: the only
# stored visuals are the Cosmos Transfer 2.5 augmented frames.
_NO_ORIGINAL_KEYS = [
    f"{PFX}/configs/manifest.json",
    f"{PFX}/cosmos_augmented/manifest.json",
    f"{PFX}/cosmos_augmented/aug-{RUN}/metadata.json",
    f"{PFX}/cosmos_augmented/aug-{RUN}/augmented_video.mp4",
    f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00000.png",
    f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00001.png",
    f"{PFX}/labeled_augmented/captions.json",
    f"{PFX}/curation/report.json",
    f"{PFX}/reports/final.json",
]


def _read_no_original(key: str):
    if key.endswith("cosmos_augmented/manifest.json"):
        return {"mode": "cosmos_transfer2.5_gpu"}
    if key.endswith("labeled_augmented/captions.json"):
        return {
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "input_path": f"s3://bucket/{PFX}/cosmos_augmented/aug-{RUN}/",
        }
    if key.endswith("configs/manifest.json"):
        return {"variables": {"road_condition": ["dry", "wet"], "weather": ["clear"]}}
    return {}


def test_origin_when_no_original_input_stored() -> None:
    origin = build_run_origin(_NO_ORIGINAL_KEYS, run_id=RUN, read_json=_read_no_original)
    assert origin["original_present"] is False
    assert origin["original_inputs"] == []
    # Earliest stored visual is the Cosmos Transfer augmented output, not an original.
    assert origin["earliest_visual"]["stage"] == "Augment"
    assert origin["earliest_visual"]["count"] == 3  # 2 frames + 1 video
    assert origin["augment"]["engine"] == "cosmos_transfer_2.5_gpu"
    assert "cosmos_augmented/aug-" in origin["labeled_from"]
    summary = origin["summary"].lower()
    assert "no separate original input image was stored" in summary
    assert "augment outputs" in summary
    assert "cosmos transfer 2.5" in summary
    # Grounds the "generated from config, not an uploaded image" story.
    assert "road_condition" in origin["summary"]
    assert "pseudo-labeled those augmented frames" in origin["summary"]


def test_origin_when_source_frames_uploaded() -> None:
    keys = [
        f"{PFX}/input/clip0/frame-00000.png",
        f"{PFX}/input/clip0/frame-00001.png",
        f"{PFX}/labeled_original/captions.json",
        f"{PFX}/cosmos_augmented/manifest.json",
        f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00000.png",
    ]
    origin = build_run_origin(keys, run_id=RUN, read_json=_read_gpu)
    assert origin["original_present"] is True
    assert len(origin["original_inputs"]) == 2
    assert origin["original_inputs"][0]["kind"] == "image"
    assert "uploaded source" in origin["summary"].lower()
    assert "input/clip0/frame-00000.png" in origin["summary"]
