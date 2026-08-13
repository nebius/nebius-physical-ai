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


def test_provenance_reflects_real_fiftyone_brain_curation() -> None:
    def read_fo(key: str):
        if key.endswith("cosmos_augmented/manifest.json"):
            return {"mode": "cosmos_transfer2.5_gpu"}
        if key.endswith("curation/report.json"):
            return {
                "curation_engine": "fiftyone-brain",
                "curated_kept": 4,
                "curated_dropped": 1,
                "fiftyone": {"fiftyone_version": "1.15.0"},
            }
        return {}

    prov = build_run_provenance(KEYS, run_id=RUN, read_json=read_fo)
    cur = next(c for c in prov["components"] if c["stage"] == "Curation")
    assert cur["component"] == "Real FiftyOne Brain curation"
    assert cur.get("engine") == "fiftyone_brain"
    assert "npa-fiftyone image" in cur["runtime"]
    assert "uniqueness" in cur["detail"] and "4 kept" in cur["detail"]
    assert cur["model"] == "fiftyone 1.15.0"


def test_provenance_flags_report_only_curation() -> None:
    def read_report_only(key: str):
        if key.endswith("curation/report.json"):
            return {"curation_engine": "report-only"}
        return {}

    prov = build_run_provenance(KEYS, run_id=RUN, read_json=read_report_only)
    cur = next(c for c in prov["components"] if c["stage"] == "Curation")
    assert cur.get("engine") == "report_only"
    assert "FiftyOne Brain did not run" in cur["detail"]


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


def test_provenance_uses_truthful_learning_phases_for_groot_offline_eval() -> None:
    run = "groot17-learning-test"
    root = f"groot-1-7-finetune/{run}"
    keys = [
        f"{root}/data/train/videos/chunk-000/observation.image/episode_000000.mp4",
        f"{root}/data/heldout/videos/chunk-000/observation.image/episode_000000.mp4",
        f"{root}/reports/split/manifest.json",
        f"{root}/checkpoints/baseline/model.safetensors",
        f"{root}/offline/baseline/evaluation.json",
        f"{root}/checkpoints/candidate/checkpoint-4/model.safetensors",
        f"{root}/checkpoints/candidate/npa_groot_finetune_manifest.json",
        f"{root}/offline/trained/evaluation.json",
        f"{root}/reports/two-gpu-pipeline-report.json",
        f"{root}/reports/offline-heldout-comparison.mp4",
        f"{root}/reports/groot-offline-evaluation.rrd",
        f"{root}/reports/groot-offline-evaluation.mcap",
        f"{root}/reports/publish-manifest.json",
        f"{root}/workflow.yaml",
    ]

    def read_learning(key: str):
        if key.endswith("reports/two-gpu-pipeline-report.json"):
            return {
                "schema": "npa.groot.learning.v1",
                "evaluation_kind": "offline held-out policy evaluation",
                "closed_loop": False,
                "dataset": {
                    "camera_names": ["front"],
                    "source_resolution": "96x96",
                    "heldout_episodes": 1,
                    "fps": 10,
                },
                "provenance": {"primary_camera": "front"},
                "training": {"distinct_gpu_count": 7, "coverage_criterion": "one complete pass"},
                "evaluation": {"metric_name": "action_mse", "real_model_forward": True},
            }
        return {}

    prov = build_run_provenance(keys, run_id=run, read_json=read_learning)
    stages = [component["stage"] for component in prov["components"]]
    assert stages == [
        "Prepare leakage-free split",
        "Baseline held-out inference",
        "Multi-GPU policy training",
        "Post-training held-out inference",
        "Compare learning",
        "Synchronized learning replay",
        "Validate and publish",
    ]
    assert "Visualize + finalize" not in prov["summary"]
    assert "not a rollout" in prov["summary"]
    assert "real Gr00tPolicy forwards" in prov["components"][1]["detail"]
    assert prov["origin"]["original_present"] is True
    assert "96x96 native" in prov["origin"]["summary"]
    assert "not a simulator/robot rollout" in prov["origin"]["summary"]


def test_groot_filename_without_authoritative_metadata_is_not_semantic_proof() -> None:
    run = "ambiguous-groot-name"
    root = f"arbitrary/{run}"
    keys = [
        f"{root}/reports/two-gpu-pipeline-report.json",
        f"{root}/reports/groot-offline-evaluation.rrd",
    ]

    prov = build_run_provenance(
        keys,
        run_id=run,
        read_json=lambda _key: {
            "schema": "attacker.chosen.v1",
            "evaluation_kind": "offline held-out policy evaluation",
            "closed_loop": False,
        },
    )

    assert "Offline held-out GR00T" not in prov["summary"]
    assert all(component["stage"] != "Synchronized learning replay" for component in prov["components"])


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


def test_origin_when_source_frames_are_operator_provided() -> None:
    keys = [
        f"{PFX}/input/clip0/frame-00000.png",
        f"{PFX}/input/clip0/frame-00001.png",
        f"{PFX}/labeled_original/captions.json",
        f"{PFX}/cosmos_augmented/manifest.json",
        f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00000.png",
    ]
    def read_with_source(key: str):
        if key.endswith("configs/manifest.json"):
            return {
                "input_source": {
                    "kind": "operator_provided",
                    "uri": f"s3://bucket/{PFX}/input/",
                }
            }
        return _read_gpu(key)

    keys.insert(0, f"{PFX}/configs/manifest.json")
    origin = build_run_origin(keys, run_id=RUN, read_json=read_with_source)
    assert origin["original_present"] is True
    assert len(origin["original_inputs"]) == 2
    assert origin["original_inputs"][0]["kind"] == "image"
    assert "user-supplied input" in origin["summary"].lower()
    assert "input/clip0/frame-00000.png" in origin["summary"]
    assert origin["input_source"]["kind"] == "operator_provided"


def test_origin_truthfully_identifies_seeded_fixture_as_run_input() -> None:
    keys = [
        f"{PFX}/configs/manifest.json",
        f"{PFX}/input/frame_0000.png",
        f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00000.png",
    ]

    def read_seeded(key: str):
        if key.endswith("configs/manifest.json"):
            return {
                "input_source": {
                    "kind": "npa_seeded_fixture",
                    "uri": f"s3://bucket/{PFX}/input/",
                    "frame_count": 8,
                }
            }
        return {}

    origin = build_run_origin(keys, run_id=RUN, read_json=read_seeded)
    assert origin["original_present"] is False
    assert origin["original_inputs"] == []
    assert origin["input_source"]["kind"] == "npa_seeded_fixture"
    assert "synthetic seeded fixture" in origin["summary"].lower()
    assert "not original real-world data" in origin["summary"].lower()
    assert "uploaded" not in origin["summary"].lower()
