"""Unit tests for Physical AI Data Factory run provenance."""

from __future__ import annotations

from npa.workflows.data_factory_provenance import build_run_provenance

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


def test_provenance_only_reports_present_stages() -> None:
    # A run with only augment artifacts must not claim curation/reports happened.
    keys = [f"{PFX}/cosmos_augmented/aug-{RUN}/frame-00000.png"]
    prov = build_run_provenance(keys, run_id=RUN)
    stages = {c["stage"] for c in prov["components"]}
    assert stages == {"Augment"}
