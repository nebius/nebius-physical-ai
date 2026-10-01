"""Prove native controls, model routing, direct prompts, and artifact binding."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workflows.video_sweep import artifacts, checkpoints, cosmos3, planning


VARIANT = {
    "prompt": "A yellow forklift carries a stable low pallet under warm lighting.",
    "seed": 17,
    "control_guidance": 1.5,
    "edge_threshold": "medium",
    "guidance": 5.0,
    "num_steps": 35,
}


@pytest.mark.parametrize(
    "key,value",
    [
        ("control_guidance", 0),
        ("control_guidance", float("nan")),
        ("guidance", True),
        ("num_steps", 0),
        ("seed", False),
        ("edge_threshold", "invented"),
        ("cfg_normalization", True),
        ("cfg_normalization", "invented"),
        ("prompt", ""),
        ("extra", "unsupported"),
    ],
)
def test_invalid_native_controls_fail_before_compute(key, value):
    with pytest.raises(ValueError):
        cosmos3.validate_variant({**VARIANT, key: value})


def test_native_manifest_requires_explicit_generation_identity():
    source = {"schema": "npa.video_sweep.sources.v1", "clips": ["source.mp4"]}
    sweep = {"schema": "npa.video_sweep.variants.v2", "variants": [VARIANT]}
    with pytest.raises(ValueError, match="explicit"):
        planning._validate_inputs(source, sweep)
    planning._validate_inputs(source, {**sweep, "generator": "cosmos3-nano"})


def test_direct_prompt_is_preserved_without_llm_rewrite(monkeypatch):
    source = {"sha256": "a" * 64, "description": "Synthetic warehouse."}
    monkeypatch.setattr(planning, "_describe_source", lambda *_: source)
    monkeypatch.setattr(planning, "_merge", lambda *_: pytest.fail("Prompt rewritten"))
    args = SimpleNamespace(root_uri="private", reasoner_model="reasoner", samples=4)
    rows = planning._expand_items(
        args, {"clips": ["source.mp4"]}, {"variants": [VARIANT]}, None
    )
    assert rows[0]["prompt"] == VARIANT["prompt"]
    assert rows[0]["merge_provenance"] == {"mode": "direct-user-prompt"}


def test_native_workflow_routes_to_cosmos3_image_and_literal_backend():
    from npa.orchestration.npa_workflow import load_spec
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
    from npa.orchestration.npa_workflow.skypilot_render import tool_image_key

    spec = load_spec(
        Path(__file__).resolve().parents[3]
        / "workflows/testing/video-variant-sweep-cosmos3.yaml"
    )
    for name in ("worker-0", "worker-1"):
        ref = spec.states[name].tool_ref
        assert tool_image_key(ref) == "cosmos3"
        argv = TOOL_CATALOG[ref].argv_template
        assert argv[argv.index("--generator") + 1] == "cosmos3-nano"
        assert TOOL_CATALOG[ref].access_capabilities == ("cosmos3",)


@pytest.mark.parametrize(
    "change",
    [
        {"guardrail_state": {"status": "passed", "effective": False}},
        {"structural_transfer": {"text_guardrail_passed": True}},
    ],
)
def test_configured_guardrails_are_not_effective_execution(change):
    result = {
        "guardrail_state": {"status": "passed", "effective": True},
        "structural_transfer": {
            "text_guardrail_passed": True,
            "video_guardrail_passed": True,
            "guardrail_postprocessing_applied": True,
        },
    }
    with pytest.raises(ValueError, match="Cosmos3 must prove"):
        cosmos3._guarded_transfer({**result, **change})


@pytest.mark.parametrize("normalization", [None, "enabled", "disabled"])
def test_native_transfer_retains_the_whole_source_contract(
    monkeypatch, tmp_path, normalization
):
    from npa.workflows import paidf_cosmos3_media as media

    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic-source")
    args = SimpleNamespace(root_uri=str(tmp_path / "run"))
    item = {
        "id": "a" * 64,
        "source": {"uri": str(source), "sha256": artifacts.file_digest(source)},
        "prompt": VARIANT["prompt"],
        "variant": {
            **VARIANT,
            **({"cfg_normalization": normalization} if normalization else {}),
        },
    }
    calls = []
    monkeypatch.setattr(media, "prepare_reference", lambda *_: {"status": "prepared"})
    monkeypatch.setattr(
        media,
        "verify_pair",
        lambda *args: calls.append(("alignment", args)) or {"status": "verified"},
    )

    def generate(**kwargs):
        calls.append(("generation", kwargs))
        return {"output_path": str(source)}

    monkeypatch.setattr("npa.workbench.cosmos.generate.run_cosmos3_generate", generate)
    monkeypatch.setattr(cosmos3, "_publish", lambda *args: {"generation": args[3]})
    cosmos3.generate(item, args, {"samples": 4})
    request = calls[0][1]
    assert request["mode"] == "video2video" and request["checkpoint"] == "Cosmos3-Nano"
    assert request["no_guardrails"] is False
    assert request["transfer"].control_guidance == 1.5
    assert request["transfer"].first_chunk_conditional_frames == 0
    assert request["transfer"].cfg_normalization == (normalization or "disabled")
    assert request["seed"] == 17 and request["num_steps"] == 35
    assert calls[1][0] == "alignment" and calls[1][1][-1] == 24


def test_native_checkpoint_detects_rebound_control_artifacts(tmp_path, monkeypatch):
    candidate = {
        "uri": "output.mp4",
        "sha256": "b" * 64,
        "video": {},
        "controls": {"edge": {"uri": "edge.mkv", "sha256": "c" * 64}},
        "reference": {"uri": "reference.mp4", "sha256": "d" * 64},
    }
    item = {"id": "a" * 64, "source": {"sha256": "e" * 64}}
    evidence = {
        "source_sha256": item["source"]["sha256"],
        "artifacts": {
            "video": {key: candidate[key] for key in ("uri", "sha256", "video")},
            "edge": copy.deepcopy(candidate["controls"]["edge"]),
            "reference": candidate["reference"],
        },
    }
    candidate["generation_sha256"] = artifacts.digest(evidence)
    monkeypatch.setattr(artifacts, "read_json", lambda _: evidence)
    monkeypatch.setattr(
        artifacts, "download", lambda *_: pytest.fail("Downloaded rebound media")
    )
    candidate["controls"]["edge"]["uri"] = "different.mkv"
    with pytest.raises(ValueError, match="evidence differs"):
        checkpoints._verify_native(candidate, item, str(tmp_path))


def test_export_parameters_omit_prompt_text_and_keep_actual_control(
    monkeypatch, tmp_path
):
    from npa.workflows.video_sweep import demo

    row = {
        "engine": "cosmos3-nano",
        "score": 0.9,
        "passed": True,
        "accepted": True,
        "controls": {"edge": {"uri": "private-control", "sha256": "a" * 64}},
    }
    item = {"variant": {**VARIANT, "prompt": "private prompt never exported"}}
    monkeypatch.setattr(demo, "_media", lambda stage, name, media: {"name": name})
    controls = []
    result = demo._candidate(tmp_path, item, row, 1, controls)
    assert result["engine"] == "Cosmos3-Nano"
    assert result["parameters"]["control_guidance"] == 1.5
    assert "prompt" not in result["parameters"]
    assert "private" not in str(result)
    assert controls == [{"name": "control-1"}]


def test_comparison_film_identifies_the_actual_generator(monkeypatch):
    from npa.workflows.video_sweep import film

    labels = []
    monkeypatch.setattr(
        film, "_text", lambda canvas, xy, text, *args: labels.append(text)
    )
    summary = {"threshold": 0.8, "judge": "Selected judge", "samples": 12}
    row = {"engine": "Cosmos3-Nano", "accepted": False, "seed": 17, "score": 0.2}
    film._comparison(summary, row, 0)
    assert "GENERATED  /  COSMOS3-NANO" in labels
    assert not any("TRANSFER 2.5" in label for label in labels)
