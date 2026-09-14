"""Decode PAIDF review evidence without publishing operator infrastructure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from npa.workflows import data_factory_viz as viz
from npa.workflows.data_factory_input import load_starter_contract
from npa.workflows.data_factory_review import _PaidfReview


ROOT = "s3://private-fixture-bucket/workflows/review-run"


def _write_json(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("reference,expected", [
    (ROOT + "/input/source.mp4", "input/source.mp4"),
    (ROOT + "/input/timeline.json", "input/timeline.json"),
    ("input/source.mp4", "input/source.mp4"),
    (ROOT + "-other/input/source.mp4", None),
    (ROOT.replace("private-fixture-bucket", "external-fixture-bucket") + "/input/source.mp4", None),
    (ROOT + "/input/source.mp4?X-Amz-Signature=private-signature", None),
    (ROOT + "/input/source.mp4#private-fragment", None),
    (ROOT + "/../external/source.mp4", None),
    (ROOT + "/%2e%2e/external/source.mp4", None),
    (ROOT + "/%2fexternal/source.mp4", None),
    ("https://private-endpoint.invalid/source.mp4", None),
    ("https://github.com/private-fixture/repository", None),
    ("https://huggingface.co/private-fixture/model", None),
    ("C:\\private-operator\\source.mp4", None),
])
def test_references_require_exact_canonical_root(tmp_path, reference, expected):
    result = _PaidfReview(tmp_path, ROOT).reference(reference)
    assert result == (expected or "[private location omitted]")


def test_grounded_public_references_and_local_paths(tmp_path):
    review = _PaidfReview(tmp_path, ROOT)
    contract = load_starter_contract()
    for field in ("authoritative_url", "asset_url", "episode_metadata_url"):
        value = contract["source"][field]
        assert review.reference(value) == value
        assert review.reference(value + "?token=private-token") == "[private location omitted]"
    local = tmp_path / "input/source.mp4"
    assert review.reference(str(local)) == "input/source.mp4"
    assert review.reference(str(tmp_path.parent / "outside.mp4")) == "[private location omitted]"


def test_projection_preserves_facts_without_arbitrary_nested_runtime_data(tmp_path):
    source = _write_json(tmp_path, "input/provenance.json", {
        "run_id": "review-run", "source_kind": "upstream_sample", "sha256": "a" * 64,
        "immutable_revision": "b" * 40, "staged_video_uri": ROOT + "/input/source.mp4",
        "media": {"duration_seconds": 3.38, "decoded_frames": 169, "hostname": "private-worker"},
        "runtime": {"project_id": "private-project", "token": "private-token"},
    })
    before = source.read_bytes()
    review = _PaidfReview(tmp_path, ROOT)
    projected = review.read(source, "input")
    assert projected["media"] == {"duration_seconds": 3.38, "decoded_frames": 169}
    assert projected["sha256"] == "a" * 64 and projected["immutable_revision"] == "b" * 40
    assert projected["staged_video_uri"] == "input/source.mp4"
    assert projected["source_report"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert source.read_bytes() == before
    assert "runtime" not in projected
    assert not any(value in json.dumps(projected) for value in ("private-worker", "private-project", "private-token"))


def test_caption_appearance_model_and_candidate_identity_remain_useful(tmp_path):
    _write_json(tmp_path, "input/provenance.json", {"hostname": "worker", "bucket": "private-fixture-bucket"})
    review = _PaidfReview(tmp_path, ROOT)
    assert review.text("a robot folds blue cloth under warm light") == "a robot folds blue cloth under warm light"
    assert "unlisted-private-token" not in review.text("blue cloth; token=unlisted-private-token")
    payload = review.payload({
        "model": "nvidia/worker-model", "run_id": "worker-run", "candidate_id": "worker-candidate",
        "variants": [{"clip": "worker-candidate", "variables": {"lighting": "warm", "surface_finish": "matte",
                                                                 "hostname": "worker", "diagnostic": {"url": "private"}}}],
    }, "augment")
    assert payload["model"] == "nvidia/worker-model" and payload["run_id"] == "worker-run"
    assert payload["variants"][0]["clip"] == "worker-candidate"
    assert payload["variants"][0]["variables"] == {"lighting": "warm", "surface_finish": "matte"}
    candidate = review.payload({"candidate_id": "iteration-1/worker-candidate"}, "candidate")
    assert candidate["candidate_id"] == "iteration-1/worker-candidate"
    variables = review.payload({"variables": {"lighting": "warm", "aws_secret_access_key": "fixture-secret"}}, "metadata")
    assert variables["variables"]["lighting"] == "warm"
    assert "fixture-secret" not in json.dumps(variables)


@pytest.mark.parametrize("private_field", ["hostname", "api_key", "aws_secret_access_key"])
def test_candidate_lookup_precedes_display_redaction(tmp_path, private_field):
    path = _write_json(tmp_path, "grade/iteration-1/ranking/cosmos_evaluator.json", {
        private_field: "candidate-a", "clips": [{"clip_id": "candidate-a", "score": 0.82, "passed": True,
                                                  "attribute_verification": {"vlm_model": "example/candidate-a"}}],
    })
    review = _PaidfReview(tmp_path, ROOT)
    selected = viz._candidate_evaluation(tmp_path, 1, "candidate-a", review)
    assert selected["passed"] is True and selected["score"] == 0.82
    assert selected["source_report"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert private_field not in selected
    if private_field == "hostname":
        assert selected["clip_id"] == "candidate-a"
        assert selected["attribute_verification"]["vlm_model"] == "example/candidate-a"
        assert review.payload({"clip_ids": ["candidate-a"]}, "curation")["clip_ids"] == ["candidate-a"]
    else:
        assert "candidate-a" not in selected["clip_id"]
        assert "candidate-a" not in selected["attribute_verification"]["vlm_model"]
    assert "candidate-a" not in review.text("The runtime host is candidate-a.")


@pytest.mark.parametrize("identity", [
    "https://private-endpoint.invalid/model?X-Amz-Signature=private-signature",
    ROOT + "/model?token=private-signature", "Bearer private-credential", "token=private-credential",
])
def test_semantic_identifiers_keep_location_and_credential_guards(tmp_path, identity):
    projected = _PaidfReview(tmp_path, ROOT).payload({"model": identity}, "augment")
    assert "private-endpoint" not in projected["model"]
    assert "private-signature" not in projected["model"]
    assert "private-credential" not in projected["model"]


def _quality_fixture(run):
    metric = {"engine": "observed-metric", "score": 0.82, "threshold": 0.75, "passed": True,
              "total_frames": 2, "regions": [{"region_id": "center", "score": 0.83,
                                               "bounds": [0, 0, 1, 1], "hostname": "private-worker"}]}
    clip = {"clip_id": "candidate-a", "score": 0.82, "passed": True, "hallucination": metric,
            "appearance_fidelity": metric, "temporal_consistency": metric,
            "attribute_verification": {"passed": True, "vlm_model": "Qwen/Qwen2.5-VL-72B-Instruct",
                                       "checks": [{"variable": "lighting", "value": "warm", "passed": True,
                                                   "question": "Is the lighting warm?", "options": {"A": "yes", "B": "no"},
                                                   "vlm_answer": "A", "error": "private-provider-error"}]}}
    evaluator = {"schema": "npa.cosmos.evaluator.v1", "status": "completed", "score": 0.82,
                 "threshold": 0.75, "clips": [clip], "result_uri": ROOT + "/grade/iteration-1/ranking/cosmos_evaluator.json",
                 "upstream": {"repo": "https://github.com/nvidia-cosmos/cosmos-evaluator", "license": "Apache-2.0"}}
    _write_json(run, "grade/iteration-1/cosmos_evaluator.json", evaluator)
    _write_json(run, "grade/iteration-1/ranking/cosmos_evaluator.json", evaluator)
    _write_json(run, "grade/quality_disposition.json", {"quality_status": "accepted", "score": 0.82,
                "threshold": 0.75, "evaluator_report_uri": evaluator["result_uri"], "hostname": "private-worker"})
    _write_json(run, "grade/iteration-1/decision.json", {"decision": "promote_checkpoint"})


def _recording_fixture(run):
    from PIL import Image

    candidate = run / "cosmos_augmented/iteration-1/candidate-a"
    candidate.mkdir(parents=True)
    Image.new("RGB", (32, 24), (15, 35, 55)).save(candidate / "frame-00000.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-loop", "1", "-i",
                    str(candidate / "frame-00000.png"), "-t", "0.2", "-pix_fmt", "yuv420p", "-y",
                    str(candidate / "augmented_video.mp4")], check=True, capture_output=True)
    _write_json(run, "input/provenance.json", {"run_id": "review-run", "source_kind": "video_uri",
                "sha256": "a" * 64, "staged_video_uri": ROOT + "/input/source.mp4",
                "timeline_uri": ROOT + "/input/timeline.json", "hostname": "private-worker",
                "media": {"duration_seconds": 3.38, "decoded_frames": 169}})
    _write_json(run, "cosmos_augmented/iteration-1/manifest.json", {
        "schema": "npa.cosmos2.transfer.v1", "mode": "cosmos_transfer2.5_gpu", "status": "executed",
        "node_count": 1, "variant_count": 1, "model": "nvidia/Cosmos-Transfer2.5-2B",
        "lineage": {"input_provenance_uri": ROOT + "/input/provenance.json",
                    "captions_uri": ROOT + "-other/private-captions.json"},
        "variants": [{"clip": "candidate-a", "variant_index": 0,
                      "augmented_video_uri": ROOT + "/cosmos_augmented/iteration-1/candidate-a/augmented_video.mp4",
                      "control_uris": {}, "seed": 42, "steps": 10}],
    })
    _write_json(run, "cosmos_augmented/iteration-1/candidate-a/metadata.json",
                {"variables": {"lighting": "warm", "hostname": "private-worker"}, "cwd": "/private/operator/work"})
    _quality_fixture(run)
    return candidate


def _additional_documents(run):
    _write_json(run, "configs/manifest.json", {"scene": "robot folding cloth",
                "augmentations": [{"lighting": "warm", "prompt": "blue cloth under warm light", "hostname": "private-worker"}]})
    _write_json(run, "labeled_original/captions.json", {"model": "Qwen/Qwen2.5-VL-72B-Instruct", "captions": [{
        "image": ROOT + "/input/frame-00000.png", "caption": "a robot folds blue cloth under warm light"}, {
        "image": "https://private-endpoint.invalid/frame.png?X-Amz-Signature=private-signature",
        "caption": "Visible cloth. Source s3://external-fixture-bucket/private/frame.png on private-worker."}]})
    _write_json(run, "labeled_augmented/captions.json", {"captions": [{"image": "/private/operator/frame.png",
                "caption": "blue cloth; token=private-token"}], "token": "private-token"})
    _write_json(run, "curation/cosmos_curator.json", {"engine": "cosmos-curator", "clip_count": 1,
                "output_uri": "s3://external-fixture-bucket/private-curator", "diagnostic": {"host": "private-worker"}})
    _write_json(run, "curation/report.json", {"augmented_clips": 1, "multiply": {"mode": "single-variant"},
                "dataset_uri": ROOT + "/curation/dataset?X-Amz-Signature=private-signature"})
    _write_json(run, "reports/final.json", {"artifact_count": 14, "has_rrd": True, "variant_count": 1,
                "input_source": {"staged_video_uri": ROOT + "/input/source.mp4", "sha256": "a" * 64},
                "runtime": {"project_id": "private-project", "hostname": "private-worker"}})


def _decoded_text(chunks):
    documents = {}
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if "TextDocument:text" in batch.schema.names:
            documents.setdefault(str(chunk.entity_path), []).extend(
                row[0] for row in batch.column("TextDocument:text").to_pylist() if row)
    return documents


def _json_document(text):
    return json.loads(re.search(r"```json\s*(.*?)```", text, re.S).group(1))


@pytest.mark.parametrize("overlapping_hostname", [False, True])
def test_real_rrd_preserves_review_evidence_and_omits_private_locations(tmp_path, monkeypatch, overlapping_hostname):
    from rerun.recording import load_recording

    run = tmp_path / "review-run"
    candidate = _recording_fixture(run)
    _additional_documents(run)
    if overlapping_hostname:
        _write_json(run, "reports/runtime.json", {"hostname": "candidate-a"})
        _write_json(run, "labeled_augmented/captions.json", {"model": "example/candidate-a",
                    "captions": [{"image": "frame.png", "caption": "Visible cloth; runtime host is candidate-a."}]})
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    out = tmp_path / "review.rrd"
    monkeypatch.setattr(viz, "_s3_inventory", lambda *_args: [])
    monkeypatch.setattr(viz, "_materialize_run", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(viz, "_publish", lambda source, uri, **_kwargs: (shutil.copyfile(source, out), uri)[1])
    result = viz.build_run_rrd(ROOT, ROOT + "/reports/review.rrd", storage_client=object())
    assert result["augmented_video_components"] == 1
    subprocess.run([str(Path(sys.executable).with_name("rerun")), "rrd", "verify", str(out)], check=True, capture_output=True)
    recording = load_recording(out)
    assert recording.recording_id() == "review-run"
    chunks = list(recording.chunks())
    docs = _decoded_text(chunks)
    text = "\n".join(item for values in docs.values() for item in values)
    for private in ("s3://", "private-fixture-bucket", "external-fixture-bucket", "private-worker", "private-project",
                    "private-signature", "private-endpoint", "private-token", "private-provider-error", "/private/operator", str(tmp_path)):
        assert private not in text
    assert "a robot folds blue cloth under warm light" in text
    assert "nvidia/Cosmos-Transfer2.5-2B" in text and "Qwen/Qwen2.5-VL-72B-Instruct" in text
    assert "https://github.com/nvidia-cosmos/cosmos-evaluator" in text
    if overlapping_hostname:
        caption = docs["/captions/labeled_augmented"][0]
        assert '"model": "example/candidate-a"' in caption
        assert "runtime host is [private identity omitted]." in caption
    _assert_recorded_facts(run, candidate, chunks, docs)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def _assert_recorded_facts(run, candidate, chunks, docs):
    provenance = _json_document(docs["/provenance/input"][0])
    assert provenance["staged_video_uri"] == "input/source.mp4"
    assert provenance["timeline_uri"] == "input/timeline.json"
    assert provenance["media"] == {"duration_seconds": 3.38, "decoded_frames": 169}
    assert provenance["sha256"] == "a" * 64
    assert provenance["source_report"]["sha256"] == hashlib.sha256((run / "input/provenance.json").read_bytes()).hexdigest()
    candidate_text = docs["/augmented/iteration-1/candidate-a/disposition"][0]
    assert "# ACCEPTED — candidate `iteration-1/candidate-a`" in candidate_text
    candidate_doc = _json_document(candidate_text)
    assert candidate_doc["candidate_id"] == "iteration-1/candidate-a"
    assert candidate_doc["output_media_entity"] == "augmented/iteration-1/candidate-a"
    assert candidate_doc["promotion_eligible"] is True and candidate_doc["score"] == 0.82
    label = docs["/augmented/iteration-1/candidate-a"][0]
    assert label.startswith("candidate-a: lighting=warm")
    assert _json_document(label)["artifact"] == "cosmos_augmented/iteration-1/candidate-a/metadata.json"
    assert candidate_doc["final_disposition"]["evaluator_report_uri"] == "grade/iteration-1/ranking/cosmos_evaluator.json"
    assert candidate_doc["hallucination"]["regions"] == [{"region_id": "center", "score": 0.83, "bounds": [0, 0, 1, 1]}]
    assert len(candidate_doc["source_reports"]) == 2
    assert candidate_doc["attribute_results"][0]["question"] == "Is the lighting warm?"
    columns = {name for chunk in chunks for name in chunk.to_record_batch().schema.names}
    assert {"frame", "video_time", "EncodedImage:blob", "AssetVideo:blob"} <= columns
    videos = [bytes(row[0]) for chunk in chunks if str(chunk.entity_path).endswith("/video")
              for row in chunk.to_record_batch().to_pydict().get("AssetVideo:blob", []) if row]
    assert videos == [(candidate / "augmented_video.mp4").read_bytes()]
