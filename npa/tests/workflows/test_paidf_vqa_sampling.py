"""Hosted request compatibility; synthetic protocols, never inference evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows import paidf_native
from test_paidf_native import (
    TOKEN_FACTORY_ENDPOINT,
    _link_label_fixtures,
    _native_identity,
    _write_fixture_json,
)


STAGES = ("visual-qa-anomaly", "visual-qa-person")


@pytest.fixture
def sampling_oracle():
    return json.loads(
        (Path(__file__).parents[1] / "fixtures/paidf_vqa_sampling_vectors.json").read_text()
    )


@pytest.mark.parametrize("stage", STAGES)
def test_vqa_contract_binds_reviewed_published_sampling(stage, sampling_oracle):
    contract = paidf_native._evg_vqa_request_media_contract(stage)
    source = sampling_oracle["source"]
    for field in ("repository", "revision"):
        assert contract[f"upstream_{field}"] == source[field]
    for field in ("path", "sha256"):
        assert contract[f"source_{field}"] == source[field]
    assert contract["value"] == contract["hosted_max_images"] == sampling_oracle["hosted_max_images"] == 10
    assert contract["sampling_strategy"] == "endpoint-inclusive-even-subsampling"
    if stage == "visual-qa-anomaly":
        assert contract["control"] == "--max-frames"
        assert contract["upstream_value"] == 16
        assert contract["sampling_function"] == "_sample_frame_ids"
        assert contract["sampling_scope"] == "candidate-frame-ids"
    else:
        assert contract["control"] == "--max-crops-per-track"
        assert contract["upstream_value"] == 12
        assert contract["sampling_function"] == "_sample_even"
        assert contract["sampling_scope"] == "track-crop-list"
    # These are outputs of the hash-verified published functions, not an NPA
    # sampling implementation. The real vendor CLI continues to own sampling.
    vectors = [v for v in sampling_oracle["cases"] if v["function"] == contract["sampling_function"]]
    assert vectors
    for vector in vectors:
        indices = vector["expected"]
        inputs = vector["inputs"]
        assert indices == sorted(set(indices)) and 0 < len(indices) <= 10
        if stage == "visual-qa-anomaly":
            assert inputs["max_frames"] == contract["value"]
            step = max(1, round(inputs["source_fps"] / inputs["sampling_fps"]))
            candidates = range(inputs["start_frame"], inputs["end_frame"] + 1, step)
            assert indices[0] == candidates[0] and indices[-1] == candidates[-1]
        else:
            assert inputs["max_crops"] == contract["value"]
            assert indices[0] == 0 and indices[-1] == inputs["candidate_count"] - 1


@pytest.fixture
def label_history(tmp_path):
    data = tmp_path / "auto_labeling/person/0"
    for name in (
        "contextual/objects.json",
        "contextual/instances.json",
        "sidecars/captioning/video_captions.json",
        "sidecars/visual_qa_anomaly/items.json",
        "sidecars/visual_qa_per_track/items.json",
        "sidecars/visual_qa_per_track/windows.normalized.json",
    ):
        _write_fixture_json(data / name, {})
    validation = _write_fixture_json(
        tmp_path / "validation.json",
        {
            **_native_identity("evg-validation", "evg"),
            "accepted": [{
                "input_key": "person", "augmentation_index": 0,
                "media_uri": str(tmp_path / "media.bmp"),
            }],
        },
    )
    labels = _write_fixture_json(
        tmp_path / "labels.json",
        {
            **_native_identity("evg-auto-label-person-attribute-search", "evg"),
            "outputs": [{"key": "person_aug0", "data_path": str(data)}],
        },
    )
    _link_label_fixtures(validation, labels)
    return validation, data


@pytest.mark.parametrize("stage", STAGES)
def test_vqa_actual_adapter_argv_and_report_share_the_hosted_cap(
    tmp_path, monkeypatch, stage, label_history
):
    validation, data = label_history
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-test-token")
    source = tmp_path / "orchestration"
    configs = source / "airflow/dags/workflows/event_video_generation_dag/configs"
    for name in ("question_bank.person_attributes.json", "question_bank.anomaly_tags.json"):
        _write_fixture_json(configs / name, {})
    monkeypatch.setattr(paidf_native, "_runtime_fetch", lambda *_args: source)
    commands = []
    monkeypatch.setattr(paidf_native, "_run_component", lambda argv, **_kwargs: commands.append(argv))
    previous = "captioning" if stage == "visual-qa-anomaly" else "visual-qa-anomaly"
    result = paidf_native.run_auto_label(
        "evg", stage, str(validation), str(data.parents[1]),
        str(tmp_path / "result.json"), TOKEN_FACTORY_ENDPOINT, "vlm-model",
        TOKEN_FACTORY_ENDPOINT, "llm-model", "unit-run",
        str(tmp_path / f"{previous}.json"),
    )
    assert len(commands) == 1 and commands[0][0] == "/app/.venv/bin/main"
    contract = result["request_media_contract"]
    argv = commands[0]
    assert argv.count(contract["control"]) == 1
    assert argv[argv.index(contract["control"]) + 1] == str(contract["value"]) == "10"
    assert argv[argv.index("--generation-mode") + 1] == "window-direct-vlm"
    if stage == "visual-qa-anomaly":
        assert "--single-window" in argv
        assert argv[argv.index("--sampling-fps") + 1] == "3.0"
    else:
        assert argv[argv.index("--track-crops-sidecar") + 1] == "detection_and_tracking/tracks.json"
    descriptor = paidf_native._producer_descriptor(str(tmp_path / "result.json"), result)
    assert descriptor["request_media_contract"] == contract
    documents = paidf_native._verified_producers(
        [*result["producers"], descriptor],
        paidf_native._lineage_kinds("evg")[:paidf_native._lineage_kinds("evg").index(f"evg-auto-label-{stage}") + 1],
        "unit-run", "evg",
    )
    assert documents[-1] == result


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("mutation", ["missing", "source_sha256", "upstream_revision", "control", "value", "hosted_max_images", "sampling_strategy", "upstream_value", "floating_value"])
def test_vqa_lineage_rejects_missing_or_rebound_request_media_contract(
    tmp_path, stage, mutation
):
    kind = f"evg-auto-label-{stage}"
    payload = {**_native_identity(kind, "evg"), "stage": stage, "producers": []}
    if mutation == "missing":
        del payload["request_media_contract"]
    elif mutation == "floating_value":
        payload["request_media_contract"]["value"] = 10.0
    else:
        old = payload["request_media_contract"][mutation]
        payload["request_media_contract"][mutation] = 11 if isinstance(old, int) else "changed"
    path = _write_fixture_json(tmp_path / "producer.json", payload)
    # Recompute the descriptor so this proves exact contract enforcement even
    # when the modified document and its declared lineage hash agree.
    descriptor = paidf_native._producer_descriptor(str(path), payload)
    with pytest.raises(paidf_native.PaidfNativeError, match="request media contract"):
        paidf_native._verified_producers([descriptor], [kind], "unit-run", "evg")


@pytest.mark.parametrize("stage", STAGES)
def test_vqa_descriptor_binds_request_media_contract_after_handoff(tmp_path, stage):
    kind = f"evg-auto-label-{stage}"
    payload = {**_native_identity(kind, "evg"), "stage": stage, "producers": []}
    path = _write_fixture_json(tmp_path / "producer.json", payload)
    descriptor = paidf_native._producer_descriptor(str(path), payload)
    payload["request_media_contract"]["value"] = 9
    _write_fixture_json(path, payload)
    with pytest.raises(paidf_native.PaidfNativeError, match="document changed"):
        paidf_native._verified_producers([descriptor], [kind], "unit-run", "evg")
