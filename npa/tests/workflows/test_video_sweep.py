"""Exercise sweep coverage, real video decoding, review gates, and publication."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import av
import httpx
import numpy as np
import pytest
from botocore.exceptions import ClientError

from npa.workflows.video_sweep import (
    artifacts,
    execution,
    planning,
    publication,
    tracking,
    vision,
)
from npa.workflows.video_sweep.__main__ import main


def test_workflow_partitions_before_review_and_tracking_before_publication(monkeypatch):
    from npa.orchestration.npa_workflow import load_spec
    from npa.orchestration.npa_workflow.skypilot_render import (
        render_skypilot_steps_yaml,
        tool_image_key,
        tool_pip_extra,
    )
    from npa.orchestration.npa_workflow.waves import build_wave_plan

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/npa")
    root = Path(__file__).resolve().parents[3]
    spec = load_spec(root / "workflows/testing/video-variant-sweep.yaml")
    waves = build_wave_plan(spec, run_id="test").waves
    assert [[step.state for step in wave.steps] for wave in waves] == [
        ["prepare"],
        ["worker-0", "worker-1"],
        ["review"],
        ["lineage"],
        ["publish"],
    ]
    assert waves[1].kind == "parallel"
    assert tool_image_key("workflow.video_sweep.generate") == "cosmos2-transfer"
    assert tool_image_key("workflow.video_sweep.review") is None
    assert tool_pip_extra("workflow.video_sweep.lineage") == "video-sweep"
    for wave in waves:
        rendered = render_skypilot_steps_yaml(
            spec, wave.steps, run_id="test", execution=wave.kind
        )
        assert "npa.workflows.video_sweep" in rendered


@pytest.mark.parametrize("status", [307, 401, 500])
def test_tracking_failure_does_not_produce_a_receipt(status):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={}))
    with httpx.Client(
        base_url="https://tracking.example/", transport=transport
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            tracking._post(client, "runs/create", {"experiment_id": "0"})


@pytest.mark.parametrize(
    "uri",
    [
        "http://tracking.example",
        "https://user:password@tracking.example",
        "https://tracking.example?token=secret",
    ],
)
def test_tracking_credentials_are_not_embedded_in_urls(monkeypatch, uri):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(ValueError):
        tracking._mlflow_client()


@pytest.mark.parametrize("code", ["PreconditionFailed", "412", "KeyAlreadyExists"])
def test_s3_conditional_replay_requires_identical_bytes(monkeypatch, code):
    def put(**kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        raise ClientError({"Error": {"Code": code}}, "PutObject")

    monkeypatch.setattr(artifacts, "_client", lambda: SimpleNamespace(put_object=put))
    monkeypatch.setattr(artifacts, "read_json", lambda _: {"value": True})
    artifacts.write_json("s3://example-bucket/receipt.json", {"value": True})
    with pytest.raises(ValueError, match="different stage result"):
        artifacts.write_json("s3://example-bucket/receipt.json", {"value": 1})


def test_local_artifacts_refuse_conflicting_rewrites(tmp_path):
    path = str(tmp_path / "receipt.json")
    artifacts.write_json(path, {"accepted": True})
    artifacts.write_json(path, {"accepted": True})
    with pytest.raises(ValueError, match="different stage result"):
        artifacts.write_json(path, {"accepted": False})


def test_storage_error_is_not_treated_as_replay(monkeypatch):
    def put(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")

    monkeypatch.setattr(artifacts, "_client", lambda: SimpleNamespace(put_object=put))
    with pytest.raises(ClientError):
        artifacts.write_json("s3://example-bucket/receipt.json", {"accepted": True})


def _video(path: Path, color: int = 80) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for step in range(16):
            pixels = np.full((64, 64, 3), color, dtype=np.uint8)
            pixels[20:36, step * 2 : step * 2 + 16] = (255, 0, 0)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class _Model:
    def list_models(self):
        return ["vision-model", "text-model"]

    def chat_completion(self, model, messages):
        instruction = messages[0]["content"][0]["text"]
        answer = "A red block moves across a gray surface."
        if "Return ONLY JSON" in instruction:
            answer = json.dumps(
                {
                    "passed": True,
                    "score": 0.95,
                    "reason": "Motion preserved in sampled frames.",
                }
            )
        return {
            "id": "synthetic-request",
            "model": model,
            "usage": {"total_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": answer}}],
        }


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    _video(source)
    sources, variants = tmp_path / "sources.json", tmp_path / "variants.json"
    artifacts.write_json(
        str(sources), {"schema": "npa.video_sweep.sources.v1", "clips": [str(source)]}
    )
    settings = {
        "hint": "Use warm lighting",
        "seed": 1,
        "control": "edge",
        "control_weight": 1.0,
        "guidance": 3.0,
    }
    artifacts.write_json(
        str(variants),
        {
            "schema": "npa.video_sweep.variants.v1",
            "variants": [settings, {**settings, "seed": 2}],
        },
    )
    args = SimpleNamespace(
        root_uri=str(tmp_path / "run"),
        run_id="test-run",
        sources_uri=str(sources),
        variants_uri=str(variants),
        workers=2,
        worker=0,
        samples=4,
        reasoner_model="vision-model",
        merge_model="text-model",
        threshold=0.8,
    )
    monkeypatch.setattr(planning, "TokenFactoryClient", _Model)
    monkeypatch.setattr(execution, "TokenFactoryClient", _Model)
    planning.prepare(args)
    calls = []

    def transfer(**kwargs):
        calls.append(kwargs)
        return {
            "video_path": kwargs["input_video"],
            "input_conditioned": True,
            "content_guardrails_enabled": True,
            "inference_seed": kwargs["seed"],
        }

    monkeypatch.setattr("npa.workbench.cosmos.transfer.run_cosmos_transfer", transfer)
    for worker in range(2):
        args.worker = worker
        execution.generate(args)
    return args, calls


def _lineage_receipt(args):
    report = artifacts.read_json(args.root_uri + "/review.json")
    artifacts.write_json(
        args.root_uri + "/lineage.json",
        {
            "review_sha256": artifacts.digest(report),
            "items": [
                {"id": row["id"], "mlflow_run_id": "synthetic-track"}
                for row in report["items"]
            ],
        },
    )


def test_complete_stage_handoffs_decode_real_media(sweep):
    args, calls = sweep
    execution.review(args)
    _lineage_receipt(args)
    publication.publish(args)
    dataset = artifacts.read_json(args.root_uri + "/dataset/manifest.json")
    assert len(dataset["clips"]) == 2
    assert {call["seed"] for call in calls} == {1, 2}
    assert all(call["prompt"] and call["input_video"] for call in calls)
    for clip in dataset["clips"]:
        _, metadata = vision.sample_video(Path(clip["uri"]), 4)
        assert metadata["frame_count"] == 16
        assert artifacts.file_digest(Path(clip["uri"])) == clip["sha256"]
    next_sources = artifacts.read_json(args.root_uri + "/dataset/next-sources.json")
    assert next_sources["clips"] == [row["uri"] for row in dataset["clips"]]


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"passed":"true","score":1,"reason":"x"}',
        '{"passed":true,"score":NaN,"reason":"x"}',
        '{"passed":true,"score":2,"reason":"x"}',
        '{"passed":true,"score":true,"reason":"x"}',
        '{"passed":true,"score":1,"reason":""}',
        '```json\n{"passed":true,"score":1,"reason":"x"}\n```',
    ],
)
def test_malformed_judgments_never_pass(text):
    with pytest.raises(ValueError):
        vision.parse_review(text, 0.8)


@pytest.mark.parametrize(
    "passed,score,accepted", [(False, 1, False), (True, 0.7, False), (True, 0.8, True)]
)
def test_boolean_and_score_are_both_required(passed, score, accepted):
    result = vision.parse_review(
        json.dumps({"passed": passed, "score": score, "reason": "observed"}), 0.8
    )
    assert result["accepted"] is accepted


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "wrong-plan", "wrong-worker"]
)
def test_worker_join_refuses_incomplete_or_foreign_results(sweep, mutation):
    args, _ = sweep
    path = Path(args.root_uri) / "workers/1.json"
    receipt = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
        with pytest.raises(FileNotFoundError):
            execution.review(args)
        return
    if mutation == "duplicate":
        receipt["items"] *= 2
    if mutation == "wrong-plan":
        receipt["plan_sha256"] = "wrong"
    if mutation == "wrong-worker":
        receipt["worker"] = 0
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        execution.review(args)
    assert not (Path(args.root_uri) / "review.json").exists()


def test_changed_candidate_bytes_block_publication(sweep):
    args, _ = sweep
    execution.review(args)
    _lineage_receipt(args)
    report = artifacts.read_json(args.root_uri + "/review.json")
    Path(report["items"][0]["uri"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        publication.publish(args)
    assert not (Path(args.root_uri) / "dataset/manifest.json").exists()


def test_lineage_is_required_before_publication(sweep):
    args, _ = sweep
    execution.review(args)
    with pytest.raises(FileNotFoundError):
        publication.publish(args)


def test_rejected_clips_are_excluded_but_kept_in_review(sweep):
    args, _ = sweep
    execution.review(args)
    report = artifacts.read_json(args.root_uri + "/review.json")
    report["items"][0].update(passed=False, accepted=False)
    Path(args.root_uri + "/review.json").write_text(json.dumps(report))
    _lineage_receipt(args)
    publication.publish(args)
    dataset = artifacts.read_json(args.root_uri + "/dataset/manifest.json")
    assert len(dataset["clips"]) == 1
    assert len(artifacts.read_json(args.root_uri + "/review.json")["items"]) == 2


def test_all_rejected_never_publishes_dataset(sweep):
    args, _ = sweep
    execution.review(args)
    report = artifacts.read_json(args.root_uri + "/review.json")
    for row in report["items"]:
        row.update(passed=False, accepted=False)
    Path(args.root_uri + "/review.json").write_text(json.dumps(report))
    _lineage_receipt(args)
    with pytest.raises(ValueError, match="No candidate passed"):
        publication.publish(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("control", "depth"),
        ("control_weight", float("nan")),
        ("hint", ""),
        ("guidance", -1),
    ],
)
def test_invalid_sweep_fails_before_model_calls(field, value):
    variant = {
        "hint": "warmer",
        "seed": 0,
        "control": "edge",
        "control_weight": 1,
        "guidance": 3,
    }
    variant[field] = value
    with pytest.raises(ValueError):
        planning._validate_variant(variant)


def test_foreign_model_or_truncated_answer_is_rejected():
    response = _Model().chat_completion(
        "vision-model", [{"content": [{"text": "describe"}]}]
    )
    for mutation in ("model", "finish"):
        altered = copy.deepcopy(response)
        if mutation == "model":
            altered["model"] = "another-model"
        else:
            altered["choices"][0]["finish_reason"] = "length"
        client = SimpleNamespace(chat_completion=lambda **_: altered)
        with pytest.raises(ValueError):
            vision.completion(client, "vision-model", [])


def test_cli_does_not_echo_private_error_content(monkeypatch, capsys):
    def fail(_):
        raise ValueError("private-prompt-and-password")

    monkeypatch.setattr("npa.workflows.video_sweep.__main__.prepare", fail)
    assert main(["prepare", "--root-uri", "unused", "--run-id", "test"]) == 1
    assert "private-prompt" not in capsys.readouterr().err


def test_demo_exports_decoded_media_without_private_metadata(sweep, tmp_path):
    from npa.workflows.video_sweep.demo import export_demo

    args, _ = sweep
    execution.review(args)
    report_path = Path(args.root_uri) / "review.json"
    report = json.loads(report_path.read_text())
    report["items"][0]["reason"] = (
        "private-review-text </script><script>alert(1)</script>"
    )
    report_path.write_text(json.dumps(report))
    _lineage_receipt(args)
    publication.publish(args)
    output = tmp_path / "demo"
    summary = export_demo(args, output)
    html = (output / "index.html").read_text()
    assert summary["accepted"] == 2
    assert summary["judge"] == "Operator-selected model"
    for forbidden in (
        args.root_uri,
        args.run_id,
        "private-review-text",
        "synthetic-track",
        "synthetic-request",
        "Use warm lighting",
    ):
        assert forbidden not in html
        assert forbidden not in (output / "summary.json").read_text()
    assert "__DEMO_DATA__" not in html
    assert "data:video/mp4;base64," in html
    assert output.stat().st_mode & 0o777 == 0o700
    with av.open(str(output / "demo.mp4")) as video:
        assert not video.streams.audio
        assert sum(1 for _ in video.decode(video=0)) > 24 * 9
    for row in summary["candidates"]:
        path = output / (row["name"] + ".mp4")
        _, metadata = vision.sample_video(path, 2)
        assert metadata["frame_count"] == row["frames"] == 16
        assert artifacts.file_digest(path) == row["sha256"]
    with pytest.raises(FileExistsError):
        export_demo(args, output)


@pytest.mark.parametrize(
    "corruption", ["dataset_hash", "lineage", "media", "missing_clip", "next_inventory"]
)
def test_demo_refuses_unverified_publication(sweep, tmp_path, corruption):
    from npa.workflows.video_sweep.demo import export_demo

    args, _ = sweep
    execution.review(args)
    _lineage_receipt(args)
    publication.publish(args)
    path = Path(args.root_uri) / "dataset/manifest.json"
    dataset = json.loads(path.read_text())
    if corruption == "dataset_hash":
        dataset["review_sha256"] = "wrong"
    elif corruption == "lineage":
        dataset["lineage_sha256"] = "wrong"
    elif corruption == "media":
        Path(dataset["clips"][0]["uri"]).write_bytes(b"changed")
    elif corruption == "next_inventory":
        (Path(args.root_uri) / "dataset/next-sources.json").write_text("{}")
    else:
        dataset["clips"].pop()
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError):
        export_demo(args, tmp_path / "demo")
    assert not (tmp_path / "demo").exists()
