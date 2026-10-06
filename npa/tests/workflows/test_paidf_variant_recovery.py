"""Verify worker recovery with explicit native-API fixtures; these are not live proof."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workbench.cosmos.guarded_inference import GUARDRAIL_STATE_SCHEMA
from npa.workflows import paidf_cosmos3 as paidf
from npa.workflows import paidf_variant_recovery as recovery

IMAGE = "ghcr.io/example/npa-cosmos3@sha256:" + "a" * 64
MODEL = {
    "schema": "npa.cosmos3.nano-model-binding.v1",
    "revision": "b" * 40,
    "repository": "nvidia/Cosmos3-Nano",
    "files": {"fixture": {"sha256": "c" * 64, "bytes": 1}},
}


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.fail_uri = ""

    def read_bytes_with_etag(self, uri):
        payload = self.objects.get(uri)
        return (
            None if payload is None else (payload, hashlib.sha256(payload).hexdigest())
        )

    def put_bytes_conditional(self, payload, uri, *, if_none_match=False, **_kwargs):
        assert if_none_match is True
        if uri in self.objects:
            raise StoragePreconditionFailed("synthetic concurrent writer")
        if self.fail_uri and uri.endswith(self.fail_uri):
            raise RuntimeError("synthetic interrupted upload")
        self.objects[uri] = bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def upload_file(self, source, uri):
        self.objects[uri] = Path(source).read_bytes()
        return uri


def _video(path, color):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=832x480:r=24",
            "-frames:v",
            "6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _inputs(root):
    configs, captions = root / "configs", root / "captions"
    configs.mkdir()
    captions.mkdir()
    (configs / "manifest.json").write_text(
        json.dumps(
            {
                "augmentations": [
                    {"prompt": f"Synthetic unit profile {index}."}
                    for index in range(12)
                ]
            }
        )
    )
    (captions / "captions.json").write_text(
        json.dumps({"captions": [{"caption": "Synthetic unit caption."}]})
    )
    return configs, captions


@pytest.fixture
def batch(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real decode fixtures require ffmpeg and ffprobe")
    monkeypatch.setattr(recovery, "_source_binding", lambda: {"fixture.py": "d" * 64})
    monkeypatch.setattr(recovery, "_hardware_binding", lambda _env: ["fixture-gpu,1,1"])
    snapshot_calls = []

    def snapshot(mode, revision, _env):
        snapshot_calls.append((mode, revision))
        return (
            {"revision": MODEL["revision"]}
            if mode == "resolve"
            else {"model_binding": copy.deepcopy(MODEL)}
        )

    monkeypatch.setattr(recovery, "_snapshot_command", snapshot)
    source = _video(tmp_path / "source.mp4", "blue")
    generated = _video(tmp_path / "generated.mp4", "red")
    configs, captions = _inputs(tmp_path)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"status": "prepared", "sha256": recovery.video_sha256(source)})
    )
    kwargs = _batch_kwargs(tmp_path, source, provenance, configs, captions)
    return kwargs, source, generated, snapshot_calls


def _batch_kwargs(tmp_path, source, provenance, configs, captions):
    kwargs = dict(
        input_video_uri=str(source),
        input_provenance_uri=str(provenance),
        captions_uri=str(captions),
        configs_uri=str(configs),
        output_uri="s3://example-bucket/unit-run/cosmos_augmented/",
        scores_uri=str(tmp_path / "scores"),
        attempt_uri=str(tmp_path / "attempt.json"),
        mode="video2video",
        checkpoint="Cosmos3-Nano",
        prompt="Synthetic unit prompt.",
        negative_prompt="Synthetic unit negative.",
        seed=17,
        guidance=5,
        steps=35,
        variant_count=12,
        variant_parallelism=1,
        retry_seed_stride=100,
        retry_guidance_delta=0,
        retry_steps_delta=0,
        parallelism_preset="latency",
        guardrails=True,
        run_id="unit-run",
        structural_control="edge",
        control_guidance=1,
        transfer_edge_threshold="low",
        transfer_rgb_weight=0.25,
        transfer_first_chunk_conditional_frames=0,
        environ={"CUDA_VISIBLE_DEVICES": "0", "NPA_TASK_IMAGE": IMAGE},
        storage=MemoryStorage(),
    )
    return kwargs


def _sample(kwargs):
    transfer = kwargs["transfer"]
    return {
        "name": kwargs["name"],
        "model_mode": "video2video",
        "prompt": kwargs["prompt"],
        "negative_prompt": kwargs["negative_prompt"],
        "seed": kwargs["seed"],
        "num_steps": kwargs["num_steps"],
        "guidance": kwargs["guidance"],
        "fps": transfer.fps,
        "num_frames": transfer.chunk_frames,
        "num_video_frames_per_chunk": transfer.chunk_frames,
        "num_conditional_frames": 5,
        "max_frames": 6,
        "control_guidance": transfer.control_guidance,
        "num_first_chunk_conditional_frames": transfer.first_chunk_conditional_frames,
        "normalize_cfg": False,
        "edge": {"preset_edge_threshold": transfer.edge_threshold},
    }


def _controls(source, directory):
    edge, rgb = directory / "edge.mkv", directory / "rgb.mkv"
    edge.write_bytes(b"explicit synthetic edge control fixture")
    rgb.write_bytes(b"explicit synthetic RGB control fixture")
    return {
        "schema": "npa.cosmos3.structural-transfer.v1",
        "source_frames": 6,
        "source_video_sha256": recovery.video_sha256(source),
        "control_loader_verified": True,
        "text_guardrail_passed": True,
        "video_guardrail_passed": True,
        "guardrail_postprocessing_applied": True,
        "control_path": str(edge),
        "control_sha256": recovery.video_sha256(edge),
        "control_pixels_sha256": "1" * 64,
        "output_fps": 24,
        "native_torch_compile": False,
        "first_chunk_conditional_frames": 0,
        "overlap_conditional_frames": 5,
        "normalize_cfg": False,
        "effective_prompts": ["Synthetic actual guard prompt."],
        "native_chunks": 1,
        "rgb_conditioning": {
            "control_loader_verified": True,
            "preset": "none",
            "weight": 0.25,
            "control_path": str(rgb),
            "control_sha256": recovery.video_sha256(rgb),
            "control_pixels_sha256": "2" * 64,
        },
    }


def _native_fixture(kwargs, source, generated):
    artifact = Path(kwargs["output_path"]) / kwargs["name"] / "vision.mp4"
    artifact.parent.mkdir(parents=True)
    shutil.copy2(generated, artifact)
    roles = {
        "prompt_input": ["unit-text-guard"],
        "generated_media": ["unit-video-guard"],
    }
    config = {"synthetic_unit_native_api_config": True}
    controls = _controls(source, artifact.parent)
    controls["guarded_output_sha256"] = recovery.video_sha256(artifact)
    return {
        "output_path": str(artifact),
        "guardrail_state": {
            "schema": GUARDRAIL_STATE_SCHEMA,
            "requested": True,
            "effective": True,
            "status": "passed",
            "discovered": roles,
            "evaluated": roles,
        },
        "native_model_selection": {
            "schema": "npa.cosmos3.native-model-selection.v1",
            "model_binding": copy.deepcopy(MODEL),
            "effective_config": config,
            "effective_config_sha256": recovery._digest(config),
        },
        "sample_outputs": {"args": _sample(kwargs), "status": "success"},
        "structural_transfer": controls,
    }


def _generator(source, generated, calls, *, fail_from=None, mutate=None):
    def run(**kwargs):
        index = int(kwargs["name"].split("-")[1])
        calls.append(index)
        if fail_from is not None and index >= fail_from:
            raise RuntimeError("synthetic worker interruption")
        assert kwargs["environ"]["NPA_COSMOS3_NANO_REVISION"] == MODEL["revision"]
        result = _native_fixture(kwargs, source, generated)
        if mutate:
            mutate(result)
        return result

    return run


def _receipts(storage):
    return {
        uri: json.loads(payload)
        for uri, payload in storage.objects.items()
        if "/_generation-recovery/" in uri and "/variant-" in uri
    }


def _interrupted(batch, completed=1):
    kwargs, source, generated, _calls = batch
    with pytest.raises(paidf.PaidfCosmos3Error):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, [], fail_from=completed)
        )


def test_five_verified_variants_survive_worker_loss_and_only_seven_are_generated(batch):
    kwargs, source, generated, snapshot_calls = batch
    _interrupted(batch, 5)
    assert len(_receipts(kwargs["storage"])) == 5
    assert not Path(kwargs["attempt_uri"]).exists()
    assert not any(
        uri.endswith("cosmos_augmented/manifest.json")
        for uri in kwargs["storage"].objects
    )
    calls = []
    result = paidf.generate_variants(
        **kwargs, generator=_generator(source, generated, calls)
    )
    assert sorted(calls) == list(range(5, 12))
    assert result["variant_count"] == 12 and result["verified_variant_recovery"] is True
    assert [variant["seed"] for variant in result["variants"]] == list(range(17, 29))
    assert len(_receipts(kwargs["storage"])) == 12
    assert snapshot_calls == [
        ("resolve", ""),
        ("materialize", MODEL["revision"]),
        ("materialize", MODEL["revision"]),
    ]
    assert (
        paidf.validate_committed_augment_manifest(result, kwargs["output_uri"])
        == result["variants"]
    )


def test_legacy_progress_and_metadata_cannot_skip_generation(batch):
    kwargs, source, generated, _calls = batch
    kwargs["storage"].objects[kwargs["output_uri"] + "generation-progress.json"] = (
        b'{"published_variant_count":12}'
    )
    kwargs["storage"].objects[kwargs["output_uri"] + "variant-0000/metadata.json"] = (
        b'{"status":"executed"}'
    )
    calls = []
    result = paidf.generate_variants(
        **kwargs, generator=_generator(source, generated, calls)
    )
    assert sorted(calls) == list(range(12)) and result["variant_count"] == 12


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", "changed"),
        ("negative_prompt", "changed"),
        ("seed", 18),
        ("guidance", 6),
        ("steps", 36),
        ("variant_count", 11),
        ("control_guidance", 2),
        ("transfer_edge_threshold", "medium"),
        ("transfer_rgb_weight", 0.5),
        ("transfer_first_chunk_conditional_frames", 1),
        ("transfer_cfg_normalization", "enabled"),
        ("parallelism_preset", "throughput"),
        ("run_id", "other-unit-run"),
    ],
)
def test_changed_settings_fail_before_generation(batch, field, value):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    kwargs[field] = value
    calls = []
    with pytest.raises(recovery.VariantRecoveryError, match="settings changed"):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


@pytest.mark.parametrize(
    "kind", ["model", "source", "hardware", "image", "caption", "config"]
)
def test_changed_model_source_runtime_and_inputs_fail_closed(batch, monkeypatch, kind):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    if kind == "model":
        monkeypatch.setattr(
            recovery,
            "_snapshot_command",
            lambda *_args: {"model_binding": {**MODEL, "files": {}}},
        )
    elif kind == "source":
        monkeypatch.setattr(
            recovery, "_source_binding", lambda: {"fixture.py": "e" * 64}
        )
    elif kind == "hardware":
        monkeypatch.setattr(
            recovery, "_hardware_binding", lambda _env: ["different-unit-gpu,1,1"]
        )
    elif kind == "image":
        kwargs["environ"]["NPA_TASK_IMAGE"] = (
            "ghcr.io/example/npa-cosmos3@sha256:" + "e" * 64
        )
    elif kind == "caption":
        (Path(kwargs["captions_uri"]) / "captions.json").write_text(
            '{"captions":[{"caption":"Changed."}]}'
        )
    else:
        path = Path(kwargs["configs_uri"]) / "manifest.json"
        value = json.loads(path.read_bytes()) | {"augmentation_seed": "changed"}
        path.write_text(json.dumps(value))
    calls = []
    with pytest.raises(recovery.VariantRecoveryError):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


@pytest.mark.parametrize(
    "kind",
    ["video", "missing-video", "metadata", "control", "frame", "native", "inventory"],
)
def test_corrupted_completion_fails_closed_before_generation(batch, kind):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    storage = kwargs["storage"]
    receipt_uri, receipt = next(iter(_receipts(storage).items()))
    names = {
        "video": "augmented_video.mp4",
        "missing-video": "augmented_video.mp4",
        "metadata": "metadata.json",
        "control": "source_edges.mkv",
        "native": "native_execution.json",
    }
    if kind == "inventory":
        storage.objects[receipt_uri] = json.dumps(receipt | {"objects": {}}).encode()
    elif kind == "missing-video":
        storage.objects.pop(receipt["prefix"] + names[kind])
    else:
        name = (
            next(name for name in receipt["objects"] if name.endswith(".png"))
            if kind == "frame"
            else names[kind]
        )
        storage.objects[receipt["prefix"] + name] = b"corrupted unit object"
    calls = []
    with pytest.raises(recovery.VariantRecoveryError):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


@pytest.mark.parametrize(
    "field",
    [
        "native_model_selection",
        "guardrail_state",
        "sample_outputs",
        "structural_transfer",
    ],
)
def test_missing_native_proof_never_commits_completion(batch, field):
    kwargs, source, generated, _calls = batch
    kwargs["variant_count"] = 1
    with pytest.raises(paidf.PaidfCosmos3Error):
        paidf.generate_variants(
            **kwargs,
            generator=_generator(
                source, generated, [], mutate=lambda result: result.pop(field)
            ),
        )
    assert _receipts(kwargs["storage"]) == {}
    assert not any(
        uri.endswith("cosmos_augmented/manifest.json")
        for uri in kwargs["storage"].objects
    )


def test_partial_upload_is_not_reusable(batch):
    kwargs, source, generated, _calls = batch
    kwargs["variant_count"] = 1
    kwargs["storage"].fail_uri = "source_rgb.mkv"
    with pytest.raises(paidf.PaidfCosmos3Error):
        paidf.generate_variants(**kwargs, generator=_generator(source, generated, []))
    assert _receipts(kwargs["storage"]) == {}
    kwargs["storage"].fail_uri = ""
    calls = []
    result = paidf.generate_variants(
        **kwargs, generator=_generator(source, generated, calls)
    )
    assert calls == [0] and result["variant_count"] == 1


@pytest.mark.parametrize(
    "image",
    [
        "",
        "ghcr.io/example/npa-cosmos3:latest",
        "ghcr.io/example/npa-cosmos3@sha256:bad",
    ],
)
def test_mutable_image_cannot_enable_recovery(image):
    if not image:
        assert (
            recovery.recovery_enabled(
                "Cosmos3-Nano", "edge", "unit-run", {"NPA_TASK_IMAGE": image}
            )
            is False
        )
    else:
        with pytest.raises(recovery.VariantRecoveryError, match="immutable"):
            recovery.recovery_enabled(
                "Cosmos3-Nano", "edge", "unit-run", {"NPA_TASK_IMAGE": image}
            )


def test_tagged_and_bare_digest_references_bind_to_the_same_image():
    tagged = "ghcr.io/example/npa-cosmos3:dev-" + "b" * 40 + "@sha256:" + "a" * 64
    assert recovery._image_binding(tagged) == recovery._image_binding(IMAGE)


def test_object_creation_race_accepts_only_identical_bytes(tmp_path):
    storage, prefix = MemoryStorage(), "s3://example-bucket/unit-run/variant/"
    publication = recovery.ImmutableVariantPublication(storage, prefix)
    source = tmp_path / "object"
    source.write_bytes(b"unit publication bytes")
    storage.objects[prefix + "object"] = source.read_bytes()
    assert publication.upload_file(str(source), prefix + "object") == prefix + "object"
    source.write_bytes(b"changed unit publication bytes")
    with pytest.raises(recovery.VariantRecoveryError, match="readback"):
        publication.upload_file(str(source), prefix + "object")


@pytest.mark.parametrize(
    "relative", ["../escape", "a/../escape", "/escape", "a\\escape", ""]
)
def test_object_cannot_escape_directory(relative):
    prefix = "s3://example-bucket/unit-run/variant/"
    with pytest.raises(recovery.VariantRecoveryError):
        recovery._relative_object(prefix + relative, prefix)


def _change_document(storage, uri, mutate):
    value = json.loads(storage.objects[uri])
    mutate(value)
    payload = json.dumps(value).encode()
    storage.objects[uri] = payload
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


@pytest.mark.parametrize(
    "field", ["seed", "guidance", "steps", "variables", "video_bytes", "frame_count"]
)
def test_descriptor_drift_cannot_reuse_an_otherwise_valid_native_receipt(batch, field):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    storage = kwargs["storage"]
    uri, receipt = next(iter(_receipts(storage).items()))
    receipt["variant"][field] = {"changed": True} if field == "variables" else 999
    storage.objects[uri] = json.dumps(receipt).encode()
    calls = []
    with pytest.raises(recovery.VariantRecoveryError):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


def test_native_rgb_claim_without_published_rgb_object_cannot_be_reused(batch):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    storage = kwargs["storage"]
    uri, receipt = next(iter(_receipts(storage).items()))
    receipt["objects"].pop("source_rgb.mkv")
    storage.objects.pop(receipt["prefix"] + "source_rgb.mkv")
    storage.objects[uri] = json.dumps(receipt).encode()
    calls = []
    with pytest.raises(recovery.VariantRecoveryError, match="inventory"):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


@pytest.mark.parametrize(
    "object_name,field",
    [
        ("metadata.json", "seed"),
        ("metadata.json", "prompt"),
        ("metadata.json", "published_video_sha256"),
        ("metadata.json", "lineage"),
        ("transfer.json", "control_uri"),
        ("transfer.json", "source_video_sha256"),
    ],
)
def test_semantic_publication_drift_fails_even_with_updated_object_hash(
    batch, object_name, field
):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    storage = kwargs["storage"]
    uri, receipt = next(iter(_receipts(storage).items()))
    object_uri = receipt["prefix"] + object_name
    receipt["objects"][object_name] = _change_document(
        storage, object_uri, lambda value: value.update({field: "changed"})
    )
    storage.objects[uri] = json.dumps(receipt).encode()
    calls = []
    with pytest.raises(recovery.VariantRecoveryError):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


@pytest.mark.parametrize("kind", ["missing", "shifted", "zero-based", "unpadded"])
def test_frame_sequence_must_use_exact_existing_canonical_names(batch, kind):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    storage = kwargs["storage"]
    uri, receipt = next(iter(_receipts(storage).items()))
    original = "frame-00001.png"
    evidence = receipt["objects"].pop(original)
    replacements = {
        "shifted": "frame-00007.png",
        "zero-based": "frame-00000.png",
        "unpadded": "frame-1.png",
    }
    if kind != "missing":
        replacement = replacements[kind]
        receipt["objects"][replacement] = evidence
        storage.objects[receipt["prefix"] + replacement] = storage.objects[
            receipt["prefix"] + original
        ]
    storage.objects[uri] = json.dumps(receipt).encode()
    calls = []
    with pytest.raises(recovery.VariantRecoveryError, match="frame sequence"):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


def test_content_address_must_equal_actual_video_bytes(batch):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    storage = kwargs["storage"]
    uri, receipt = next(iter(_receipts(storage).items()))
    old = receipt["prefix"]
    new = old.rsplit("/", 2)[0] + "/" + "f" * 64 + "/"
    for name in receipt["objects"]:
        storage.objects[new + name] = storage.objects[old + name]
    receipt["prefix"] = new
    receipt["variant"]["augmented_video_uri"] = new + "augmented_video.mp4"
    storage.objects[uri] = json.dumps(receipt).encode()
    calls = []
    with pytest.raises(recovery.VariantRecoveryError, match="actual video bytes"):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


def test_compare_and_swap_batch_winner_cannot_be_silently_replaced(batch, monkeypatch):
    kwargs, source, generated, _calls = batch
    storage = kwargs["storage"]
    original = storage.put_bytes_conditional

    def race(payload, uri, **conditions):
        if uri.endswith("/batch.json"):
            candidate = json.loads(payload)
            candidate["model_binding"]["revision"] = "e" * 40
            storage.objects[uri] = json.dumps(candidate).encode()
        return original(payload, uri, **conditions)

    monkeypatch.setattr(storage, "put_bytes_conditional", race)
    calls = []
    with pytest.raises(recovery.VariantRecoveryError, match="model or request changed"):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, calls)
        )
    assert calls == []


def test_invalid_semantic_descriptor_is_never_sealed_as_complete(batch, monkeypatch):
    kwargs, source, generated, _calls = batch
    kwargs["variant_count"] = 1
    original = paidf._publish_variant

    def drift(**arguments):
        return original(**arguments) | {"seed": 999}

    monkeypatch.setattr(paidf, "_publish_variant", drift)
    with pytest.raises(paidf.PaidfCosmos3Error):
        paidf.generate_variants(**kwargs, generator=_generator(source, generated, []))
    assert _receipts(kwargs["storage"]) == {}


def test_all_twelve_verified_completions_recover_without_any_generator_calls(
    batch, monkeypatch
):
    kwargs, source, generated, _calls = batch
    original = paidf._write_json

    def lose_worker_after_variants(value, uri, **options):
        if uri.endswith("cosmos_augmented/manifest.json"):
            raise RuntimeError("synthetic loss after last immutable completion")
        return original(value, uri, **options)

    monkeypatch.setattr(paidf, "_write_json", lose_worker_after_variants)
    with pytest.raises(RuntimeError, match="last immutable completion"):
        paidf.generate_variants(**kwargs, generator=_generator(source, generated, []))
    assert len(_receipts(kwargs["storage"])) == 12
    assert not Path(kwargs["attempt_uri"]).exists()
    monkeypatch.setattr(paidf, "_write_json", original)
    calls = []
    result = paidf.generate_variants(
        **kwargs, generator=_generator(source, generated, calls)
    )
    assert calls == []
    assert result["recovered_variant_count"] == 12
    assert len(result["variants"]) == 12


@pytest.mark.parametrize("kind", ["source", "video", "guard-schema", "sample-status"])
def test_inconsistent_native_execution_cannot_seal_completion(batch, kind):
    kwargs, source, generated, _calls = batch
    kwargs["variant_count"] = 1

    def drift(result):
        if kind == "source":
            result["structural_transfer"]["source_video_sha256"] = "e" * 64
        elif kind == "video":
            result["structural_transfer"]["guarded_output_sha256"] = "e" * 64
        elif kind == "guard-schema":
            result["guardrail_state"]["schema"] = "unbound-schema"
        else:
            result["sample_outputs"]["status"] = "running"

    with pytest.raises(paidf.PaidfCosmos3Error):
        paidf.generate_variants(
            **kwargs, generator=_generator(source, generated, [], mutate=drift)
        )
    assert _receipts(kwargs["storage"]) == {}


def test_valid_other_worker_completion_wins_without_overwriting_objects(
    batch, monkeypatch, tmp_path
):
    kwargs, source, generated, _calls = batch
    _interrupted(batch)
    storage = kwargs["storage"]
    receipt_uri, prior = next(iter(_receipts(storage).items()))
    prior_payload = storage.objects.pop(receipt_uri)
    kwargs["variant_count"] = 12
    original = storage.put_bytes_conditional

    def race(payload, uri, **conditions):
        if uri == receipt_uri:
            storage.objects[uri] = prior_payload
        return original(payload, uri, **conditions)

    monkeypatch.setattr(storage, "put_bytes_conditional", race)
    different_video = _video(tmp_path / "other-worker.mp4", "green")
    calls = []
    result = paidf.generate_variants(
        **kwargs, generator=_generator(source, different_video, calls)
    )
    assert sorted(calls) == list(range(12))
    assert result["variants"][0] == prior["variant"]
    assert storage.objects[receipt_uri] == prior_payload


def test_native_failure_report_preserves_safe_reason_and_never_echoes_stderr(
    tmp_path, monkeypatch
):
    from npa.workbench.cosmos import generate

    monkeypatch.setattr(generate, "cosmos3_repo", lambda _env: tmp_path)

    def failed(command, **_options):
        destination = Path(command[command.index("--receipt-path") + 1])
        destination.write_text(
            json.dumps(
                {
                    "schema": "npa.cosmos3.nano-failure.v1",
                    "reason": "Native default sound checkpoint contract changed",
                }
            )
        )
        return subprocess.CompletedProcess(
            command, 1, b"", b"synthetic-secret-token /private/operator-path"
        )

    monkeypatch.setattr(recovery.subprocess, "run", failed)
    with pytest.raises(recovery.VariantRecoveryError) as error:
        recovery._snapshot_command("materialize", "a" * 40, {})
    message = str(error.value)
    assert (
        "native exit code 1" in message
        and "sound checkpoint contract changed" in message
    )
    assert "synthetic-secret" not in message and "/private/" not in message
