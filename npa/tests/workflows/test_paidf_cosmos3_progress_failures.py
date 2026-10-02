"""Failure-injection regressions for Cosmos 3 variant progress reporting.

``_publish_completed_variants`` must drain and publish every sibling result even
when a generation, publication, or generation-progress.json upload fails, then
fail closed without a canonical manifest, while keeping raw exception text out
of JSON artifacts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workflows import paidf_cosmos3 as c3

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None, reason="ffmpeg is required for video fixture tests"
)

PROGRESS_KEY = "/generation-progress.json"
# Stands in for content that must never be serialized into artifacts.
INJECTED_TOKEN = "injected-upload-error-token-XYZ"


def _tiny_video(path: Path, *, color: str = "blue") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x64:d=2:r=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


class _ProgressFailureStorage:
    """In-memory object store that can fail progress uploads on demand.

    ``progress_failures=None`` never fails; an integer is the number of
    remaining progress uploads that raise before recovery (``float("inf")``
    models a persistent outage).  ``on_first_progress_write``, when supplied,
    is set just before the first failing progress upload, letting tests gate a
    sibling on the collector having reached that write.
    """

    def __init__(
        self,
        progress_failures: float | None,
        on_first_progress_write: threading.Event | None = None,
    ) -> None:
        self.objects: dict[str, bytes] = {}
        self.s3 = SimpleNamespace(list_objects_v2=self._list)
        self.progress_failures = progress_failures
        self.on_first_progress_write = on_first_progress_write

    def upload_file(self, source: str, uri: str) -> str:
        if uri.endswith(PROGRESS_KEY) and self.progress_failures:
            if self.on_first_progress_write is not None:
                self.on_first_progress_write.set()
            self.progress_failures -= 1
            raise RuntimeError(f"upload refused: {INJECTED_TOKEN}")
        self.objects[uri] = Path(source).read_bytes()
        return uri

    def _list(self, *, Bucket: str, Prefix: str, **_kwargs):
        stem = f"s3://{Bucket}/"
        return {
            "Contents": [
                {"Key": uri.removeprefix(stem)}
                for uri in self.objects
                if uri.startswith(stem + Prefix)
            ],
            "IsTruncated": False,
        }


def _generation_inputs(tmp_path: Path) -> dict[str, Path]:
    source = _tiny_video(tmp_path / "source.mp4")
    configs = tmp_path / "configs"
    captions = tmp_path / "captions"
    scores = tmp_path / "scores"
    configs.mkdir()
    captions.mkdir()
    scores.mkdir()
    (configs / "manifest.json").write_text(
        json.dumps(
            {
                "augmentations": [
                    {"lighting": "bright daylight", "prompt": "Use bright daylight."},
                    {"lighting": "warm lamp light", "prompt": "Use warm lamp light."},
                ]
            }
        ),
        encoding="utf-8",
    )
    (captions / "captions.json").write_text(
        json.dumps({"captions": [{"caption": "a robot arm moves a cube"}]}),
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"status": "prepared"}), encoding="utf-8")
    return {
        "source": source,
        "configs": configs,
        "captions": captions,
        "scores": scores,
        "provenance": provenance,
        "attempt": configs / "attempt.json",
    }


def _generate_kwargs(paths, storage, generator) -> dict:
    return dict(
        input_video_uri=str(paths["source"]),
        input_provenance_uri=str(paths["provenance"]),
        captions_uri=str(paths["captions"]),
        configs_uri=str(paths["configs"]),
        output_uri="s3://example-bucket/run/cosmos_augmented/",
        scores_uri=str(paths["scores"]),
        attempt_uri=str(paths["attempt"]),
        mode="video2video",
        checkpoint="Cosmos3-Nano",
        prompt="Preserve the robot motion.",
        negative_prompt="distortion",
        seed=10,
        guidance=5.0,
        steps=20,
        variant_count=2,
        variant_parallelism=2,
        retry_seed_stride=100,
        retry_guidance_delta=-0.5,
        retry_steps_delta=2,
        parallelism_preset="latency",
        guardrails=True,
        run_id="test-run",
        storage=storage,
        environ={"CUDA_VISIBLE_DEVICES": "0,1"},
        generator=generator,
    )


def _healthy_generator(healthy_video: Path):
    def generator(**kwargs):
        artifact = Path(kwargs["output_path"]) / kwargs["name"] / "vision.mp4"
        artifact.parent.mkdir(parents=True)
        shutil.copy2(healthy_video, artifact)
        return {"output_path": str(artifact), "output_bytes": artifact.stat().st_size}

    return generator


PROGRESS_URI = "s3://example-bucket/run/cosmos_augmented/generation-progress.json"


def _assert_no_leaked_token(storage) -> None:
    leaked = [
        uri
        for uri, body in storage.objects.items()
        if uri.endswith(".json") and INJECTED_TOKEN.encode() in body
    ]
    assert leaked == []


def _assert_all_clips_published(storage, clips) -> None:
    for clip in clips:
        assert any(f"/{clip}/" in uri for uri in storage.objects)


def _assert_failed_batch_contract(storage, attempt_path) -> None:
    """Siblings retained, but never a canonical manifest or attempt advance."""
    _assert_all_clips_published(storage, ("variant-0000", "variant-0001"))
    assert not any(
        uri.endswith("cosmos_augmented/manifest.json") for uri in storage.objects
    )
    assert not attempt_path.exists()
    _assert_no_leaked_token(storage)


def _stored_progress(storage) -> dict:
    return json.loads(storage.objects[PROGRESS_URI])


@requires_ffmpeg
def test_single_progress_write_failure_fails_stage_but_keeps_evidence(
    tmp_path: Path,
) -> None:
    """One failed progress write fails the stage; siblings stay as evidence.

    The later successful write must persist a document that agrees with the
    failed stage: status "failed" with an additive count of the earlier
    progress-write failure, never "completed".
    """
    paths = _generation_inputs(tmp_path)
    healthy_video = _tiny_video(tmp_path / "generated.mp4", color="red")
    storage = _ProgressFailureStorage(progress_failures=1)

    with pytest.raises(c3.PaidfCosmos3Error) as excinfo:
        c3.generate_variants(
            **_generate_kwargs(paths, storage, _healthy_generator(healthy_video))
        )

    message = str(excinfo.value)
    assert "write failed after 1 completed variant(s)" in message
    assert "2 published variants retained" in message
    assert INJECTED_TOKEN not in message
    assert INJECTED_TOKEN in str(excinfo.value.__cause__)
    _assert_failed_batch_contract(storage, paths["attempt"])
    progress = _stored_progress(storage)
    assert progress["status"] == "failed"
    assert progress["published_variant_count"] == 2
    assert progress["failed_variant_count"] == 0
    assert progress["progress_write_failure_count"] == 1


@requires_ffmpeg
def test_persistent_progress_write_failure_retains_siblings_and_fails_closed(
    tmp_path: Path,
) -> None:
    """Every progress write fails: siblings still publish, no manifest."""
    paths = _generation_inputs(tmp_path)
    healthy_video = _tiny_video(tmp_path / "generated.mp4", color="green")
    storage = _ProgressFailureStorage(progress_failures=float("inf"))

    with pytest.raises(c3.PaidfCosmos3Error) as excinfo:
        c3.generate_variants(
            **_generate_kwargs(paths, storage, _healthy_generator(healthy_video))
        )

    message = str(excinfo.value)
    assert "write failed after 2 completed variant(s)" in message
    assert "2 published variants retained" in message
    assert INJECTED_TOKEN not in message
    assert INJECTED_TOKEN in str(excinfo.value.__cause__)
    _assert_failed_batch_contract(storage, paths["attempt"])


@requires_ffmpeg
def test_first_observed_error_is_the_raised_cause(tmp_path: Path) -> None:
    """The cause is exactly the first observed error object, by identity.

    The sibling cannot finish generation until the storage signals the first
    progress write, which happens only after the collector has drained the
    failing variant and recorded its error, so the identity assertion is
    deterministic rather than dependent on as_completed ordering.
    """
    paths = _generation_inputs(tmp_path)
    healthy_video = _tiny_video(tmp_path / "generated.mp4", color="red")
    first_write_attempted = threading.Event()
    storage = _ProgressFailureStorage(
        progress_failures=float("inf"),
        on_first_progress_write=first_write_attempted,
    )
    first_error = RuntimeError(f"generation exploded first: {INJECTED_TOKEN}")

    def ordered_generator(**kwargs):
        if kwargs["name"] == "variant-0000":
            raise first_error
        assert first_write_attempted.wait(timeout=10)
        return _healthy_generator(healthy_video)(**kwargs)

    with pytest.raises(c3.PaidfCosmos3Error) as excinfo:
        c3.generate_variants(**_generate_kwargs(paths, storage, ordered_generator))

    assert excinfo.value.__cause__ is first_error
    message = str(excinfo.value)
    assert "1 of 2 variants failed" in message
    assert "write failed after 2 completed variant(s)" in message
    assert INJECTED_TOKEN not in message
    # The healthy sibling is retained, but never a manifest or attempt advance.
    assert any("/variant-0001/" in uri for uri in storage.objects)
    assert not any(
        uri.endswith("cosmos_augmented/manifest.json") for uri in storage.objects
    )
    assert not paths["attempt"].exists()
    _assert_no_leaked_token(storage)


@requires_ffmpeg
def test_generation_failure_retains_sibling_without_manifest(tmp_path: Path) -> None:
    """A variant inference failure is typed, and its sibling is retained."""
    paths = _generation_inputs(tmp_path)
    healthy_video = _tiny_video(tmp_path / "generated.mp4", color="red")
    storage = _ProgressFailureStorage(progress_failures=None)

    def flaky_generator(**kwargs):
        if kwargs["name"] == "variant-0001":
            raise RuntimeError(f"generation exploded: {INJECTED_TOKEN}")
        return _healthy_generator(healthy_video)(**kwargs)

    with pytest.raises(c3.PaidfCosmos3Error, match="1 of 2 variants failed"):
        c3.generate_variants(**_generate_kwargs(paths, storage, flaky_generator))

    failure = _stored_progress(storage)["failures"][0]
    assert failure["clip"] == "variant-0001"
    assert failure["phase"] == "generation"
    assert failure["error_type"] == "RuntimeError"
    assert INJECTED_TOKEN not in json.dumps(_stored_progress(storage))
    assert any("/variant-0000/" in uri for uri in storage.objects)
    assert not any(
        uri.endswith("cosmos_augmented/manifest.json") for uri in storage.objects
    )
    _assert_no_leaked_token(storage)


@requires_ffmpeg
def test_publication_failure_is_recorded_in_publication_phase(tmp_path: Path) -> None:
    """A post-generation publish failure is attributed to the publication phase."""
    paths = _generation_inputs(tmp_path)
    healthy_video = _tiny_video(tmp_path / "generated.mp4", color="blue")
    storage = _ProgressFailureStorage(progress_failures=None)

    def empty_artifact_generator(**kwargs):
        if kwargs["name"] != "variant-0000":
            return _healthy_generator(healthy_video)(**kwargs)
        artifact = Path(kwargs["output_path"]) / kwargs["name"] / "vision.mp4"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"")
        return {"output_path": str(artifact), "output_bytes": 0}

    with pytest.raises(c3.PaidfCosmos3Error, match="1 of 2 variants failed"):
        c3.generate_variants(
            **_generate_kwargs(paths, storage, empty_artifact_generator)
        )

    failure = _stored_progress(storage)["failures"][0]
    assert failure["clip"] == "variant-0000"
    assert failure["phase"] == "publication"
    assert any("/variant-0001/" in uri for uri in storage.objects)
    assert not any(
        uri.endswith("cosmos_augmented/manifest.json") for uri in storage.objects
    )
    _assert_no_leaked_token(storage)
