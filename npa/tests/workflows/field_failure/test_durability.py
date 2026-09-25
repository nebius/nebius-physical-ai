"""Verify import safety, sealed execution identity, and permanent attempt ownership."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from npa.workflows.field_failure.adapters import _invoke
from npa.workflows.field_failure.artifacts import _digest
from npa.workflows.field_failure.contracts import _Adapter
from npa.workflows.field_failure.stages import run_stage


def test_bad_digest_never_executes_module_or_package_initializer(monkeypatch, tmp_path):
    package = tmp_path / "untrusted_adapter"
    package.mkdir()
    parent_marker, leaf_marker = tmp_path / "parent-ran", tmp_path / "leaf-ran"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(parent_marker)!r}).touch()\n"
    )
    (package / "worker.py").write_text(
        f"from pathlib import Path\nPath({str(leaf_marker)!r}).touch()\ndef execute(request): return {{}}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    identity = _Adapter(
        entrypoint="untrusted_adapter.worker:execute",
        source_sha256="0" * 64,
        runtime_image="registry.example.invalid/adapter@sha256:" + "a" * 64,
    )
    with pytest.raises(ValueError, match="source SHA-256"):
        _invoke(identity, {})
    assert not parent_marker.exists()
    assert not leaf_marker.exists()


def test_same_file_callable_substitution_is_rejected_before_execution(navigation):
    navigation.run("validate")
    with pytest.raises(ValueError, match="configured callable"):
        navigation.run("reconstruct", adapter="synthetic_navigation_fixture:substitute")
    assert not navigation.requests
    assert (
        navigation.root + "/claims/reconstruction.json" not in navigation.store.objects
    )


@pytest.mark.parametrize(
    "image",
    [
        "registry.example.invalid/navigation:latest",
        "registry.example.invalid/navigation@sha256:" + "b" * 64,
        "",
    ],
)
def test_runtime_identity_mismatch_is_rejected_before_execution(navigation, image):
    navigation.run("validate")
    with pytest.raises(ValueError, match="runtime image"):
        navigation.run("reconstruct", image=image)
    assert not navigation.requests


def test_mutable_image_cannot_enter_bundle(navigation):
    identity = navigation.bundle["adapters"]["reconstruct"].copy()
    identity["runtime_image"] = "registry.example.invalid/navigation:latest"
    with pytest.raises(ValueError):
        _Adapter.model_validate(identity)


def test_all_completed_stages_reuse_verified_outputs(evaluated):
    before = evaluated.store.objects.copy()
    calls = len(evaluated.requests)
    for stage in [
        "validate",
        "reconstruct",
        "train",
        "baseline-evaluate",
        "candidate-evaluate",
    ]:
        evaluated.run(stage)
    assert len(evaluated.requests) == calls
    assert evaluated.store.objects == before
    evaluated.run("compare")
    decision = evaluated.store.objects[evaluated.root + "/decision.json"]
    evaluated.run("compare")
    assert evaluated.store.objects[evaluated.root + "/decision.json"] == decision


@pytest.mark.parametrize(
    "stage,record,artifact",
    [
        ("reconstruct", "reconstruction", lambda r: r["scenes"][0]["asset"]),
        ("train", "training", lambda r: r["candidate"]["checkpoint"]),
        (
            "baseline-evaluate",
            "baseline-evaluation",
            lambda r: r["episodes"][0]["evidence"],
        ),
        (
            "candidate-evaluate",
            "candidate-evaluation",
            lambda r: r["episodes"][0]["evidence"],
        ),
    ],
)
def test_retry_refuses_corrupt_outputs_without_rerunning(
    evaluated, stage, record, artifact
):
    reference = artifact(evaluated.store.read(evaluated.root + f"/{record}.json"))
    evaluated.store.objects[reference["uri"]] = b"corrupted generation"
    calls = len(evaluated.requests)
    with pytest.raises(ValueError, match="SHA-256"):
        evaluated.run(stage)
    assert len(evaluated.requests) == calls


def test_failed_adapter_attempt_is_never_taken_over(navigation):
    navigation.run("validate")
    calls = []

    def fail(request):
        calls.append(request)
        navigation.store.asset(request["output_prefix"] + "partial.bin", b"partial")
        raise RuntimeError("interrupted adapter")

    navigation.runtime.reconstruct = fail
    with pytest.raises(RuntimeError, match="interrupted"):
        navigation.run("reconstruct")
    with pytest.raises(ValueError, match="active, incomplete, or conflicting"):
        navigation.run("reconstruct")
    assert len(calls) == 1
    assert navigation.root + "/reconstruction.json" not in navigation.store.objects


def test_concurrent_stage_invokes_only_claim_owner(navigation):
    navigation.run("validate")
    started, release = threading.Event(), threading.Event()
    original = navigation.runtime.reconstruct

    def blocked(request):
        started.set()
        assert release.wait(10)
        return original(request)

    navigation.runtime.reconstruct = blocked
    with ThreadPoolExecutor() as executor:
        first = executor.submit(navigation.run, "reconstruct")
        assert started.wait(10)
        try:
            second = executor.submit(navigation.run, "reconstruct")
            with pytest.raises(ValueError, match="active, incomplete, or conflicting"):
                second.result()
        finally:
            release.set()
        first.result()
    navigation.run("reconstruct")
    assert len(navigation.requests) == 1


def test_late_writer_cannot_publish_after_claim_changes(navigation):
    navigation.run("validate")
    original = navigation.runtime.reconstruct
    attempted = []

    def changed_claim(request):
        attempted.append(request["output_prefix"])
        result = original(request)
        claim_uri = navigation.root + "/claims/reconstruction.json"
        claim = navigation.store.read(claim_uri)
        claim["attempt_id"] = "b" * 32
        navigation.store.asset(claim_uri, claim)
        navigation.store.asset(
            navigation.root + "/reconstruction/attempt-" + "b" * 32 + "/sentinel",
            b"new attempt",
        )
        return result

    navigation.runtime.reconstruct = changed_claim
    with pytest.raises(ValueError, match="claim changed"):
        navigation.run("reconstruct")
    assert navigation.root + "/reconstruction.json" not in navigation.store.objects
    assert "attempt-" + "b" * 32 not in attempted[0]
    assert (
        navigation.store.objects[
            navigation.root + "/reconstruction/attempt-" + "b" * 32 + "/sentinel"
        ]
        == b"new attempt"
    )


def test_conflicting_input_cannot_reuse_completed_run(evaluated):
    bundle = evaluated.store.read(evaluated.bundle_uri)
    bundle["metrics"][0]["minimum_improvement"] = 0.4
    reference = evaluated.store.asset("s3://fixture/input/new-bundle.json", bundle)
    calls = len(evaluated.requests)
    identity = bundle["adapters"]["reconstruct"]
    with pytest.raises(ValueError, match="receipt differs"):
        run_stage(
            "reconstruct",
            reference["uri"],
            reference["sha256"],
            evaluated.root,
            "fixture-run",
            identity["entrypoint"],
            identity["runtime_image"],
        )
    assert len(evaluated.requests) == calls


def test_claim_input_tampering_blocks_completed_record_reuse(evaluated):
    uri = evaluated.root + "/claims/training.json"
    claim = evaluated.store.read(uri)
    claim["inputs_sha256"]["reconstruction"] = _digest(b"other scene")
    evaluated.store.asset(uri, claim)
    calls = len(evaluated.requests)
    with pytest.raises(ValueError, match="conflicts with permanent claim"):
        evaluated.run("train")
    assert len(evaluated.requests) == calls
