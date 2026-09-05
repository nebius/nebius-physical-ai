"""Exact model cache and secure NLTK handoff for the EVG generation service."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.cosmos import transfer
from npa.workflows import paidf_guardrails as guardrails
from npa.workflows import paidf_evg_tokenizer as tokenizer


def _expected_files(snapshot):
    return {
        p.relative_to(snapshot).as_posix(): {
            "content_hash": p.resolve().name,
            "hash_algorithm": "sha256" if len(p.resolve().name) == 64 else "git-sha1",
            "size_bytes": p.stat().st_size,
        }
        for p in snapshot.rglob("*")
        if p.is_file()
    }


_QWEN_PROTOCOL_FIXTURE = """class Guard:
    def extract_label_and_categories(self, content):
        if isinstance(content, Exception):
            raise content
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        safe_label_match = re.search(safe_pattern, content)
        label = safe_label_match.group(1) if safe_label_match else None
        return label.lower() != "unsafe", label

    def is_safe(self, content):
        try:
            return self.extract_label_and_categories(content)
        except Exception as e:
            return True, "Unexpected error occurred when running Qwen3Guard guardrail."
"""


@pytest.mark.parametrize(
    "verdict,allowed", [("Safe", True), ("Unsafe", False), ("Controversial", True)]
)
def test_qwen_adaptation_preserves_published_verdict_policy(verdict, allowed):
    namespace = {"re": re}
    exec(
        guardrails._qwen_guardrail_patch_bytes(_QWEN_PROTOCOL_FIXTURE.encode()),
        namespace,
    )
    assert namespace["Guard"]().is_safe(f"Safety: {verdict}\nCategories: None") == (
        allowed,
        verdict,
    )


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Safety: Safely",
        "Safety: Unknown",
        "Safety: Safe\nSafety: Unsafe",
        "Safety: Safe\nSafety : Unsafe",
        RuntimeError("inference unavailable"),
    ],
)
def test_qwen_adaptation_rejects_missing_malformed_duplicate_or_failed_verdicts(output):
    namespace = {"re": re}
    exec(
        guardrails._qwen_guardrail_patch_bytes(_QWEN_PROTOCOL_FIXTURE.encode()),
        namespace,
    )
    with pytest.raises(RuntimeError, match="failed closed"):
        namespace["Guard"]().is_safe(output)


def test_qwen_adaptation_refuses_unreviewed_installed_source():
    with pytest.raises(guardrails.PaidfGuardrailError, match="reviewed image"):
        guardrails._patch_qwen_guardrail_source(_QWEN_PROTOCOL_FIXTURE.encode())


@pytest.mark.parametrize("setting", [False, None, "true", 1, {}])
def test_disabled_or_malformed_request_is_rejected_before_model_fetch(
    tmp_path, monkeypatch, setting
):
    from npa.workflows import paidf_native as native

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "endpoints": [
                    {
                        "role": "image2video",
                        "url": "http://127.0.0.1:8000/v1",
                        "model": guardrails.COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
                        "api_key_env": "GENERATION_API_KEY",
                    }
                ],
                "augmentation": {
                    "parameters": {"extra_params": {"guardrails": setting}}
                },
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "npa.paidf.native.evg-configs.v1",
                "workflow": "evg",
                "run_id": "test-run",
                "configs": [{"config_uri": str(config)}],
            }
        )
    )
    monkeypatch.setattr(
        guardrails,
        "prepare_evg_generation_environment",
        lambda: pytest.fail("disabled request fetched models"),
    )
    monkeypatch.setattr(
        native.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("disabled request started service"),
    )
    with pytest.raises(native.PaidfNativeError, match="explicit enabled guardrails"):
        native.run_local_augmentation(
            str(manifest),
            "unused",
            guardrails.COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
            guardrails.COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
            "image2video",
            8000,
            2,
            "test-run",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "missing",
        "revision",
        "online",
        "no_guardrails",
        "tokenizer",
        "linked_nltk",
        "empty",
        "hash",
    ],
)
def test_runtime_handoff_requires_exact_complete_enabled_contract(mutation):
    value = json.loads(
        (Path(__file__).parent / "fixtures/paidf-evg-runtime.json").read_text()
    )
    if mutation == "none":
        guardrails.require_evg_generation_runtime(value)
        return
    if mutation == "missing":
        value["models"].pop()
    elif mutation == "revision":
        value["models"][1]["revision"] = "c" * 40
    elif mutation == "online":
        value["offline"] = False
    elif mutation == "no_guardrails":
        value["guardrails_enabled"] = False
    elif mutation == "tokenizer":
        value["tokenizer_source_adaptation"]["tokenizer_type"] = "unreviewed"
        value["tokenizer_source_adaptation"]["patch_sha256"] = guardrails._digest_document(
            {k: v for k, v in value["tokenizer_source_adaptation"].items() if k != "patch_sha256"}
        )
    elif mutation == "linked_nltk":
        value["nltk_data"]["regular_files"] = False
    elif mutation == "empty":
        value["models"][0]["file_count"] = 0
    if mutation != "hash":
        value["contract_sha256"] = guardrails._digest_document(
            {k: v for k, v in value.items() if k != "contract_sha256"}
        )
    else:
        value["contract_sha256"] = "f" * 64
    with pytest.raises(guardrails.PaidfGuardrailError):
        guardrails.require_evg_generation_runtime(value)


def _snapshot(hub: Path, repository: str, revision: str, *, git_blob=False) -> Path:
    package = hub / ("models--" + repository.replace("/", "--"))
    snapshot = package / "snapshots" / revision
    blobs = package / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for relative, content in {
        "config.json": b'{"model_type":"test"}',
        "blocklist/nltk_data/tokenizers/punkt_tab/english/collocations.tab": b"test\tword\n",
    }.items():
        digest = (
            hashlib.sha1(
                f"blob {len(content)}\0".encode() + content, usedforsecurity=False
            )
            if git_blob
            else hashlib.sha256(content)
        ).hexdigest()
        blob = blobs / digest
        blob.write_bytes(content)
        entry = snapshot / relative
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.symlink_to(os.path.relpath(blob, entry.parent))
    return snapshot


@pytest.mark.parametrize("git_blob", [False, True])
def test_snapshot_verifies_hugging_face_content_hashes(tmp_path, git_blob):
    revision = "a" * 40
    snapshot = _snapshot(tmp_path, "example/model", revision, git_blob=git_blob)
    expected = _expected_files(snapshot)
    inventory = guardrails._snapshot_inventory(
        snapshot, tmp_path, "example/model", revision, expected
    )
    assert len(inventory) == 2
    assert (
        inventory["config.json"]["sha256"]
        == hashlib.sha256(b'{"model_type":"test"}').hexdigest()
    )
    (snapshot / "config.json").write_text("tampered")
    with pytest.raises(guardrails.PaidfGuardrailError, match="content hash"):
        guardrails._snapshot_inventory(
            snapshot, tmp_path, "example/model", revision, expected
        )


@pytest.mark.parametrize("escape", ["foreign_blob", "directory_link", "wrong_revision"])
def test_snapshot_refuses_cache_identity_or_confinement_changes(tmp_path, escape):
    revision = "a" * 40
    snapshot = _snapshot(tmp_path, "example/model", revision)
    expected = _expected_files(snapshot)
    if escape == "foreign_blob":
        foreign = _snapshot(tmp_path, "other/model", revision)
        entry = snapshot / "config.json"
        entry.unlink()
        entry.symlink_to((foreign / "config.json").resolve())
    elif escape == "directory_link":
        (snapshot / "linked").symlink_to(
            snapshot / "blocklist", target_is_directory=True
        )
    else:
        revision = "b" * 40
    with pytest.raises(guardrails.PaidfGuardrailError):
        guardrails._snapshot_inventory(
            snapshot, tmp_path, "example/model", revision, expected
        )


@pytest.mark.parametrize(
    "mutation", ["valid_blob_retarget", "missing", "extra", "size"]
)
def test_snapshot_requires_exact_pinned_path_hash_size_and_set(tmp_path, mutation):
    revision = "a" * 40
    snapshot = _snapshot(tmp_path, "example/model", revision)
    expected = _expected_files(snapshot)
    config = snapshot / "config.json"
    if mutation == "valid_blob_retarget":
        other = (
            snapshot
            / "blocklist/nltk_data/tokenizers/punkt_tab/english/collocations.tab"
        )
        config.unlink()
        config.symlink_to(other.resolve())
    elif mutation == "missing":
        config.unlink()
    elif mutation == "extra":
        (snapshot / "extra.json").symlink_to(config.resolve())
    else:
        expected["config.json"]["size_bytes"] += 1
    with pytest.raises(guardrails.PaidfGuardrailError, match="exact"):
        guardrails._snapshot_inventory(
            snapshot, tmp_path, "example/model", revision, expected
        )


@pytest.mark.parametrize(
    "mutation",
    ["none", "revision", "repository", "duplicate", "hash", "size", "escape"],
)
def test_official_revision_manifest_requires_precise_identity(
    tmp_path, monkeypatch, mutation
):
    repository, revision = "example/model", "a" * 40
    document = {
        "id": repository,
        "sha": revision,
        "siblings": [
            {
                "rfilename": "blocklist/data",
                "blobId": "b" * 40,
                "size": 3,
                "lfs": {"sha256": "c" * 64, "size": 3},
            },
            {"rfilename": "excluded", "blobId": "d" * 40, "size": 1},
        ],
    }
    if mutation == "revision":
        document["sha"] = "f" * 40
    elif mutation == "repository":
        document["id"] = "foreign/model"
    elif mutation == "duplicate":
        document["siblings"].append(document["siblings"][0])
    elif mutation == "hash":
        document["siblings"][0]["lfs"]["sha256"] = "bad"
    elif mutation == "size":
        document["siblings"][0]["lfs"]["size"] = 4
    elif mutation == "escape":
        document["siblings"][0]["rfilename"] = "../outside"

    def request(url):
        assert (
            url
            == f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
        )
        response = io.StringIO(json.dumps(document))
        response.geturl = lambda: url
        return response

    monkeypatch.setattr(guardrails.urllib.request, "urlopen", request)
    if mutation != "none":
        with pytest.raises(guardrails.PaidfGuardrailError):
            guardrails._load_snapshot_manifest(repository, revision, ("blocklist/**",))
    else:
        assert guardrails._load_snapshot_manifest(
            repository, revision, ("blocklist/**",)
        ) == {
            "blocklist/data": {
                "content_hash": "c" * 64,
                "hash_algorithm": "sha256",
                "size_bytes": 3,
            }
        }


def test_default_reference_is_exact_and_drift_fails(tmp_path):
    revision = "a" * 40
    guardrails._pin_cached_default(tmp_path, "example/model", revision)
    reference = tmp_path / "models--example--model/refs/main"
    assert reference.read_text() == revision
    guardrails._pin_cached_default(tmp_path, "example/model", revision)
    with pytest.raises(guardrails.PaidfGuardrailError, match="approved revision"):
        guardrails._pin_cached_default(tmp_path, "example/model", "b" * 40)
    reference.unlink()
    reference.symlink_to(tmp_path / "missing")
    with pytest.raises(guardrails.PaidfGuardrailError, match="symlink"):
        guardrails._pin_cached_default(tmp_path, "example/model", revision)


@pytest.mark.parametrize(
    "redirect", ["repository", "snapshot_subtree", "locks", "lock_file"]
)
def test_pre_download_cache_checks_refuse_existing_redirects(tmp_path, redirect):
    revision = "a" * 40
    outside = tmp_path / "outside"
    outside.mkdir()
    hub = tmp_path / "hub"
    hub.mkdir()
    if redirect == "repository":
        (hub / "models--example--model").symlink_to(outside, target_is_directory=True)
    elif redirect == "snapshot_subtree":
        snapshot = _snapshot(hub, "example/model", revision)
        (snapshot / "redirect").symlink_to(outside, target_is_directory=True)
    elif redirect == "locks":
        (hub / ".locks").symlink_to(outside, target_is_directory=True)
    else:
        lock_root = hub / ".locks/models--example--model"
        lock_root.mkdir(parents=True)
        (lock_root / ("f" * 64 + ".lock")).symlink_to(outside / "marker")
    with pytest.raises(guardrails.PaidfGuardrailError, match="redirect"):
        guardrails._prepare_model_cache(hub, "example/model", revision)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("access", ["missing_token", "denied"])
def test_access_failure_precedes_download_or_cache_creation(
    tmp_path, monkeypatch, access
):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_TOKEN", "operator-test-token" if access == "denied" else "")
    monkeypatch.setattr(
        guardrails, "validate_hf_access", lambda *_: SimpleNamespace(ok=False)
    )
    monkeypatch.setattr(
        guardrails.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("download before access"),
    )
    with pytest.raises(guardrails.PaidfGuardrailError):
        guardrails.prepare_evg_generation_environment()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("tamper_nltk", [False, True])
def test_runtime_stages_exact_cli_revisions_and_regular_nltk_offline(
    tmp_path, monkeypatch, tamper_nltk
):
    token = "operator-test-token"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.test")
    monkeypatch.setenv("HUGGINGFACE_CO_STAGING", "1")
    aliases = ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH")
    for name in aliases:
        monkeypatch.setenv(name, "synthetic-private-access")
    calls = []
    accesses = []

    def access(*args):
        accesses.append(args)
        return SimpleNamespace(ok=True)

    def download(argv, **kwargs):
        assert len(accesses) == 2
        assert kwargs["env"]["HF_TOKEN"] == token
        assert token not in argv
        calls.append(argv)
        hub = Path(argv[argv.index("--cache-dir") + 1])
        _snapshot(hub, argv[2], argv[argv.index("--revision") + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(guardrails, "validate_hf_access", access)
    monkeypatch.setattr(guardrails.subprocess, "run", download)
    reference = _snapshot(tmp_path / "reference", "example/model", "a" * 40)
    monkeypatch.setattr(
        guardrails, "_load_snapshot_manifest", lambda *_: _expected_files(reference)
    )
    monkeypatch.setattr(
        guardrails, "_prepare_qwen_guardrail_overlay", lambda *_: tmp_path / "overlay"
    )
    monkeypatch.setattr(
        tokenizer, "prepare_evg_tokenizer_overlay", lambda *_: tmp_path / "tokenizer-overlay"
    )
    if tamper_nltk:
        original_prepare = guardrails.prepare_guardrail_nltk_data

        def changed_local_cache(**kwargs):
            original_prepare(**kwargs)
            destination = transfer._guardrail_nltk_data_path(
                kwargs["hf_home"],
                repository=kwargs["repository"],
                revision=kwargs["revision"],
            )
            data = destination / "tokenizers/punkt_tab/english/collocations.tab"
            data.chmod(0o644)
            data.write_bytes(b"changed local bytes")
            marker = destination / transfer.GUARDRAIL_NLTK_READY_MARKER
            marker.chmod(0o644)
            transfer._write_guardrail_nltk_ready_marker(
                destination,
                repository=kwargs["repository"],
                revision=kwargs["revision"],
            )
            assert original_prepare(**kwargs) == 1  # Self-consistent local manifest.
            return 1

        monkeypatch.setattr(
            guardrails, "prepare_guardrail_nltk_data", changed_local_cache
        )
        with pytest.raises(
            guardrails.PaidfGuardrailError, match="exact pinned model snapshot"
        ):
            guardrails.prepare_evg_generation_environment()
        return
    environment, manifest = guardrails.prepare_evg_generation_environment()
    assert len(calls) == 3
    assert [call[2] for call in calls] == [
        model[0] for model in guardrails.EVG_RUNTIME_MODELS
    ]
    guardrail_cli = calls[1]
    assert guardrail_cli[guardrail_cli.index("--include") :] == [
        "--include",
        "blocklist/**",
        "--include",
        "face_blur_filter/Resnet50_Final.pth",
    ]
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
    assert not any(name in environment for name in ("HF_TOKEN", *aliases))
    assert os.environ["HF_TOKEN"] == token
    assert environment["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(tmp_path / "tokenizer-overlay"), str(tmp_path / "overlay")
    ]
    assert environment["HF_HOME"] != str(tmp_path)
    assert environment["HF_ENDPOINT"] == "https://huggingface.co"
    assert "HUGGINGFACE_CO_STAGING" not in environment
    assert (
        manifest["guardrail_source_adaptation"]
        == guardrails.qwen_guardrail_source_adaptation()
    )
    assert manifest["tokenizer_source_adaptation"] == tokenizer.tokenizer_source_adaptation()
    nltk = Path(environment["NLTK_DATA"])
    assert not any(path.is_symlink() for path in nltk.rglob("*"))
    assert (
        nltk / "tokenizers/punkt_tab/english/collocations.tab"
    ).read_bytes() == b"test\tword\n"
    assert manifest["guardrails_enabled"] is True
    assert manifest["offline"] is True
    assert manifest["nltk_data"]["regular_files"] is True
    assert manifest["nltk_data"]["file_count"] == 1
    assert token not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(manifest)
    for model in manifest["models"]:
        assert model["file_count"] == 2 and model["size_bytes"] > 0
        reference = (
            Path(environment["HF_HUB_CACHE"])
            / ("models--" + model["repository"].replace("/", "--"))
            / "refs/main"
        )
        assert reference.read_text() == model["revision"]
    contract = manifest.pop("contract_sha256")
    assert contract == guardrails._digest_document(manifest)


def test_parameterized_nltk_cache_binds_repository_revision_and_each_byte(tmp_path):
    repository = guardrails.COSMOS_GUARDRAIL_MODEL
    revision = guardrails.COSMOS_GUARDRAIL_REVISION
    snapshot = _snapshot(tmp_path / "hub", repository, revision)
    options = dict(
        hf_home=str(tmp_path),
        repository=repository,
        revision=revision,
        snapshot_path=snapshot,
    )
    assert transfer.prepare_guardrail_nltk_data(**options) == 1
    destination = transfer._guardrail_nltk_data_path(
        str(tmp_path), repository=repository, revision=revision
    )
    assert destination != transfer._guardrail_nltk_data_path(str(tmp_path))
    assert transfer.prepare_guardrail_nltk_data(**options) == 1
    payload = destination / "tokenizers/punkt_tab/english/collocations.tab"
    payload.chmod(0o644)
    payload.write_bytes(b"changed")
    with pytest.raises(transfer.GuardrailNLTKDataError, match="verified manifest"):
        transfer.prepare_guardrail_nltk_data(**options)


def test_parameterized_nltk_refuses_foreign_snapshot_before_copy(tmp_path):
    snapshot = _snapshot(tmp_path / "hub", guardrails.COSMOS_GUARDRAIL_MODEL, "b" * 40)
    with pytest.raises(
        transfer.GuardrailNLTKDataError, match="exact repository revision"
    ):
        transfer.prepare_guardrail_nltk_data(
            hf_home=str(tmp_path),
            repository=guardrails.COSMOS_GUARDRAIL_MODEL,
            revision=guardrails.COSMOS_GUARDRAIL_REVISION,
            snapshot_path=snapshot,
        )


@pytest.mark.parametrize("ancestor", ["root", "repository"])
def test_nltk_materialization_refuses_ancestor_redirect_before_writing(
    tmp_path, ancestor
):
    repository, revision = (
        guardrails.COSMOS_GUARDRAIL_MODEL,
        guardrails.COSMOS_GUARDRAIL_REVISION,
    )
    snapshot = _snapshot(tmp_path / "hub", repository, revision)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = transfer._guardrail_nltk_data_path(
        str(tmp_path), repository=repository, revision=revision
    )
    redirect = (
        destination.parent
        if ancestor == "repository"
        else tmp_path / transfer.GUARDRAIL_NLTK_MATERIALIZED_DIR
    )
    redirect.parent.mkdir(parents=True, exist_ok=True)
    redirect.symlink_to(outside, target_is_directory=True)
    with pytest.raises(transfer.GuardrailNLTKDataError, match="ancestor link"):
        transfer.prepare_guardrail_nltk_data(
            hf_home=str(tmp_path),
            repository=repository,
            revision=revision,
            snapshot_path=snapshot,
        )
    assert list(outside.iterdir()) == []
