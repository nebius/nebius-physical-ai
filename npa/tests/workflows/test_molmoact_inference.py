"""Prove the private MolmoAct offline inference contract without payloads."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.resources
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import socket
import stat
import subprocess
import sys
import sysconfig
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from npa.workflows.byof import molmoact_inference as contract


NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)
IMAGE = "registry.example/research/molmoact@sha256:" + "a" * 64
ACTION_TOKENS = ("弎", "弐", "弌", "弍", "弑", "弒", "弓")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _write_private(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    path.chmod(0o600)


def _observation(**changes: object) -> contract.MountObservation:
    values = {
        "mount_id": 17,
        "st_dev": 2049,
        "major": 8,
        "minor": 1,
        "mount_options": ("ro", "nosuid"),
        "superblock_options": ("ro",),
        "statvfs_read_only": True,
    }
    values.update(changes)
    return contract.MountObservation(**values)


def _isolation_data(
    lock_hash: str,
    decision_id: str = "decision-1",
    observation: contract.MountObservation | None = None,
) -> dict[str, object]:
    observed = observation or _observation()
    return {
        "schema": "npa.runtime.egress-isolation-receipt.v2",
        "runtime_lock_sha256": lock_hash,
        "decision_id": decision_id,
        "declared_candidate_image": IMAGE,
        "scope": "scientific_research_and_development",
        "enforcement_mechanism": "external test assertion",
        "issuer_assertion": "external issuer assertion",
        "issued_at": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "process_egress_denied": True,
        "subprocess_egress_denied": True,
        "consumer_view_manifest_sha256": (
            "83508fd23ba77c29b9df1d48c56c18eb80aa7f47869434f2d3acc93443f0c3b9"
        ),
        "consumer_view_mount_id": observed.mount_id,
        "consumer_view_st_dev": observed.st_dev,
        "consumer_view_major": observed.major,
        "consumer_view_minor": observed.minor,
        "consumer_view_read_only": True,
        "consumer_view_remount_denied": True,
        "cap_sys_admin_absent": True,
        "exclusive_one_shot_process": True,
        "same_euid_concurrency_denied": True,
    }


def _decision_data(
    lock: contract.ValidatedMachineRecord,
    isolation_hash: str,
) -> dict[str, object]:
    inputs = lock.data["input_snapshot"]
    return {
        "schema": "npa.molmoact.use-scope-decision.v1",
        "decision_id": "decision-1",
        "status": "approved_for_bounded_research",
        "scope": "scientific_research_and_development",
        "operation": "one_shot_offline_inference",
        "maximum_generations": 1,
        "robot_actuation": False,
        "issued_at": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "runtime_lock_sha256": lock.sha256,
        "model": {
            "repository": "allenai/MolmoAct-7B-O-0812",
            "revision": "b614d9041b68b1e79bd1337a0308ad74bd5df8ae",
            "executable_code_revision": ("b614d9041b68b1e79bd1337a0308ad74bd5df8ae"),
        },
        "qwen": {
            "repository": "Qwen/Qwen2-7B",
            "revision": "453ed1575b739b5b03ce3758b23befdb0967f40e",
            "closure_sha256": (
                "beb6a0e21024812e339920bae9513d3ed6b6f4656327eec5b66b16d9e9f4732e"
            ),
        },
        "inputs": {
            "repository": inputs["repository"],
            "revision": inputs["revision"],
            "files": [
                {"path": item["path"], "sha256": item["identity"]}
                for item in inputs["files"]
            ],
        },
        "instruction": "close the box",
        "prompt_sha256": (
            "29a1cc6b4d29bf5c3cb6e9654d2ca5dac99ca91126325f3ae8f6a31966a346f8"
        ),
        "operation_sha256": (
            "3a016011b6dc3448c0ef41c4e1246502f6557415210ab160a45078a2148696c6"
        ),
        "terms": [
            {
                "id": term_id,
                "reference": f"https://example.invalid/{term_id}",
                "immutable_revision_or_sha256": "b" * 64,
                "reviewed_at": "2026-01-01T00:00:00Z",
                "disposition": "approved_for_this_bounded_operation",
            }
            for term_id in (
                "model_card_and_terms",
                "model_executable_remote_code",
                "input_card_and_images",
                "qwen_license_and_tokenizer",
                "apache_2_0",
                "ai2_responsible_use_guidelines",
            )
        ],
        "declared_candidate_image": IMAGE,
        "egress_isolation_receipt_sha256": isolation_hash,
        "authored_by_assertion": "external author assertion",
        "reviewed_by_assertion": "external reviewer assertion",
    }


def _valid_records() -> tuple[
    contract.ValidatedMachineRecord,
    bytes,
    bytes,
    contract.ValidatedMachineRecord,
]:
    lock = contract.load_runtime_lock()
    isolation_bytes = _canonical(_isolation_data(lock.sha256))
    decision_bytes = _canonical(
        _decision_data(lock, hashlib.sha256(isolation_bytes).hexdigest())
    )
    decision = contract.validate_decision(
        decision_bytes,
        lock=lock.data,
        lock_sha256=lock.sha256,
        now=NOW,
    )
    return lock, decision_bytes, isolation_bytes, decision


def _result_envelope(label: str = "baseline") -> dict[str, object]:
    return contract.build_result_envelope(
        {
            "schema": "npa.molmoact.inference-payload.v1",
            "test_vector": label,
            "external_publication_receipt_required": True,
        }
    )


def _independent_consumer_manifest(lock: dict[str, Any]) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for prefix, key in (
        ("molmoact_model", "model_snapshot"),
        ("inputs", "input_snapshot"),
        ("qwen_tokenizer", "qwen_snapshot"),
    ):
        for item in lock[key]["files"]:
            path = f"{prefix}/{item['path']}"
            files.append(
                {
                    "identity": item["identity"],
                    "identity_kind": item["identity_kind"],
                    "kind": "payload",
                    "path": path,
                    "role": item["role"],
                    "size": item["size"],
                    "source_path": path,
                }
            )
    files.extend(_independent_generated_cache_entries(lock))
    return {
        "directories": [
            ".",
            "hf_modules",
            "hf_modules/transformers_modules",
            "hf_modules/transformers_modules/molmoact_model",
            "inputs",
            "molmoact_model",
            "qwen_tokenizer",
        ],
        "files": sorted(files, key=lambda item: item["path"]),
        "schema": "npa.molmoact.consumer-view-manifest.v1",
    }


def _independent_generated_cache_entries(
    lock: dict[str, Any],
) -> list[dict[str, object]]:
    entries = [
        {
            "identity": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "identity_kind": "sha256",
            "kind": "generated_empty",
            "path": path,
            "role": "dynamic_cache_init",
            "size": 0,
            "source_path": "generated:empty",
        }
        for path in (
            "hf_modules/__init__.py",
            "hf_modules/transformers_modules/__init__.py",
            "hf_modules/transformers_modules/molmoact_model/__init__.py",
        )
    ]
    model_files = {item["path"]: item for item in lock["model_snapshot"]["files"]}
    for name in (
        "configuration_molmoact.py",
        "image_processing_molmoact.py",
        "modeling_molmoact.py",
        "processing_molmoact.py",
    ):
        source = model_files[name]
        entries.append(
            {
                "identity": source["identity"],
                "identity_kind": source["identity_kind"],
                "kind": "cache_copy",
                "path": f"hf_modules/transformers_modules/molmoact_model/{name}",
                "role": "dynamic_cache_copy",
                "size": source["size"],
                "source_path": f"molmoact_model/{name}",
            }
        )
    return entries


def _private_directory(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    return path


def _content_identity(payload: bytes, identity_kind: str) -> str:
    if identity_kind == "git_blob_sha1":
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(b"blob " + str(len(payload)).encode("ascii") + b"\x00")
    else:
        assert identity_kind in {"lfs_sha256", "sha256"}
        digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _synthetic_consumer_view(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], str, SimpleNamespace]:
    lock = deepcopy(contract.load_runtime_lock().data)
    layout = lock["consumer_view_layout"]
    root = _private_directory(tmp_path / "consumer-view")
    for relative in layout["directories"]:
        if relative != ".":
            (root / relative).mkdir(parents=True, exist_ok=True)

    payloads: dict[str, bytes] = {}
    for item in layout["files"]:
        if item["kind"] != "cache_copy":
            payloads[item["path"]] = (
                b""
                if item["kind"] == "generated_empty"
                else f"synthetic:{item['path']}\n".encode()
            )
    for item in layout["files"]:
        if item["kind"] == "cache_copy":
            payloads[item["path"]] = payloads[item["source_path"]]

    for item in layout["files"]:
        payload = payloads[item["path"]]
        item["size"] = len(payload)
        item["identity"] = _content_identity(payload, item["identity_kind"])
        destination = root / item["path"]
        destination.write_bytes(payload)
        destination.chmod(0o444)
    for relative in sorted(
        layout["directories"], key=lambda value: value.count("/"), reverse=True
    ):
        (root if relative == "." else root / relative).chmod(0o555)

    details = root.stat()
    major = os.major(details.st_dev)
    minor = os.minor(details.st_dev)
    mountinfo = f"42 1 {major}:{minor} / {root} ro,nosuid - tmpfs tmpfs ro\n"
    return root, lock, mountinfo, SimpleNamespace(f_flag=os.ST_RDONLY)


def _generated_text(
    *, quoted: bool = False, tokens: tuple[str, ...] = ACTION_TOKENS
) -> str:
    depth = "<DEPTH_START>" + "<DEPTH_1>" * 100 + "<DEPTH_END>"
    rendered = [f"'{token}'" if quoted else token for token in tokens]
    return f"{depth} [[1,2],[3,4]] [{','.join(rendered)}]"


class _Tokenizer:
    vocab_size = 1000
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return 999 - ACTION_TOKENS.index(token)


class _ParserModel:
    bin_centers = list(range(7))

    def parse_depth(self, text: str) -> list[str]:
        start = text.index("<DEPTH_START>")
        end = text.index("<DEPTH_END>") + len("<DEPTH_END>")
        return [text[start:end]]

    def parse_trace(self, text: str) -> list[list[list[int]]]:
        del text
        return [[[1, 2], [3, 4]]]

    def get_action_dim(self, key: str) -> int:
        assert key == "molmoact"
        return 7

    def parse_action(self, text: str, *, unnorm_key: str) -> list[list[float]]:
        assert text
        assert unnorm_key == "molmoact"
        return [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_module_import_is_stdlib_clean_in_fresh_interpreter() -> None:
    source = Path(contract.__file__).parents[3]
    script = (
        "import json,sys;"
        f"sys.path.insert(0,{str(source)!r});"
        "import npa.workflows.byof.molmoact_inference;"
        "p=('PIL','numpy','torch','transformers','huggingface_hub','kernels');"
        "print(json.dumps(sorted(n for n in sys.modules "
        "if any(n==x or n.startswith(x+'.') for x in p))))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_runtime_lock_exact_vectors_and_packaged_resource() -> None:
    lock = contract.load_runtime_lock()
    operation = lock.data["operation"]
    operation_projection = {
        "input_files": [
            {"path": item["path"], "sha256": item["identity"]}
            for item in lock.data["input_snapshot"]["files"]
        ],
        "input_repository": lock.data["input_snapshot"]["repository"],
        "input_revision": lock.data["input_snapshot"]["revision"],
        "instruction": operation["instruction"],
        "model_repository": lock.data["model_snapshot"]["repository"],
        "model_revision": lock.data["model_snapshot"]["revision"],
        "operation": "one_shot_offline_inference",
        "prompt_sha256": operation["prompt_sha256"],
        "qwen_repository": lock.data["qwen_snapshot"]["repository"],
        "qwen_revision": lock.data["qwen_snapshot"]["revision"],
    }
    assert (
        _canonical(operation_projection).decode()
        == operation["operation_canonical_json"]
    )
    assert (
        hashlib.sha256(_canonical(operation_projection)).hexdigest()
        == operation["operation_sha256"]
    )
    qwen_projection = {
        "files": [
            {
                "identity": item["identity"],
                "identity_kind": item["identity_kind"],
                "path": item["path"],
                "role": item["role"],
                "size": item["size"],
            }
            for item in lock.data["qwen_snapshot"]["files"]
        ],
        "repository": lock.data["qwen_snapshot"]["repository"],
        "revision": lock.data["qwen_snapshot"]["revision"],
    }
    assert (
        _canonical(qwen_projection).decode() == operation["qwen_closure_canonical_json"]
    )
    assert (
        hashlib.sha256(_canonical(qwen_projection)).hexdigest()
        == operation["qwen_closure_sha256"]
    )
    manifest = _independent_consumer_manifest(lock.data)
    assert manifest["files"] == lock.data["consumer_view_layout"]["files"]
    projection = _canonical(manifest)
    assert len(projection) == 9585
    assert (
        hashlib.sha256(
            b"npa.molmoact.consumer-view-manifest.v1\x00" + projection
        ).hexdigest()
        == lock.data["consumer_view_layout"]["manifest_sha256"]
    )
    resource = importlib.resources.files("npa.workflows.byof").joinpath(
        "molmoact_runtime_lock.json"
    )
    assert resource.read_bytes() == contract.RUNTIME_LOCK_PATH.read_bytes()


def test_all_frozen_snapshot_vectors_have_exact_counts() -> None:
    lock = contract.load_runtime_lock().data
    assert len(lock["model_snapshot"]["files"]) == 24
    assert len(lock["qwen_snapshot"]["files"]) == 4
    assert len(lock["input_snapshot"]["files"]) == 2
    assert len(lock["consumer_view_layout"]["files"]) == 37
    assert [item["path"] for item in lock["consumer_view_layout"]["files"]] == sorted(
        item["path"] for item in lock["consumer_view_layout"]["files"]
    )


def test_runtime_lock_uses_the_strict_reader_without_terminal_bytes(
    tmp_path: Path,
) -> None:
    raw = contract.RUNTIME_LOCK_PATH.read_bytes()
    assert raw.endswith(b"}")
    assert not raw.endswith((b"\n", b"\r", b"\t", b" "))
    hostile = tmp_path / "runtime-lock-with-newline.json"
    _write_private(hostile, raw + b"\n")
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.load_runtime_lock(hostile)
    assert caught.value.code == "MOLMOACT_LOCK_INVALID"


def test_compact_manifest_and_payload_digest_vectors() -> None:
    compact = {
        "directories": ["."],
        "files": [
            {
                "identity": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
                "identity_kind": "sha256",
                "kind": "generated_empty",
                "path": "__init__.py",
                "role": "dynamic_cache_init",
                "size": 0,
                "source_path": "generated:empty",
            }
        ],
        "schema": "npa.molmoact.consumer-view-manifest.v1",
    }
    assert (
        hashlib.sha256(
            b"npa.molmoact.consumer-view-manifest.v1\x00" + _canonical(compact)
        ).hexdigest()
        == "17658753374c950bb2159dadd6b169a8876f41184565ba680de57c44c94462d8"
    )
    payload = {
        "schema": "npa.molmoact.inference-payload.v1",
        "test_vector": "payload-digest-domain",
    }
    envelope = contract.build_result_envelope(payload)
    assert envelope["payload_digest"]["value"] == (
        "6945b59b3088bfff24666d141a750ccbcae10650b0ed44f0c88d7cd4ff620231"
    )
    with pytest.raises(ValueError):
        contract.build_result_envelope(
            {
                "schema": "npa.molmoact.inference-payload.v1",
                "nonfinite": float("nan"),
            }
        )


def test_gate_a_serializer_and_record_vectors() -> None:
    payload = {
        "cpu_only_assertion": True,
        "created_at": "2026-01-01T00:00:00Z",
        "distributions": [],
        "interpreter": {
            "abi": "cpython-312-x86_64-linux-gnu",
            "implementation": "CPython",
            "path": "/opt/npa-gate-a/bin/python",
            "python_version": "3.12.0",
            "sha256": "0" * 64,
        },
        "network_disabled_assertion": True,
    }
    assert len(_canonical(payload)) == 334
    assert (
        hashlib.sha256(
            b"npa.molmoact.gate-a-environment-receipt.v1\x00" + _canonical(payload)
        ).hexdigest()
        == "1cdc8b441276e9086e21acc86eda16eac9d27cb4c1f3600deac2f87f7da47b0a"
    )
    assert (
        hashlib.sha256(b"npa.molmoact.gate-a-record-rows.v1\x00[]").hexdigest()
        == "6debb7c8e28f5c4dffd7b6c4ece7ec11a6007677ba6f31e1c45d0257274351d5"
    )


def test_gate_a_accepts_the_running_venv_projection() -> None:
    projection = contract.gate_a_distribution_projection()
    assert projection
    assert projection == sorted(projection, key=lambda item: item["normalized_name"])

    executable = sys.executable
    value = {
        "abi": str(sysconfig.get_config_var("SOABI") or ""),
        "implementation": platform.python_implementation(),
        "path": executable,
        "python_version": platform.python_version(),
        "sha256": hashlib.sha256(Path(executable).resolve().read_bytes()).hexdigest(),
    }
    contract._validate_gate_a_interpreter(value, executable)


def test_canonical_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        contract.canonical_json_bytes({"value": float("nan")})


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"nested":{"a":1,"a":2}}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1} trailing',
        b" {}\n",
        b"{}\n",
        b"{}\t",
        b"\xef\xbb\xbf{}",
        b"[]",
        b"\xff",
    ],
)
def test_strict_json_rejects_hostile_machine_records(raw: bytes) -> None:
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.read_strict_json(raw, error_code="MOLMOACT_LOCK_INVALID")
    assert caught.value.code == "MOLMOACT_LOCK_INVALID"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value["model_snapshot"]["files"][0].update(size=True),
        lambda value: value["qwen_snapshot"]["files"][0].update(role="retained notice"),
        lambda value: value["operation"].update(expected_action_dim=True),
        lambda value: value["operation"].update(prompt=1),
        lambda value: value["consumer_view_layout"]["files"].reverse(),
        lambda value: value["consumer_view_layout"].update(manifest_sha256="0" * 64),
    ],
)
def test_runtime_lock_mutations_fail_exactly(mutate) -> None:
    value = deepcopy(contract.load_runtime_lock().data)
    mutate(value)
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_runtime_lock(value)
    assert caught.value.code == "MOLMOACT_LOCK_INVALID"


def test_git_blob_framing_is_not_raw_sha1() -> None:
    payload = b"synthetic snapshot bytes\n"
    expected = hashlib.sha1(
        b"blob " + str(len(payload)).encode() + b"\x00" + payload,
        usedforsecurity=False,
    ).hexdigest()
    assert contract.git_blob_sha1(payload) == expected
    assert (
        contract.git_blob_sha1(payload)
        != hashlib.sha1(payload, usedforsecurity=False).hexdigest()
    )


def _synthetic_snapshot(
    root: Path,
    name: str,
    payload: bytes,
    *,
    identity_kind: str = "git_blob_sha1",
) -> dict[str, object]:
    root.mkdir()
    file_path = root / name
    file_path.write_bytes(payload)
    file_path.chmod(0o444)
    root.chmod(0o555)
    identity = contract.git_blob_sha1(payload)
    if identity_kind == "sha256":
        identity = hashlib.sha256(payload).hexdigest()
    return {
        "repository": "example/synthetic",
        "revision": "1" * 40,
        "files": [
            {
                "path": name,
                "role": "synthetic",
                "size": len(payload),
                "identity_kind": identity_kind,
                "identity": identity,
            }
        ],
    }


def _verified_input(root: Path, payload: bytes) -> contract.VerifiedSnapshot:
    snapshot = _synthetic_snapshot(
        root,
        "frame.bin",
        payload,
        identity_kind="sha256",
    )
    return contract.verify_snapshot(root, snapshot, category="input")


def _rewrite_retained_input(root: Path, payload: bytes) -> None:
    source = root / "frame.bin"
    source.chmod(0o600)
    source.write_bytes(payload)
    source.chmod(0o444)


def test_retained_input_reads_the_verified_bytes(tmp_path: Path) -> None:
    root = tmp_path / "input"
    verified = _verified_input(root, b"frame-a")
    try:
        assert contract._read_verified_inputs(verified) == [b"frame-a"]
    finally:
        verified.close()
        root.chmod(0o755)


def test_retained_input_rejects_changed_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "input"
    verified = _verified_input(root, b"frame-a")
    expected = verified.records[0]["identity"]
    _rewrite_retained_input(root, b"frame-longer")
    monkeypatch.setattr(
        contract.hashlib,
        "sha256",
        lambda *_: SimpleNamespace(hexdigest=lambda: expected),
    )
    try:
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract._read_verified_inputs(verified)
        assert caught.value.code == "MOLMOACT_INPUT_IDENTITY"
        assert caught.value.detail == "input changed during retained read"
    finally:
        verified.close()
        root.chmod(0o755)


def test_retained_input_rejects_same_size_changed_hash(tmp_path: Path) -> None:
    root = tmp_path / "input"
    verified = _verified_input(root, b"frame-a")
    _rewrite_retained_input(root, b"frame-b")
    try:
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract._read_verified_inputs(verified)
        assert caught.value.code == "MOLMOACT_INPUT_IDENTITY"
        assert caught.value.detail == "input changed during retained read"
    finally:
        verified.close()
        root.chmod(0o755)


def test_snapshot_exact_set_and_git_identity(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    snapshot = _synthetic_snapshot(root, "config.json", b"{}")
    try:
        verified = contract.verify_snapshot(root, snapshot, category="model")
        assert (
            verified.records[0]["verified_identity"] == snapshot["files"][0]["identity"]
        )
        verified.close()
        root.chmod(0o755)
        extra = root / "extra"
        extra.write_bytes(b"x")
        extra.chmod(0o444)
        root.chmod(0o555)
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.verify_snapshot(root, snapshot, category="model")
        assert caught.value.code == "MOLMOACT_SNAPSHOT_SHAPE"
    finally:
        root.chmod(0o755)


def test_snapshot_rejects_wrong_size_and_raw_sha1(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    snapshot = _synthetic_snapshot(root, "config.json", b"{}")
    try:
        wrong_size = deepcopy(snapshot)
        wrong_size["files"][0]["size"] = 3
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.verify_snapshot(root, wrong_size, category="model")
        assert caught.value.code == "MOLMOACT_SNAPSHOT_SIZE"
        wrong_hash = deepcopy(snapshot)
        wrong_hash["files"][0]["identity"] = hashlib.sha1(
            b"{}", usedforsecurity=False
        ).hexdigest()
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.verify_snapshot(root, wrong_hash, category="model")
        assert caught.value.code == "MOLMOACT_GIT_BLOB_IDENTITY"
    finally:
        root.chmod(0o755)


def test_qwen_categorical_weight_rejection_precedes_exact_set(tmp_path: Path) -> None:
    root = tmp_path / "qwen"
    snapshot = _synthetic_snapshot(root, "vocab.json", b"{}")
    root.chmod(0o755)
    weight = root / "unexpected.safetensors"
    weight.write_bytes(b"not a weight")
    weight.chmod(0o444)
    root.chmod(0o555)
    try:
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.verify_snapshot(root, snapshot, category="qwen")
        assert caught.value.code == "MOLMOACT_QWEN_WEIGHT_PRESENT"
    finally:
        root.chmod(0o755)


def test_snapshot_rejects_symlinked_root_and_file(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    snapshot = _synthetic_snapshot(root, "config.json", b"{}")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    try:
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.verify_snapshot(alias, snapshot, category="model")
        assert caught.value.code == "MOLMOACT_SNAPSHOT_SHAPE"
        root.chmod(0o755)
        (root / "config.json").unlink()
        (root / "target.json").write_bytes(b"{}")
        (root / "target.json").chmod(0o444)
        (root / "config.json").symlink_to("target.json")
        root.chmod(0o555)
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.verify_snapshot(root, snapshot, category="model")
        assert caught.value.code == "MOLMOACT_SNAPSHOT_SHAPE"
    finally:
        root.chmod(0o755)


def test_isolated_lock_copy_mutation_does_not_touch_repository(tmp_path: Path) -> None:
    source_module = Path(contract.__file__)
    source_lock = contract.RUNTIME_LOCK_PATH
    before = {
        source_module: hashlib.sha256(source_module.read_bytes()).hexdigest(),
        source_lock: hashlib.sha256(source_lock.read_bytes()).hexdigest(),
    }
    copied = tmp_path / "copy"
    copied.mkdir()
    shutil.copyfile(source_module, copied / source_module.name)
    value = json.loads(source_lock.read_text())
    value["operation"]["expected_action_dim"] = 8
    (copied / source_lock.name).write_text(json.dumps(value))
    script = (
        "import importlib.util,sys;"
        f"p={str(copied / source_module.name)!r};"
        "s=importlib.util.spec_from_file_location('mutant',p);"
        "m=importlib.util.module_from_spec(s);sys.modules['mutant']=m;"
        "s.loader.exec_module(m);m.load_runtime_lock()"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode != 0
    assert "MOLMOACT_LOCK_INVALID" in result.stderr
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before
    } == before


def test_decision_and_isolation_records_bind_exactly() -> None:
    lock, _, isolation_bytes, decision = _valid_records()
    assert contract.validate_image_declaration(IMAGE, decision.data) == IMAGE
    receipt = contract.validate_isolation_receipt(
        isolation_bytes,
        decision=decision,
        lock_sha256=lock.sha256,
        observation=_observation(),
        now=NOW,
    )
    assert receipt.sha256 == decision.data["egress_isolation_receipt_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("maximum_generations", True, "MOLMOACT_DECISION_INVALID"),
        ("status", "approved", "MOLMOACT_DECISION_INVALID"),
        ("runtime_lock_sha256", "0" * 64, "MOLMOACT_DECISION_BINDING"),
        ("operation_sha256", "0" * 64, "MOLMOACT_DECISION_BINDING"),
    ],
)
def test_decision_mutations_have_stable_codes(
    field: str, value: object, code: str
) -> None:
    lock, decision_bytes, _, _ = _valid_records()
    decision = json.loads(decision_bytes)
    decision[field] = value
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_decision(
            _canonical(decision),
            lock=lock.data,
            lock_sha256=lock.sha256,
            now=NOW,
        )
    assert caught.value.code == code


def test_decision_rejects_mutable_terms_reference() -> None:
    lock, decision_bytes, _, _ = _valid_records()
    decision = json.loads(decision_bytes)
    decision["terms"][0]["immutable_revision_or_sha256"] = "main"
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_decision(
            _canonical(decision),
            lock=lock.data,
            lock_sha256=lock.sha256,
            now=NOW,
        )
    assert caught.value.code == "MOLMOACT_DECISION_INVALID"


@pytest.mark.parametrize(
    "reference",
    [
        "mutable-reference",
        " https://example.invalid/terms",
        "http://example.invalid/terms",
        "https://user@example.invalid/terms",
        "https://example.invalid/bad path",
        "repository\\path",
        "https://example.invalid/\x00terms",
    ],
)
def test_decision_rejects_malformed_term_reference(reference: str) -> None:
    lock, decision_bytes, _, _ = _valid_records()
    decision = json.loads(decision_bytes)
    decision["terms"][0]["reference"] = reference
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_decision(
            _canonical(decision),
            lock=lock.data,
            lock_sha256=lock.sha256,
            now=NOW,
        )
    assert caught.value.code == "MOLMOACT_DECISION_INVALID"


def test_expired_decision_and_image_mismatch_are_distinct() -> None:
    lock, decision_bytes, _, decision = _valid_records()
    value = json.loads(decision_bytes)
    value["expires_at"] = "2026-01-02T00:00:00Z"
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_decision(
            _canonical(value),
            lock=lock.data,
            lock_sha256=lock.sha256,
            now=NOW,
        )
    assert caught.value.code == "MOLMOACT_DECISION_EXPIRED"
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_image_declaration(
            "registry.example/research/other@sha256:" + "b" * 64,
            decision.data,
        )
    assert caught.value.code == "MOLMOACT_IMAGE_DECLARATION"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("process_egress_denied", False),
        ("subprocess_egress_denied", False),
        ("consumer_view_read_only", 1),
        ("consumer_view_remount_denied", False),
        ("cap_sys_admin_absent", False),
        ("exclusive_one_shot_process", False),
        ("same_euid_concurrency_denied", False),
        ("consumer_view_mount_id", True),
        ("consumer_view_st_dev", 124),
        ("consumer_view_major", 9),
        ("consumer_view_minor", 9),
        ("consumer_view_manifest_sha256", "0" * 64),
        ("runtime_lock_sha256", "0" * 64),
        ("decision_id", "other-decision"),
        (
            "declared_candidate_image",
            "registry.example/research/other@sha256:" + "b" * 64,
        ),
        ("scope", "other-scope"),
        ("schema", "npa.runtime.egress-isolation-receipt.v1"),
        ("enforcement_mechanism", ""),
        ("issuer_assertion", ""),
        ("expires_at", "2026-01-02T00:00:00Z"),
    ],
)
def test_isolation_receipt_controls_fail_closed(field: str, value: object) -> None:
    lock, decision_bytes, isolation_bytes, _ = _valid_records()
    receipt = json.loads(isolation_bytes)
    receipt[field] = value
    altered = _canonical(receipt)
    decision_value = json.loads(decision_bytes)
    decision_value["egress_isolation_receipt_sha256"] = hashlib.sha256(
        altered
    ).hexdigest()
    decision = contract.validate_decision(
        _canonical(decision_value),
        lock=lock.data,
        lock_sha256=lock.sha256,
        now=NOW,
    )
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_isolation_receipt(
            altered,
            decision=decision,
            lock_sha256=lock.sha256,
            observation=_observation(),
            now=NOW,
        )
    assert caught.value.code == "MOLMOACT_EGRESS_EVIDENCE"


def test_gate_a_absence_is_a_failure_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NPA_MOLMOACT_CONTRACT_PYTHON", raising=False)
    monkeypatch.delenv("NPA_MOLMOACT_GATE_A_RECEIPT", raising=False)
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_configured_gate_a_environment()
    assert caught.value.code == "MOLMOACT_OFFLINE_STATE"
    missing = tmp_path / "missing-receipt.json"
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_gate_a_environment_receipt(missing)
    assert caught.value.code == "MOLMOACT_OFFLINE_STATE"


def test_mount_receipt_logic_rejects_local_writable_observation() -> None:
    lock, decision_bytes, isolation_bytes, _ = _valid_records()
    decision = contract.validate_decision(
        decision_bytes,
        lock=lock.data,
        lock_sha256=lock.sha256,
        now=NOW,
    )
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_isolation_receipt(
            isolation_bytes,
            decision=decision,
            lock_sha256=lock.sha256,
            observation=_observation(statvfs_read_only=False),
            now=NOW,
        )
    assert caught.value.code == "MOLMOACT_EGRESS_EVIDENCE"


def test_mountinfo_observation_binds_mount_and_device(tmp_path: Path) -> None:
    root = _private_directory(tmp_path / "view")
    root.chmod(0o555)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        details = os.fstat(descriptor)
        major = os.major(details.st_dev)
        minor = os.minor(details.st_dev)
        mountinfo = f"42 1 {major}:{minor} / {root} ro,nosuid - ext4 /dev/test ro\n"
        observation = contract.observe_consumer_mount(
            str(root),
            descriptor,
            mountinfo_text=mountinfo,
            statvfs_result=SimpleNamespace(f_flag=os.ST_RDONLY),
        )
        assert observation.mount_id == 42
        assert (observation.major, observation.minor) == (major, minor)
        assert observation.statvfs_read_only is True
    finally:
        os.close(descriptor)
        root.chmod(0o755)


def test_mountinfo_observation_rejects_a_nested_mount(tmp_path: Path) -> None:
    root = _private_directory(tmp_path / "view")
    root.chmod(0o555)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        details = os.fstat(descriptor)
        major = os.major(details.st_dev)
        minor = os.minor(details.st_dev)
        mountinfo = (
            f"42 1 {major}:{minor} / {root} ro,nosuid - ext4 /dev/test ro\n"
            f"43 42 {major}:{minor} / {root}/molmoact_model rw - tmpfs tmpfs rw\n"
        )
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.observe_consumer_mount(
                str(root),
                descriptor,
                mountinfo_text=mountinfo,
                statvfs_result=SimpleNamespace(f_flag=os.ST_RDONLY),
            )
        assert caught.value.code == "MOLMOACT_EGRESS_EVIDENCE"
    finally:
        os.close(descriptor)
        root.chmod(0o755)


@pytest.mark.parametrize(
    ("mount_options", "superblock_options", "major_delta", "read_only"),
    [
        ("rw,nosuid", "ro", 0, True),
        ("ro,nosuid", "rw", 0, True),
        ("ro,nosuid", "ro", 1, True),
        ("ro,nosuid", "ro", 0, False),
    ],
)
def test_mountinfo_observation_rejects_each_writable_or_device_mismatch(
    tmp_path: Path,
    mount_options: str,
    superblock_options: str,
    major_delta: int,
    read_only: bool,
) -> None:
    root = _private_directory(tmp_path / "view")
    root.chmod(0o555)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        details = os.fstat(descriptor)
        major = os.major(details.st_dev)
        minor = os.minor(details.st_dev)
        mountinfo = (
            f"42 1 {major + major_delta}:{minor} / {root} {mount_options} "
            f"- ext4 /dev/test {superblock_options}\n"
        )
        flags = os.ST_RDONLY if read_only else 0
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            contract.observe_consumer_mount(
                str(root),
                descriptor,
                mountinfo_text=mountinfo,
                statvfs_result=SimpleNamespace(f_flag=flags),
            )
        assert caught.value.code == "MOLMOACT_EGRESS_EVIDENCE"
    finally:
        os.close(descriptor)
        root.chmod(0o755)


def test_consumer_view_verifies_exact_37_entries_and_cache_copies(
    tmp_path: Path,
) -> None:
    root, lock, mountinfo, statvfs_result = _synthetic_consumer_view(tmp_path)
    observation = contract.verify_consumer_view(
        root,
        lock,
        mountinfo_text=mountinfo,
        statvfs_result=statvfs_result,
    )
    assert observation.mount_id == 42
    assert len(lock["consumer_view_layout"]["files"]) == 37
    assert (
        sum(
            1
            for item in lock["consumer_view_layout"]["files"]
            if item["kind"] == "cache_copy"
        )
        == 4
    )


@pytest.mark.parametrize("fault", ["extra", "bytes"])
def test_consumer_view_rejects_extra_and_byte_faults(
    tmp_path: Path, fault: str
) -> None:
    root, lock, mountinfo, statvfs_result = _synthetic_consumer_view(tmp_path)
    if fault == "extra":
        root.chmod(0o755)
        extra = root / "unexpected"
        extra.write_bytes(b"x")
        extra.chmod(0o444)
        root.chmod(0o555)
    else:
        item = next(
            item
            for item in lock["consumer_view_layout"]["files"]
            if item["kind"] == "payload" and item["size"] > 0
        )
        target = root / item["path"]
        payload = target.read_bytes()
        target.chmod(0o644)
        target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        target.chmod(0o444)

    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.verify_consumer_view(
            root,
            lock,
            mountinfo_text=mountinfo,
            statvfs_result=statvfs_result,
        )
    assert caught.value.code in {
        "MOLMOACT_INPUT_IDENTITY",
        "MOLMOACT_SNAPSHOT_SHAPE",
        "MOLMOACT_GIT_BLOB_IDENTITY",
        "MOLMOACT_LFS_IDENTITY",
    }


def test_consumer_view_rejects_individually_valid_but_different_cache_copy(
    tmp_path: Path,
) -> None:
    root, lock, mountinfo, statvfs_result = _synthetic_consumer_view(tmp_path)
    item = next(
        item
        for item in lock["consumer_view_layout"]["files"]
        if item["kind"] == "cache_copy"
    )
    target = root / item["path"]
    payload = target.read_bytes()
    changed = bytes([payload[0] ^ 1]) + payload[1:]
    target.chmod(0o644)
    target.write_bytes(changed)
    target.chmod(0o444)
    item["identity"] = _content_identity(changed, item["identity_kind"])

    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.verify_consumer_view(
            root,
            lock,
            mountinfo_text=mountinfo,
            statvfs_result=statvfs_result,
        )
    assert caught.value.code == "MOLMOACT_SNAPSHOT_SHAPE"


def test_generated_only_grammar_and_tokenizer_injection() -> None:
    model = _ParserModel()
    tokenizer = _Tokenizer()
    result = contract.validate_generated_text(
        _generated_text(), model=model, tokenizer=tokenizer
    )
    assert model._qwen_tokenizer is tokenizer
    assert result["depth_token_values"] == [1] * 100
    assert result["trace"] == [[1, 2], [3, 4]]
    assert result["action_token_ids"] == [999, 998, 997, 996, 995, 994, 993]
    assert result["action_values"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_generated_action_tokens_accept_ascii_when_identity_known() -> None:
    tokens = ("act_a", "act_b", "act_c", "act_d", "act_e", "act_f", "act_g")
    tokenizer = SimpleNamespace(
        vocab_size=1000,
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: 999 - tokens.index(token),
    )
    result = contract.validate_generated_text(
        _generated_text(tokens=tokens),
        model=_ParserModel(),
        tokenizer=tokenizer,
    )
    assert result["action_tokens"] == list(tokens)
    assert result["action_token_ids"] == [999, 998, 997, 996, 995, 994, 993]


@pytest.mark.parametrize("coerced", [True, 1.0])
def test_upstream_trace_rejects_type_coercion(coerced: object) -> None:
    class CoerciveTraceModel(_ParserModel):
        def parse_trace(self, text: str) -> list[list[list[object]]]:
            del text
            return [[[coerced, 2], [3, 4]]]

    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_generated_text(
            _generated_text(),
            model=CoerciveTraceModel(),
            tokenizer=_Tokenizer(),
        )
    assert caught.value.code == "MOLMOACT_TRACE_GRAMMAR"


def test_generated_action_tokens_accept_one_matching_quote_layer() -> None:
    result = contract.validate_generated_text(
        _generated_text(quoted=True), model=_ParserModel(), tokenizer=_Tokenizer()
    )
    assert result["action_tokens"] == list(ACTION_TOKENS)
    assert result["action_token_ids"] == [999, 998, 997, 996, 995, 994, 993]


def test_generated_trace_accepts_strict_json_whitespace_after_outer_bracket() -> None:
    text = _generated_text().replace("[[1,2]", "[ [1,2]")
    result = contract.validate_generated_text(
        text, model=_ParserModel(), tokenizer=_Tokenizer()
    )
    assert result["trace"] == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    ("replace", "code"),
    [
        (
            lambda text: text.replace("<DEPTH_1>", "<DEPTH_129>", 1),
            "MOLMOACT_DEPTH_GRAMMAR",
        ),
        (
            lambda text: (
                text.replace("<DEPTH_START>", "<TEMP>", 1)
                .replace("<DEPTH_END>", "<DEPTH_START>", 1)
                .replace("<TEMP>", "<DEPTH_END>", 1)
            ),
            "MOLMOACT_DEPTH_GRAMMAR",
        ),
        (
            lambda text: text.replace("[[1,2],[3,4]]", "[[1.0,2]]"),
            "MOLMOACT_TRACE_GRAMMAR",
        ),
        (
            lambda text: text.replace("[[1,2],[3,4]]", "[[256,2]]"),
            "MOLMOACT_TRACE_GRAMMAR",
        ),
        (
            lambda text: text.replace(",弓", ""),
            "MOLMOACT_ACTION_TOKEN_GRAMMAR",
        ),
        (
            lambda text: text.replace(
                f"[{','.join(ACTION_TOKENS)}]",
                "[A0,A1,A2,A3,A4,A5,A6]",
            ),
            "MOLMOACT_ACTION_TOKEN_GRAMMAR",
        ),
        (
            lambda text: text.replace("弎", "'弎\""),
            "MOLMOACT_ACTION_TOKEN_GRAMMAR",
        ),
        (
            lambda text: text + " [弫]",
            "MOLMOACT_ACTION_TOKEN_GRAMMAR",
        ),
    ],
)
def test_generated_grammar_mutations_fail_exactly(replace, code: str) -> None:
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_generated_text(
            replace(_generated_text()), model=_ParserModel(), tokenizer=_Tokenizer()
        )
    assert caught.value.code == code


def test_action_parser_rejects_nonfinite_values() -> None:
    model = _ParserModel()
    model.parse_action = lambda *args, **kwargs: [[0, 1, 2, 3, 4, 5, float("nan")]]
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.validate_generated_text(
            _generated_text(), model=model, tokenizer=_Tokenizer()
        )
    assert caught.value.code == "MOLMOACT_ACTION_PARSE"


class _FakeTensor:
    def __init__(self, shape: tuple[int, int], device: str = "cpu") -> None:
        self.shape = shape
        self.device = device

    def to(self, device: str) -> "_FakeTensor":
        return _FakeTensor(self.shape, device)

    def __getitem__(self, key) -> "_FakeTensor":
        rows, columns = key
        assert rows == slice(None)
        assert isinstance(columns, slice)
        start = columns.start or 0
        return _FakeTensor((self.shape[0], self.shape[1] - start), self.device)


class _FakeImage:
    size = (32, 24)
    mode = "RGB"

    def __enter__(self) -> "_FakeImage":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def load(self) -> None:
        return None

    def convert(self, mode: str) -> "_FakeImage":
        assert mode == "RGB"
        return self


class _FakeProcessor:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.calls.append(("chat", (messages, kwargs)))
        return "chat text"

    def __call__(self, **kwargs) -> dict[str, _FakeTensor]:
        self.calls.append(("processor", kwargs))
        return {
            "input_ids": _FakeTensor((1, 3)),
            "pixel_values": _FakeTensor((1, 4)),
        }

    def batch_decode(self, value, **kwargs) -> list[str]:
        self.calls.append(("decode", (value, kwargs)))
        return [_generated_text()]


class _FakeRuntimeModel(_ParserModel):
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls
        self._qwen_tokenizer = None

    def eval(self) -> "_FakeRuntimeModel":
        self.calls.append(("eval", None))
        return self

    def parameters(self):
        yield SimpleNamespace(device="cuda:0")
        yield SimpleNamespace(device="cuda:0")

    def generate(self, **kwargs) -> _FakeTensor:
        self.calls.append(("generate", kwargs))
        return _FakeTensor((1, 5), "cuda:0")

    def parse_action(self, text: str, *, unnorm_key: str) -> list[list[float]]:
        if self._qwen_tokenizer is None:
            self._qwen_tokenizer = self.fallback_loader()
        return super().parse_action(text, unnorm_key=unnorm_key)


def _fake_runtime_libraries(
    calls: list[tuple[str, object]],
) -> SimpleNamespace:
    processor = _FakeProcessor(calls)
    model = _FakeRuntimeModel(calls)
    tokenizer = _Tokenizer()

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls.append(("processor_loader", (path, kwargs)))
            return processor

    class ModelLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls.append(("model_loader", (path, kwargs)))
            return model

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls.append(("tokenizer_loader", (path, kwargs)))
            return tokenizer

    model.fallback_loader = lambda: TokenizerLoader.from_pretrained("Qwen/Qwen2-7B")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda device: calls.append(("synchronize", device)),
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
        get_device_properties=lambda index: SimpleNamespace(
            name="synthetic", total_memory=123
        ),
        max_memory_allocated=lambda device: 10,
        max_memory_reserved=lambda device: 20,
    )
    torch = SimpleNamespace(
        __version__="synthetic",
        bfloat16=object(),
        cuda=cuda,
        version=SimpleNamespace(cuda="synthetic"),
        _C=SimpleNamespace(_cuda_getDriverVersion=lambda: 1),
        is_tensor=lambda value: isinstance(value, _FakeTensor),
        inference_mode=lambda: nullcontext(),
        autocast=lambda **kwargs: nullcontext(),
    )
    return SimpleNamespace(
        image=SimpleNamespace(open=lambda stream: _FakeImage()),
        torch=torch,
        transformers=SimpleNamespace(__version__="4.52.3"),
        AutoProcessor=ProcessorLoader,
        AutoModelForImageTextToText=ModelLoader,
        Qwen2Tokenizer=TokenizerLoader,
        is_hub_kernels_available=lambda: False,
        runtime_model=model,
        runtime_tokenizer=tokenizer,
    )


def test_mocked_runtime_uses_exact_call_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []
    libraries = _fake_runtime_libraries(calls)
    view = str(tmp_path / "view")
    cache = f"{view}/hf_modules"
    monkeypatch.setattr(contract, "_observe_path_openat2", lambda path: None)
    result = contract._execute_runtime_call_graph(
        contract.load_runtime_lock().data,
        view,
        [b"side", b"wrist"],
        libraries,
    )
    try:
        processor_call = next(
            value for name, value in calls if name == "processor_loader"
        )
        model_call = next(value for name, value in calls if name == "model_loader")
        tokenizer_call = next(
            value for name, value in calls if name == "tokenizer_loader"
        )
        assert processor_call == (
            f"{view}/molmoact_model",
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "padding_side": "left",
            },
        )
        assert model_call[0] == f"{view}/molmoact_model"
        assert model_call[1]["trust_remote_code"] is True
        assert model_call[1]["local_files_only"] is True
        assert model_call[1]["torch_dtype"] is libraries.torch.bfloat16
        assert model_call[1]["device_map"] == {"": "cuda:0"}
        assert tokenizer_call == (
            f"{view}/qwen_tokenizer",
            {"local_files_only": True},
        )
        processor_kwargs = next(value for name, value in calls if name == "processor")
        chat_messages, chat_kwargs = next(
            value for name, value in calls if name == "chat"
        )
        assert chat_messages == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": contract.load_runtime_lock().data["operation"][
                            "prompt"
                        ],
                    }
                ],
            }
        ]
        assert chat_kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert processor_kwargs["images"][0][0].size == (32, 24)
        assert len(processor_kwargs["images"]) == 1
        assert len(processor_kwargs["images"][0]) == 2
        assert processor_kwargs["padding"] is True
        assert processor_kwargs["return_tensors"] == "pt"
        assert sum(name == "processor_loader" for name, _ in calls) == 1
        assert sum(name == "model_loader" for name, _ in calls) == 1
        assert sum(name == "tokenizer_loader" for name, _ in calls) == 1
        assert sum(name == "generate" for name, _ in calls) == 1
        generation = next(value for name, value in calls if name == "generate")
        assert generation["max_new_tokens"] == 256
        assert generation["do_sample"] is False
        assert generation["num_beams"] == 1
        assert generation["num_return_sequences"] == 1
        assert generation["use_cache"] is True
        assert generation["input_ids"].device == "cuda:0"
        assert generation["pixel_values"].device == "cuda:0"
        decode_value, decode_kwargs = next(
            value for name, value in calls if name == "decode"
        )
        assert decode_value.shape == (1, 2)
        assert decode_kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        assert libraries.runtime_model._qwen_tokenizer is libraries.runtime_tokenizer
        libraries.runtime_model._qwen_tokenizer = None
        with pytest.raises(
            contract.MolmoActInferenceError,
            match="mutable Qwen tokenizer fallback is forbidden",
        ):
            libraries.runtime_model.parse_action(
                _generated_text(), unnorm_key="molmoact"
            )
        assert result["generated_token_count"] == 2
    finally:
        while cache in sys.path:
            sys.path.remove(cache)


def test_hub_kernels_and_python_network_are_fail_closed() -> None:
    libraries = _fake_runtime_libraries([])
    libraries.is_hub_kernels_available = lambda: True
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract._validate_runtime_libraries(libraries)
    assert caught.value.code == "MOLMOACT_HUB_KERNELS"
    with contract.network_tripwire():
        with pytest.raises(contract.MolmoActInferenceError) as caught:
            socket.getaddrinfo("example.invalid", 443)
        assert caught.value.code == "MOLMOACT_NETWORK_ATTEMPT"


def test_preimport_rejection_covers_every_heavy_prefix() -> None:
    source = Path(contract.__file__).parents[3]
    for name in ("PIL", "numpy", "torch", "transformers", "huggingface_hub", "kernels"):
        script = (
            "import sys,types;"
            f"sys.path.insert(0,{str(source)!r});"
            f"sys.modules[{name!r}]=types.ModuleType({name!r});"
            "from npa.workflows.byof import molmoact_inference as m;"
            "m._check_heavy_preimports()"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert result.returncode != 0
        assert "MOLMOACT_HEAVY_PREIMPORTED" in result.stderr


def test_offline_ordering_and_scratch_cleanup_are_child_contained(
    tmp_path: Path,
) -> None:
    source = Path(contract.__file__).parents[3]
    scratch_parent = _private_directory(tmp_path / "scratch")
    keys = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_MODULES_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
        "PYTHONDONTWRITEBYTECODE",
    )
    parent_environment = {key: os.environ.get(key) for key in keys}
    script = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path
        import socket
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, {str(source)!r})
        from npa.workflows.byof import molmoact_inference as contract

        parent = Path({str(scratch_parent)!r})
        events = []
        context = SimpleNamespace(
            lock=SimpleNamespace(data=None),
            decision=None,
            isolation=None,
            candidate_image="candidate",
            consumer_path="/verified/view",
            mount=None,
            snapshots=[None, None, object()],
            verification_ns=1,
        )

        def read_inputs(snapshot):
            assert snapshot is context.snapshots[2]
            assert os.environ["HF_HOME"].startswith(str(parent) + "/molmoact-")
            assert os.environ["HF_HUB_CACHE"].startswith(str(parent) + "/molmoact-")
            assert os.environ["TRANSFORMERS_CACHE"].startswith(
                str(parent) + "/molmoact-"
            )
            assert os.environ["HF_MODULES_CACHE"] == "/verified/view/hf_modules"
            for key in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "DO_NOT_TRACK",
                "PYTHONDONTWRITEBYTECODE",
            ):
                assert os.environ[key] == "1"
            assert sys.dont_write_bytecode is True
            events.append("read")
            return [b"side", b"wrist"]

        def execute_runtime(lock, consumer_view, image_bytes):
            assert lock is None
            assert consumer_view == "/verified/view"
            assert image_bytes == [b"side", b"wrist"]
            try:
                socket.getaddrinfo("example.invalid", 443)
            except contract.MolmoActInferenceError as exc:
                assert exc.code == "MOLMOACT_NETWORK_ATTEMPT"
            else:
                raise AssertionError("network tripwire was not active")
            events.append("runtime")
            return {{"synthetic": True}}

        def result_payload(**kwargs):
            assert kwargs["runtime"] == {{"synthetic": True}}
            assert list(parent.iterdir()) == []
            assert events == ["read", "runtime"]
            return {{"events": list(events)}}

        contract._read_verified_inputs = read_inputs
        contract._load_and_execute_runtime = execute_runtime
        contract._result_payload = result_payload
        assert contract._execute_in_scratch(context, str(parent)) == {{
            "events": ["read", "runtime"]
        }}
        assert list(parent.iterdir()) == []

        def fail_inputs(snapshot):
            del snapshot
            raise contract.MolmoActInferenceError(
                "MOLMOACT_INPUT_IDENTITY", "injected input failure"
            )

        contract._read_verified_inputs = fail_inputs
        try:
            contract._execute_in_scratch(context, str(parent))
        except contract.MolmoActInferenceError as exc:
            assert exc.code == "MOLMOACT_INPUT_IDENTITY"
        else:
            raise AssertionError("injected failure was not preserved")
        assert list(parent.iterdir()) == []

        real_cleanup = contract._cleanup_scratch

        def fail_cleanup(scratch, anchor):
            real_cleanup(scratch, anchor)
            raise contract.MolmoActInferenceError(
                "MOLMOACT_OFFLINE_STATE", "injected cleanup failure"
            )

        contract._cleanup_scratch = fail_cleanup
        try:
            contract._execute_in_scratch(context, str(parent))
        except contract.MolmoActInferenceError as exc:
            assert exc.code == "MOLMOACT_INPUT_IDENTITY"
            assert any("scratch cleanup also failed" in note for note in exc.__notes__)
        else:
            raise AssertionError("cleanup failure did not take precedence")
        assert list(parent.iterdir()) == []
        print(json.dumps({{"events": events, "scratch_entries": 0}}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "events": ["read", "runtime"],
        "scratch_entries": 0,
    }
    assert {key: os.environ.get(key) for key in keys} == parent_environment


def test_network_tripwire_covers_runtime_envelope_and_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_getaddrinfo = socket.getaddrinfo
    events: list[str] = []

    def require_tripwire(stage: str) -> None:
        try:
            socket.getaddrinfo(None, 0)
        except contract.MolmoActInferenceError as exc:
            assert exc.code == "MOLMOACT_NETWORK_ATTEMPT"
        else:
            raise AssertionError(f"network tripwire absent during {stage}")
        events.append(stage)

    snapshot = SimpleNamespace(close=lambda: events.append("close"))
    context = SimpleNamespace(snapshots=[snapshot])
    monkeypatch.setattr(contract, "_check_heavy_preimports", lambda: None)
    monkeypatch.setattr(contract, "_preflight_inference", lambda **kwargs: context)

    def execute(current: object, scratch: object) -> dict[str, object]:
        assert current is context
        assert scratch == "/scratch"
        require_tripwire("runtime")
        return {"payload": True}

    def envelope(payload: object) -> dict[str, object]:
        assert payload == {"payload": True}
        require_tripwire("envelope")
        return {"envelope": True}

    def publish(output: object, value: object) -> dict[str, object]:
        assert output == "/output"
        assert value == {"envelope": True}
        require_tripwire("publication")
        return {"receipt": True}

    monkeypatch.setattr(contract, "_execute_in_scratch", execute)
    monkeypatch.setattr(contract, "build_result_envelope", envelope)
    monkeypatch.setattr(contract, "publish_result", publish)
    assert contract.run_inference(
        decision_path="/decision",
        isolation_receipt_path="/isolation",
        consumer_view="/consumer",
        scratch_parent="/scratch",
        output_path="/output",
    ) == {"receipt": True}
    assert events == ["runtime", "envelope", "publication", "close"]
    assert socket.getaddrinfo is original_getaddrinfo


def test_publication_uses_real_openat2_and_canonical_readback(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"
    receipt = contract.publish_result(destination, _result_envelope())
    assert receipt["publication_verified"] is True
    assert receipt["lexical_destination_verified_at_openat2"] is True
    raw = destination.read_bytes()
    assert raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == receipt["final_file_sha256"]
    assert not list(parent.glob(".molmoact-*.tmp"))


def test_publication_receipt_is_durable_and_never_replaced(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "output")
    result = parent / "result.json"
    receipt_path = parent / "publication-receipt.json"
    receipt = contract.publish_result(result, _result_envelope())

    contract.persist_publication_receipt(receipt_path, receipt)
    assert receipt_path.read_bytes() == _canonical(receipt) + b"\n"
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.persist_publication_receipt(receipt_path, receipt)
    assert caught.value.code == "MOLMOACT_OUTPUT_COLLISION"


def test_main_returns_machine_errors_without_tracebacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert contract.main([]) == 1
    missing = json.loads(capsys.readouterr().err)
    assert missing["error"]["code"] == "MOLMOACT_OFFLINE_STATE"

    parent = _private_directory(tmp_path / "output")
    receipt_path = parent / "receipt.json"
    monkeypatch.setattr(contract, "preflight_destination", lambda path: None)

    def unexpected(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise OSError("private detail that must not escape")

    monkeypatch.setattr(contract, "run_inference", unexpected)
    args = [
        "--decision", "/decision", "--isolation-receipt", "/isolation",
        "--consumer-view", "/consumer", "--scratch-parent", "/scratch",
        "--output", "/output", "--publication-receipt", str(receipt_path),
    ]
    assert contract.main(args) == 2
    rendered = capsys.readouterr().err
    assert "Traceback" not in rendered and "private detail" not in rendered
    assert json.loads(rendered)["error"]["code"] == "MOLMOACT_UNEXPECTED"


def test_publication_fsyncs_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"
    original = os.fsync
    modes: list[int] = []

    def observe(descriptor: int) -> None:
        modes.append(os.fstat(descriptor).st_mode)
        original(descriptor)

    monkeypatch.setattr(contract.os, "fsync", observe)
    contract.publish_result(destination, _result_envelope())
    assert any(stat.S_ISREG(mode) for mode in modes)
    assert any(stat.S_ISDIR(mode) for mode in modes)


def test_publication_rejects_tampered_temporary_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"
    original = contract._write_all

    def write_tampered(descriptor: int, raw_bytes: bytes) -> None:
        original(descriptor, raw_bytes + b"x")

    monkeypatch.setattr(contract, "_write_all", write_tampered)
    stages: list[str] = []
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(destination, _result_envelope(), fault=stages.append)
    assert caught.value.code == "MOLMOACT_OUTPUT_READBACK"
    assert caught.value.context["final_state"] == "absent"
    assert "before_link" not in stages
    assert not destination.exists()
    assert not list(parent.glob(".molmoact-*.tmp"))


def test_publication_never_replaces_existing_destination(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"
    destination.write_bytes(b"winner")
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(destination, _result_envelope())
    assert caught.value.code == "MOLMOACT_OUTPUT_COLLISION"
    assert destination.read_bytes() == b"winner"
    assert not list(parent.glob(".molmoact-*.tmp"))


def test_publication_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    real = _private_directory(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(alias / "result.json", _result_envelope())
    assert caught.value.code == "MOLMOACT_DESTINATION"
    assert not (real / "result.json").exists()


@pytest.mark.parametrize(
    ("stage", "exception_type", "expected_code"),
    [
        ("before_write", OSError, "MOLMOACT_OUTPUT_DURABILITY"),
        ("before_file_fsync", OSError, "MOLMOACT_OUTPUT_DURABILITY"),
        ("before_temp_readback", ValueError, "MOLMOACT_OUTPUT_READBACK"),
        ("before_link", OSError, "MOLMOACT_OUTPUT_DURABILITY"),
        ("after_link", OSError, "MOLMOACT_OUTPUT_DURABILITY"),
        ("before_temp_unlink", OSError, "MOLMOACT_OUTPUT_DURABILITY"),
        ("before_directory_fsync", OSError, "MOLMOACT_OUTPUT_DURABILITY"),
        ("before_openat2", OSError, "MOLMOACT_OUTPUT_DURABILITY"),
    ],
)
def test_publication_failures_clean_destination_and_temporary(
    tmp_path: Path,
    stage: str,
    exception_type: type[BaseException],
    expected_code: str,
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"

    def fail(current: str) -> None:
        if current == stage:
            raise exception_type("injected")

    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(destination, _result_envelope(), fault=fail)
    assert caught.value.code == expected_code
    assert caught.value.context["final_state"] == "absent"
    assert not destination.exists()
    assert not list(parent.glob(".molmoact-*.tmp"))


def test_openat2_unavailable_fails_and_removes_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"

    def unavailable(*args, **kwargs):
        raise OSError(38, "ENOSYS")

    monkeypatch.setattr(contract, "_openat2_from_root", unavailable)
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(destination, _result_envelope())
    assert caught.value.code == "MOLMOACT_OUTPUT_DURABILITY"
    assert caught.value.context["final_state"] == "absent"
    assert not destination.exists()


@pytest.mark.parametrize(
    "swap_stage",
    ["before_link", "after_link", "before_directory_fsync", "before_openat2"],
)
def test_parent_swap_never_returns_a_receipt(tmp_path: Path, swap_stage: str) -> None:
    parent = _private_directory(tmp_path / "output")
    moved = tmp_path / "moved-output"
    destination = parent / "result.json"
    swapped = False

    def swap(stage: str) -> None:
        nonlocal swapped
        if stage == swap_stage and not swapped:
            parent.rename(moved)
            _private_directory(parent)
            swapped = True

    with pytest.raises(contract.MolmoActInferenceError):
        contract.publish_result(destination, _result_envelope(), fault=swap)
    assert not destination.exists()
    assert not (moved / "result.json").exists()
    assert not list(moved.glob(".molmoact-*.tmp"))


def test_wrong_linked_inode_never_returns_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"
    decoy = parent / "decoy"
    decoy.write_bytes(b"decoy")
    original_link = os.link

    def link_decoy(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        del source, src_dir_fd
        original_link(
            decoy.name,
            target,
            src_dir_fd=dst_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(contract.os, "link", link_decoy)
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(destination, _result_envelope())
    assert caught.value.code == "MOLMOACT_OUTPUT_DURABILITY"
    assert caught.value.context["final_state"] == "residual_unverified_candidate"
    assert destination.read_bytes() == b"decoy"
    destination.unlink()
    decoy.unlink()
    assert not list(parent.glob(".molmoact-*.tmp"))


def test_exception_after_successful_link_cleans_the_committed_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"
    original_link = os.link

    def link_then_raise(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        original_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        raise OSError("injected post-commit interruption")

    monkeypatch.setattr(contract.os, "link", link_then_raise)
    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(destination, _result_envelope())
    assert caught.value.code == "MOLMOACT_OUTPUT_DURABILITY"
    assert caught.value.context["final_state"] == "absent"
    assert not destination.exists()
    assert not list(parent.glob(".molmoact-*.tmp"))


def test_unprovable_postlink_cleanup_is_truthful_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = parent / "result.json"
    original = contract._unlink_exact

    def deny_destination(directory: int, name: str, inode: tuple[int, int]) -> bool:
        if name == destination.name:
            return False
        return original(directory, name, inode)

    monkeypatch.setattr(contract, "_unlink_exact", deny_destination)

    def fail_after_link(stage: str) -> None:
        if stage == "after_link":
            raise OSError("injected")

    with pytest.raises(contract.MolmoActInferenceError) as caught:
        contract.publish_result(destination, _result_envelope(), fault=fail_after_link)
    assert caught.value.code == "MOLMOACT_OUTPUT_DURABILITY"
    assert caught.value.context["final_state"] == "residual_unverified_candidate"
    assert destination.exists()
    destination.unlink()


def _publisher_worker(
    destination: str,
    barrier,
    queue,
) -> None:
    barrier.wait()
    try:
        receipt = contract.publish_result(destination, _result_envelope("race"))
        queue.put(("success", receipt["final_file_sha256"]))
    except contract.MolmoActInferenceError as exc:
        queue.put(("error", exc.code))


def _crash_publisher_worker(destination: str, stage: str) -> None:
    def crash(current: str) -> None:
        if current == stage:
            os._exit(73)

    contract.publish_result(destination, _result_envelope("crash"), fault=crash)


@pytest.mark.parametrize(
    ("stage", "destination_expected"),
    [("before_link", False), ("after_link", True)],
)
def test_publication_crash_points_never_create_a_success_receipt(
    tmp_path: Path, stage: str, destination_expected: bool
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = str(parent / "result.json")
    process = multiprocessing.get_context("fork").Process(
        target=_crash_publisher_worker,
        args=(destination, stage),
    )
    process.start()
    process.join(10)
    assert process.exitcode == 73
    assert Path(destination).exists() is destination_expected
    temporaries = list(parent.glob(".molmoact-*.tmp"))
    assert len(temporaries) == 1
    for path in [Path(destination), *temporaries]:
        path.unlink(missing_ok=True)


def test_two_process_publication_race_has_one_complete_winner(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "output")
    destination = str(parent / "result.json")
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [
        context.Process(target=_publisher_worker, args=(destination, barrier, queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    results = sorted(queue.get(timeout=2) for _ in processes)
    assert [kind for kind, _ in results].count("success") == 1
    assert ("error", "MOLMOACT_OUTPUT_COLLISION") in results
    raw = Path(destination).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == next(
        value for kind, value in results if kind == "success"
    )
    assert not list(parent.glob(".molmoact-*.tmp"))
