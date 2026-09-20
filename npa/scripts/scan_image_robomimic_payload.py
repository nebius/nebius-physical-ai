#!/usr/bin/env python3
"""Fail closed on CUDA/runtime/data/cache/output bytes in a robomimic image."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_image_wan_payload as walker  # noqa: E402


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "torch_or_triton_distribution",
        re.compile(
            r"(?:^|/)(?:torch|torchvision|triton)-[^/!]+\.whl(?:$|!/)|"
            r"(?:^|!/)(?:torch|torchvision|triton)(?:/|$)|"
            r"(?:^|!/)(?:torch|torchvision|triton)-[^/!]+\.dist-info(?:/|$)|"
            r"(?:^|/)(?:site-packages|dist-packages)/(?:torch(?:/|vision/)|"
            r"torch(?:vision)?-[^/]*\.dist-info/|triton(?:/|-[^/]*\.dist-info/))",
            re.I,
        ),
    ),
    (
        "nvidia_python_distribution",
        re.compile(
            r"(?:^|/)nvidia[-_][^/!]+\.whl(?:$|!/)|"
            r"(?:^|!/)nvidia(?:/|_)|"
            r"(?:^|!/)(?:nvidia[-_])[^/!]+\.dist-info(?:/|$)|"
            r"(?:^|/)(?:site-packages|dist-packages)/(?:nvidia(?:/|_)|"
            r"nvidia_[^/]*\.dist-info/)",
            re.I,
        ),
    ),
    (
        "cuda_library",
        re.compile(
            r"(?:^|/)(?:lib)?(?:(?:[a-z0-9]+_)*(?:cuda|cudart|cublas|cudnn|"
            r"nccl|nvrtc|nvjitlink|nvtx|nvtoolsext|nvfatbin|nvptxcompiler|cupti|"
            r"cufile|cusparse|cusolver|curand|cufft|npp[a-z]*)(?:[a-z0-9_-]*)|"
            r"nvidia(?:-[a-z0-9_-]+)?|(?:glx|egl)_nvidia|"
            r"nv(?:cuvid|optix|encode|decode))(?:[^/]*)\.(?:so(?:\.|$)|a$)",
            re.I,
        ),
    ),
    (
        "cuda_tool_or_header",
        re.compile(
            r"(?:^|/)(?:(?:usr/local|opt)/cuda(?:-[0-9.]+)?(?:/|$)|"
            r"usr/include/(?:cuda|cublas|cudnn|nccl|nvrtc|nvToolsExt|npp|cupti|"
            r"cufft|cusparse|cusolver|curand)[^/]*\.h$|(?:nvcc|ptxas|cuobjdump|"
            r"compute-sanitizer|nvidia-smi)(?:$|\.))",
            re.I,
        ),
    ),
    (
        "checkpoint_or_weight",
        re.compile(
            r"(?:\.(?:safetensors|ckpt|pt|bin|onnx|msgpack|"
            r"gguf|engine|plan|tflite|mlmodel)$|"
            r"(?:^|/)(?:weights?|checkpoints?)(?:/|$)|"
            r"(?:^|/)(?:models?|policies?)/[^!]*\.(?:npy|npz|pth)$|"
            r"(?:^|/)[^/]*(?:weight|checkpoint|policy|model)[^/]*\."
            r"(?:npy|npz|pth)$)",
            re.I,
        ),
    ),
    (
        "dataset_payload",
        re.compile(r"\.(?:hdf5|h5)$", re.I),
    ),
    (
        "populated_runtime_cache",
        re.compile(
            r"(?:^|/)(?:opt/npa-runtime/robomimic|\.cache/(?:pip|uv|huggingface)|"
            r"pip-cache|wheelhouse)/(?:.+)",
            re.I,
        ),
    ),
    (
        "run_output_or_proof",
        re.compile(
            r"(?:^|/)workspace/(?:byof-inputs|byof-runs)/.+|"
            r"(?:^|/)robomimic-smoke\.json$",
            re.I,
        ),
    ),
    (
        "credential_file",
        re.compile(
            r"(?:^|/)(?:\.aws/credentials|\.docker/config\.json|\.git-credentials|"
            r"kubeconfig|etc/ssh/ssh_host_(?:rsa|ecdsa|ed25519)_key)$",
            re.I,
        ),
    ),
)

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cuda_or_pytorch_base", re.compile(r"\b(?:nvidia/cuda|pytorch/pytorch):", re.I)),
    (
        "cuda_install_at_build",
        re.compile(
            r"(?:download\.pytorch\.org/whl/cu|"
            r"(?:pip|uv)(?:\s+pip)?\s+install(?:(?!&&|\|\||;)[^\n])*"
            r"(?:nvidia-|torch==[^\s;&|]*\+cu|torchvision==[^\s;&|]*\+cu))",
            re.I,
        ),
    ),
    (
        "runtime_population_at_build",
        re.compile(
            r"\brobomimic-runtime\s+(?:ensure|install|fetch|warm|exec)\b",
            re.I | re.S,
        ),
    ),
    (
        "dataset_fetch_at_build",
        re.compile(
            r"(?:robomimic_datasets|low_dim_v15\.hdf5|huggingface-cli\s+download)",
            re.I | re.S,
        ),
    ),
    (
        "invented_acceptance_proxy",
        re.compile(
            r"\b(?:ACCEPT_EULA|CUDA_ACCEPT|CUDNN_ACCEPT|NPA_ROBOMIMIC_ACCEPT)"
            r"[A-Z0-9_]*\b",
            re.I,
        ),
    ),
)


SOURCE_ROOT = "usr/share/npa/robomimic/corresponding-source/"
IMAGE_ROOT = Path(__file__).resolve().parents[1] / "docker/workbench/robomimic"


@contextmanager
def _audited_codec_fixtures(audited, verified_sources, observed=None):
    """Keep raw checks, but bound expansion of exact upstream error fixtures."""

    original = walker._scan_nested_archive
    original_file = walker._scan_file_stream
    original_path = walker._add_forbidden_path_findings
    observed = observed if observed is not None else {}
    occurrences: dict[str, int] = {}

    def record_path(path, source, findings):
        if path in audited:
            occurrences[path] = occurrences.get(path, 0) + 1
        original_path(path, source, findings)

    def scan_file(**kwargs):
        path = kwargs["path"]
        if path not in audited:
            return original_file(**kwargs)
        with tempfile.TemporaryFile() as captured:
            digest = hashlib.sha256()
            while chunk := kwargs["stream"].read(1024 * 1024):
                digest.update(chunk)
                captured.write(chunk)
            member_type = (
                "decompressed-file" if path.endswith("!/<decompressed>") else "file"
            )
            observed.setdefault(path, []).append((digest.hexdigest(), member_type))
            captured.seek(0)
            original_file(**{**kwargs, "stream": captured})

    def scan_fixture(**kwargs):
        path = kwargs["parent_path"]
        archive_path = path.partition("!/")[0]
        disposition = audited.get(path, {})
        if (
            archive_path in verified_sources
            and "nested_archive_unreadable" in disposition.get("kinds", [])
            and disposition.get("archive_sha256") == archive_path.split("/")[-2]
        ):
            stream = kwargs["stream"]
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            if digest == disposition["member_sha256"]:
                kwargs["findings"].append(
                    walker.Finding(
                        "nested_archive_unreadable",
                        path,
                        "hash-verified upstream codec/archive error fixture; raw byte checks retained",
                    )
                )
                return
        original(**kwargs)

    # Like the existing policy context, this CLI-only scope is not thread-safe.
    walker._scan_nested_archive = scan_fixture
    walker._scan_file_stream = scan_file
    walker._add_forbidden_path_findings = record_path
    try:
        yield occurrences
    finally:
        walker._scan_nested_archive = original
        walker._scan_file_stream = original_file
        walker._add_forbidden_path_findings = original_path


def _directory_identities(stream, member_parts: list[str]) -> list[tuple[str, str]]:
    """Observe the exact metadata of listed directory fixtures, including duplicates."""

    identities = []
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if member.name.lstrip("/") != member_parts[0]:
                continue
            if len(member_parts) == 1:
                identities.append(
                    (
                        hashlib.sha256(b"").hexdigest(),
                        "directory" if member.isdir() else "not-directory",
                    )
                )
            elif member.isfile():
                with archive.extractfile(member) as nested:
                    identities.extend(_directory_identities(nested, member_parts[1:]))
    return identities


def _scan_report(tars: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    """Keep raw findings and attribute only exact locked public source fixtures.

    Each hash-verified source archive receives the walker's existing archive
    budget. The lock bounds their total compressed bytes and count; duplicate
    source delivery is refused. All other image bytes retain per-layer budgets.
    """

    lock = json.loads((IMAGE_ROOT / "corresponding-source.lock.json").read_text())
    artifacts = [lock["cpython"]] + [
        item for source in lock["sources"] for item in source["artifacts"]
    ]
    expected = {
        SOURCE_ROOT + item["sha256"] + "/" + item["filename"]: item
        for item in artifacts
    }
    dispositions = json.loads(
        (IMAGE_ROOT / "source-fixture-dispositions.json").read_text()
    )
    audited = {item["path"]: item for item in dispositions["fixtures"]}
    raw: list[walker.Finding] = []
    attributed: list[walker.Finding] = []
    observed: dict[str, list[tuple[str, str]]] = {}
    matched_dispositions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="robomimic-source-scan-") as tmp:
        sources: dict[str, Path] = {}
        with (
            walker.payload_policy(
                forbidden_paths=FORBIDDEN_PATHS,
                forbidden_history=FORBIDDEN_HISTORY,
                audited_secret_files=walker.AUDITED_SECRET_LITERAL_FILE_SHA256,
                audited_libraries=walker.AUDITED_LITERAL_LIBRARY_SHA256,
            ),
            _audited_codec_fixtures(audited, sources, observed) as occurrences,
        ):
            for layer in tars:
                budget = walker._NestedArchiveBudget(
                    walker.MAX_NESTED_UNCOMPRESSED_BYTES,
                    walker.MAX_NESTED_ARCHIVE_MEMBERS,
                )
                with tarfile.open(layer, "r:*") as archive:
                    for member in archive:
                        path = walker._normalize_archive_path(member.name)
                        if not member.isfile():
                            walker._add_forbidden_path_findings(path, layer.name, raw)
                            continue
                        with archive.extractfile(member) as stream:
                            selected_budget = budget
                            if path in expected:
                                item = expected[path]
                                if path in sources or member.size != item["size"]:
                                    raise ValueError(
                                        "duplicate or changed corresponding-source archive"
                                    )
                                captured = Path(tmp) / str(len(sources))
                                digest = hashlib.sha256()
                                with captured.open("wb") as output:
                                    while chunk := stream.read(1024 * 1024):
                                        digest.update(chunk)
                                        output.write(chunk)
                                if digest.hexdigest() != item["sha256"]:
                                    raise ValueError(
                                        "corresponding-source archive bytes differ from lock"
                                    )
                                sources[path] = captured
                                selected_budget = walker._NestedArchiveBudget(
                                    walker.MAX_NESTED_UNCOMPRESSED_BYTES,
                                    dispositions.get("archive_member_limits", {}).get(
                                        item["sha256"],
                                        walker.MAX_NESTED_ARCHIVE_MEMBERS,
                                    ),
                                )
                                with captured.open("rb") as source_stream:
                                    walker._scan_file_stream(
                                        path=path,
                                        stream=source_stream,
                                        source=layer.name,
                                        findings=raw,
                                        depth=0,
                                        budget=selected_budget,
                                    )
                            else:
                                walker._scan_file_stream(
                                    path=path,
                                    stream=stream,
                                    source=layer.name,
                                    findings=raw,
                                    depth=0,
                                    budget=selected_budget,
                                )
            raw.extend(walker.scan_tars([], config))
        for finding in raw:
            disposition = audited.get(finding.path)
            if disposition is None or finding.kind not in disposition["kinds"]:
                continue
            archive_path, separator, member_path = finding.path.partition("!/")
            if not separator or archive_path not in sources:
                continue
            if expected[archive_path]["sha256"] != disposition["archive_sha256"]:
                continue
            if (
                disposition["member_type"] == "directory"
                and finding.path not in observed
            ):
                with sources[archive_path].open("rb") as stream:
                    observed[finding.path] = _directory_identities(
                        stream, member_path.split("!/")
                    )
            identity = (disposition["member_sha256"], disposition["member_type"])
            if occurrences.get(finding.path) == 1 and observed.get(finding.path) == [
                identity
            ]:
                attributed.append(finding)
                matched_dispositions.append(
                    {
                        **disposition,
                        "finding": asdict(finding),
                        "observed_member_sha256": identity[0],
                        "observed_member_type": identity[1],
                    }
                )
    findings = [finding for finding in raw if finding not in attributed]
    return {
        "findings": findings,
        "raw_findings": raw,
        "attributed_source_fixtures": attributed,
        "matched_source_dispositions": matched_dispositions,
    }


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[walker.Finding]:
    """Scan every layer and OCI config, retaining only unresolved findings."""

    return _scan_report(tars, config)["findings"]


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[walker.Finding]:
    """Scan one test rootfs plus an OCI config."""

    return scan_tars([rootfs_tar], config)


docker_save_material = walker.docker_save_material


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        sum(bool(value) for value in (args.image, args.rootfs_tar, args.docker_save))
        != 1
    ):
        parser.error("provide exactly one IMAGE, --rootfs-tar, or --docker-save")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")

    try:
        with tempfile.TemporaryDirectory(prefix="npa-robomimic-byte-scan-") as tmp:
            if args.image:
                tars, config = walker.remote_material(args.image, Path(tmp))
            elif args.docker_save:
                tars, config = docker_save_material(args.docker_save, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            report = _scan_report(tars, config)
            findings = report["findings"]
    except Exception as exc:  # noqa: BLE001 - every unreadable artifact fails closed
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    result = {
        "format": "npa_robomimic_image_byte_scan_v1",
        "image": args.image
        or ("docker-save" if args.docker_save else "offline-rootfs"),
        "status": "pass" if not findings else "fail",
        "archives_scanned": len(tars),
        "source_lock_sha256": hashlib.sha256(
            (IMAGE_ROOT / "corresponding-source.lock.json").read_bytes()
        ).hexdigest(),
        "source_fixture_dispositions_sha256": hashlib.sha256(
            (IMAGE_ROOT / "source-fixture-dispositions.json").read_bytes()
        ).hexdigest(),
        "findings": [asdict(item) for item in findings],
        "raw_findings": [asdict(item) for item in report["raw_findings"]],
        "attributed_source_fixtures": [
            asdict(item) for item in report["attributed_source_fixtures"]
        ],
        "matched_source_dispositions": report["matched_source_dispositions"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
