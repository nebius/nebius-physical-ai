from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "scan_image_alpamayo2_payload.py"
)
SPEC = importlib.util.spec_from_file_location("scan_image_alpamayo2_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def test_dockerfile_parses_torch_arch_flags_as_tokens() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "alpamayo2-super"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "_cuda_getArchFlags().split()" in dockerfile
    assert "{'sm_90', 'sm_100', 'sm_120'} <= flags" in dockerfile
    assert "npa-workflows/sim2real.yaml" in dockerfile
    assert "npa-workflows/physical-ai-data-factory.yaml" in dockerfile
    assert "chown -R ubuntu:ubuntu /opt/alpamayo2" not in dockerfile
    assert "'uvicorn==0.41.0' /opt/npa-src" in dockerfile


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _saved_image(
    tmp_path: Path, members: dict[str, bytes], *, config: bytes = b"{}"
) -> Path:
    layer = _tar(tmp_path / "layer.tar", members)
    return _tar(
        tmp_path / "image.tar",
        {
            "manifest.json": json.dumps(
                [
                    {
                        "Config": "config.json",
                        "RepoTags": ["test:latest"],
                        "Layers": ["layer.tar"],
                    }
                ]
            ).encode(),
            "config.json": config,
            "layer.tar": layer.read_bytes(),
        },
    )


def test_clean_source_and_empty_cache_pass(tmp_path: Path) -> None:
    findings, layers = scanner.scan_saved_image(
        _saved_image(tmp_path, {"opt/alpamayo2/LICENSE": b"Apache License 2.0"})
    )
    assert layers == 1
    assert findings == []


def test_symlink_under_runtime_tree_is_not_read_as_file(tmp_path: Path) -> None:
    layer_path = tmp_path / "layer.tar"
    with tarfile.open(layer_path, "w") as archive:
        link = tarfile.TarInfo("opt/alpamayo2/.venv/bin/python")
        link.type = tarfile.SYMTYPE
        link.linkname = "/usr/bin/python3.12"
        archive.addfile(link)
    assert scanner.scan_layer(layer_path, layer="layer.tar") == []


def test_weight_and_dataset_payload_fail_even_in_lower_layer(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            {
                "workspace/.cache/huggingface/hub/models--nvidia/model.safetensors": b"x",
                "opt/alpamayo2/notebooks/clip_ids.parquet": b"x",
            },
        )
    )
    assert {finding.kind for finding in findings} == {
        "model_weight",
        "populated_hf_cache",
        "physical_ai_av_dataset_payload",
    }


def test_secret_content_in_source_fails(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path, {"opt/alpamayo2/config": b"hf_abcdefghijklmnopqrstuvwxyz"}
        )
    )
    assert {finding.kind for finding in findings} == {"credential_content"}


def test_dependency_fixtures_do_not_impersonate_operator_payload(
    tmp_path: Path,
) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            {
                "opt/alpamayo2/.venv/site-packages/pkg/testing.py": b"hf_abcdefghijklmnopqrstuvwxyz",
                "opt/alpamayo2/.venv/site-packages/pyarrow/tests/example.parquet": b"x",
            },
        )
    )
    assert findings == []


def test_secret_in_image_environment_fails(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            {},
            config=b'{"config":{"Env":["HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"]}}',
        )
    )
    assert {finding.kind for finding in findings} == {"credential_content"}
