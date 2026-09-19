"""Complete-layer negative controls for the SeedVR2 runtime-payload scanner."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "scan_image_seedvr2_payload.py"
)
SPEC = importlib.util.spec_from_file_location("scan_image_seedvr2_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _saved_image(
    tmp_path: Path,
    layers: list[dict[str, bytes]],
    *,
    config: bytes = b"{}",
) -> Path:
    layer_names = []
    payloads = {
        "manifest.json": b"",
        "config.json": config,
    }
    for index, members in enumerate(layers):
        name = f"layer-{index}.tar"
        payloads[name] = _tar(tmp_path / name, members).read_bytes()
        layer_names.append(name)
    payloads["manifest.json"] = json.dumps(
        [
            {
                "Config": "config.json",
                "RepoTags": ["seedvr2:test"],
                "Layers": layer_names,
            }
        ]
    ).encode()
    return _tar(tmp_path / "image.tar", payloads)


def test_clean_source_runtime_and_python_pth_file_pass(tmp_path: Path) -> None:
    findings, layers = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/seedvr2/LICENSE": b"Apache License 2.0",
                    "opt/seedvr2-venv/lib/python3.12/site-packages/runtime.pth": b"/opt",
                }
            ],
        )
    )
    assert layers == 1
    assert findings == []


def test_weights_cache_and_sensor_media_fail_in_lower_layer(tmp_path: Path) -> None:
    findings, layers = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "workspace/.cache/huggingface/hub/models--seedvr/model.pth": b"x",
                    "opt/seedvr2/test_videos/customer.mp4": b"x",
                },
                {
                    "workspace/.cache/huggingface/hub/models--seedvr/.wh.model.pth": b"",
                    "opt/seedvr2/test_videos/.wh.customer.mp4": b"",
                },
            ],
        )
    )
    assert layers == 2
    kinds = {finding.kind for finding in findings}
    assert {"model_weight", "populated_hf_cache", "sensor_or_output_media"} <= kinds


def test_secret_in_application_or_image_environment_fails(tmp_path: Path) -> None:
    token = b"hf_abcdefghijklmnopqrstuvwxyz"
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"opt/npa-src/config.json": token}],
            config=b'{"config":{"Env":["HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"]}}',
        )
    )
    assert [finding.kind for finding in findings].count("credential_content") == 2


def test_dependency_secret_shaped_fixture_is_not_operator_payload(
    tmp_path: Path,
) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/seedvr2-venv/lib/python3.12/site-packages/pkg/test.py": (
                        b"hf_abcdefghijklmnopqrstuvwxyz"
                    )
                }
            ],
        )
    )
    assert findings == []
