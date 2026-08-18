from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/scan_content_agents_image.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scan_content_agents_image", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scanner = _load()


def _runner(*, redistribution: str = "restricted", findings: bool = False):
    def run(argv):
        if argv[1:3] == ["image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Id": "sha256:" + "1" * 64,
                        "RepoDigests": [
                            "private.invalid/npa-content-agents@sha256:" + "2" * 64
                        ],
                        "Config": {
                            "User": "ubuntu",
                            "Labels": {
                                "npa.tool": "content-agents",
                                "npa.redistribution": redistribution,
                                "npa.driver_provisioning": (
                                    "gpu-operator-host-mounted"
                                ),
                                "npa.driver_capabilities": (
                                    "compute,utility,graphics,display"
                                ),
                                "org.opencontainers.image.revision": (
                                    "36dbf3f274f8e256637230a05a085853f65cc175"
                                ),
                                "org.opencontainers.image.version": "0.5.2",
                                "npa.source_revision": "3" * 40,
                                "org.opencontainers.image.licenses": (
                                    "Apache-2.0 AND LicenseRef-NVIDIA-Proprietary-OVRTX"
                                ),
                            },
                        },
                    }
                ]
            )
        script = argv[-1]
        if "inspect_runtime" in script:
            return json.dumps(
                {
                    "status": "ready",
                    "ovrtx": {"version": "0.3.0.312915", "isolated_venv": True},
                    "ovphysx": False,
                    "scene_optimizer_core": False,
                }
            )
        if "cli_contract" in script:
            return json.dumps(
                {
                    "cli_contract": {
                        "executable_present": True,
                        "version_exit_code": 0,
                        "version_output_present": True,
                    }
                }
            )
        if "config_parse" in script:
            return json.dumps(
                {
                    "config_parse": {
                        "material": {
                            "dry_run_exit_code": 0,
                            "plan_rendered": True,
                        },
                        "physics": {
                            "dry_run_exit_code": 0,
                            "plan_rendered": True,
                        },
                    }
                }
            )
        return json.dumps(
            {
                "forbidden_exact": ["/opt/content-agents/.git"] if findings else [],
                "forbidden_dirs": [],
                "weight_files": [],
            }
        )

    return run


def test_built_image_audit_passes_only_the_expected_restricted_boundary() -> None:
    result = scanner.audit_image(
        "private.invalid/image@sha256:digest", runner=_runner()
    )
    assert result["status"] == "passed"
    assert result["runtime"]["ovrtx"]["version"] == "0.3.0.312915"
    assert result["npa_source_revision"] == "3" * 40
    assert result["cli_contract"]["cli_contract"] == {
        "executable_present": True,
        "version_exit_code": 0,
        "version_output_present": True,
    }
    assert result["config_parse"]["config_parse"]["material"] == {
        "dry_run_exit_code": 0,
        "plan_rendered": True,
    }
    assert result["inventory"] == {
        "forbidden_exact": [],
        "forbidden_dirs": [],
        "weight_files": [],
    }


def test_built_image_audit_refuses_a_public_classification() -> None:
    with pytest.raises(scanner.ImageAuditError, match="npa.redistribution"):
        scanner.audit_image("image", runner=_runner(redistribution="public"))


def test_built_image_audit_fails_on_excluded_payload() -> None:
    with pytest.raises(scanner.ImageAuditError, match="forbidden payload"):
        scanner.audit_image("image", runner=_runner(findings=True))


def test_built_image_audit_enforces_requested_npa_source_checkpoint() -> None:
    with pytest.raises(scanner.ImageAuditError, match="requested checkpoint"):
        scanner.audit_image(
            "image", expected_npa_source_sha="4" * 40, runner=_runner()
        )


def test_built_image_audit_requires_the_npa_console_entrypoint() -> None:
    runner = _runner()

    def missing_cli(argv):
        if argv[1:3] != ["image", "inspect"] and "cli_contract" in argv[-1]:
            return json.dumps(
                {
                    "cli_contract": {
                        "executable_present": False,
                        "version_exit_code": 1,
                        "version_output_present": False,
                    }
                }
            )
        return runner(argv)

    with pytest.raises(scanner.ImageAuditError, match="console entry point"):
        scanner.audit_image("image", runner=missing_cli)


def test_scanner_distinguishes_python_path_hooks_from_nested_pth_weights() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "path.parent.name != 'site-packages'" in source
    assert "{'.pt', '.ckpt', '.safetensors', '.onnx', '.gguf'}" in source
