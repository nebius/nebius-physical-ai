"""Shell-script smoke tests for sim2real customer demo (Mac/bash 3.2 safe paths)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = REPO_ROOT / "ops" / "private" / "sim2real-rtxpro"
LIB = OPS / "lib"


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_npa_repo_root_from_lib_dir() -> None:
    result = _bash(
        f"source '{LIB}/operator-config.sh' && npa_repo_root '{LIB}'",
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT


def test_npa_repo_root_from_ops_script_dir() -> None:
    result = _bash(
        f"source '{LIB}/operator-config.sh' && npa_repo_root '{OPS}'",
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT


def test_demo_common_root_points_at_checkout_not_ops_npa() -> None:
    result = _bash(f"source '{LIB}/demo-common.sh' && demo_common_root")
    assert result.returncode == 0, result.stderr
    root = Path(result.stdout.strip())
    assert root == REPO_ROOT
    assert (root / "npa" / "pyproject.toml").is_file()


def test_npa_read_lines_bash32_compatible() -> None:
    result = _bash(
        f"source '{LIB}/operator-config.sh' && "
        "npa_read_lines lines printf '%s\\n' one two three && "
        'printf "%s\\n" "${lines[@]}"',
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines() == ["one", "two", "three"]


def test_run_demo_script_exists_and_is_executable() -> None:
    run_demo = OPS / "run-demo.sh"
    assert run_demo.is_file()
    assert os.access(run_demo, os.X_OK)


def test_run_local_demo_aliases_run_demo() -> None:
    text = (OPS / "run-local-demo.sh").read_text()
    assert "run-demo.sh" in text


@pytest.mark.parametrize(
    "script",
    [
        "submit-k8s-staged-job.sh",
        "prestage-offline-run.sh",
        "setup-local-operator.sh",
        "monitor-k8s-job.sh",
    ],
)
def test_scripts_use_npa_repo_root(script: str) -> None:
    content = (OPS / script).read_text()
    assert "npa_repo_root" in content


def test_trigger_pipeline_script_exists() -> None:
    path = OPS / "trigger-pipeline.sh"
    assert path.is_file()
    assert os.access(path, os.X_OK)
    text = path.read_text()
    assert "TRIGGER_DATASET_URI" in text
    assert "trigger_preflight_s3" in text


def test_submit_passes_trigger_dataset_uri() -> None:
    content = (OPS / "submit-k8s-staged-job.sh").read_text()
    assert "NPA_SIM2REAL_TRIGGER_DATASET_URI" in content
    assert "--trigger-dataset-uri" in content


def test_submit_workbench_script_exists() -> None:
    script = OPS / "submit-workbench-job.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK)
    text = script.read_text()
    assert "npa workbench workflow submit" in text
    assert "operator_parse_submit_run_id" in (LIB / "operator-config.sh").read_text()


def test_operator_normalizes_staged_run_id() -> None:
    content = (LIB / "operator-config.sh").read_text()
    assert "operator_normalize_staged_run_id" in content
    assert "operator_parse_submit_run_id" in content
    assert "us-central1" in content


def test_operator_exports_kubeconfig_helper() -> None:
    content = (LIB / "operator-config.sh").read_text()
    assert "operator_export_kubeconfig" in content
    assert "operator_find_nebius_cli" in content
