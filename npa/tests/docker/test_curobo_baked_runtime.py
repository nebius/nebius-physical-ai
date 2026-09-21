"""The baked NPA interpreter and source identity survive SkyPilot setup."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


DOCKERFILE = (
    Path(__file__).resolve().parents[3] / "npa/docker/workbench/curobo/Dockerfile"
)
PIP_BOOTSTRAP = DOCKERFILE.with_name("pip-bootstrap.lock")
RUNTIME_IMPORT_CHECK = DOCKERFILE.with_name("verify_runtime_imports.py")
RUNTIME_PAYLOAD = DOCKERFILE.with_name("runtime-payload.json")
SKYPILOT_CORE_BOOTSTRAP_PACKAGES = (
    "curl",
    "fuse",
    "gcc",
    "netcat-openbsd",
    "openssh-server",
    "patch",
    "pciutils",
    "rsync",
    "wget",
)
IMPORT_SPEC = importlib.util.spec_from_file_location(
    "curobo_verify_runtime_imports", RUNTIME_IMPORT_CHECK
)
assert IMPORT_SPEC and IMPORT_SPEC.loader
IMPORT_CHECK = importlib.util.module_from_spec(IMPORT_SPEC)
IMPORT_SPEC.loader.exec_module(IMPORT_CHECK)


def test_baked_identity_uses_checked_build_input_and_absolute_interpreter():
    text = DOCKERFILE.read_text()
    assert "ARG NPA_SOURCE_SHA" in text
    assert "ARG UBUNTU_SNAPSHOT=20260920T000000Z" in text
    assert "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" in text
    assert "archive.ubuntu.com" not in text
    assert "security.ubuntu.com" not in text
    assert "/etc/apt/sources.list.d/*.sources" in text
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in text
    assert "NPA_BAKED_PYTHON=/opt/npa-venv/bin/python" in text
    assert "PYTHONPATH=" not in text
    assert text.index('RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]') < text.index(
        "RUN pip install"
    )


def test_bakes_proved_skypilot_core_bootstrap_closure():
    text = DOCKERFILE.read_text()
    install_layer = text.split("apt-get install -y --no-install-recommends", 1)[
        1
    ].split("&& dpkg-query -W", 1)[0]
    inventory_check = text.split("&& dpkg-query -W", 1)[1].split("&& rm -f", 1)[0]

    # Reuse the SkyPilot 0.12.2 core closure proved by the OpenPI image contract.
    for package in SKYPILOT_CORE_BOOTSTRAP_PACKAGES:
        assert package in install_layer.split()
        assert package in inventory_check.split()
    assert "fuse3" not in install_layer.split()


def test_pip_bootstrap_distribution_is_content_pinned():
    text = DOCKERFILE.read_text()
    assert (
        PIP_BOOTSTRAP.read_text()
        == "# PyPI wheel: https://files.pythonhosted.org/packages/f3/6e/"
        "1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/"
        "pip-26.2.1-py3-none-any.whl\n"
        "pip==26.2.1 \\\n"
        "    --hash=sha256:71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e\n"
    )
    assert "COPY docker/workbench/curobo/pip-bootstrap.lock" in text
    assert "--require-hashes -r /opt/pip-bootstrap.lock" in text
    assert "pip install --no-cache-dir --upgrade 'pip==" not in text


def test_image_build_imports_pinocchio_upstream_boundary_after_pinned_sources():
    text = DOCKERFILE.read_text()
    invocation = "python /opt/verify_runtime_imports.py"
    assert "libgomp1=14.2.0-4ubuntu2~24.04.1" in text
    assert (
        "COPY docker/workbench/curobo/verify_runtime_imports.py "
        "/opt/verify_runtime_imports.py"
    ) in text
    assert text.index("codeload.github.com/NVlabs/curobo") < text.index(invocation)
    assert text.index("codeload.github.com/fishbotics/robometrics") < text.index(
        invocation
    )
    assert text.index("pip install --no-deps") < text.index(invocation)
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "MPLCONFIGDIR=/tmp/npa-curobo-import-cache/matplotlib" in text


def _install_dataset_modules(monkeypatch, *, motion_rows=800, mpinets_rows=1800):
    package = ModuleType("robometrics")
    package.__path__ = []
    datasets = ModuleType("robometrics.datasets")
    datasets.motion_benchmaker_raw = lambda: {
        f"motion-{index}": range(motion_rows // 8) for index in range(8)
    }
    datasets.mpinets_raw = lambda: {
        f"mpinets-{index}": range(mpinets_rows // 12) for index in range(12)
    }
    package.datasets = datasets
    monkeypatch.setitem(sys.modules, "robometrics", package)
    monkeypatch.setitem(sys.modules, "robometrics.datasets", datasets)


def test_runtime_import_check_records_exact_real_boundary_receipt(
    monkeypatch, tmp_path
):
    from npa.workbench.curobo import runner

    calls = []
    monkeypatch.setattr(runner, "_benchmark_module", lambda: calls.append("imported"))
    _install_dataset_modules(monkeypatch)
    receipt_path = tmp_path / "runtime-import.json"
    monkeypatch.setattr(IMPORT_CHECK, "_RECEIPT_PATH", receipt_path)

    IMPORT_CHECK.main()

    payload = receipt_path.read_bytes()
    contract = json.loads(RUNTIME_PAYLOAD.read_text())["runtime_import_receipt"]
    assert calls == ["imported"]
    assert len(payload) == contract["size"]
    assert hashlib.sha256(payload).hexdigest() == contract["sha256"]
    assert json.loads(payload)["datasets"] == {
        "motion_benchmaker": {"groups": 8, "rows": 800},
        "mpinets": {"groups": 12, "rows": 1800},
    }


def test_runtime_import_check_rejects_dataset_population_drift(monkeypatch, tmp_path):
    from npa.workbench.curobo import runner

    monkeypatch.setattr(runner, "_benchmark_module", lambda: object())
    _install_dataset_modules(monkeypatch, motion_rows=792)
    receipt_path = tmp_path / "runtime-import.json"
    monkeypatch.setattr(IMPORT_CHECK, "_RECEIPT_PATH", receipt_path)

    with pytest.raises(RuntimeError, match="population changed"):
        IMPORT_CHECK.main()
    assert not receipt_path.exists()


@pytest.mark.parametrize("sha", ["", "a" * 39, "a" * 41, "g" * 40, "a" * 40])
def test_docker_source_gate_executes_and_rejects_nonfull_sha(sha):
    instructions = DOCKERFILE.read_text().replace("\\\n", " ").splitlines()
    instruction = next(line for line in instructions if line.startswith("RUN [[ "))
    result = subprocess.run(
        ["/bin/bash", "-c", instruction.removeprefix("RUN ")],
        env={
            "NPA_SOURCE_SHA": sha,
            "SOURCE_DATE_EPOCH": "1700000000",
            "UBUNTU_SNAPSHOT": "20260920T000000Z",
        },
        check=False,
        capture_output=True,
    )
    assert (result.returncode == 0) is (sha == "a" * 40)


@pytest.mark.parametrize(
    "snapshot", ["", "20260919T000000Z", "20260920T000000Z", "latest"]
)
def test_docker_source_gate_rejects_unreviewed_snapshot(snapshot):
    instructions = DOCKERFILE.read_text().replace("\\\n", " ").splitlines()
    instruction = next(line for line in instructions if line.startswith("RUN [[ "))
    result = subprocess.run(
        ["/bin/bash", "-c", instruction.removeprefix("RUN ")],
        env={
            "NPA_SOURCE_SHA": "a" * 40,
            "SOURCE_DATE_EPOCH": "1700000000",
            "UBUNTU_SNAPSHOT": snapshot,
        },
        check=False,
        capture_output=True,
    )
    assert (result.returncode == 0) is (snapshot == "20260920T000000Z")
