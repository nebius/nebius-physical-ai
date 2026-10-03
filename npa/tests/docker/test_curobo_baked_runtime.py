"""The baked NPA interpreter and source identity survive SkyPilot setup."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


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


def test_full_distro_source_and_notice_closure_is_baked_and_byte_bound():
    source_path = DOCKERFILE.with_name("distro-source-closure.json")
    raw = source_path.read_bytes()
    source = json.loads(raw)
    binding = json.loads(RUNTIME_PAYLOAD.read_text())["distro_source_closure"]
    assert source["schema_version"] == "npa.curobo.distro-source-closure.v1"
    assert binding["manifest"] == {
        "path": "usr/share/doc/npa-curobo/distro-source-closure.json",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert binding["package_databases"] == source["dpkg_databases"]
    assert binding["copyright_files"] == source["notice_files"]
    assert binding["final_dpkg_database_sha256"] == source["final_dpkg_database_sha256"]
    assert len(source["packages"]) == 250
    assert len(source["dpkg_databases"]) == 4
    assert len(source["notice_files"]) == 164
    assert len(source["sources"]) == 143
    sources = {(row["package"], row["version"]) for row in source["sources"]}
    vendor = {
        (row["package"], row["version"]) for row in source["vendor_source_exceptions"]
    }
    assert len(sources) == 143 and len(vendor) == 2 and not sources & vendor
    assert {
        (row["source"], row["source_version"]) for row in source["packages"]
    } == sources | vendor
    for row in source["sources"]:
        assert row["artifacts"]
        assert any(item["filename"].endswith(".dsc") for item in row["artifacts"])
        for artifact in row["artifacts"]:
            assert artifact["url"].startswith("https://snapshot.ubuntu.com/ubuntu/")
            assert artifact["url"].endswith("/" + artifact["filename"])
            assert artifact["bytes"] > 0
            assert len(bytes.fromhex(artifact["sha256"])) == 32
    notices = {row["path"]: row for row in source["notice_files"]}
    assert len(notices) == 164
    for row in source["vendor_source_exceptions"]:
        assert row["notice_path"] in notices
    for row in notices.values():
        assert {"size": row["size"], "sha256": row["sha256"]} in row[
            "ancestor_versions"
        ]
    dockerfile = DOCKERFILE.read_text()
    assert (
        "COPY --chmod=0444 docker/workbench/curobo/distro-source-closure.json "
        "/usr/share/doc/npa-curobo/distro-source-closure.json"
    ) in dockerfile
    assert "GPL-2.0-or-later AND GPL-3.0-or-later AND LGPL-2.1-or-later" in dockerfile
    assert "(GPL-3.0-or-later WITH GCC-exception-3.1)" in dockerfile


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


def test_runtime_http_dependency_retains_fixed_streaming_and_proxy_boundary():
    requirements = DOCKERFILE.with_name("requirements.in").read_text()
    lock = DOCKERFILE.with_name("requirements.lock").read_text()
    # GHSA-vxq7-64xx-v4gw, GHSA-gh4c-6fx4-qh6g and GHSA-8988-9cw3-xx77:
    # preserve upstream streaming/proxy fixes without changing planner pins.
    assert "urllib3>=2.8.0,<3" in requirements.splitlines()
    assert "urllib3==2.8.0 \\\n" in lock
    assert (
        "--hash=sha256:0cf3cae568d36aa9576b28dfb35f11328f1cb974ca7647d9475ebb86c75ac6e3"
        in lock
    )
    assert (
        "--hash=sha256:63bf2ead4c879426ebf22ef2a781eeb4aa3b4ae798a0435506f8687fd5bb9b63"
        in lock
    )


def test_runtime_lock_satisfies_current_npa_base_dependencies():
    project = DOCKERFILE.parents[3] / "pyproject.toml"
    dependencies = tomllib.loads(project.read_text())["project"]["dependencies"]
    lock = DOCKERFILE.with_name("requirements.lock").read_text()
    pins = {
        canonicalize_name(name): version
        for name, version in re.findall(
            r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", lock, re.MULTILINE
        )
    }
    environment = default_environment() | {
        "python_version": "3.12",
        "python_full_version": "3.12.0",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "extra": "",
    }
    for value in dependencies:
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        assert name in pins, f"image lock omits NPA dependency {requirement}"
        assert pins[name] in requirement.specifier, (
            f"image lock {name}=={pins[name]} conflicts with {requirement}"
        )


def test_final_image_checks_all_dependencies_after_installing_npa():
    text = DOCKERFILE.read_text()
    installed = text.index(
        "RUN pip install --no-deps --no-build-isolation --no-cache-dir /opt/npa-src"
    )
    checked = text.index("&& pip check", installed)
    assert installed < checked < text.index("python /opt/verify_runtime_imports.py")


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
