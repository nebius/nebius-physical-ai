"""The baked NPA interpreter and source identity survive SkyPilot setup."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
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


def runtime_producer():
    return (
        DOCKERFILE.read_text()
        .split("\nFROM scratch AS runtime", 1)[0]
        .rsplit("FROM ", 1)[1]
    )


def test_clean_root_recipe_retains_measured_cuda_and_launch_configuration():
    text = DOCKERFILE.read_text()
    final = text.split("\nFROM scratch AS runtime\n", 1)[1]
    instructions = re.sub(r"\\\n\s*", " ", final).splitlines()
    assert [line for line in instructions if line.startswith("COPY ")] == [
        "COPY --from=runtime-producer / /"
    ]
    assert not any(line.startswith(("RUN ", "ADD ")) for line in instructions)
    raw = DOCKERFILE.with_name("clean-root-config.json").read_bytes()
    expected = json.loads(raw)
    payload = json.loads(RUNTIME_PAYLOAD.read_text())
    assert payload["clean_root_config_sha256"] == hashlib.sha256(raw).hexdigest()
    environment = next(line for line in instructions if line.startswith("ENV "))
    assert shlex.split(environment)[1:] == expected["config"]["Env"]
    for name, key in [("USER", "User"), ("WORKDIR", "WorkingDir")]:
        assert f"{name} {expected['config'][key]}" in instructions
    for name, key in [("SHELL", "Shell"), ("ENTRYPOINT", "Entrypoint"), ("CMD", "Cmd")]:
        value = next(
            line[len(name) + 1 :]
            for line in instructions
            if line.startswith(name + " ")
        )
        assert json.loads(value) == expected["config"][key]
    health = next(line for line in instructions if line.startswith("HEALTHCHECK "))
    assert "--interval=30s --timeout=5s --retries=3 CMD " in health
    assert (
        json.loads(health.split(" CMD ", 1)[1])
        == expected["config"]["Healthcheck"]["Test"][1:]
    )
    assert "EXPOSE 8080" in instructions
    assert expected["layer_count"] == 1
    assert expected["optional_empty_metadata_layer"] == {
        "blob_sha256": "4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1",
        "blob_bytes": 32,
        "diff_id": "sha256:" + hashlib.sha256(b"\x00" * 1024).hexdigest(),
    }
    assert expected["platform"] == {"os": "linux", "architecture": "amd64"}


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
    text = runtime_producer()
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
    assert binding["package_databases"] == [
        row
        for row in source["dpkg_databases"]
        if row["sha256"] == source["final_dpkg_database_sha256"]
    ]
    assert len(binding["package_databases"]) == 1
    for selected, original in zip(
        binding["copyright_files"], source["notice_files"], strict=True
    ):
        assert {
            key: value for key, value in selected.items() if key != "ancestor_versions"
        } == {
            key: value for key, value in original.items() if key != "ancestor_versions"
        }
        assert selected["ancestor_versions"] == [
            {"sha256": selected["sha256"], "size": selected["size"]}
        ]
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
    assert "AS secure-pip-builder" in text
    assert "COPY docker/workbench/common/secure_pip" in text
    assert "--work-dir /opt/secure-pip-work --output-dir /opt/secure-pip-wheels" in text
    final = runtime_producer()
    assert "pip-bootstrap.lock" not in final
    assert 'hashlib.sha256(w.read_bytes()).hexdigest() == r["sha256"]' in final
    assert "pip-26.2.1+npa.1-py3-none-any.whl" in final
    instructions = re.sub(r"\\\n\s*", " ", final).splitlines()
    apt = next(line for line in instructions if line.startswith("RUN --mount="))
    for boundary in (
        "apt-get install",
        "python3.12 /opt/install_security_apt.py",
        "rm /usr/share/python-wheels/pip-24.0-py3-none-any.whl",
        "install -m 0444 /opt/secure-pip-wheels/pip-26.2.1+npa.1-py3-none-any.whl",
        "python3.12 -m venv /opt/npa-venv",
        'assert pip.__version__ == "26.2.1+npa.1"',
    ):
        assert boundary in apt
    assert apt.index("rm /usr/share/python-wheels/") < apt.index("python3.12 -m venv")
    assert 'assert urllib3.__version__ == "2.8.0"' in apt
    assert 'assert msgpack.__version__ == "1.2.1"' in apt
    assert "COPY --from=secure-pip-builder --chmod=0444" in final
    assert "pip install --no-cache-dir --upgrade 'pip==" not in text


def test_full_setuptools_seed_has_its_own_fixed_wheel_lock():
    seed_lock = DOCKERFILE.with_name("setuptools-seed.lock").read_text()
    lock_lines = [line for line in seed_lock.splitlines() if not line.startswith("#")]
    digest = "51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670"
    assert lock_lines == ["setuptools==84.0.0 \\", f"    --hash=sha256:{digest}"]
    runtime_lock = DOCKERFILE.with_name("requirements.lock").read_text()
    assert f"setuptools==84.0.0 \\\n    --hash=sha256:{digest}" in runtime_lock
    text = DOCKERFILE.read_text()
    builder = text.split("AS runtime-producer", 1)[0]
    final = runtime_producer()
    assert (
        "COPY docker/workbench/curobo/setuptools-seed.lock /opt/setuptools-seed.lock"
    ) in builder
    assert "python -m pip --isolated download --no-cache-dir --no-deps" in builder
    assert "--only-binary=:all: --require-hashes" in builder
    assert "--index-url https://pypi.org/simple" in builder
    assert (
        "--requirement /opt/setuptools-seed.lock --dest /opt/setuptools-seed" in builder
    )
    # The full vulnerable package must not be reused as a seed just because its
    # separate pkg_resources-only donor remains a reviewed installer input.
    manifest = json.loads(
        (DOCKERFILE.parent.parent / "common/secure_pip/inputs.json").read_text()
    )
    donor = next(row for row in manifest["vendors"] if row["name"] == "setuptools")
    assert donor["filename"] == "setuptools-80.9.0-py3-none-any.whl"
    assert donor["filename"] not in final
    assert donor["sha256"] not in final


def test_full_seed_replacement_occurs_before_the_apt_layer_is_committed():
    final = runtime_producer()
    instructions = re.sub(r"\\\n\s*", " ", final).splitlines()
    apt = next(line for line in instructions if line.startswith("RUN --mount="))
    wheel = "setuptools-84.0.0-py3-none-any.whl"
    digest = "51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670"
    mount = f"source=/opt/setuptools-seed/{wheel},target=/opt/{wheel},ro"
    verified = f"{digest}  /opt/{wheel}"
    old_removed = "rm /usr/share/python-wheels/pip-24.0-py3-none-any.whl"
    installed = f"install -m 0444 /opt/{wheel} /usr/share/python-wheels/"
    assert mount in apt
    assert apt.index(verified) < apt.index(old_removed) < apt.index(installed)
    assert apt.index(installed) < apt.index("python3.12 -m venv /opt/npa-venv")
    assert "setuptools-68.1.2-py3-none-any.whl" in apt
    assert final.count(installed) == 1


def test_runtime_http_dependency_retains_fixed_streaming_and_proxy_boundary():
    requirements = DOCKERFILE.with_name("requirements.in").read_text()
    lock = DOCKERFILE.with_name("requirements.lock").read_text()
    # GHSA-vxq7-64xx-v4gw, GHSA-gh4c-6fx4-qh6g and GHSA-8988-9cw3-xx77:
    # preserve upstream streaming/proxy fixes at main's exact safe runtime pin.
    assert "urllib3==2.8.0" in requirements.splitlines()
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
