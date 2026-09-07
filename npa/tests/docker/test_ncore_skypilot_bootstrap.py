"""Real dpkg semantics and the pinned SkyPilot bootstrap filesystem boundary."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


PACKAGING = Path(__file__).parents[2] / "docker/workbench/ncore"


@pytest.fixture
def base_sources():
    spec = importlib.util.spec_from_file_location(
        "ncore_bootstrap_sources", PACKAGING / "base_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_file_selection_is_not_falsely_reported_installed(
    base_sources, tmp_path
):
    """SkyPilot's real dpkg/grep loop must request the missing full packages."""
    dpkg = shutil.which("dpkg")
    if dpkg is None:
        pytest.skip("requires the real Debian dpkg executable")
    base_sources.assemble_package_state(tmp_path)
    state = tmp_path / "var/lib/dpkg"
    # A selected curl ELF is a loose file, not a configured Debian curl package.
    # No fabricated Package/Version/Status or synthetic per-package lists.
    assert (state / "status").read_bytes() == b""
    assert list((state / "info").iterdir()) == []
    result = subprocess.run(
        [dpkg, f"--admindir={state}", "-l"], capture_output=True, check=True
    )
    assert result.stdout == b""
    assert result.stderr == b""
    result = subprocess.run(
        [
            "bash",
            "-c",
            '"$1" --admindir="$2" -l | grep -q "^ii  curl "',
            "probe",
            dpkg,
            str(state),
        ],
        capture_output=True,
    )
    assert result.returncode == 1
    result = subprocess.run(
        [dpkg, f"--admindir={state}", "--audit"], capture_output=True, check=True
    )
    assert result.stdout == b""
    assert result.stderr == b""


def test_bootstrap_apt_policy_uses_real_apt_configuration(base_sources, tmp_path):
    apt_config = shutil.which("apt-config")
    if apt_config is None:
        pytest.skip("requires the real Debian APT configuration parser")
    base_sources.assemble_package_state(tmp_path)
    config = tmp_path / "etc/apt/apt.conf.d/90npa-bootstrap"
    result = subprocess.run(
        [
            apt_config,
            "-c",
            str(config),
            "shell",
            "RECOMMENDS",
            "APT::Install-Recommends/b",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert result.stdout == "RECOMMENDS='false'\n"


def test_bootstrap_lock_covers_package_manager_helpers_and_interpreters():
    lock = json.loads((PACKAGING / "base-source-lock.json").read_text())
    files = {f["path"] for p in lock["debian_binaries"] for f in p["files"]}
    assert {
        "usr/bin/apt",
        "usr/bin/apt-get",
        "usr/bin/apt-key",
        "usr/bin/gpgv",
        "usr/bin/curl",
        "usr/bin/dpkg",
        "usr/bin/dpkg-query",
        "usr/bin/dpkg-deb",
        "usr/bin/dpkg-split",
        "usr/bin/dpkg-trigger",
        "usr/bin/dpkg-divert",
        "usr/bin/update-alternatives",
        "usr/bin/perl",
        "usr/bin/seq",
        "usr/bin/whoami",
        "usr/sbin/ldconfig",
        "usr/lib/apt/methods/https",
        "usr/lib/apt/methods/gpgv",
        "usr/share/dpkg/tupletable",
    } <= files
    # Runtime fetch stays separate from Debian bootstrap.
    assert lock["python_wheels"] == []
    assert lock["python_distributions"] == []
    assert lock["base_diff_ids"] == []


def test_actual_pinned_skypilot_apt_startup(base_sources):
    """Opt-in network/filesystem probe; use a disposable copy of the final root.

    NPA_NCORE_BOOTSTRAP_ROOT must have runtime /dev/null, /dev/urandom and DNS,
    ubuntu/sudo, and a standard policy-rc.d that refuses automatic daemon starts.
    NPA_NCORE_SKYPILOT_SOURCE points to downloaded v0.12.2 template/provisioner
    files. No cloud, image build, service start, wheels or runtime are involved.
    APT changes this disposable root; NEVER run against the publication root.
    """
    import ast
    import os
    import textwrap

    root_value = os.environ.get("NPA_NCORE_BOOTSTRAP_ROOT")
    source_value = os.environ.get("NPA_NCORE_SKYPILOT_SOURCE")
    if not root_value or not source_value:
        pytest.skip("requires an explicit disposable root and pinned SkyPilot sources")
    root, source = Path(root_value).resolve(), Path(source_value).resolve()
    assert root != Path("/")
    assert (root / "var/lib/dpkg/status").read_bytes() == b""
    assert (root / "usr/sbin/policy-rc.d").read_text() == "#!/bin/sh\nexit 101\n"
    lock = json.loads((PACKAGING / "base-source-lock.json").read_text())
    upstream = {}
    for item in lock["bootstrap"]["upstream"]:
        name = item["url"].rsplit("/", 1)[1]
        raw = (source / name).read_bytes()
        assert base_sources.digest(raw) == item["sha256"]
        upstream[name] = raw.decode()
    # Exercise all selected Debian bytes, not a host package manager fallback.
    for package in lock["debian_binaries"]:
        for item in package["files"]:
            path = root / item["path"]
            if "link" in item:
                assert path.is_symlink() and os.readlink(path) == item["link"]
            else:
                assert base_sources.file_digest(path) == item["sha256"]

    def run(script):
        command = [
            "chroot",
            "--userspec=1000:1000",
            str(root),
            "/usr/bin/env",
            "-i",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME=/home/ubuntu",
            "LANG=C",
            "LC_ALL=C",
            "/bin/bash",
            "-ec",
            script,
        ]
        if os.geteuid() != 0:
            command = ["sudo", "-n", *command]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    assert run("dpkg --print-architecture").strip() == "amd64"
    assert "libcurl/" in run("curl --version")
    template = upstream["kubernetes-ray.yml.j2"]
    prefix = next(
        line.strip()
        for line in template.splitlines()
        if line.strip().startswith("prefix_cmd() {")
    )
    start = template.index("                APT_ACQUIRE_TIMEOUT=")
    end = template.index("                {% if k8s_enable_docker_all", start)
    # Execute the original functions, package queries and installs verbatim.
    # Upstream's own retries are preserved; this test adds no execution limits.
    apt_step = textwrap.dedent(template[start:end])
    assert "{{" not in apt_step and "{%" not in apt_step
    run(prefix + "\n" + apt_step)
    assert "Fetched" in (root / "tmp/apt-update.log").read_text()

    # The provisioner always runs apt install even after pod initialization.
    tree = ast.parse(upstream["instance.py"])
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "install_ssh_k8s_cmd"
            for target in node.targets
        )
    )
    ssh_command = ast.literal_eval(assignment.value)
    apt_only, _ = ssh_command.split("$(prefix_cmd) mkdir -p /var/run/sshd;", 1)
    run(apt_only)
    # These are the template's actual readiness predicates, with real dpkg state.
    for package in ("curl", "patch", "openssh-server", "rsync"):
        run(f'dpkg -l | grep -q "^ii  {package} "')
        assert list((root / "var/lib/dpkg/info").glob(package + ".list"))
    run('dpkg -l | grep -q "^ii  \\(netcat\\|netcat-openbsd\\|netcat-traditional\\) "')
    run("sudo -n apt-get check")
    assert run("sudo -n dpkg --audit").strip() == ""


@pytest.mark.parametrize("metadata", ["status", "info/curl.list", "triggers/File"])
def test_publication_refuses_fabricated_or_inherited_package_state(
    base_sources, tmp_path, metadata
):
    lock = json.loads((PACKAGING / "base-source-lock.json").read_text())
    files = {"var/lib/dpkg/status": {"sha256": base_sources.digest(b"")}}
    files["var/lib/dpkg/" + metadata] = {
        "sha256": base_sources.digest(b"builder state")
    }
    inventory = {
        "layers": [
            {
                "diff_id": "",
                "files": files,
                "debian_packages": [],
                "python_packages": [],
            }
        ],
        "final_files": files,
    }
    with pytest.raises(ValueError, match="dpkg"):
        base_sources.verify_coverage(lock, inventory, tmp_path)
