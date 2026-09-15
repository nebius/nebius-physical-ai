"""Exercise generated bootstrap package checks with real Bash and dpkg queries."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from ncore_publication import bootstrap  # noqa: E402

PACKAGES = ("curl", "patch", "openssh-server", "rsync")
APT = "APT_ACQUIRE_TIMEOUT=300\nprintf 'upstream-apt\\n'\n"
SSH = "printf 'upstream-ssh\\n'; "
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


@pytest.fixture
def apt_script(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    template = "prefix_cmd() { printf '%s' sudo; }\n"
    template += "".join("                " + line for line in APT.splitlines(keepends=True))
    template += "                {% if k8s_enable_docker_all %}\n"
    ssh = SSH + "$(prefix_cmd) mkdir -p /var/run/sshd; exit 99"
    sources = {"kubernetes-ray.yml.j2": template, "instance.py": f"install_ssh_k8s_cmd = {ssh!r}\n"}
    records = []
    for name, contents in sources.items():
        raw = contents.encode()
        (upstream / name).write_bytes(raw)
        records.append({"url": "https://example.invalid/" + name, "sha256": hashlib.sha256(raw).hexdigest()})
    lock = tmp_path / "npa/docker/workbench/ncore/base-source-lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"bootstrap": {"upstream": records}}))
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    return bootstrap._apt_script(upstream)


def _package_record(name, status="install ok installed"):
    return (
        f"Package: {name}\nStatus: {status}\nPriority: optional\nSection: misc\n"
        "Installed-Size: 42\nMaintainer: Test <test@example.invalid>\n"
        "Architecture: amd64\nVersion: 1.0-1\n"
        "Description: Synthetic package inventory for deterministic pipe overflow\n\n"
    )


@pytest.fixture
def package_database(tmp_path):
    for name in ("bash", "dpkg", "dpkg-query", "grep"):
        if shutil.which(name, path=ENVIRONMENT["PATH"]) is None:
            pytest.skip(f"requires the real Debian {name} executable")
    database = tmp_path / "dpkg"
    database.mkdir()
    (database / "status").write_text("".join(_package_record(name) for name in PACKAGES))
    return database


@pytest.fixture
def overflowing_database(package_database):
    # More than 1 MiB of ordinary dpkg output keeps the producer writing after
    # grep finds curl, even when Bash's pipe has the usual maximum capacity.
    with (package_database / "status").open("a") as stream:
        for index in range(12000):
            stream.write(_package_record(f"zz-test-package-{index:05d}"))
    return package_database


def _bash(script, database):
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script, "bootstrap-packages", str(database)],
        cwd=ROOT, env=ENVIRONMENT, capture_output=True, text=True, check=False,
    )


def _bootstrap_probe(apt_script, database, query_override=""):
    # Only the upstream setup and privileged integrity checks are synthetic;
    # package queries use the real parser against an isolated dpkg database.
    setup = """set -euo pipefail
database=$1
dpkg() { command dpkg --admindir="$database" "$@"; }
dpkg-query() { command dpkg-query --admindir="$database" "$@"; }
sudo() {
  case "$*" in
    '-n apt-get check') printf 'apt-get-check\\n' ;;
    '-n dpkg --audit') printf 'dpkg-audit\\n' ;;
    *) return 97 ;;
  esac
}
"""
    return _bash(setup + query_override + apt_script + "printf 'bootstrap-ready\\n'\n", database)


def test_legacy_package_pipeline_fails_after_successful_match(overflowing_database):
    inventory = _bash('set -euo pipefail\ndpkg --admindir="$1" -l\n', overflowing_database)
    assert inventory.returncode == 0, inventory.stderr
    assert len(inventory.stdout.encode()) > 1024 * 1024
    for package in PACKAGES:
        assert any(line.startswith(f"ii  {package} ") for line in inventory.stdout.splitlines())
    script = """set -euo pipefail
trap 'printf "pipeline=%s statuses=%s\\n" "$?" "${PIPESTATUS[*]}"' ERR
dpkg --admindir="$1" -l | grep -q "^ii  curl "
printf 'unexpected-success\\n'
"""
    result = _bash(script, overflowing_database)
    assert result.returncode == 141
    assert result.stdout == "pipeline=141 statuses=141 0\n"
    assert result.stderr == ""


def test_generated_bootstrap_accepts_installed_packages_in_large_inventory(apt_script, overflowing_database):
    assert APT in apt_script and SSH in apt_script
    result = _bootstrap_probe(apt_script, overflowing_database)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "upstream-apt", "upstream-ssh", "apt-get-check", "dpkg-audit", "bootstrap-ready",
    ]


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("status", [
    None, "deinstall ok config-files", "install ok not-installed", "install ok unpacked",
    "install ok half-installed", "install ok half-configured", "install ok triggers-awaited",
    "install ok triggers-pending", "install reinstreq installed",
])
def test_generated_bootstrap_rejects_each_missing_or_unconfigured_package(
    apt_script, package_database, package, status,
):
    records = [_package_record(name) for name in PACKAGES if name != package]
    if status is not None:
        records.append(_package_record(package, status))
    (package_database / "status").write_text("".join(records))
    result = _bootstrap_probe(apt_script, package_database)
    assert result.returncode != 0
    assert result.stdout.splitlines() == ["upstream-apt", "upstream-ssh"]


@pytest.mark.parametrize("package", PACKAGES)
def test_generated_bootstrap_rejects_query_failure_after_installed_output(apt_script, package_database, package):
    query = f"""dpkg-query() {{
  command dpkg-query --admindir="$database" "$@"
  if [ "${{@: -1}}" = "{package}" ]; then
    printf 'query-failed-after-output\\n' >&2
    return 23
  fi
}}
"""
    result = _bootstrap_probe(apt_script, package_database, query)
    assert result.returncode == 23
    assert result.stdout.splitlines() == ["upstream-apt", "upstream-ssh"]
    assert result.stderr == "query-failed-after-output\n"


def test_generated_bootstrap_rejects_real_query_database_error(apt_script, package_database):
    (package_database / "status").write_text("invalid dpkg status record\n")
    result = _bootstrap_probe(apt_script, package_database)
    assert result.returncode == 2
    assert "parsing file" in result.stderr
    assert result.stdout.splitlines() == ["upstream-apt", "upstream-ssh"]
