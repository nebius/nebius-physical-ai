"""Neither LeRobot version may retain corrected dependency bytes in an ancestor."""

from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lerobot/Dockerfile"
RECIPE = DOCKERFILE.parent
CORRECTION = "/opt/lerobot/remove-scikit-image-recipe.py"
SYSTEM_PKCS12_TEMPLATE = (
    "/usr/share/cmake-3.22/Templates/Windows/Windows_TemporaryKey.pfx"
)
VENV_PKCS12_TEMPLATE = (
    "/opt/lerobot/venv/lib/python3.12/site-packages/"
    "cmake/data/share/cmake-4.1/Templates/Windows/Windows_TemporaryKey.pfx"
)
VENV_PKCS12_GUARD = (
    "find /opt/lerobot/venv \\( -iname '*.p12' -o -iname '*.pfx' \\) -print -quit"
)
SYSTEM_PKCS12_GUARD = (
    "find / -xdev \\( -iname '*.p12' -o -iname '*.pfx' \\) -print -quit"
)


def _instructions(text):
    return [
        (match.group(1), match.group(2))
        for line in text.replace("\\\n", " ").splitlines()
        if (match := re.match(r"^([A-Z]+)\s+(.*)$", line))
    ]


def _installation(text):
    instructions = _instructions(text)
    index, body = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and "/opt/lerobot/requirements-060.lock" in body
    )
    assert any(
        kind == "ENV" and "PIP_NO_CACHE_DIR=1" in value
        for kind, value in instructions[:index]
    )
    assert any(
        kind == "COPY"
        and "docker/workbench/curobo/remove_scikit_image_recipe.py" in value
        and CORRECTION in value
        for kind, value in instructions[:index]
    )
    assert any(
        kind == "COPY"
        and "bootstrap-requirements.lock" in value
        and "requirements-051.lock" in value
        and "requirements-060.lock" in value
        for kind, value in instructions[:index]
    )
    bootstrap, versioned_branches = body.split(
        '&& if [ "${LEROBOT_VERSION}" = "0.6.0" ]; then', 1
    )
    optional_branch, default_branch = versioned_branches.split(
        'elif [ "${LEROBOT_VERSION}" = "0.5.1" ]; then', 1
    )
    default_branch = default_branch.split("else", 1)[0]
    assert "prepare-secure-wheel-060.py" in optional_branch
    assert "requirements-060.lock" in optional_branch
    assert "requirements-051.lock" in default_branch
    assert "--require-hashes --no-deps" in optional_branch
    assert "--require-hashes --no-deps" in default_branch
    assert "--only-binary=:all:" in bootstrap
    assert "--only-binary=:all:" not in optional_branch
    assert "--only-binary=:all:" not in default_branch
    assert "--no-build-isolation" in optional_branch
    assert "--no-build-isolation" in default_branch
    assert "pip install --upgrade" not in body
    assert '"wandb>=0.30.0"' not in body
    assert "pip uninstall -y wandb" not in optional_branch
    assert "pip uninstall -y wandb" not in default_branch
    assert f"python {CORRECTION}" in optional_branch
    assert f"python {CORRECTION}" in default_branch
    assert (
        "--site-packages /opt/lerobot/venv/lib/python3.12/site-packages"
        in default_branch
    )
    assert (
        "--site-packages /opt/lerobot/venv/lib/python3.12/site-packages"
        in optional_branch
    )
    assert "--capability lerobot" in optional_branch
    assert "--capability lerobot" in default_branch
    assert "dependency-source-correction.json" in optional_branch
    assert "dependency-source-correction.json" in default_branch
    cleanup = re.compile(
        rf"test -f {re.escape(VENV_PKCS12_TEMPLATE)}\s+"
        rf"&& rm {re.escape(VENV_PKCS12_TEMPLATE)}\s+"
        rf"&& test ! -e {re.escape(VENV_PKCS12_TEMPLATE)}\s+"
        rf"&& test ! -L {re.escape(VENV_PKCS12_TEMPLATE)}\s+"
        rf"&& test -z \"\$\({re.escape(VENV_PKCS12_GUARD)}\)\""
    )
    match = cleanup.search(body)
    assert match is not None
    assert match.start() < body.index("/opt/lerobot/venv/bin/pip check")
    return body


def test_both_versions_correct_the_reviewed_source_in_the_original_install_layer():
    text = DOCKERFILE.read_text()
    _installation(text)
    assert "NPA_LEROBOT_INTEGRATION_PROFILE" not in text
    receipt_dir = "/usr/share/doc/npa-lerobot"
    assert f"install -d -m 0755 -o root -g root {receipt_dir}" in text
    assert f"{receipt_dir}/source-integration.json" in text
    assert f"{receipt_dir}/dependency-source-correction.json" in text
    assert "install -o root -g root -m 0644" in text


def test_root_owned_dependency_audit_precedes_later_source_copies():
    """Catch backend output outside the ubuntu-owned virtualenv before later COPYs."""

    instructions = _instructions(DOCKERFILE.read_text())
    dependency_index = next(
        index
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and "/opt/lerobot/requirements-060.lock" in body
    )
    root_user_index = next(
        index
        for index, (kind, body) in enumerate(
            instructions[dependency_index + 1 :], dependency_index + 1
        )
        if kind == "USER" and body == "root"
    )
    audit_kind, audit_body = instructions[root_user_index + 1]
    assert audit_kind == "RUN"
    assert audit_body == f'test -z "$({SYSTEM_PKCS12_GUARD})"'
    assert "rm " not in audit_body
    first_runtime_copy = next(
        index
        for index, (kind, body) in enumerate(
            instructions[root_user_index + 1 :], root_user_index + 1
        )
        if kind == "COPY" and "src/npa /opt/npa/src/npa" in body
    )
    assert dependency_index < root_user_index < root_user_index + 1 < first_runtime_copy


def test_bootstrap_sshd_is_baked_before_apt_hardening_and_has_no_host_keys():
    """Keep the non-root SkyPilot contract runnable after libc header hardening."""

    instructions = _instructions(DOCKERFILE.read_text())
    installer_copy_index = next(
        index
        for index, (kind, body) in enumerate(instructions)
        if kind == "COPY"
        and "install_immutable_system_packages.sh" in body
        and "/usr/local/bin/install-immutable-lerobot-system-packages" in body
    )
    installer_run_kind, installer_run = instructions[installer_copy_index + 1]
    assert installer_run_kind == "RUN"
    assert "/usr/local/bin/install-immutable-lerobot-system-packages" in installer_run
    prereq_index, prereq = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and "rm -f /etc/ssh/ssh_host_*" in body
    )
    hardening_index, hardening = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and "dpkg --purge --force-depends linux-libc-dev" in body
    )

    system_packages = (RECIPE / "install_immutable_system_packages.sh").read_text()
    assert "openssh-server=1:8.9p1-3ubuntu0.16" in system_packages
    assert "netcat-openbsd=1.218-4ubuntu1" in system_packages
    assert "python3=3.10.6-1~22.04.1" in system_packages
    assert "python3=3.10.4-0ubuntu2" not in system_packages
    assert "https://snapshot.ubuntu.com/ubuntu/20260820T000000Z" in system_packages
    assert (
        f'readonly CMAKE_WINDOWS_TEMPLATE="{SYSTEM_PKCS12_TEMPLATE}"' in system_packages
    )
    assert 'test -f "${CMAKE_WINDOWS_TEMPLATE}"' in system_packages
    assert 'rm "${CMAKE_WINDOWS_TEMPLATE}"' in system_packages
    assert 'test ! -e "${CMAKE_WINDOWS_TEMPLATE}"' in system_packages
    assert 'test ! -L "${CMAKE_WINDOWS_TEMPLATE}"' in system_packages
    assert f'test -z "$({SYSTEM_PKCS12_GUARD})"' in system_packages
    assert system_packages.index(
        "cmake=3.22.1-1ubuntu1.22.04.2"
    ) < system_packages.index('test -f "${CMAKE_WINDOWS_TEMPLATE}"')
    assert system_packages.index(
        "apt-get install -y --no-install-recommends /tmp/npa-python312/*.deb"
    ) < system_packages.index(f'test -z "$({SYSTEM_PKCS12_GUARD})"')
    assert "rm -f /etc/ssh/ssh_host_*" in system_packages
    assert (
        "test -z \"$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*'"
        in system_packages
    )
    assert "rm -f /etc/ssh/ssh_host_*" in prereq
    assert (
        "test -z \"$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' "
        '-print -quit)"'
    ) in prereq
    assert prereq_index < hardening_index
    assert "linux-libc-dev" in hardening


def test_shared_python_bootstraps_bind_the_same_exact_debian_artifacts():
    common = (
        ROOT / "npa/docker/workbench/common/install_python312_jammy.sh"
    ).read_text()
    lerobot = (RECIPE / "install_immutable_system_packages.sh").read_text()
    pattern = r"^((?:lib)?python3\.12[^\s|]*\.deb)[ |]([0-9a-f]{64})$"
    common_pins = re.findall(pattern, common, re.MULTILINE)
    lerobot_pins = re.findall(pattern, lerobot, re.MULTILINE)

    assert len(common_pins) == len(lerobot_pins) == 6
    assert set(common_pins) == set(lerobot_pins)
    assert all("_3.12.16-1+jammy1_" in name for name, _ in common_pins)
    assert "sha256sum --check --strict" in common
    assert "sha256sum --check --status" in lerobot


@pytest.mark.parametrize("patch_offset", [-1, 0, 1])
def test_system_python_guard_accepts_only_the_pinned_release(patch_offset):
    """Exercise the build's version check against its actual Debian package pin."""
    system_packages = (RECIPE / "install_immutable_system_packages.sh").read_text()
    pinned_version = re.search(
        r"^python3\.12_([0-9.]+)-[^_]+_amd64\.deb ", system_packages, re.MULTILINE
    ).group(1)
    major, minor, patch = map(int, pinned_version.split("."))
    installed_version = f"{major}.{minor}.{patch + patch_offset}"
    guard = next(
        line
        for line in system_packages.splitlines()
        if line.startswith('test "$(python3.12 --version)"')
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'installed_version="$1"\n'
            'python3.12() { printf "Python %s\\n" "$installed_version"; }\n' + guard,
            "version-guard",
            installed_version,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if patch_offset == 0 else 1), (
        f"build guard must accept only pinned Python {pinned_version}; "
        f"installed Python {installed_version} returned {result.returncode}"
    )


def test_build_dependency_closure_has_no_moving_repository_or_python_resolution():
    """Prevent a future build from reintroducing a moving bootstrap input."""

    text = DOCKERFILE.read_text()
    system_packages = (RECIPE / "install_immutable_system_packages.sh").read_text()
    assert "add-apt-repository" not in text
    assert "nodesource.com" not in text
    assert "apt-get install" not in text
    assert "pip install --upgrade" not in text
    assert "--require-hashes --no-deps" in text
    assert "python3.12_3.12.16-1+jammy1_amd64.deb" in system_packages
    assert "sha256sum --check --status" in system_packages
    assert "apt-get install -y --no-install-recommends" in system_packages
    bootstrap_lock = (RECIPE / "bootstrap-requirements.lock").read_text()
    assert (
        "packaging==25.0 "
        "--hash=sha256:29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484"
    ) in bootstrap_lock
    assert ".venv-local/" in (ROOT / "npa/.dockerignore").read_text()
    assert "/npa/.venv-local/" in (ROOT / ".gitignore").read_text()
    for name, local_wheel, wheel_hash in (
        (
            "requirements-051.lock",
            "lerobot-0.5.1+npa.secure4-py3-none-any.whl",
            "fce5dce533558d694c463e4a3c0cd5dbc8f9eb3dedbbd4ecd4719e5f3044fe7b",
        ),
        (
            "requirements-060.lock",
            "lerobot-0.6.0+npa.secure4-py3-none-any.whl",
            "e339d769b2501233de5d219caa52cc12adf0b543bada54903ab14c99324b5d2c",
        ),
    ):
        lock = (RECIPE / name).read_text()
        assert "file:///tmp" not in lock
        version = local_wheel.removeprefix("lerobot-").removesuffix("-py3-none-any.whl")
        assert f"lerobot=={version}" in lock
        assert "wandb==0.30.0" in lock
        assert "datasets==5.0.1" in lock
        assert "fsspec==2026.6.0" in lock
        assert "cmake==4.1.3" in lock
        assert f"--hash=sha256:{wheel_hash}" in lock
        assert "--hash=sha256:" in lock
    assert "--find-links /opt/lerobot/secure-wheel" in text


@pytest.mark.parametrize(
    ("lock_name", "requirements"),
    [
        (
            "requirements-051.lock",
            (
                "certifi==2026.10.19",
                "setuptools==84.2.1",
                "transformers==5.10.0",
            ),
        ),
        (
            "requirements-060.lock",
            (
                "certifi==2026.10.19",
                "diffusers==0.38.0",
                "pillow==13.0.0",
                "setuptools==84.2.1",
                "torch==2.13.0",
                "transformers==5.10.0",
            ),
        ),
    ],
)
def test_security_remediation_pins_are_hash_locked(lock_name, requirements):
    """Keep the CI-remediated dependency fixes immutable in both image closures."""

    lock = (RECIPE / lock_name).read_text().splitlines()
    for requirement in requirements:
        start = lock.index(requirement + " \\")
        hashes = []
        for line in lock[start + 1 :]:
            if not line.startswith("    --hash=sha256:"):
                break
            hashes.append(line)
        assert hashes


@pytest.mark.parametrize(
    "unsafe_change",
    ["later-layer", "pkcs12-later-layer", "wheel-cache", "unreviewed-lock"],
)
def test_unsafe_dependency_installation_is_rejected(unsafe_change):
    text = DOCKERFILE.read_text()
    if unsafe_change == "later-layer":
        text = text.replace(
            f"    && /opt/lerobot/venv/bin/python {CORRECTION}",
            f"\nRUN /opt/lerobot/venv/bin/python {CORRECTION}",
        )
    elif unsafe_change == "pkcs12-later-layer":
        cleanup = (
            f"    && test -f {VENV_PKCS12_TEMPLATE} \\\n"
            f"    && rm {VENV_PKCS12_TEMPLATE} \\\n"
            f"    && test ! -e {VENV_PKCS12_TEMPLATE} \\\n"
            f"    && test ! -L {VENV_PKCS12_TEMPLATE} \\\n"
            f'    && test -z "$({VENV_PKCS12_GUARD})" \\\n'
        )
        assert cleanup in text
        text = text.replace(
            cleanup,
            f"    && /opt/lerobot/venv/bin/pip check \\\n"
            f"\nRUN test -f {VENV_PKCS12_TEMPLATE} \\\n"
            f"    && rm {VENV_PKCS12_TEMPLATE} \\\n"
            f"    && test ! -e {VENV_PKCS12_TEMPLATE} \\\n"
            f"    && test ! -L {VENV_PKCS12_TEMPLATE} \\\n"
            f'    && test -z "$({VENV_PKCS12_GUARD})" \\\n'
            f"    && /opt/lerobot/venv/bin/pip check \\\n",
        )
    elif unsafe_change == "wheel-cache":
        text = text.replace("PIP_NO_CACHE_DIR=1", "PIP_NO_CACHE_DIR=0")
    else:
        text = text.replace("requirements-051.lock", "unreviewed-requirements.lock")
    with pytest.raises((AssertionError, ValueError)):
        _installation(text)
