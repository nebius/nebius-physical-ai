"""Tests for the runtime Isaac bootstrap.

The four Isaac workbench images ship no NVIDIA Isaac bytes; Isaac Sim and Isaac Lab are
fetched on first run from ``pypi.nvidia.com`` with EULA acceptance enabled by default.
Two properties of that mechanism are load-bearing and must not regress:

1. **Default and opt-out behavior.** An unset value defaults to NVIDIA's documented
   ``ACCEPT_EULA=Y`` so workflows remain non-interactive. Recognized negative values
   opt out, invalid values fail distinctly, and the bootstrap must download nothing.
2. **Concurrency safety.** Eight GPUs per node means up to eight pods racing one cache
   volume. A partially-written cache would be an extremely unpleasant bug to debug on a
   customer's cluster.

These run offline: ``pip`` and ``git`` are replaced with fakes on ``PATH``, so the tests
exercise the real script's control flow without touching the network. They assert the
fakes were *not* invoked where the script must refuse, which is stronger than asserting
an exit code alone.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON = REPO_ROOT / "npa" / "docker" / "workbench" / "common"
BOOTSTRAP = COMMON / "isaac_bootstrap.sh"
SHIM = COMMON / "isaac_python.sh"
BASE_INSTALLER = COMMON / "install_isaac_runtime_base.sh"
NPA_CLI = COMMON.parent / "isaac-lab" / "npa_cli.sh"
WHEELS = COMMON / "isaac-nvidia-wheels.txt"
OSS_DEPS = COMMON / "isaac-oss-deps.txt"

EX_CONFIG = 78
EX_UNAVAILABLE = 69

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to exercise the bootstrap"
)


# --------------------------------------------------------------------------------------
# Harness: a fake python/pip/git so the real script runs end to end with no network.
# --------------------------------------------------------------------------------------


ISAAC_LAB_SRC_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"


class Harness:
    """A sandbox that simulates the image side of the bootstrap without any network.

    The fake ``python3.11`` stands in for the image's baked interpreter. It is a real
    simulator rather than a stub: it reports a controlled ``purelib``, emulates ``.pth``
    processing so the script's "can the cache venv see the image's torch?" check is a
    genuine test of the layering, and its ``pip`` materialises dist-info for whatever the
    requirements file names, so the script's post-install verification is genuinely
    exercised too. A fake ``git`` fabricates the Isaac Lab source layout.
    """

    def __init__(
        self, tmp_path: Path, *, pip_fails: bool = False, pip_delay: int = 0
    ) -> None:
        self.root = tmp_path
        self.cache = tmp_path / "cache"
        self.bin = tmp_path / "bin"
        self.base_site = tmp_path / "image-site-packages"
        self.calls = tmp_path / "calls.log"
        for directory in (self.cache, self.bin, self.base_site):
            directory.mkdir(parents=True, exist_ok=True)

        # A stand-in for the image's baked torch, so the layering check has something
        # real to find via the .pth the bootstrap writes.
        (self.base_site / "torch").mkdir(exist_ok=True)
        (self.base_site / "torch" / "__init__.py").write_text(
            '__version__ = "2.9.0+cu128"\n', encoding="utf-8"
        )

        self._write(
            self.bin / "python3.11",
            f"""#!/usr/bin/env bash
echo "python3.11 $*" >> {self.calls}
real={sys.executable}
self_dir="$(cd "$(dirname "$0")" && pwd)"
# A venv copy of this script lives at <venv>/bin/python, so its purelib is its own
# site-packages; the harness copy on PATH reports the image's site-packages.
if [ -d "$self_dir/../lib/python3.11/site-packages" ]; then
  purelib="$(cd "$self_dir/../lib/python3.11/site-packages" && pwd)"
else
  purelib="{self.base_site}"
fi

case "${{1:-}}" in
  -m)
    case "${{2:-}}" in
      venv)
        target="${{@: -1}}"
        mkdir -p "$target/bin" "$target/lib/python3.11/site-packages"
        cp "{self.bin}/python3.11" "$target/bin/python"
        chmod +x "$target/bin/python"
        exit 0
        ;;
      pip)
        echo "PIP $*" >> {self.calls}
        {"sleep " + str(pip_delay) if pip_delay else "true"}
        if [ "{1 if pip_fails else 0}" = "1" ]; then exit 1; fi
        # Materialise dist-info for every pinned requirement, so the script's own
        # post-install verification runs for real instead of being skipped.
        reqfile=""
        prev=""
        for arg in "$@"; do
          [ "$prev" = "-r" ] && reqfile="$arg"
          prev="$arg"
        done
        if [ -n "$reqfile" ] && [ -f "$reqfile" ]; then
          while IFS= read -r line; do
            case "$line" in \\#*|"") continue ;; esac
            case "$line" in *==*) ;; *) continue ;; esac
            name="${{line%%==*}}"
            rest="${{line#*==}}"
            version="${{rest%% *}}"
            version="${{version%%\\\\*}}"
            module="$(echo "$name" | tr '.-' '__')"
            mkdir -p "$purelib/${{module}}"
            : > "$purelib/${{module}}/__init__.py"
            dist="$purelib/${{name//-/_}}-${{version}}.dist-info"
            mkdir -p "$dist"
            printf 'Metadata-Version: 2.1\\nName: %s\\nVersion: %s\\n' "$name" "$version" \\
              > "$dist/METADATA"
            printf 'Wheel-Version: 1.0\\n' > "$dist/WHEEL"
          done < "$reqfile"
        fi
        exit 0
        ;;
    esac
    ;;
  -c)
    # The bootstrap asks for purelib; answer from this interpreter's identity.
    case "${{2:-}}" in
      *purelib*) echo "$purelib"; exit 0 ;;
    esac
    ;;
esac

# Emulate site's .pth handling so `import torch` in the cache venv only succeeds if the
# bootstrap actually wrote the layering file. This is the point of the check.
extra=""
[ -f "$purelib/_npa_image_site.pth" ] && extra="$(cat "$purelib/_npa_image_site.pth")"
PYTHONPATH="$purelib${{extra:+:$extra}}${{PYTHONPATH:+:$PYTHONPATH}}" exec "$real" "$@"
""",
        )
        # A fake git that fabricates the Isaac Lab source layout at the pinned commit.
        self._write(
            self.bin / "git",
            f"""#!/usr/bin/env bash
echo "GIT $*" >> {self.calls}
for arg in "$@"; do
  case "$arg" in
    clone) mode=clone ;;
    rev-parse) mode=revparse ;;
  esac
done
case "${{mode:-}}" in
  clone)
    target="${{@: -1}}"
    mkdir -p "$target/.git" \\
             "$target/scripts/reinforcement_learning/rsl_rl" \\
             "$target/source" "$target/apps"
    : > "$target/scripts/reinforcement_learning/rsl_rl/train.py"
    exit 0
    ;;
  revparse) echo "{ISAAC_LAB_SRC_COMMIT}"; exit 0 ;;
esac
exit 0
""",
        )

    @staticmethod
    def _write(path: Path, body: str) -> None:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)

    def env(self, **overrides: str) -> dict[str, str]:
        env = {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "HOME": str(self.root),
            "NPA_ISAAC_CACHE_DIR": str(self.cache),
            "NPA_ISAAC_BASE_PYTHON": str(self.bin / "python3.11"),
            "NPA_ISAAC_WHEELS_FILE": str(WHEELS),
        }
        env.update(overrides)
        return env

    def run(self, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(BOOTSTRAP), *args],
            env=self.env(**overrides),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    @property
    def call_log(self) -> str:
        return self.calls.read_text(encoding="utf-8") if self.calls.exists() else ""

    def downloaded_anything(self) -> bool:
        return "PIP " in self.call_log or "GIT " in self.call_log

    def stamp_dir(self) -> Path:
        (trees,) = list((self.cache / "v").glob("*")) or [None]  # type: ignore[list-item]
        assert trees is not None, "no cache tree was created"
        return trees

    def fake_ready_tree(self) -> Path:
        """Produce a cache tree the fast path will accept, without running an install."""
        result = self.run("status")
        expected = next(
            line.split("=", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("expected_tree=")
        )
        tree = Path(expected)
        (tree / "venv" / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy(self.bin / "python3.11", tree / "venv" / "bin" / "python")
        (tree / "venv" / "bin" / "python").chmod(0o755)
        (tree / ".complete").touch()
        return tree


# --------------------------------------------------------------------------------------
# Default acceptance and explicit opt-out
# --------------------------------------------------------------------------------------


def test_bootstrap_accepts_and_downloads_when_eula_is_unset(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run("ensure")

    assert result.returncode == 0, result.stderr
    assert harness.downloaded_anything()
    assert result.stdout.strip()


def test_refusal_links_the_terms_the_operator_is_accepting(tmp_path: Path) -> None:
    result = Harness(tmp_path).run("ensure", ACCEPT_EULA="")
    assert "nvidia.com" in result.stderr
    assert "Omniverse" in result.stderr and "Isaac Sim" in result.stderr


@pytest.mark.parametrize("value", ["Y", "YES", "yes", "y", "1", "true"])
def test_bootstrap_migrates_affirmative_values(tmp_path: Path, value: str) -> None:
    harness = Harness(tmp_path)
    result = harness.run("ensure", ACCEPT_EULA=value)
    assert result.returncode == 0, result.stderr
    assert harness.downloaded_anything()


@pytest.mark.parametrize("value", ["no", "N", "0", "false", "", "  "])
def test_bootstrap_preserves_recognized_opt_out(tmp_path: Path, value: str) -> None:
    harness = Harness(tmp_path)
    result = harness.run("ensure", ACCEPT_EULA=value)
    assert result.returncode == EX_CONFIG
    assert "explicitly disabled" in result.stderr
    assert not harness.downloaded_anything()


def test_bootstrap_rejects_unrecognized_value_distinctly(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run("ensure", ACCEPT_EULA="maybe")
    assert result.returncode == EX_CONFIG
    assert "invalid ACCEPT_EULA value" in result.stderr
    assert not harness.downloaded_anything()


def test_status_reports_default_acceptance_and_uses_no_network(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run("status")

    assert result.returncode == 0, result.stderr
    assert "eula_accepted=yes" in result.stdout
    assert "eula_state=accepted" in result.stdout
    assert "ready=no" in result.stdout
    assert "isaacsim=" in result.stdout and "isaaclab=" in result.stdout
    assert not harness.downloaded_anything()


@pytest.mark.parametrize("value", ["Y", "YES", "yes", "1", "TRUE", "true"])
def test_status_uses_ensure_parser_for_affirmative_values(
    tmp_path: Path, value: str
) -> None:
    result = Harness(tmp_path).run("status", ACCEPT_EULA=value)

    assert result.returncode == 0, result.stderr
    assert "eula_state=accepted" in result.stdout
    assert "eula_accepted=yes" in result.stdout


@pytest.mark.parametrize("value", ["", "N", "no", "0", "FALSE", "false"])
def test_status_uses_ensure_parser_for_opt_out_values(
    tmp_path: Path, value: str
) -> None:
    result = Harness(tmp_path).run("status", ACCEPT_EULA=value)

    assert result.returncode == 0, result.stderr
    assert "eula_state=opt-out" in result.stdout
    assert "eula_accepted=no" in result.stdout


def test_status_reports_invalid_acceptance_without_network(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run("status", ACCEPT_EULA="maybe")

    assert result.returncode == 0, result.stderr
    assert "eula_state=invalid" in result.stdout
    assert "eula_accepted=no" in result.stdout
    assert not harness.downloaded_anything()


# --------------------------------------------------------------------------------------
# Idempotency, atomicity, and the offline / read-only postures
# --------------------------------------------------------------------------------------


def test_ensure_is_idempotent_and_makes_no_calls_when_warm(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run("ensure", ACCEPT_EULA="Y")
    assert first.returncode == 0, first.stderr

    harness.calls.unlink()
    second = harness.run("ensure", ACCEPT_EULA="Y")

    assert second.returncode == 0, second.stderr
    assert second.stdout == first.stdout, "the same cache tree must be reused"
    assert not harness.downloaded_anything(), "a warm cache must not re-download"


def test_warm_cache_honors_explicit_opt_out(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    harness.fake_ready_tree()
    result = harness.run("ensure", ACCEPT_EULA="")
    assert result.returncode == EX_CONFIG
    assert not harness.downloaded_anything()


def test_offline_mode_refuses_rather_than_reaching_the_network(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run(
        "ensure",
        NPA_ISAAC_BOOTSTRAP_OFFLINE="1",
        ACCEPT_EULA="Y",
    )
    assert result.returncode == EX_UNAVAILABLE
    assert "warm" in result.stderr
    assert not harness.downloaded_anything()


def test_offline_mode_succeeds_against_a_warm_cache(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    tree = harness.fake_ready_tree()
    result = harness.run("ensure", NPA_ISAAC_BOOTSTRAP_OFFLINE="1", ACCEPT_EULA="Y")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tree)


def test_readonly_mode_never_attempts_a_write(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run(
        "ensure",
        NPA_ISAAC_CACHE_READONLY="1",
        ACCEPT_EULA="Y",
    )
    assert result.returncode == EX_UNAVAILABLE
    assert not list((harness.cache / "v").glob("*.tmp.*"))
    assert not harness.downloaded_anything()


def test_a_failed_install_publishes_nothing(tmp_path: Path) -> None:
    """Fail loudly rather than leaving a half-installed cache for the next pod."""
    harness = Harness(tmp_path, pip_fails=True)
    result = harness.run("ensure", ACCEPT_EULA="Y")

    assert result.returncode != 0
    assert not (harness.cache / "current").exists(), (
        "current must not point at a failure"
    )
    assert not list((harness.cache / "v").glob("*/.complete")), (
        "no tree may be marked complete"
    )
    assert not list((harness.cache / "v").glob("*.tmp.*")), (
        "temp trees must be cleaned up"
    )


def test_manifest_records_what_was_installed(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run("ensure", ACCEPT_EULA="Y")
    assert result.returncode == 0, result.stderr

    manifest = json.loads((Path(result.stdout.strip()) / "MANIFEST.json").read_text())
    assert manifest["format"] == "npa_isaac_runtime_cache_v1"
    assert manifest["isaacsim_version"] == "5.1.0.0"
    assert manifest["isaaclab_version"] == "2.3.2.post1"
    assert manifest["index_url"] == "https://pypi.nvidia.com"
    assert len(manifest["cache_stamp"]) == 64
    assert Path(result.stdout.strip()).name == manifest["cache_stamp"]
    # The digest of the wheel manifest ties a cache tree to a reviewed pin set.
    assert len(manifest["wheels_file_sha256"]) == 64
    assert len(manifest["bootstrap_sha256"]) == 64
    assert len(manifest["oss_dependencies_sha256"]) == 64


def test_current_symlink_points_at_the_completed_tree(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    result = harness.run("ensure", ACCEPT_EULA="Y")
    assert result.returncode == 0, result.stderr
    current = harness.cache / "current"
    assert current.is_symlink()
    assert current.resolve() == Path(result.stdout.strip()).resolve()


@pytest.mark.parametrize(
    "override",
    [
        {"ISAAC_SIM_VERSION": "6.0.1.0"},
        {"ISAAC_LAB_VERSION": "3.0.0"},
        {"NPA_ISAAC_LAB_SRC_COMMIT": "0" * 40},
        {"NPA_ISAAC_INDEX_URL": "https://mirror.example.com/simple"},
    ],
)
def test_changing_a_pin_changes_the_cache_stamp(tmp_path: Path, override: dict) -> None:
    """A pin change must build a NEW tree, never mutate one a running pod is using."""
    harness = Harness(tmp_path)

    def stamp(**env: str) -> str:
        result = harness.run("status", **env)
        return next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("expected_tree=")
        )

    assert stamp() != stamp(**override), override


def test_changing_bootstrap_source_changes_the_cache_stamp(tmp_path: Path) -> None:
    """A controller cannot reuse a closure warmed by older bootstrap logic."""

    harness = Harness(tmp_path / "harness")
    copied_common = tmp_path / "common"
    shutil.copytree(COMMON, copied_common)
    copied_bootstrap = copied_common / "isaac_bootstrap.sh"

    def stamp() -> str:
        result = subprocess.run(
            ["bash", str(copied_bootstrap), "status"],
            env=harness.env(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("expected_tree=")
        )

    before = stamp()
    copied_bootstrap.write_text(
        copied_bootstrap.read_text(encoding="utf-8") + "\n# cache contract v2\n",
        encoding="utf-8",
    )
    assert before != stamp()


# --------------------------------------------------------------------------------------
# Concurrency: up to 8 pods per GPU node race one cache volume
# --------------------------------------------------------------------------------------


def test_eight_concurrent_installs_produce_one_tree(tmp_path: Path) -> None:
    # A slow "install" so the other seven genuinely contend for the lock rather than
    # each finding a warm cache and making the test vacuous.
    harness = Harness(tmp_path, pip_delay=3)
    env = harness.env(ACCEPT_EULA="Y")
    processes = [
        subprocess.Popen(  # noqa: S603 - fixed argv, test-local
            ["bash", str(BOOTSTRAP), "ensure"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    results = [process.communicate(timeout=600) for process in processes]
    codes = [process.returncode for process in processes]

    assert codes == [0] * 8, [err for _, err in results]
    trees = {out.strip() for out, _ in results}
    assert len(trees) == 1, f"pods disagreed about the cache tree: {trees}"
    assert len(list((harness.cache / "v").glob("*.tmp.*"))) == 0, "temp trees leaked"
    installs = sum("installing pinned NVIDIA" in err for _, err in results)
    assert installs == 1, f"expected exactly one installer, {installs} installed"


# --------------------------------------------------------------------------------------
# The pin manifests
# --------------------------------------------------------------------------------------


def test_every_nvidia_wheel_is_version_and_hash_pinned() -> None:
    """The runtime fetch is attack surface; nothing may be unpinned."""
    lines = [
        line.strip()
        for line in WHEELS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    requirements = [line for line in lines if not line.startswith("--hash")]
    hashes = [line for line in lines if line.startswith("--hash")]

    assert requirements, "no requirements found"
    assert len(requirements) == len(hashes), "every requirement needs exactly one hash"
    for requirement in requirements:
        assert "==" in requirement, f"unpinned requirement: {requirement}"
    for digest in hashes:
        assert digest.startswith("--hash=sha256:")
        assert len(digest.removeprefix("--hash=sha256:")) == 64


def test_wheel_manifest_matches_the_repo_pins() -> None:
    text = WHEELS.read_text(encoding="utf-8")
    assert "isaacsim==5.1.0.0" in text
    assert "isaaclab==2.3.2.post1" in text
    # isaacsim[all,extscache] + isaacsim-kernel + isaaclab
    assert text.count("--hash=sha256:") == 26


def test_wheel_manifest_is_fetched_only_from_nvidias_index() -> None:
    """--index-url, not --extra-index-url: the set must not be shadowable from PyPI.

    Checked against the script's instructions, not its prose — the surrounding comment
    explains the distinction and therefore names both spellings.
    """
    instructions = "\n".join(
        line
        for line in BOOTSTRAP.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "--index-url" in instructions
    assert "--extra-index-url" not in instructions
    assert "--require-hashes" in instructions
    assert "--no-deps" in instructions


def test_oss_deps_carry_no_nvidia_isaac_package() -> None:
    """The baked list must stay OSS-only; an Isaac wheel here would defeat the design."""
    for line in OSS_DEPS.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip().lower()
        if not requirement:
            continue
        assert not requirement.replace("_", "-").startswith(("isaacsim", "isaaclab")), (
            line
        )


@pytest.mark.parametrize(
    ("package", "why"),
    [
        (
            "scipy",
            "isaacsim.core.utils.numpy.rotations imports scipy.spatial.transform",
        ),
        (
            "matplotlib",
            "isaaclab.envs -> .ui -> widgets.image_plot imports matplotlib.cm",
        ),
        ("opencv-python-headless", "isaaclab_assets.sensors.gelsight imports cv2"),
        ("boto3", "omni.replicator.core imports botocore.exceptions"),
        ("coverage==7.13.5", "omni.kit.test.test_coverage imports coverage"),
    ],
)
def test_oss_deps_include_every_undeclared_isaac_dependency(
    package: str, why: str
) -> None:
    """Regression pins for dependencies nothing in the Isaac wheels declares.

    These used to arrive free with the nvcr.io base image, and each was found by running
    Isaac on a GPU and reading its startup diagnostics - not by reading requirements. Kit
    can log an ImportError during extension startup and carry on partially loaded, so these
    dependencies must remain explicit in the thin runtime.
    """
    assert package in OSS_DEPS.read_text(encoding="utf-8"), f"{package}: {why}"


def test_isaac_image_normalizes_bootstrap_scripts_for_the_non_root_user() -> None:
    dockerfile = (COMMON.parent / "isaac-lab" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert (
        "find /opt/npa/docker/workbench/common -type d -exec chmod 0755" in dockerfile
    )
    assert (
        "find /opt/npa/docker/workbench/common -type f -exec chmod 0644" in dockerfile
    )
    assert "chmod 0755 /opt/npa/docker/workbench/common/*.sh" in dockerfile


def test_base_installer_upgrades_linux_headers_from_the_fixed_snapshot() -> None:
    """Do not retain the stale kernel-header package inherited from CUDA.

    ``linux-libc-dev`` contains userspace development headers, not the cluster's
    running kernel.  Even so, the immutable Ubuntu snapshot carries a fixed build,
    so keeping the older inherited package would leave avoidable critical Trivy
    findings in every thin Isaac runtime.
    """

    installer = BASE_INSTALLER.read_text(encoding="utf-8")
    assert 'UBUNTU_SNAPSHOT="${NPA_UBUNTU_SNAPSHOT:-20260801T053000Z}"' in installer
    assert "linux-libc-dev=5.15.0-186.196" in installer


# --------------------------------------------------------------------------------------
# The shim
# --------------------------------------------------------------------------------------


def test_shim_and_bootstrap_are_valid_bash() -> None:
    for script in (BOOTSTRAP, SHIM, BASE_INSTALLER, NPA_CLI):
        subprocess.run(["bash", "-n", str(script)], check=True, timeout=60)


def test_isaac_image_installs_a_cli_for_skypilot_setup() -> None:
    dockerfile = (COMMON.parent / "isaac-lab" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    launcher = NPA_CLI.read_text(encoding="utf-8")

    assert "npa_cli.sh /usr/local/bin/npa" in dockerfile
    assert "env -u PYTHONPATH python3 -c 'import npa'" in dockerfile
    assert (
        'exec "${NPA_BAKED_PYTHON:-/opt/npa/sim/venv/bin/python}" -m npa "$@"'
        in launcher
    )


def test_readonly_runtime_redirects_kit_portable_state_to_scratch() -> None:
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    shim = SHIM.read_text(encoding="utf-8")
    expected = "--portable-root /tmp/npa-isaac-kit"

    assert expected in bootstrap
    assert "kit_args=os.environ.get(" in bootstrap
    assert expected in shim
    assert "export NPA_ISAAC_KIT_ARGS=" in shim
    for variable in (
        "OMNI_USER_DIR",
        "OMNI_LOG_DIR",
        "XDG_RUNTIME_DIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    ):
        assert f"export {variable}=" in shim


def test_shim_derives_internal_kit_acceptance_only_after_shared_parser() -> None:
    """The bootstrap subprocess cannot export into the launcher that starts Kit."""

    shim = SHIM.read_text(encoding="utf-8")
    assert 'ACCEPT_EULA="${ACCEPT_EULA-Y}"' in shim
    assert 'TREE="$("$BOOTSTRAP" ensure)"' in shim
    assert "export OMNI_KIT_ACCEPT_EULA=YES" in shim
    assert "PRIVACY_CONSENT" not in shim


def test_shim_defaults_internal_kit_acceptance_without_manual_env(
    tmp_path: Path,
) -> None:
    harness = Harness(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SHIM),
            "-c",
            "import os; print(os.environ.get('OMNI_KIT_ACCEPT_EULA', ''))",
        ],
        env=harness.env(NPA_ISAAC_BOOTSTRAP=str(BOOTSTRAP)),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "YES"


def test_shim_propagates_the_refusal_exit_code(tmp_path: Path) -> None:
    """`/isaac-sim/python.sh` must fail closed, not fall back to a system python."""
    harness = Harness(tmp_path)
    result = subprocess.run(
        ["bash", str(SHIM), "-c", "print('THIS MUST NOT RUN')"],
        env=harness.env(NPA_ISAAC_BOOTSTRAP=str(BOOTSTRAP), ACCEPT_EULA=""),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == EX_CONFIG
    assert "THIS MUST NOT RUN" not in result.stdout
    assert "ACCEPT_EULA=Y" in result.stderr


def test_shim_keeps_stdout_clean_for_callers(tmp_path: Path) -> None:
    """The SkyPilot Isaac templates read hydra overrides out of this interpreter's
    stdout in a `while read` loop, so bootstrap chatter must go to stderr only."""
    harness = Harness(tmp_path)
    harness.fake_ready_tree()
    result = subprocess.run(
        ["bash", str(SHIM), "--version"],
        env=harness.env(NPA_ISAAC_BOOTSTRAP=str(BOOTSTRAP), ACCEPT_EULA="Y"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "isaac-bootstrap:" not in result.stdout


def test_base_installer_proves_the_refusal_at_build_time() -> None:
    """The build must assert the absence of a baked install, not the presence of one."""
    text = BASE_INSTALLER.read_text(encoding="utf-8")
    assert "NPA_ISAAC_BOOTSTRAP_REFUSES_WITHOUT_EULA_OK" in text
    assert "NPA_NO_BAKED_ISAAC_OK" in text
    assert "-ne 78" in text, "the build must require the documented EX_CONFIG exit code"


def test_base_installer_never_snapshots_ssh_host_private_keys() -> None:
    """Package installation must not create one host identity for every image pull."""

    text = BASE_INSTALLER.read_text(encoding="utf-8")
    sentinel = "touch \\\n    /etc/ssh/ssh_host_rsa_key"
    cleanup = "rm -f /etc/ssh/ssh_host_*_key"
    install = "openssh-client openssh-server"
    assert sentinel in text
    assert cleanup in text
    assert text.index(sentinel) < text.index(install) < text.index(cleanup)


def test_base_installer_uses_immutable_system_and_bootstrap_inputs() -> None:
    """The GPU image build must not resolve Python or packaging from moving indexes."""
    text = BASE_INSTALLER.read_text(encoding="utf-8")
    assert "NPA_UBUNTU_SNAPSHOT:-20260801T053000Z" in text
    assert "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" in text
    assert "add-apt-repository" not in text
    assert "sha256sum --check --strict" in text
    assert text.count("3.11.15-1+jammy1") >= 9
    for requirement in (
        "pip==26.2.1",
        "setuptools==84.0.0",
        "wheel==0.47.0",
        "packaging==26.3",
    ):
        assert requirement in text
    assert "pip install --no-cache-dir --no-deps" in text
    assert '"$ISAAC_VENV/bin/python" -m pip check' in text
    assert "pip uninstall --yes wheel" in text

    closure = (COMMON / "isaac-oss-deps.txt").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in closure.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(lines) >= 120
    assert all(line.count("==") == 1 for line in lines)
    assert "packaging==23.2" in lines
    assert not any(line.lower().startswith("wheel==") for line in lines)
