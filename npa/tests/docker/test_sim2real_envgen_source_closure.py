"""Keep EnvGen's source and dependency closure independent of its parent."""

from pathlib import Path
import subprocess
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker/workbench/sim2real-envgen/Dockerfile"
REQUIREMENTS = ROOT / "docker/workbench/common/sim2real-envgen-requirements.txt"
README = ROOT / "docker/workbench/sim2real-envgen/README.md"
CONTAINER_CATALOG = ROOT.parent / "docs/workbench/container-image-catalog.md"
GOLDEN_EVALS = ROOT / "src/npa/smoke/golden_evals.yaml"
PACKAGING_CONTRACT = ROOT / "docker/workbench/packaging-contract.yaml"
PYPROJECT = ROOT / "pyproject.toml"
INHERITED_FILES = (
    "src/npa/orchestration/skypilot/capacity.py",
    "workflows/stale/catalog.yaml",
    "venv/runtime-marker",
    "legacy-unrelated-byte",
)


def _cleanup_command() -> str:
    text = DOCKERFILE.read_text().replace("\\\n", "")
    commands = [
        line.removeprefix("RUN ")
        for line in text.splitlines()
        if line.startswith("RUN test -d /opt/npa")
    ]
    assert len(commands) == 1
    assert "test ! -L /opt/npa" in commands[0]
    assert "rm -rf /opt/npa" in commands[0]
    assert "rm -rf /opt/npa/src" not in commands[0]
    assert text.index("RUN test -d /opt/npa") < text.index("COPY --chown")
    assert text.index("RUN test -d /opt/npa") < text.index("FROM scratch")
    return commands[0]


def _write(path: Path, content: str = "inherited") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _run_cleanup(root: Path) -> subprocess.CompletedProcess[str]:
    # Substitute only the fixture-owned absolute prefix, not shell syntax.
    command = _cleanup_command().replace("/opt/npa", str(root))
    return subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True)


def test_cleanup_removes_the_complete_inherited_npa_tree_before_copy(tmp_path):
    root = tmp_path / "npa"
    for relative in INHERITED_FILES:
        _write(root / relative)
    _write(tmp_path / "outside-marker", "preserved outside")

    result = _run_cleanup(root)

    assert result.returncode == 0, result.stderr
    assert not root.exists()
    assert (tmp_path / "outside-marker").read_text() == "preserved outside"

    _write(root / "src/npa/current.py", "current source")
    _write(root / "workflows/current.yaml", "current catalog")
    assert sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    ) == ["src/npa/current.py", "workflows/current.yaml"]


@pytest.mark.parametrize("kind", ["missing", "regular_file", "symlink"])
def test_invalid_source_root_refuses_before_deleting_external_bytes(tmp_path, kind):
    root = tmp_path / "npa"
    outside = tmp_path / "external"
    _write(outside / "keep", "untouched")
    if kind == "regular_file":
        root.write_text("not a directory")
    elif kind == "symlink":
        root.symlink_to(outside, target_is_directory=True)

    result = _run_cleanup(root)

    assert result.returncode != 0
    assert (outside / "keep").read_text() == "untouched"


def test_owned_tree_with_leaf_symlink_is_removed_without_following_it(tmp_path):
    root = tmp_path / "npa"
    root.mkdir()
    outside = tmp_path / "external"
    _write(outside / "keep", "untouched")
    (root / "legacy-link").symlink_to(outside, target_is_directory=True)

    result = _run_cleanup(root)

    assert result.returncode == 0, result.stderr
    assert not root.exists()
    assert (outside / "keep").read_text() == "untouched"


def test_envgen_lock_is_secure_complete_and_excludes_unneeded_tool_stacks():
    lines = [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all(line.count("==") == 1 for line in lines)
    expected = {
        "pillow==13.0.0",
        "fsspec==2026.6.0",
        "cryptography==50.0.2",
        "PyJWT==2.15.1",
        "certifi==2026.10.23",
    }
    assert expected <= set(lines)
    lowered = {line.partition("==")[0].lower() for line in lines}
    assert lowered.isdisjoint({"datasets", "lerobot", "genesis-world", "torch"})


def test_envgen_lock_satisfies_every_direct_npa_runtime_requirement():
    """The role lock is a reviewed closure, not a hand-picked security subset."""
    locked = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        assert separator and name and version
        locked[canonicalize_name(name)] = Version(version)

    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    direct = [Requirement(value) for value in project["project"]["dependencies"]]
    missing = [
        requirement.name
        for requirement in direct
        if canonicalize_name(requirement.name) not in locked
    ]
    assert not missing
    incompatible = [
        f"{requirement.name}=={locked[canonicalize_name(requirement.name)]}"
        for requirement in direct
        if locked[canonicalize_name(requirement.name)] not in requirement.specifier
    ]
    assert not incompatible


def test_envgen_runtime_uses_its_own_python312_venv_and_exact_source_pth():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ENVGEN_VENV=/opt/npa/envgen-venv" in text
    assert "ARG ENVGEN_PYTHON_VERSION=3.12.14" in text
    assert (
        "ghcr.io/astral-sh/uv:0.12.5@sha256:"
        "e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1" in text
    )
    assert 'uv python install "${ENVGEN_PYTHON_VERSION}"' in text
    assert 'uv venv --python "${ENVGEN_PYTHON_VERSION}" "${ENVGEN_VENV}"' in text
    assert "sys.version_info[:3] == (3, 12, 14)" in text
    assert "UV_PYTHON_INSTALL_DIR=/opt/npa/python" in text
    assert "UV_NO_MODIFY_PATH=1" in text
    assert "rm -f /usr/local/bin/uv /usr/local/bin/uvx" in text
    assert "test ! -e /usr/local/bin/uv && test ! -e /usr/local/bin/uvx" in text
    assert "test ! -e /root/.local/bin/python3.12" in text
    assert "sim2real-envgen-requirements.txt" in text
    assert "npa-exact-source.pth" in text
    assert 'env -u PYTHONPATH "${ENVGEN_VENV}/bin/python" -c' in text
    assert "NPA_BAKED_PYTHON=/opt/npa/envgen-venv/bin/python" in text
    assert "installed.isdisjoint(absent)" in text
    assert "NPA_GENESIS_HOME" not in text
    assert "/opt/genesis/venv" not in text


def test_envgen_documented_python_version_matches_its_runtime_contract():
    assert "Resolver-closed Python 3.12 runtime" in REQUIREMENTS.read_text(
        encoding="utf-8"
    )
    assert "Python 3.12 venv" in README.read_text(encoding="utf-8")
    envgen_catalog_row = next(
        line
        for line in CONTAINER_CATALOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Sim2Real EnvGen 0.1.2 |")
    )
    assert (
        "source-attested Python 3.12 NPA-only scenario-record runtime"
        in envgen_catalog_row
    )
    assert "A dedicated Python 3.12 EnvGen venv" in GOLDEN_EVALS.read_text(
        encoding="utf-8"
    )
    packaging_contract = PACKAGING_CONTRACT.read_text(encoding="utf-8")
    assert "creates a dedicated Python 3.12 EnvGen venv" in packaging_contract
    assert "role does not use that closure and must be independently qualified" in (
        packaging_contract
    )
    assert "source Genesis or Envgen image is qualified or publishable" not in (
        packaging_contract
    )
