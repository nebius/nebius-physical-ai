"""Check reproducible CI dependency pins and their refresh contract."""

from pathlib import Path
import shutil
import sys

from packaging.requirements import Requirement
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ci_requirements  # noqa: E402


def _copy_inputs(root: Path) -> None:
    for name in ("pyproject.toml", "ci/constraints.in", "ci/requirements.txt"):
        target = root / "npa" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ci_requirements._ROOT / "npa" / name, target)


def test_committed_ci_pins_match_dependency_inputs() -> None:
    """Require contributors to refresh pins after CI dependency changes.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: The dependency fingerprint is stale.
    """
    ci_requirements._check(ci_requirements._ROOT)


@pytest.mark.parametrize("name", ["pyproject.toml", "ci/constraints.in"])
def test_dependency_changes_require_refresh(tmp_path: Path, name: str) -> None:
    """Reject stale pins when core requirements or the CPU constraint change.

    Args:
        tmp_path: Isolated input repository.
        name: Dependency source to change.
    Returns:
        None.
    Raises:
        AssertionError: An outdated lock passes validation.
    """
    _copy_inputs(tmp_path)
    path = tmp_path / "npa" / name
    source = path.read_text()
    path.write_text(
        source.replace("typer>=0.24", "typer>=0.25")
        if name == "pyproject.toml"
        else source + "onnx==1.22.0\n"
    )
    with pytest.raises(ValueError, match="CI dependencies changed"):
        ci_requirements._check(tmp_path)


def test_non_dependency_changes_do_not_require_refresh(tmp_path: Path) -> None:
    """Allow unrelated package metadata and image configuration edits.

    Args:
        tmp_path: Isolated input repository.
    Returns:
        None.
    Raises:
        ValueError: Unrelated metadata unexpectedly invalidates the pins.
    """
    _copy_inputs(tmp_path)
    path = tmp_path / "npa/pyproject.toml"
    path.write_text(path.read_text().replace('version = "0.1.0"', 'version = "0.1.1"'))
    ci_requirements._check(tmp_path)


@pytest.mark.parametrize("version", ["3.10", "3.12", "3.14"])
def test_ci_pins_select_one_cpu_runtime_per_interpreter(version: str) -> None:
    """Resolve marker-qualified pins without admitting a CUDA dependency.

    Args:
        version: Supported CI interpreter.
    Returns:
        None.
    Raises:
        AssertionError: Pins overlap or select a different checkpoint runtime.
    """
    lines = (ci_requirements._ROOT / "npa/ci/requirements.txt").read_text().splitlines()
    requirements = [
        Requirement(line) for line in lines if line and not line.startswith("#")
    ]
    environment = {
        "python_version": version,
        "python_full_version": version + ".0",
        "sys_platform": "linux",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
    }
    active = [
        item
        for item in requirements
        if not item.marker or item.marker.evaluate(environment)
    ]
    names = [item.name for item in active]
    assert len(names) == len(set(names))
    assert all(
        item.url is None and str(item.specifier).startswith("==") for item in active
    )
    assert not any(name.startswith(("nvidia-", "triton")) for name in names)
    assert (
        str(next(item for item in active if item.name == "torch").specifier)
        == "==2.13.0+cpu"
    )


def test_refresh_preserves_pins_unless_upgrade_requested(
    tmp_path: Path, monkeypatch
) -> None:
    """Keep routine dependency refreshes separate from bulk version upgrades.

    Args:
        tmp_path: Isolated input repository.
        monkeypatch: Replaces the external compiler.
    Returns:
        None.
    Raises:
        AssertionError: A normal refresh upgrades every dependency or omits CPU routing.
    """
    _copy_inputs(tmp_path)
    commands = []
    monkeypatch.setattr(
        ci_requirements.subprocess, "check_output", lambda *a, **k: "uv 0.12.5"
    )
    monkeypatch.setattr(
        ci_requirements.subprocess, "run", lambda args, **kwargs: commands.append(args)
    )
    monkeypatch.setattr(ci_requirements, "_validate_linux_wheels", lambda root: None)
    ci_requirements._update(tmp_path, upgrade=False)
    ci_requirements._update(tmp_path, upgrade=True)
    assert "--upgrade" not in commands[0]
    assert commands[1][-1] == "--upgrade"
    assert "--torch-backend" in commands[0] and "cpu" in commands[0]
    assert "--universal" in commands[0]
    ci_requirements._check(tmp_path)
