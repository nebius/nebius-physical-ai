"""Focused runtime identity and CUDA-diagnostic regressions for Isaac Arena."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.isaac_arena import runtime
from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.identity import (
    ISAAC_ARENA_ARCHIVE_SHA256,
    ISAAC_ARENA_REVISION,
    ISAAC_ARENA_VERSION,
    SOURCE_IDENTITY_SCHEMA,
)
from npa.workbench.isaac_arena.runtime_identity import assert_runtime_identity


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "isaac-arena"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "isaaclab_arena"\nversion = "{ISAAC_ARENA_VERSION}"\n',
        encoding="utf-8",
    )
    (root / ".npa-source-identity.json").write_text(
        json.dumps(
            {
                "schema": SOURCE_IDENTITY_SCHEMA,
                "version": ISAAC_ARENA_VERSION,
                "revision": ISAAC_ARENA_REVISION,
                "archive_sha256": ISAAC_ARENA_ARCHIVE_SHA256,
            }
        ),
        encoding="utf-8",
    )
    return root


def _identity_environment(root: Path) -> dict[str, str]:
    return {
        "NPA_LIGHT_WORKBENCH_TOOL": "isaac-arena",
        "ISAAC_ARENA_ROOT": str(root),
        "ISAAC_ARENA_VERSION": ISAAC_ARENA_VERSION,
        "ISAAC_ARENA_REVISION": ISAAC_ARENA_REVISION,
    }


def test_runtime_identity_asserts_source_revision_and_sdk_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_root(tmp_path)
    monkeypatch.setattr(
        "npa.workbench.isaac_arena.runtime_identity.metadata.version",
        lambda name: "1.0.3" if name == "lightwheel-sdk" else "unexpected",
    )
    observed = assert_runtime_identity(
        root=root, environment=_identity_environment(root)
    )
    assert observed["asserted"] is True
    assert observed["arena_revision"] == ISAAC_ARENA_REVISION
    assert observed["lightwheel_sdk_version"] == "1.0.3"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ISAAC_ARENA_REVISION", "0" * 40, "ISAAC_ARENA_REVISION"),
        ("ISAAC_ARENA_VERSION", "9.9.9", "ISAAC_ARENA_VERSION"),
    ],
)
def test_runtime_identity_rejects_image_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    root = _source_root(tmp_path)
    monkeypatch.setattr(
        "npa.workbench.isaac_arena.runtime_identity.metadata.version",
        lambda _name: "1.0.3",
    )
    environment = _identity_environment(root)
    environment[field] = value
    with pytest.raises(IsaacArenaError, match=message):
        assert_runtime_identity(root=root, environment=environment)


def test_runtime_identity_rejects_lightwheel_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_root(tmp_path)
    monkeypatch.setattr(
        "npa.workbench.isaac_arena.runtime_identity.metadata.version",
        lambda _name: "1.0.4",
    )
    with pytest.raises(IsaacArenaError, match="Lightwheel SDK version"):
        assert_runtime_identity(root=root, environment=_identity_environment(root))


def test_gpu_info_treats_absent_optional_torch_as_unavailable(monkeypatch) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("No module named torch", name="torch")

    monkeypatch.setattr(runtime, "import_module", missing)
    assert runtime._gpu_info() == {
        "available": False,
        "device_name": "",
        "compute_capability": [],
    }


def test_gpu_info_reports_real_cuda_query_failure(monkeypatch) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _index: (_ for _ in ()).throw(RuntimeError("driver failed")),
        get_device_capability=lambda _index: (10, 0),
    )
    monkeypatch.setattr(runtime, "import_module", lambda _name: SimpleNamespace(cuda=cuda))
    with pytest.raises(IsaacArenaError, match="CUDA driver/device query failed"):
        runtime._gpu_info()


def test_gpu_info_reports_broken_torch_dependency(monkeypatch) -> None:
    def broken(_name: str):
        raise ModuleNotFoundError("No module named typing_extensions", name="typing_extensions")

    monkeypatch.setattr(runtime, "import_module", broken)
    with pytest.raises(IsaacArenaError, match="dependency is missing"):
        runtime._gpu_info()
