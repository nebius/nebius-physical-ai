"""Exercise the read-only Isaac source inspection command."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from npa.workflows.behavior_challenge import simulator_source_inspect as inspect
from npa.workflows.behavior_challenge import simulator_startup


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "isaac"
    apps = root / "apps"
    apps.mkdir(parents=True)
    (root / "VERSION").write_text("Isaac Sim fixture\n")
    (apps / "fixture.kit").write_text("[package]\n")
    (apps / "logo.png").write_bytes(b"png")
    (root / "exts").mkdir()
    (root / "extscache").mkdir()
    owner = tmp_path / "owner"
    owner.mkdir()
    return root, owner, owner / "isaac-view"


def test_inspection_uses_production_apps_claim_without_writes(
    tmp_path: Path, monkeypatch
) -> None:
    root, owner, view = _source(tmp_path)
    calls = []
    original = simulator_startup._apps_claim

    def validate(spec):
        calls.append(spec)
        return original(spec)

    monkeypatch.setattr(simulator_startup, "_apps_claim", validate)
    result = inspect.inspect_isaac_source(root, owner, view)

    assert len(calls) == 1
    assert result["apps_spec"]["linked_directories"] == ["exts", "extscache"]
    assert result["apps_spec"]["absent_directories"] == [
        "extsDeprecated",
        "extsPhysics",
        "extsUser",
    ]
    assert [row["path"] for row in result["observed"]["applications"]] == [
        "fixture.kit",
        "logo.png",
    ]
    assert result["effects"] == {
        "simulator_started": False,
        "policy_loaded": False,
        "writable_view_created": False,
    }
    assert not view.exists()


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_nonregular_application_rejects(tmp_path: Path, kind: str) -> None:
    root, owner, view = _source(tmp_path)
    path = root / "apps/bad"
    if kind == "symlink":
        path.symlink_to("fixture.kit")
    else:
        path.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        inspect.inspect_isaac_source(root, owner, view)


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_nondirectory_extension_rejects(tmp_path: Path, kind: str) -> None:
    root, owner, view = _source(tmp_path)
    path = root / "extsUser"
    if kind == "symlink":
        path.symlink_to("exts", target_is_directory=True)
        message = "symlink"
    else:
        path.write_text("not a directory")
        message = "not a directory"
    with pytest.raises(ValueError, match=message):
        inspect.inspect_isaac_source(root, owner, view)


def test_internal_cli_emits_machine_readable_contract(tmp_path: Path) -> None:
    root, owner, view = _source(tmp_path)
    source = Path(__file__).parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source)
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "npa.workflows.behavior_challenge",
            "simulator-source-inspect",
            "--isaac-root",
            str(root),
            "--owner-root",
            str(owner),
            "--view-root",
            str(view),
        ],
        text=True,
        env=environment,
    )
    result = json.loads(output)
    assert result["status"] == "source_observed_and_production_apps_claim_validated"
    assert result["production_apps_claim"]["applications"] == [
        {
            "path": "fixture.kit",
            **simulator_startup._identity(root / "apps/fixture.kit"),
        },
        {
            "path": "logo.png",
            **simulator_startup._identity(root / "apps/logo.png"),
        },
    ]
    assert not view.exists()
