"""Exercise writable Isaac startup and bounded cleanup without Isaac Sim."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from npa.workflows.behavior_challenge.simulator_startup import (
    FileIdentity,
    IsaacAppsSpec,
    OwnedTreeCleanupError,
    SimulatorStartupSpec,
    hold_owned_tree,
    prepared_evaluator_environment,
    prepare_writable_isaac_apps,
    remove_owned_tree,
    run_simulator_startup,
    validate_simulator_startup_receipt,
    write_simulator_startup_receipt,
)
from npa.workflows.behavior_challenge import simulator_startup as startup_module


def _file_identity(path: Path) -> FileIdentity:
    raw = path.read_bytes()
    return FileIdentity(len(raw), hashlib.sha256(raw).hexdigest())


def _fake_omnigibson() -> str:
    return """import os
import sys
from pathlib import Path
if 'boto3' in sys.modules:
    raise RuntimeError('startup child imported campaign storage dependencies')
app = None
sim = None
class _Sim:
    scenes = []
def launch():
    global app, sim
    app, sim = object(), _Sim()
    view = Path(os.environ['EXP_PATH']).parent
    appdata = Path(os.environ['OMNIGIBSON_APPDATA_PATH'])
    (view / 'generated' / 'icons').mkdir(parents=True)
    (view / 'generated' / 'icons' / 'logo.png').write_bytes(b'png')
    (appdata / 'screenshots').mkdir()
    (appdata / 'screenshots' / 'frame.txt').write_text('frame')
    (appdata / 'screenshots').chmod(0o2000)
def shutdown():
    os._exit(0)
"""


def _apps(tmp_path: Path, module_source: str | None = None) -> IsaacAppsSpec:
    source = tmp_path / "isaac"
    (source / "apps").mkdir(parents=True)
    (source / "apps/robot.kit").write_text("kit")
    (source / "VERSION").write_text("5.1")
    (source / "exts").mkdir()
    module = source / "OmniGibson/omnigibson"
    module.mkdir(parents=True)
    module.joinpath("__init__.py").write_text(module_source or _fake_omnigibson())
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=NPA",
            "-c",
            "user.email=npa@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    owner = tmp_path / "run"
    owner.mkdir()
    return IsaacAppsSpec(
        isaac_root=source,
        owner_root=owner,
        view_root=owner / "view",
        version_file="VERSION",
        version=_file_identity(source / "VERSION"),
        applications={"robot.kit": _file_identity(source / "apps/robot.kit")},
        linked_directories=("exts",),
        absent_directories=("extsPhysics",),
    )


def _apps_value(apps: IsaacAppsSpec) -> dict[str, object]:
    return {
        "isaac_root": str(apps.isaac_root),
        "owner_root": str(apps.owner_root),
        "view_root": str(apps.view_root),
        "version_file": apps.version_file,
        "version": apps.version.__dict__,
        "applications": {
            key: value.__dict__ for key, value in apps.applications.items()
        },
        "linked_directories": list(apps.linked_directories),
        "absent_directories": list(apps.absent_directories),
    }


def _spec_value(apps: IsaacAppsSpec) -> dict[str, object]:
    marker = apps.owner_root / "startup.json"
    request = apps.owner_root / "shutdown.json"
    return {
        "schema": "npa.workbench.simulator-startup-spec.v1",
        "apps": _apps_value(apps),
        "appdata_root": str(apps.owner_root / "appdata"),
        "command": [
            sys.executable,
            "-m",
            "npa.workflows.behavior_challenge",
            "simulator-startup-child",
            "--upstream-root",
            str(apps.isaac_root.resolve()),
            "--marker-path",
            str(marker.resolve()),
            "--shutdown-request-path",
            str(request.resolve()),
        ],
        "marker_path": str(marker),
        "shutdown_request_path": str(request),
        "log_path": str(apps.owner_root / "startup.log"),
        "environment": {},
        "evaluation_context": {
            "upstream_root": str(apps.isaac_root),
            "evaluator_python": sys.executable,
            "data_root": str(apps.isaac_root),
        },
    }


def _startup_spec(apps: IsaacAppsSpec) -> SimulatorStartupSpec:
    value = _spec_value(apps)
    return SimulatorStartupSpec(
        apps=apps,
        appdata_root=apps.owner_root / "appdata",
        command=tuple(value["command"]),
        marker_path=Path(value["marker_path"]),
        shutdown_request_path=Path(value["shutdown_request_path"]),
        log_path=Path(value["log_path"]),
        environment={},
        evaluation_context=value["evaluation_context"],
    )


def _write_startup(apps: IsaacAppsSpec) -> tuple[Path, dict[str, object]]:
    spec_path = apps.owner_root / "startup-spec.json"
    receipt_path = apps.owner_root / "receipt.json"
    spec_path.write_text(json.dumps(_spec_value(apps)))
    receipt = write_simulator_startup_receipt(spec_path, receipt_path)
    return receipt_path, receipt


def test_writable_apps_are_exact_and_source_remains_unchanged(tmp_path: Path) -> None:
    spec = _apps(tmp_path)

    receipt = prepare_writable_isaac_apps(spec)

    assert Path(receipt["exp_path"]) == spec.view_root / "apps"
    assert (spec.view_root / "apps/robot.kit").read_text() == "kit"
    assert (spec.view_root / "exts").resolve() == (spec.isaac_root / "exts").resolve()
    assert (spec.isaac_root / "apps/robot.kit").read_text() == "kit"


def test_writable_apps_reject_changed_source_and_unsafe_names(tmp_path: Path) -> None:
    spec = _apps(tmp_path)
    (spec.isaac_root / "apps/robot.kit").write_text("changed")
    with pytest.raises(ValueError, match="application differs"):
        prepare_writable_isaac_apps(spec)

    unsafe = _apps(tmp_path / "other")
    object.__setattr__(unsafe, "linked_directories", ("../exts",))
    with pytest.raises(ValueError, match="one relative path component"):
        prepare_writable_isaac_apps(unsafe)


def test_version_file_cannot_escape_through_an_intermediate_symlink(
    tmp_path: Path,
) -> None:
    spec = _apps(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "VERSION").write_text("5.1")
    (spec.isaac_root / "metadata").symlink_to(outside, target_is_directory=True)
    object.__setattr__(spec, "version_file", "metadata/VERSION")

    with pytest.raises(ValueError, match="contains a symlink"):
        prepare_writable_isaac_apps(spec)


def test_cleanup_repairs_owned_directories_and_derives_inventory(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "owner"
    root = owner / "tree"
    nested = root / "restricted"
    nested.mkdir(parents=True)
    (nested / "payload").write_text("value")
    nested.chmod(0o2000)
    guard = hold_owned_tree(root, owner)
    held_fd = guard.root_fd

    receipt = remove_owned_tree(guard)

    assert not root.exists()
    assert receipt["inventory"]["counts"] == {
        "directory": 2,
        "regular": 1,
        "symlink": 0,
    }
    assert receipt["repairs"][0]["mode_before"] == "0o2000"
    with pytest.raises(OSError):
        os.fstat(held_fd)
    with pytest.raises(ValueError, match="already consumed"):
        remove_owned_tree(guard)


def test_cleanup_never_follows_external_symlink(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    root = owner / "tree"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("retained")
    (root / "link").symlink_to(outside)

    receipt = remove_owned_tree(hold_owned_tree(root, owner))

    assert outside.read_text() == "retained"
    assert receipt["inventory"]["counts"]["symlink"] == 1


def test_cleanup_rejects_root_replacement_before_permission_repair(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "owner"
    root = owner / "tree"
    root.mkdir(parents=True)
    guard = hold_owned_tree(root, owner)
    held_fd = guard.root_fd
    root.rmdir()
    root.mkdir(mode=0o000)

    with pytest.raises(OwnedTreeCleanupError) as raised:
        remove_owned_tree(guard)

    assert root.stat().st_mode & 0o777 == 0
    assert raised.value.evidence["inventory"]["counts"]["directory"] == 0
    with pytest.raises(OSError):
        os.fstat(held_fd)
    with pytest.raises(ValueError, match="already consumed"):
        remove_owned_tree(guard)
    root.chmod(0o700)
    root.rmdir()


def test_child_failure_releases_retained_directory_fds(
    tmp_path: Path, monkeypatch
) -> None:
    source = """app = None
sim = None
def launch():
    raise RuntimeError('startup failed')
def shutdown():
    raise RuntimeError('not reached')
"""
    apps = _apps(tmp_path, source)
    original = startup_module.hold_owned_tree
    held_fds = []

    def tracked(*args):
        guard = original(*args)
        held_fds.append(guard.root_fd)
        return guard

    monkeypatch.setattr(startup_module, "hold_owned_tree", tracked)
    with pytest.raises(ValueError, match="child exit differs"):
        run_simulator_startup(_startup_spec(apps))

    assert len(held_fds) == 2
    for held_fd in held_fds:
        with pytest.raises(OSError):
            os.fstat(held_fd)


def test_real_child_fast_exit_produces_derived_cleanup_receipt(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    spec = _startup_spec(apps)

    receipt_path, receipt = _write_startup(apps)

    assert not apps.view_root.exists() and not spec.appdata_root.exists()
    view_members = receipt["cleanup"]["view"]["inventory"]["members"]
    assert any(row["path"].endswith("generated/icons/logo.png") for row in view_members)
    assert receipt["cleanup"]["appdata"]["repairs"][0]["mode_after"] == "0o2700"
    assert validate_simulator_startup_receipt(receipt_path) == receipt

    changed = dict(receipt["evaluation_context"])
    changed["data_root"] = str(tmp_path / "different")
    with pytest.raises(ValueError, match="evaluator context"):
        validate_simulator_startup_receipt(receipt_path, changed)

    source_app = apps.isaac_root / "apps/robot.kit"
    source_app.write_text("changed")
    with pytest.raises(
        ValueError, match="(evaluator source identity|Isaac source application)"
    ):
        validate_simulator_startup_receipt(receipt_path)
    source_app.write_text("kit")
    Path(receipt["evidence"]["log"]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="log changed"):
        validate_simulator_startup_receipt(receipt_path)


def test_actual_evaluator_attempts_get_fresh_bounded_views(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    spec = _startup_spec(apps)
    receipt = run_simulator_startup(spec)

    for _ in range(2):
        with prepared_evaluator_environment(receipt, apps.owner_root) as environment:
            assert Path(environment["EXP_PATH"]).joinpath("robot.kit").is_file()
            assert Path(environment["OMNIGIBSON_APPDATA_PATH"]).is_dir()
            assert environment["OMNIGIBSON_DATA_PATH"] == str(apps.isaac_root.resolve())
            assert environment["OMNIGIBSON_HEADLESS"] == "1"
            assert environment["PYTHONDONTWRITEBYTECODE"] == "1"

    assert not list(apps.owner_root.glob("simulator-evaluator-view-*"))
    assert not list(apps.owner_root.glob("simulator-appdata-*"))
    assert len(list(apps.owner_root.glob("simulator-evaluator-cleanup-*.json"))) == 2


def test_evaluator_attempt_search_has_no_fixed_retry_limit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(startup_module.itertools, "count", lambda: iter([10_000]))

    paths = startup_module._fresh_evaluator_attempt(tmp_path)

    assert all("10000" in path.name for path in paths)


def _rewrite_json(path: Path, value: dict[str, object]) -> dict[str, object]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("target", ["marker", "shutdown_request"])
def test_rehashed_status_mutation_is_rejected(tmp_path: Path, target: str) -> None:
    apps = _apps(tmp_path)
    _, receipt = _write_startup(apps)
    marker_path = Path(receipt["evidence"]["marker"]["path"])
    request_path = Path(receipt["evidence"]["shutdown_request"]["path"])
    embedded = "startup_marker" if target == "marker" else target
    receipt[embedded]["status"] = "forged_status"
    if target == "marker":
        receipt["evidence"]["marker"].update(
            _rewrite_json(marker_path, receipt[embedded])
        )
        receipt["shutdown_request"]["startup_marker"] = {
            key: receipt["evidence"]["marker"][key] for key in ("bytes", "sha256")
        }
    receipt["evidence"]["shutdown_request"].update(
        _rewrite_json(request_path, receipt["shutdown_request"])
    )
    path = apps.owner_root / "mutated-receipt.json"
    path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="(marker|shutdown request) differs"):
        validate_simulator_startup_receipt(path)


def test_cleanup_evidence_is_bound_to_owned_roots_guards_and_repairs(
    tmp_path: Path,
) -> None:
    apps = _apps(tmp_path)
    _, receipt = _write_startup(apps)
    mutations = []
    wrong_root = copy.deepcopy(receipt)
    wrong_root["cleanup"]["view"]["root"] = "/etc"
    mutations.append(wrong_root)
    wrong_uid = copy.deepcopy(receipt)
    wrong_uid["cleanup"]["view"]["guard"]["uid"] += 1
    mutations.append(wrong_uid)
    escaped_repair = copy.deepcopy(receipt)
    escaped_repair["cleanup"]["appdata"]["repairs"][0]["path"] = "../../outside"
    mutations.append(escaped_repair)

    for index, mutation in enumerate(mutations):
        path = apps.owner_root / f"cleanup-mutation-{index}.json"
        path.write_text(json.dumps(mutation))
        with pytest.raises(ValueError, match="cleanup"):
            validate_simulator_startup_receipt(path)


def test_cleanup_inventory_rejects_rehashed_traversal_member(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    _, receipt = _write_startup(apps)
    cleanup = receipt["cleanup"]["view"]
    cleanup["inventory"]["members"][-1]["path"] = "view/../../outside"
    cleanup["inventory"] = startup_module._inventory_summary(
        cleanup["inventory"]["members"]
    )
    path = apps.owner_root / "cleanup-traversal.json"
    path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="cleanup member"):
        validate_simulator_startup_receipt(path)


def test_receipt_rejects_rehashed_inventory_tamper(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    _, receipt = _write_startup(apps)
    receipt["cleanup"]["view"]["inventory"] = startup_module._inventory_summary([])
    path = apps.owner_root / "inventory-tamper.json"
    path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="cleanup root member"):
        validate_simulator_startup_receipt(path)


def test_receipt_rejects_command_not_bound_to_stored_spec(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    _, receipt = _write_startup(apps)
    receipt["command"] = [sys.executable, "-c", "raise SystemExit(0)"]
    path = apps.owner_root / "command-mutation.json"
    path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="differs from its specification"):
        validate_simulator_startup_receipt(path)


def test_receipt_rejects_alternate_isaac_root_and_links(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    _, receipt = _write_startup(apps)
    alternate = tmp_path / "alternate-isaac"
    shutil.copytree(apps.isaac_root, alternate, symlinks=True)
    (alternate / "exts/unbound.py").write_text("different extension")
    receipt["apps"]["source_root"] = str(alternate.resolve())
    receipt["apps"]["linked_directories"][0]["target"] = str(
        (alternate / "exts").resolve()
    )
    path = apps.owner_root / "alternate-root.json"
    path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="differs from its specification"):
        validate_simulator_startup_receipt(path)


def test_arbitrary_startup_command_is_rejected_before_execution(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    value = _spec_value(apps)
    touched = tmp_path / "arbitrary-command-ran"
    value["command"] = [sys.executable, "-c", f"open({str(touched)!r}, 'w').close()"]
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="built-in probe"):
        write_simulator_startup_receipt(spec, apps.owner_root / "receipt.json")

    assert not touched.exists()


def test_clean_exit_before_real_startup_marker_is_failure(tmp_path: Path) -> None:
    source = """import os
app = None
sim = None
def launch():
    raise SystemExit(0)
def shutdown():
    os._exit(0)
"""
    apps = _apps(tmp_path, source)
    spec = tmp_path / "spec.json"
    receipt = apps.owner_root / "receipt.json"
    spec.write_text(json.dumps(_spec_value(apps)))

    with pytest.raises(ValueError, match="required regular file differs"):
        write_simulator_startup_receipt(spec, receipt)

    assert not receipt.exists()
    failure = apps.owner_root / "receipt-failure.json"
    assert json.loads(failure.read_text())["success_receipt_written"] is False


def test_public_cli_runs_the_source_bound_startup(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    spec = tmp_path / "spec.json"
    receipt = apps.owner_root / "receipt.json"
    spec.write_text(json.dumps(_spec_value(apps)))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "npa.workflows.behavior_challenge",
            "simulator-startup",
            "--spec-path",
            str(spec),
            "--receipt-path",
            str(receipt),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        validate_simulator_startup_receipt(receipt, expected_spec_path=spec)[
            "case_started"
        ]
        is False
    )


def test_child_cli_does_not_import_campaign_storage_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    apps = _apps(tmp_path)
    hooks = tmp_path / "import-hooks"
    hooks.mkdir()
    hooks.joinpath("sitecustomize.py").write_text(
        """import importlib.abc
import sys
class RejectStorage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'boto3' or fullname == 'npa.clients.storage':
            raise RuntimeError(f'forbidden startup dependency: {fullname}')
sys.meta_path.insert(0, RejectStorage())
"""
    )
    monkeypatch.setenv("PYTHONPATH", str(hooks))
    spec = tmp_path / "spec.json"
    receipt = apps.owner_root / "receipt.json"
    spec.write_text(json.dumps(_spec_value(apps)))

    write_simulator_startup_receipt(spec, receipt)

    assert validate_simulator_startup_receipt(receipt)["child"]["returncode"] == 0


def test_changed_spec_cannot_reuse_existing_receipt(tmp_path: Path) -> None:
    apps = _apps(tmp_path)
    spec = tmp_path / "spec.json"
    receipt = apps.owner_root / "receipt.json"
    value = _spec_value(apps)
    spec.write_text(json.dumps(value))
    write_simulator_startup_receipt(spec, receipt)
    value["environment"] = {"NEW_SETTING": "changed"}
    spec.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="specification identity"):
        validate_simulator_startup_receipt(receipt, expected_spec_path=spec)
