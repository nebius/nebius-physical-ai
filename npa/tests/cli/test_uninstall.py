from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli import uninstall
from npa.cli import uninstall_helper


runner = CliRunner()


def _synthetic_environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "npa").mkdir()
    (repo / "npa" / "pyproject.toml").write_text(
        "[project]\nname='npa'\n", encoding="utf-8"
    )
    target = repo / "npa" / ".venv"
    (target / "bin").mkdir(parents=True)
    (target / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    executable = target / "bin" / "python"
    executable.symlink_to("/usr/bin/python3")
    return repo, target, executable


def _clean_git(*args, **kwargs):  # noqa: ANN001
    return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")


def _inspect(tmp_path: Path, monkeypatch):  # noqa: ANN001
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    repo, target, executable = _synthetic_environment(tmp_path)
    inspection = uninstall.inspect_repository_environment(
        executable=executable,
        cwd=repo,
        current_prefix=target,
        base_prefix=Path("/usr"),
        runner=_clean_git,
        scan_processes=False,
    )
    return repo, target, executable, inspection


def test_standard_venv_python_symlink_resolves_to_repository_environment(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    repo, target, executable, inspection = _inspect(tmp_path, monkeypatch)

    assert executable.is_symlink()
    assert inspection.safe
    assert inspection.target == target
    assert inspection.repo_root == repo


def test_documented_repo_root_venv_is_also_an_exact_supported_layout(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "npa").mkdir()
    (repo / "npa" / "pyproject.toml").write_text("[project]\nname='npa'\n")
    target = repo / ".venv"
    (target / "bin").mkdir(parents=True)
    (target / "pyvenv.cfg").write_text("home = /usr/bin\n")
    executable = target / "bin" / "python"
    executable.symlink_to("/usr/bin/python3")

    inspection = uninstall.inspect_repository_environment(
        executable=executable,
        cwd=repo,
        current_prefix=target,
        base_prefix=Path("/usr"),
        runner=_clean_git,
        scan_processes=False,
    )

    assert inspection.safe
    assert inspection.repo_root == repo


def test_dry_run_is_default_and_machine_readable(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    _repo, target, _executable, inspection = _inspect(tmp_path, monkeypatch)
    monkeypatch.setattr(uninstall, "inspect_repository_environment", lambda: inspection)

    result = runner.invoke(app, ["uninstall", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "dry_run"
    assert payload["changed"] is False
    assert payload["required_flags"] == ["--remove-environment", "--yes"]
    assert target.exists()


def test_both_confirmation_flags_are_required(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    _repo, target, _executable, inspection = _inspect(tmp_path, monkeypatch)
    monkeypatch.setattr(uninstall, "inspect_repository_environment", lambda: inspection)
    monkeypatch.setattr(
        uninstall,
        "launch_deferred_uninstall",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("launched")),
    )

    for args in (["--yes"], ["--remove-environment"]):
        result = runner.invoke(app, ["uninstall", *args])
        assert result.exit_code == 0, result.output
        assert "No files were removed" in result.output
        assert target.exists()


def test_confirmed_uninstall_only_schedules_a_deferred_exact_path_plan(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _repo, target, _executable, inspection = _inspect(tmp_path, monkeypatch)
    receipt_root = tmp_path / "receipts"
    monkeypatch.setenv("NPA_UNINSTALL_RECEIPT_DIR", str(receipt_root))
    monkeypatch.setattr(uninstall, "inspect_repository_environment", lambda: inspection)
    launched: list[tuple[Path, str]] = []

    def launch(path: Path, payload: dict, **kwargs) -> int:
        launched.append((path, payload["target"]))
        return 12345

    monkeypatch.setattr(uninstall, "launch_deferred_uninstall", launch)

    result = runner.invoke(
        app, ["uninstall", "--remove-environment", "--yes", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "scheduled"
    assert launched == [(Path(payload["receipt_path"]), str(target))]
    assert target.exists()
    stored = json.loads(Path(payload["receipt_path"]).read_text(encoding="utf-8"))
    assert stored["target"] == str(target)
    assert stored["state"] == "scheduled"


@pytest.mark.parametrize(
    "kind", ["conda", "externally-managed", "pipx", "dirty", "active"]
)
def test_shared_or_active_environments_are_refused(
    monkeypatch, tmp_path: Path, kind: str
) -> None:  # noqa: ANN001
    repo, target, executable = _synthetic_environment(tmp_path)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    run = _clean_git
    scan = False
    if kind == "conda":
        (target / "conda-meta").mkdir()
    elif kind == "externally-managed":
        (target / "EXTERNALLY-MANAGED").write_text("managed\n", encoding="utf-8")
    elif kind == "pipx":
        monkeypatch.setenv("PIPX_HOME", str(tmp_path / "pipx"))
    elif kind == "dirty":
        run = lambda *args, **kwargs: subprocess.CompletedProcess(  # noqa: E731
            args[0], 0, stdout="?? npa/.venv/local-change\n", stderr=""
        )
    else:
        scan = True
        monkeypatch.setattr(
            uninstall,
            "_active_environment_processes",
            lambda _target: ["pid 77 exe=/repo/npa/.venv/bin/python"],
        )

    inspection = uninstall.inspect_repository_environment(
        executable=executable,
        cwd=repo,
        current_prefix=target,
        base_prefix=Path("/usr"),
        runner=run,
        scan_processes=scan,
    )

    assert not inspection.safe
    assert inspection.reasons


def test_symlink_escape_and_arbitrary_path_are_refused(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    arbitrary = tmp_path / "shared" / "bin"
    arbitrary.mkdir(parents=True)
    executable = arbitrary / "python"
    executable.symlink_to("/usr/bin/python3")

    inspection = uninstall.inspect_repository_environment(
        executable=executable,
        cwd=tmp_path,
        runner=_clean_git,
        scan_processes=False,
    )

    assert not inspection.safe
    assert any(
        "not an exact supported repository-local" in item for item in inspection.reasons
    )


def test_helper_revalidates_and_removes_only_a_synthetic_venv(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _repo, target, _executable, inspection = _inspect(tmp_path, monkeypatch)
    receipt_root = tmp_path / "receipts"
    monkeypatch.setenv("NPA_UNINSTALL_RECEIPT_DIR", str(receipt_root))
    path, payload = uninstall._new_plan(inspection)
    payload["parent_pid"] = 0
    payload["parent_start"] = ""
    uninstall._write_atomic(path, payload)
    monkeypatch.setattr(uninstall_helper, "_other_processes_using", lambda target: [])

    code = uninstall_helper.run(path, payload["nonce"])

    completed = json.loads(path.read_text(encoding="utf-8"))
    assert code == 0, completed
    assert not target.exists()
    assert completed["state"] == "succeeded"


def test_helper_inode_race_preserves_replacement_and_failure_receipt(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _repo, target, _executable, inspection = _inspect(tmp_path, monkeypatch)
    receipt_root = tmp_path / "receipts"
    monkeypatch.setenv("NPA_UNINSTALL_RECEIPT_DIR", str(receipt_root))
    path, payload = uninstall._new_plan(inspection)
    payload["parent_pid"] = 0
    payload["parent_start"] = ""
    uninstall._write_atomic(path, payload)
    original = target.with_name(".venv-original")
    target.rename(original)
    (target / "bin").mkdir(parents=True)
    (target / "pyvenv.cfg").write_text("replacement\n", encoding="utf-8")

    code = uninstall_helper.run(path, payload["nonce"])

    assert code == 1
    assert target.exists()
    assert original.exists()
    failed = json.loads(path.read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert "inode/device changed" in failed["error"]
    assert failed["recovery_command"].startswith("npa uninstall --remove-environment")


def test_helper_rechecks_opened_directory_inode_after_final_name_lookup(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _repo, target, _executable, inspection = _inspect(tmp_path, monkeypatch)
    payload = uninstall._new_plan(inspection)[1]
    original = target.with_name(".venv-original")
    real_stat = uninstall_helper.os.stat
    swapped = False

    def swap_after_final_stat(path, *args, **kwargs):  # noqa: ANN001
        nonlocal swapped
        result = real_stat(path, *args, **kwargs)
        if not swapped and kwargs.get("dir_fd") is not None and path == target.name:
            swapped = True
            target.rename(original)
            target.mkdir()
            (target / "replacement").write_text("preserve\n", encoding="utf-8")
        return result

    monkeypatch.setattr(uninstall_helper.os, "stat", swap_after_final_stat)

    with pytest.raises(RuntimeError, match="between final check and directory open"):
        uninstall_helper._remove_exact_target(target, payload)

    assert target.exists()
    assert (target / "replacement").is_file()
    assert original.exists()


def test_status_hides_nonce_and_marker_digest(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    _repo, _target, _executable, inspection = _inspect(tmp_path, monkeypatch)
    receipt_root = tmp_path / "receipts"
    monkeypatch.setenv("NPA_UNINSTALL_RECEIPT_DIR", str(receipt_root))
    _path, payload = uninstall._new_plan(inspection)

    result = runner.invoke(
        app, ["uninstall", "--status", payload["receipt_id"], "--json"]
    )

    assert result.exit_code == 0, result.output
    public = json.loads(result.output)
    assert public["state"] == "scheduled"
    assert "nonce" not in public
    assert "pyvenv_sha256" not in public
