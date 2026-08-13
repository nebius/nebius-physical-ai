from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import stat

import pytest

from npa.orchestration.npa_workflow import first_run_state


def _fixed_identity(project: str) -> tuple[str, str, str]:
    return f"stable:{project}", project, "configured_project_id"


def test_run_state_is_scoped_by_project_and_workflow(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(first_run_state, "resolve_project_identity", _fixed_identity)

    project_a = first_run_state.prepare_run(
        project="project-a", workflow_identity="paidf", state_root=tmp_path
    )
    project_b = first_run_state.prepare_run(
        project="project-b", workflow_identity="paidf", state_root=tmp_path
    )
    other_workflow = first_run_state.prepare_run(
        project="project-a", workflow_identity="other", state_root=tmp_path
    )

    assert (
        len({project_a.state_path, project_b.state_path, other_workflow.state_path})
        == 3
    )
    assert len({project_a.run_id, project_b.run_id, other_workflow.run_id}) == 3


def test_stable_project_identity_survives_alias_rename(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        first_run_state,
        "resolve_project_identity",
        lambda project: ("project-stable-id", project, "configured_project_id"),
    )

    old_alias = first_run_state.prepare_run(
        project="old-alias", workflow_identity="paidf", state_root=tmp_path
    )
    new_alias = first_run_state.prepare_run(
        project="new-alias", workflow_identity="paidf", state_root=tmp_path
    )

    assert old_alias.state_path == new_alias.state_path
    assert new_alias.previous_run is not None
    assert new_alias.previous_run["run_id"] == old_alias.run_id
    assert new_alias.run_id != old_alias.run_id


def test_existing_state_age_and_explicit_resume_are_reported(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(first_run_state, "resolve_project_identity", _fixed_identity)
    initial = first_run_state.prepare_run(
        project="project-a", workflow_identity="paidf", state_root=tmp_path
    )

    resumed = first_run_state.prepare_run(
        project="project-a",
        workflow_identity="paidf",
        resume_run=initial.run_id,
        state_root=tmp_path,
    )

    assert resumed.generated_new is False
    assert resumed.previous_run is not None
    assert resumed.previous_run["run_id"] == initial.run_id
    assert resumed.previous_run["age_seconds"] is not None
    payload = json.loads(Path(resumed.state_path).read_text())
    assert payload["source"] == "explicit_resume"
    assert payload["project_identity"] == "stable:project-a"


def test_legacy_global_file_warns_but_is_never_reused_or_deleted(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(first_run_state, "resolve_project_identity", _fixed_identity)
    legacy = tmp_path / "paidf-first-run-id"
    legacy.write_text("legacy-two-day-old-run\n", encoding="utf-8")

    prepared = first_run_state.prepare_run(
        project="project-a",
        workflow_identity="paidf",
        state_root=tmp_path / "scoped",
        legacy_path=legacy,
    )

    assert prepared.run_id != "legacy-two-day-old-run"
    assert "was not reused or deleted" in prepared.warning
    assert legacy.read_text(encoding="utf-8") == "legacy-two-day-old-run\n"


def test_concurrent_generation_is_locked_atomic_and_owner_only(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(first_run_state, "resolve_project_identity", _fixed_identity)

    def prepare(_: int):
        return first_run_state.prepare_run(
            project="project-a", workflow_identity="paidf", state_root=tmp_path
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(prepare, range(24)))

    assert len({item.run_id for item in prepared}) == 24
    path = Path(prepared[-1].state_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == first_run_state.STATE_SCHEMA
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_resume_and_new_id_are_mutually_exclusive(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(first_run_state, "resolve_project_identity", _fixed_identity)

    with pytest.raises(ValueError, match="mutually exclusive"):
        first_run_state.prepare_run(
            project="project-a",
            workflow_identity="paidf",
            resume_run="old-run",
            new_run_id="new-run",
            state_root=tmp_path,
        )
