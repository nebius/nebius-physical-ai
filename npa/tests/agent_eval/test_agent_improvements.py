"""Real persistence, process contention and evidence-bound improvement tests."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from npa.agent_backend.actions import run_action_loop
from npa.agent_backend.improvements import (
    ImprovementError, ImprovementScope, ImprovementStore, find_improvements,
    lesson_context, store_from_config,
)
from npa.agent_backend.improvement_routes import (
    ImprovementDeps, ImprovementRuntime, register_improvement_routes,
)


def _scope(component="trajectory", files=("src/adapter.py",)):
    return ImprovementScope(
        scope_id="adapter-fix", component=component, files=files,
        base_revision="a" * 40, required_checks=("reproducer", "privacy"),
        lesson_keys=("trajectory_observation_conservation",),
    )


def _store(root, scopes=None, literals=()):
    repo = Path(root) / "repo"
    repo.mkdir(exist_ok=True)
    return ImprovementStore(
        Path(root) / "queue", repository=repo, evidence_directory=Path(root) / "evidence",
        scopes=scopes or [_scope()], reviewers=("independent-reviewer", "builder"),
        private_literals=literals,
    )


@pytest.fixture
def store(tmp_path):
    result = _store(tmp_path)
    (tmp_path / "repo/src").mkdir()
    (tmp_path / "repo/src/adapter.py").write_text("candidate = 1\n")
    return result


def _observation(store, episode="episode-one", component="trajectory", evidence=None):
    return store.observe(component=component, kind="trajectory_adapter_mismatch", episode_id=episode,
                         event_index=0, evidence=evidence or {"expected": "error", "observed": "ok"})


def _claim(store, item=None):
    item = item or _observation(store)
    claim = store.claim(item["id"], owner="builder", version=item["version"])
    return claim, {key: claim[key] for key in ("owner", "generation", "claim_token")}


def _validate(store, claim, ownership, exit_code=0):
    candidate = store.begin_candidate(claim["id"], changed_files=["src/adapter.py"], **ownership)
    refs = []
    for check in ("reproducer", "privacy"):
        # A real isolated process supplies the exit status and report bytes.
        completed = subprocess.run([sys.executable, "-c", f"print('objective check report'); raise SystemExit({exit_code})"],
                                   capture_output=True, check=False)
        reference = store.write_validation_receipt(candidate, check=check, completed=completed, report=completed.stdout)
        result = store.record_validation(claim["id"], evidence_ref=reference, **ownership)
        refs.append(reference)
    return result, refs


def _verify(store):
    claim, ownership = _claim(store)
    result, refs = _validate(store, claim, ownership)
    assert result["state"] == "ready_for_review"
    review = store.write_review_receipt(claim["id"], reviewer="independent-reviewer",
                                       lesson_key="trajectory_observation_conservation",
                                       accepted=True, report=b"Independent reproducer and scope review passed.\n")
    return store.review(claim["id"], evidence_ref=review), refs, review


def test_detector_consumes_actual_action_shape_and_preserves_error_args():
    def tool(args):
        raise RuntimeError("synthetic failure")

    calls = iter([{"tool": "retrieval_search", "args": {"query": "adapter fields"}}, {"final": "failed"}])

    def planner(*args, **kwargs):
        return {"choices": [{"message": {"content": json.dumps(next(calls))}}]}

    result = run_action_loop("inspect adapter", tools={"retrieval_search": tool}, model_call=planner)
    finding = find_improvements(result)[0]
    assert finding["component"] == "retrieval_search"
    assert finding["kind"] == "tool_error"
    assert finding["evidence"]["args"] == {"query": "adapter fields"}
    assert finding["evidence"]["status"] == "error"


def test_terminal_empty_and_recovered_errors_are_not_defects():
    assert find_improvements({"ok": True, "steps": [{"phase": "call", "status": "empty", "terminal_observation": True}]}) == []
    assert find_improvements({"ok": True, "steps": [{"tool": "trajectory", "status": "error"},
                                                   {"tool": "trajectory", "status": "ok"}]}) == []
    assert find_improvements({"ok": True, "needs_confirmation": True, "steps": [{"phase": "confirm", "status": "rejected"}]}) == []


def test_dedupe_occurrences_restart_and_changed_signature(store, tmp_path):
    first = _observation(store, evidence={"expected": "error", "observed": "ok"})
    same = _observation(store, evidence={"observed": "ok", "expected": "error"})
    assert same == first
    second = _observation(store, episode="episode-two")
    assert second["id"] == first["id"] and second["occurrences"] == 2
    other = store.observe(component="trajectory", kind="privacy_rule", episode_id="episode-one", event_index=0, evidence={"rule": "headers"})
    assert other["id"] != first["id"]
    reopened = _store(tmp_path)
    history = reopened.history(first["id"])
    assert len(history["occurrences"]) == 2 and len(history["events"]) == 2


def _process_claim(root, item_id, version, worker):
    store = _store(root)
    try:
        result = store.claim(item_id, owner=worker, version=version)
        return {key: result[key] for key in ("owner", "generation", "claim_token")}
    except ImprovementError:
        return None


def test_multiprocess_claim_contention_and_restart(store, tmp_path):
    item = _observation(store)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as workers:
        futures = [workers.submit(_process_claim, str(tmp_path), item["id"], item["version"], f"worker-{i}") for i in range(2)]
        results = [future.result() for future in futures]
    winner = [value for value in results if value is not None]
    assert len(winner) == 1
    assert _store(tmp_path).history(item["id"])["item"]["owner"] == winner[0]["owner"]
    assert "claim_token" not in json.dumps(store.history(item["id"]))


def test_overlapping_scope_claims_and_release_fences(tmp_path):
    store = _store(tmp_path, scopes=[_scope(), _scope("other-tool")])
    first, ownership = _claim(store)
    second = _observation(store, component="other-tool")
    with pytest.raises(ImprovementError, match="overlapping"):
        store.claim(second["id"], owner="second", version=second["version"])
    released = store.release(first["id"], **ownership)
    next_claim = store.claim(first["id"], owner="builder", version=released["version"])
    assert next_claim["generation"] == first["generation"] + 1
    with pytest.raises(ImprovementError, match="fence"):
        store.release(first["id"], **ownership)


@pytest.mark.parametrize("filename", ["/outside", "../escape", "src/../escape", "src/*.py", "src//x.py", "src\\x.py"])
def test_invalid_scopes_fail_before_claim(tmp_path, filename):
    with pytest.raises(ImprovementError):
        _store(tmp_path, scopes=[_scope(files=(filename,))])


def test_symlink_scope_and_database_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").symlink_to(tmp_path)
    with pytest.raises(ImprovementError, match="symlink"):
        _store(tmp_path)
    (repo / "src").unlink()
    queue = tmp_path / "queue"
    (queue / "improvements.sqlite3").symlink_to(tmp_path / "outside")
    with pytest.raises(ImprovementError, match="symlink"):
        _store(tmp_path)
    assert not (tmp_path / "outside").exists()


def test_private_payload_absent_from_database_and_reports(tmp_path):
    private_name = "synthetic customer name"
    private_prefix = "private-collection-area"
    store = _store(tmp_path, literals=(private_name, private_prefix))
    item = _observation(store, evidence={
        "observation": "authorization=synthetic-value " + private_name,
        private_prefix: "https://example.invalid/private/result",
        "environment": {"SOME_VALUE": "not-safe-to-persist"},
    })
    serialized = json.dumps(store.history(item["id"]))
    for secret in (private_name, private_prefix, "synthetic-value", "not-safe-to-persist", "example.invalid"):
        assert secret not in serialized
        assert secret.encode() not in store.path.read_bytes()
    assert store.path.stat().st_mode & 0o077 == 0
    assert store.directory.stat().st_mode & 0o077 == 0
    with pytest.raises(Exception):
        _observation(store, evidence={"bytes": b"inline"})


def test_failed_then_corrected_validation_no_early_lesson(store):
    claim, ownership = _claim(store)
    failed, _ = _validate(store, claim, ownership, exit_code=1)
    assert failed["state"] == "validation_failed"
    assert store.matching_verified_lessons(["trajectory"]) == []
    with pytest.raises(ImprovementError):
        store.write_review_receipt(claim["id"], reviewer="independent-reviewer", lesson_key="trajectory_observation_conservation", accepted=True, report=b"review")
    passed, _ = _validate(store, claim, ownership)
    assert passed["state"] == "ready_for_review"
    assert store.matching_verified_lessons(["trajectory"]) == []


def test_incomplete_and_out_of_scope_validation_cannot_verify(store):
    claim, ownership = _claim(store)
    with pytest.raises(ImprovementError, match="escape"):
        store.begin_candidate(claim["id"], changed_files=["src/other.py"], **ownership)
    candidate = store.begin_candidate(claim["id"], changed_files=["src/adapter.py"], **ownership)
    with pytest.raises(ImprovementError, match="completed process"):
        store.write_validation_receipt(candidate, check="reproducer", completed={"passed": True}, report=b"asserted")
    with pytest.raises(ImprovementError, match="nonempty"):
        store.write_validation_receipt(candidate, check="reproducer", completed=subprocess.CompletedProcess([], 0), report=b"")
    reference = store.write_validation_receipt(candidate, check="reproducer", completed=subprocess.CompletedProcess([], 0), report=b"done")
    partial = store.record_validation(claim["id"], evidence_ref=reference, **ownership)
    assert partial["state"] == "claimed"
    with pytest.raises(ImprovementError):
        store.review(claim["id"], evidence_ref=reference)


@pytest.mark.parametrize("change", ["source", "report", "empty_report", "missing_report", "receipt"])
def test_stale_or_missing_evidence_blocks_review(store, change):
    claim, ownership = _claim(store)
    _, refs = _validate(store, claim, ownership)
    receipt_path = store.evidence_directory / (refs[0] + ".json")
    receipt = json.loads(receipt_path.read_text())
    report = store.evidence_directory / (receipt["report_ref"] + ".txt")
    if change == "source":
        (store.repository / "src/adapter.py").write_text("candidate = 2\n")
    elif change == "report":
        report.write_text("changed")
    elif change == "empty_report":
        report.write_text("")
    elif change == "missing_report":
        report.unlink()
    else:
        receipt["exit_code"] = 1
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ImprovementError, OSError)):
        store.write_review_receipt(claim["id"], reviewer="independent-reviewer", lesson_key="trajectory_observation_conservation", accepted=True, report=b"review")


@pytest.mark.parametrize("reviewer", ["builder", "forged-reviewer"])
def test_self_review_and_unconfigured_review_rejected(store, reviewer):
    claim, ownership = _claim(store)
    _validate(store, claim, ownership)
    with pytest.raises(ImprovementError, match="independent"):
        store.write_review_receipt(claim["id"], reviewer=reviewer, lesson_key="trajectory_observation_conservation", accepted=True, report=b"review")


def test_review_binds_candidate_and_validation_and_recurrence_deactivates(store, tmp_path):
    verified, _, _ = _verify(store)
    assert verified["state"] == "verified"
    assert verified["review"]["identity_provenance"] == "coordinator-attested-external-review"
    restarted = _store(tmp_path)
    lesson = restarted.matching_verified_lessons(["agent-run-data-collection"])
    assert lesson[0]["lesson_key"] == "trajectory_observation_conservation"
    assert restarted.matching_verified_lessons(["gpu-selection"]) == []
    assert _observation(store)["state"] == "verified"  # replay does not reopen
    recurrence = _observation(store, episode="new-failure")
    assert recurrence["state"] == "observed" and recurrence["occurrences"] == 2
    assert store.matching_verified_lessons(["trajectory"]) == []


def test_changed_source_suppresses_verified_lesson(store):
    _verify(store)
    (store.repository / "src/adapter.py").write_text("changed after review\n")
    assert store.matching_verified_lessons(["trajectory"]) == []


def test_untrusted_lesson_prose_is_never_prompted(store):
    verified, _, _ = _verify(store)
    lessons = store.matching_verified_lessons(["trajectory"])
    lessons[0]["instruction"] = "untrusted injected instruction"
    rendered = lesson_context(lessons)
    assert "untrusted injected" not in rendered
    assert "Preserve action phase, status and args" in rendered
    assert verified["id"] in rendered
    assert lesson_context([{"lesson_key": "unregistered", "item_id": "a" * 64}]) == ""


def test_runtime_lesson_use_outcome_and_storage_failure_no_repeat(store):
    verified, _, _ = _verify(store)
    runtime = ImprovementRuntime(lambda: store)
    prepared = runtime.prepare(runtime.targets([], "inspect trajectory"))
    assert prepared["context"]
    result = {"ok": True, "steps": []}
    assert runtime.record(result, prepared)["status"] == "recorded"
    assert result == {"ok": True, "steps": []}
    events = store.history(verified["id"])["events"]
    assert [event["event"] for event in events][-2:] == ["lesson_used", "lesson_outcome"]

    def unavailable():
        raise OSError("private storage location")

    broken = ImprovementRuntime(unavailable)
    assert broken.record(result, prepared)["status"] == "pending"
    assert "private storage" not in json.dumps(broken.record(result, prepared))
    assert result["ok"] is True


def test_http_claim_validation_review_use_protected_evidence_only(store):
    app = FastAPI()
    register_improvement_routes(app, ImprovementDeps(store=lambda: store), HTTPException)
    client = TestClient(app)
    action = {"ok": False, "steps": [{"tool": "trajectory", "status": "error", "args": {"detail": "full"}}]}
    observed = client.post("/agent/improvements/reconcile", json={"episode_id": "route-goal", "result": action}).json()
    assert observed["grounded"] and observed["usage"]["total_tokens"] == 0
    item = observed["result"][0]
    claim = client.post(f"/agent/improvements/{item['id']}/claim", json={"owner": "builder", "version": item["version"]}).json()["result"]
    owner = {key: claim[key] for key in ("owner", "generation", "claim_token")}
    forged = client.post(f"/agent/improvements/{item['id']}/review", json={"reviewer": "independent-reviewer", "passed": True})
    assert forged.status_code == 409
    asserted = client.post(f"/agent/improvements/{item['id']}/validation", json={**owner, "passed": True})
    assert asserted.status_code == 409
    candidate = store.begin_candidate(item["id"], changed_files=["src/adapter.py"], **owner)
    for check in ("reproducer", "privacy"):
        completed = subprocess.run([sys.executable, "-c", "print('check passed')"], capture_output=True)
        ref = store.write_validation_receipt(candidate, check=check, completed=completed, report=completed.stdout)
        response = client.post(f"/agent/improvements/{item['id']}/validation", json={**owner, "evidence_ref": ref})
        assert response.status_code == 200
    review = store.write_review_receipt(item["id"], reviewer="independent-reviewer", accepted=True,
                                       lesson_key="trajectory_observation_conservation", report=b"independent report")
    verified = client.post(f"/agent/improvements/{item['id']}/review", json={"evidence_ref": review})
    assert verified.json()["result"]["state"] == "verified"
    assert client.get("/agent/improvements/lessons", params={"target": "trajectory"}).json()["result"]
    assert client.get(f"/agent/improvements/{item['id']}").json()["result"]["occurrences"][0]["evidence"]["args"] == {"detail": "full"}
    assert "claim_token" not in client.get("/agent/improvements").text


def test_runtime_config_is_opt_in_and_owner_only(tmp_path, monkeypatch):
    monkeypatch.delenv("NPA_AGENT_IMPROVEMENT_CONFIG", raising=False)
    assert ImprovementRuntime().prepare([])["status"] == "disabled"
    config = tmp_path / "runtime.json"
    config.write_text("{}")
    config.chmod(0o644)
    with pytest.raises(ImprovementError, match="owner-only"):
        store_from_config(config)
    monkeypatch.setenv("NPA_AGENT_IMPROVEMENT_CONFIG", str(config))
    assert ImprovementRuntime().prepare([])["status"] == "pending"


def test_episode_links_are_hashed_and_session_bound(store):
    result = {"ok": False, "steps": [{"tool": "trajectory", "status": "error"}]}
    item = store.observe_action(result, episode_id="episode-source", session_id="parent-session")[0]
    evidence = store.history(item["id"])["occurrences"][0]
    assert evidence["episode_ref"] == hashlib.sha256(json.dumps("episode-source").encode()).hexdigest()
    assert evidence["session_ref"]


def test_private_evidence_fifo_rejected_without_open(store, monkeypatch):
    from npa.agent_backend.improvements import _read_private
    fifo = store.evidence_directory / "input"
    os.mkfifo(fifo, 0o600)
    calls = []
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ImprovementError, match="regular file"):
        _read_private(fifo)
    assert calls == []


def test_database_fifo_rejected_before_sqlite_connect(tmp_path, monkeypatch):
    import sqlite3
    queue = tmp_path / "queue"
    queue.mkdir(mode=0o700)
    os.mkfifo(queue / "improvements.sqlite3", 0o600)
    calls = []
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ImprovementError):
        _store(tmp_path)
    assert calls == []


def test_invalid_storage_config_creates_no_source_directories(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ImprovementError, match="outside"):
        ImprovementStore(repo / "queue", repository=repo, evidence_directory=repo / "evidence", scopes=[_scope()], reviewers=[])
    assert list(repo.iterdir()) == []


def test_changed_evidence_disables_already_verified_lesson(store):
    _, refs, _ = _verify(store)
    (store.evidence_directory / (refs[0] + ".json")).unlink()
    assert store.matching_verified_lessons(["trajectory"]) == []


def test_actual_exhaustion_reason_creates_finding():
    result = {"ok": False, "steps": [], "stopped_reason": "max_steps"}
    assert find_improvements(result)[0]["kind"] == "max_steps_exhausted"


@pytest.mark.parametrize("stage,kind", [("gate", "drive_error"), ("adjust", "drive_adjust_error"), ("diagnose", "drive_diagnosis_error")])
def test_detector_consumes_real_drive_error_shapes(stage, kind):
    from npa.agent_backend.sim2real_loop import drive_sim2real_loop

    def failed(*args):
        raise RuntimeError("synthetic component failure")

    callbacks = {
        "gate": lambda *args: {"success_rate": 0.2, "threshold": 0.8},
        "diagnose": lambda *args: {"failure_mode": "synthetic"},
        "adjust": lambda config, diagnosis: config,
    }
    callbacks[stage] = failed
    result = drive_sim2real_loop(
        "evaluate actual drive trace", config={"run_id": "synthetic-run"},
        launch=lambda config: {"run_id": "synthetic-run"},
        status=lambda run: {"ok": True, "run": {"run_id": run}, "sim_viz": {"run_id": run, "stage": "evaluation"}},
        confirm_token="synthetic-consent", session_token="synthetic-consent", **callbacks,
    )
    assert isinstance(result["iterations"][0]["status"], dict)
    finding = find_improvements(result)[0]
    assert finding["kind"] == kind and finding["component"] == "sim2real-drive"


def test_sensitive_check_and_source_names_preserve_hash_evidence(tmp_path):
    scope = ImprovementScope(scope_id="safe-checks", component="trajectory", files=("src/secret_guard.py",),
                             base_revision="a" * 40, required_checks=("secret_scan",),
                             lesson_keys=("trajectory_observation_conservation",))
    store = _store(tmp_path, scopes=[scope])
    (store.repository / "src").mkdir()
    (store.repository / "src/secret_guard.py").write_text("safe = True\n")
    claim, ownership = _claim(store)
    candidate = store.begin_candidate(claim["id"], changed_files=["src/secret_guard.py"], **ownership)
    completed = subprocess.run([sys.executable, "-c", "print('scan passed')"], capture_output=True)
    reference = store.write_validation_receipt(candidate, check="secret_scan", completed=completed, report=completed.stdout)
    result = store.record_validation(claim["id"], evidence_ref=reference, **ownership)
    assert result["state"] == "ready_for_review"
    review = store.write_review_receipt(claim["id"], reviewer="independent-reviewer", accepted=True,
                                       lesson_key="trajectory_observation_conservation", report=b"reviewed")
    assert store.review(claim["id"], evidence_ref=review)["state"] == "verified"
    assert store.matching_verified_lessons(["trajectory"])


def test_claim_credential_is_removed_by_trajectory_sanitizer(store):
    from npa.agent_backend.trajectory import redact
    claim, _ = _claim(store)
    assert redact(claim)["claim_token"] == "<redacted>"
    assert claim["claim_token"] not in store.path.read_text(errors="ignore")


def test_runtime_reloads_same_stat_config_and_revalidates_permissions(tmp_path, monkeypatch):
    store = _store(tmp_path)
    config = tmp_path / "runtime.json"
    payload = {"directory": str(store.directory), "repository": str(store.repository),
               "evidence_directory": str(store.evidence_directory), "scopes": [asdict(_scope())],
               "reviewers": ["reviewer-one"]}
    config.write_text(json.dumps(payload))
    config.chmod(0o600)
    monkeypatch.setenv("NPA_AGENT_IMPROVEMENT_CONFIG", str(config))
    runtime = ImprovementRuntime()
    assert runtime.store().reviewers == {"reviewer-one"}
    original = config.stat()
    payload["reviewers"] = ["reviewer-two"]
    config.write_text(json.dumps(payload))
    os.utime(config, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert config.stat().st_size == original.st_size
    assert runtime.store().reviewers == {"reviewer-two"}
    config.chmod(0o644)
    with pytest.raises(ImprovementError):
        runtime.store()
    config.chmod(0o600)
    target = tmp_path / "replacement.json"
    config.rename(target)
    config.symlink_to(target)
    with pytest.raises(ImprovementError):
        runtime.store()


def test_database_replacement_fifo_rejected_on_each_transaction(store, monkeypatch):
    import sqlite3
    store.path.unlink()
    os.mkfifo(store.path, 0o600)
    calls = []
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ImprovementError):
        store.list_items()
    assert calls == []


def test_dot_segments_cannot_create_queue_inside_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ImprovementError, match="outside"):
        ImprovementStore(tmp_path / "other/../repo/queue", repository=repo,
                         evidence_directory=tmp_path / "evidence", scopes=[_scope()], reviewers=[])
    assert list(repo.iterdir()) == []


def test_target_matching_failure_is_visible_without_private_payload(caplog):
    def unavailable():
        raise OSError("private-runtime-config synthetic-customer-value")

    runtime = ImprovementRuntime(unavailable)
    with caplog.at_level("DEBUG", logger="npa.agent_backend.improvement_routes"):
        assert runtime.targets(["agent-development"], "untrusted input") == ["agent-development"]
    assert "Improvement target matching unavailable; retaining selected skills" in caplog.text
    assert "private-runtime-config" not in caplog.text
    assert "synthetic-customer-value" not in caplog.text
    assert "untrusted input" not in caplog.text
