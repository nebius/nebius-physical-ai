"""Tier-0 tests for quantitative viewer signals + cross-run memory (Phase F)."""

from __future__ import annotations

from npa.cli import agent_memory as M
from npa.cli import agent_visual_feedback as V


# ── Gap 4: quantitative viewer eval ──────────────────────────────────────────


def test_extract_signals_reads_success_rate_and_threshold():
    sig = V.extract_quantitative_signals({"success_rate": 0.75, "threshold": 0.8})
    assert sig["has_signal"] is True
    assert sig["success_rate"] == 0.75
    assert sig["meets_threshold"] is False


def test_extract_signals_detects_policy_collapse():
    sig = V.extract_quantitative_signals({"success_rate": 0.0, "threshold": 0.8})
    assert sig["policy_collapse"] is True
    assert any("collapse" in n for n in sig["notes"])


def test_extract_signals_detects_degenerate_from_per_env():
    sig = V.extract_quantitative_signals({"per_env": [0.0, 0.0, 0.0, 0.0]})
    assert sig["degenerate"] is True
    assert sig["num_envs"] == 4
    assert sig["passed_envs"] == 0


def test_extract_signals_no_signal_when_empty():
    sig = V.extract_quantitative_signals({})
    assert sig["has_signal"] is False
    assert any("no numeric signal" in n for n in sig["notes"])


def test_compare_rollouts_flags_regression():
    cmp = V.compare_rollouts({"success_rate": 0.8}, {"success_rate": 0.5})
    assert cmp["regressed"] is True
    assert cmp["verdict"] == "regression"
    assert cmp["delta_success_rate"] == -0.3


def test_compare_rollouts_flags_improvement():
    cmp = V.compare_rollouts({"success_rate": 0.5}, {"success_rate": 0.9})
    assert cmp["improved"] is True
    assert cmp["verdict"] == "improvement"


def test_compare_rollouts_collapse_is_regression_even_without_baseline_delta():
    cmp = V.compare_rollouts({"success_rate": 0.4}, {"success_rate": 0.0})
    assert cmp["regressed"] is True


# ── Gap 5: cross-run memory ──────────────────────────────────────────────────


def test_record_and_get_run_roundtrip():
    mem = M.RunMemory(M.InMemoryStore())
    mem.record_run("run-a", {"success_rate": 0.8, "config": {"num_envs": 4}})
    got = mem.get_run("run-a")
    assert got["run_id"] == "run-a"
    assert got["success_rate"] == 0.8


def test_list_runs_is_most_recent_first():
    mem = M.RunMemory(M.InMemoryStore())
    mem.record_run("run-1", {"success_rate": 0.1})
    mem.record_run("run-2", {"success_rate": 0.2})
    assert mem.list_runs()[0] == "run-2"
    assert set(mem.list_runs()) == {"run-1", "run-2"}


def test_compare_runs_uses_injected_comparator():
    mem = M.RunMemory(M.InMemoryStore(), comparator=V.compare_rollouts)
    mem.record_run("base", {"success_rate": 0.9})
    mem.record_run("cand", {"success_rate": 0.6})
    result = mem.compare_runs("base", "cand")
    assert result["ok"] is True
    assert result["regressed"] is True


def test_compare_runs_missing_metadata_is_grounded_error():
    mem = M.RunMemory(M.InMemoryStore())
    mem.record_run("base", {"success_rate": 0.9})
    result = mem.compare_runs("base", "nope")
    assert result["ok"] is False
    assert "nope" in result["error"]


def test_explain_regression_is_grounded_text():
    mem = M.RunMemory(M.InMemoryStore(), comparator=V.compare_rollouts)
    mem.record_run("run-a", {"success_rate": 0.85})
    mem.record_run("run-b", {"success_rate": 0.55})
    text = mem.explain_regression("run-b", "run-a")
    assert "regression" in text.lower()
    assert "delta_success_rate" in text


def test_run_id_path_traversal_is_contained(tmp_path):
    base = tmp_path / "memory"
    store = M.JsonFileStore(str(base))
    mem = M.RunMemory(store)
    # A crafted run_id must not escape the memory store directory.
    mem.record_run("../../evil", {"success_rate": 0.5})
    escaped = tmp_path / "evil.json"
    assert not escaped.exists()
    # And the sanitized record is retrievable + confined under base.
    files = list(base.rglob("*.json"))
    assert files, "record should be written under the store base dir"
    for f in files:
        assert str(base) in str(f.resolve())


def test_record_run_stamps_provenance_source():
    mem = M.RunMemory(M.InMemoryStore())
    api_rec = mem.record_run("r-api", {"success_rate": 0.5})
    drive_rec = mem.record_run("r-drive", {"success_rate": 0.9}, source="drive")
    assert api_rec["source"] == "api"
    assert drive_rec["source"] == "drive"


def test_run_index_is_capped():
    mem = M.RunMemory(M.InMemoryStore())
    for i in range(M.MAX_INDEX_ENTRIES + 25):
        mem.record_run(f"run-{i}", {"success_rate": 0.1})
    # The full index never exceeds the cap, and the most-recent run stays first.
    all_runs = mem.list_runs(limit=10_000)
    assert len(all_runs) <= M.MAX_INDEX_ENTRIES
    assert all_runs[0] == f"run-{M.MAX_INDEX_ENTRIES + 24}"


def test_json_file_store_roundtrip(tmp_path):
    store = M.JsonFileStore(str(tmp_path / "memory"))
    mem = M.RunMemory(store)
    mem.record_run("run-x", {"success_rate": 0.7})
    # New RunMemory over the same dir reads persisted data.
    mem2 = M.RunMemory(M.JsonFileStore(str(tmp_path / "memory")))
    assert mem2.get_run("run-x")["success_rate"] == 0.7
    assert "run-x" in mem2.list_runs()
