from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import pytest

from npa.benchmarks.sim2real_model_agent import (
    CHECKPOINT_MARKER,
    EmptyStreamError,
    _context_checkpoint,
    _inside,
    _last_request_index,
    _load_transcript,
    _run_tool,
    _stream_chat,
    _workspace_preflight,
)
from npa.benchmarks.sim2real_model_server import render_server_resources
from npa.benchmarks.sim2real_success import VerificationError, _lift_evidence


def _manifest(*, duration: float = 2.0, timestamped: bool = True) -> dict:
    rows = []
    for index, when in enumerate((0.0, 1.0, duration)):
        row = {
            "step": index,
            "sim_step": index,
            "simulator_ground_truth": {
                "stable_grasp": True,
                "object_lift_m": 0.051,
            },
        }
        if timestamped:
            row["sim_time_seconds"] = when
        rows.append(row)
    return {
        "schema": "npa.sim2real.action_rollout.v1",
        "source": "byo_isaac_policy_rollout",
        "sim_backend": "isaac",
        "policy_trained": True,
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 10,
        "rollout_id": "rollout-1",
        "simulation_step_seconds": 1.0,
        "actions": rows,
    }


def test_lift_evidence_requires_physical_two_second_dwell(tmp_path: Path) -> None:
    found = _lift_evidence(
        tmp_path / "manifest.json",
        _manifest(),
        minimum_lift_m=0.05,
        minimum_hold_seconds=2.0,
    )
    assert found is not None
    assert found.duration_seconds == 2.0
    assert found.minimum_lift_m == pytest.approx(0.051)


def test_lift_evidence_rejects_short_or_broken_hold(tmp_path: Path) -> None:
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            _manifest(duration=1.99),
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )
    broken = _manifest()
    broken["actions"][1]["simulator_ground_truth"]["stable_grasp"] = False
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            broken,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )


def test_lift_evidence_refuses_sample_count_as_time(tmp_path: Path) -> None:
    manifest = _manifest(timestamped=False)
    manifest.pop("simulation_step_seconds")
    with pytest.raises(VerificationError, match="simulation_step_seconds"):
        _lift_evidence(
            tmp_path / "manifest.json",
            manifest,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )


def test_lift_evidence_rejects_gaps_in_temporal_coverage(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["actions"][-1]["sim_step"] = 3
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            manifest,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )


def test_file_tools_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _inside(tmp_path, "../outside")
    result = _run_tool(
        "write_file", {"path": "notes/result.txt", "content": "ok"}, tmp_path, {}
    )
    assert result["bytes"] == 2
    assert (tmp_path / "notes/result.txt").read_text() == "ok"


def test_tool_schema_round_trips_json_arguments(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"ok": True}))
    result = _run_tool("read_file", {"path": "data.json"}, tmp_path, {})
    assert json.loads(result["content"]) == {"ok": True}


def test_complete_workflow_uses_authoritative_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps({"status": "SUCCEEDED"}), ""
        ),
    )
    result = _run_tool(
        "complete_workflow",
        {"run_id": "sim2real-run-1"},
        tmp_path,
        {"NPA_PROJECT": "private-alias"},
    )
    assert result["terminal"] is True
    assert result["workflow_succeeded"] is True
    assert result["status"] == "SUCCEEDED"


def test_complete_workflow_rejects_untrusted_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid format"):
        _run_tool(
            "complete_workflow",
            {"run_id": "run; touch escaped"},
            tmp_path,
            {"NPA_PROJECT": "private-alias"},
        )


def test_empty_stream_is_a_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyResponse:
        def __enter__(self) -> EmptyResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(urllib.request, "urlopen", lambda _request: EmptyResponse())
    with pytest.raises(EmptyStreamError, match="empty event stream"):
        _stream_chat("http://model.example/v1", "key", {"messages": []})


def test_resume_loads_transcript_and_request_index(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"role": "assistant", "content": None, "tool_calls": []},
                {"role": "user", "content": "continue"},
            )
        )
        + "\n"
    )
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps({"request_index": 3})
        + "\n"
        + json.dumps({"request_index": 8, "transport_error": "empty"})
        + "\n"
    )

    assert [item["role"] for item in _load_transcript(transcript)] == [
        "assistant",
        "user",
    ]
    assert _last_request_index(requests) == 8


def test_context_checkpoint_is_bounded_and_becomes_resume_boundary(
    tmp_path: Path,
) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old-" + "x" * 100},
        {"role": "user", "content": "recent-1"},
        {"role": "assistant", "content": "recent-2"},
    ]
    compacted, checkpoint = _context_checkpoint(messages, max_recent_chars=80)
    assert compacted[:2] == messages[:2]
    assert compacted[2] == checkpoint
    assert checkpoint["content"].startswith(CHECKPOINT_MARKER)
    assert "recent-2" in checkpoint["content"]
    assert "old-" not in checkpoint["content"]

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(messages[2])
        + "\n"
        + json.dumps(checkpoint)
        + "\n"
        + json.dumps({"role": "assistant", "content": "after"})
        + "\n"
    )
    loaded = _load_transcript(transcript)
    assert loaded == [checkpoint, {"role": "assistant", "content": "after"}]


def test_workspace_resume_preserves_dirty_detached_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    subprocess.run(["git", "checkout", "--detach", "-q"], cwd=tmp_path, check=True)
    tracked.write_text("model change\n")

    with pytest.raises(ValueError, match="and clean"):
        _workspace_preflight(tmp_path, commit, require_clean=True)
    status = _workspace_preflight(tmp_path, commit, require_clean=False)
    assert "tracked.txt" in status


def test_model_server_renderer_pins_image_and_isolates_multinode_endpoint() -> None:
    model = {
        "repository": "org/model",
        "revision": "a" * 40,
        "server": "sglang",
        "server_image": "registry/server@sha256:" + "b" * 64,
        "tool_call_parser": "parser",
        "reasoning_parser": "reasoner",
        "context_limit": 1000,
        "tensor_parallel_size": 16,
        "server_arguments": ["--random-seed=7"],
    }
    resources = render_server_resources(model, namespace="trial-a", service_name="m")
    endpoint, statefulset = resources[2], resources[3]
    assert endpoint["spec"]["selector"] == {"statefulset.kubernetes.io/pod-name": "m-0"}
    assert statefulset["spec"]["replicas"] == 2
    container = statefulset["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].startswith("registry/server@sha256:")
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert "ORDINAL=${POD_NAME##*-}" in container["command"][-1]
    assert "--node-rank ${ORDINAL}" in container["command"][-1]
    cache_env = {
        item["name"]: item.get("value")
        for item in container["env"]
        if item["name"].endswith(("CACHE", "HOME"))
    }
    assert cache_env["HF_HUB_CACHE"].startswith("/mnt/data/model-cache/")
    assert cache_env["TRANSFORMERS_CACHE"] == cache_env["HF_HUB_CACHE"]
    network_env = {item["name"]: item.get("value") for item in container["env"]}
    assert network_env["GLOO_SOCKET_IFNAME"] == "eth0"
    assert network_env["NCCL_SOCKET_IFNAME"] == "eth0"
    pod_spec = statefulset["spec"]["template"]["spec"]
    assert pod_spec["hostNetwork"] is True
    assert any(volume["name"] == "infiniband" for volume in pod_spec["volumes"])


def test_glm_catalog_uses_release_with_blackwell_glm_support() -> None:
    npa_root = Path(__file__).resolve().parents[2]
    catalog = json.loads(
        (npa_root / "benchmarks/sim2real-three-model/models.json").read_text()
    )
    arguments = catalog["models"][0]["server_arguments"]
    assert catalog["models"][0]["server_image"].endswith(
        "@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155"
    )
    assert "--kv-cache-dtype=bfloat16" in arguments
    assert "--dsa-prefill-backend=flashmla_sparse" in arguments
    assert "--dsa-decode-backend=flashmla_kv" in arguments
    assert "--enforce-disable-flashinfer-allreduce-fusion" in arguments
    assert catalog["models"][0]["context_limit"] == 262144
