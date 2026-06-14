"""Tests for swappable Sim2Real policy container images."""

from __future__ import annotations

import json
from pathlib import Path

from npa.workflows.sim2real.engine import _component_job_script
from npa.workflows.sim2real_envgen import (
    EnvGenConfig,
    _policy_action_amplitude,
    build_policy_image_contract,
    write_action_conditioned_envs,
)


def test_policy_image_contract_documents_swap_points() -> None:
    contract = build_policy_image_contract(
        train_envs_uri="s3://bucket/run/envs/train/envs.jsonl",
        output_uri="s3://bucket/run/actions/train/",
        default_policy_image="cr.example/npa-sim2real-reference-policy:0.1.1",
    )
    assert contract["schema"] == "npa.sim2real.policy_image_contract.v1"
    assert "train_envs_uri" in contract["input"]
    assert "action_conditioned_envs_uri" in contract["output"]
    assert "--policy-image" in contract["overrides"]


def test_policy_action_amplitude_variants(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_POLICY_VARIANT", "reference")
    assert _policy_action_amplitude() == 0.025
    monkeypatch.setenv("NPA_SIM2REAL_POLICY_VARIANT", "explore")
    assert _policy_action_amplitude() == 0.085


def test_reference_and_explore_policy_variants_emit_distinct_actions(
    tmp_path, monkeypatch
) -> None:
    rows = [{"env_id": "env-0000", "seed": 1}]

    class _FakeClient:
        def download_path(self, *_args, **_kwargs):
            return tmp_path / "train-envs.jsonl"

        def upload_file(self, local_path, uri):
            return uri

    monkeypatch.setattr(
        "npa.workflows.sim2real_envgen.StorageClient.from_environment",
        lambda **_kwargs: _FakeClient(),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_envgen._read_jsonl",
        lambda _path: rows,
    )

    def _max_abs(values: list[list[float]]) -> float:
        return max(abs(component) for row in values for component in row[:3])

    def _read_actions(path: Path) -> list[list[float]]:
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return lines[0]["actions"]["values"]

    monkeypatch.setenv("NPA_SIM2REAL_POLICY_VARIANT", "reference")
    ref = write_action_conditioned_envs(
        EnvGenConfig(run_id="r", output_uri="s3://bucket/run/", env_count=1, seed=7),
        tmp_path / "ref",
        policy_image="cr.example/npa-sim2real-reference-policy:0.1.1",
        limit=1,
        train_envs_uri="s3://bucket/run/envs/train/envs.jsonl",
    )
    ref_amp = _max_abs(_read_actions(tmp_path / "ref" / "action-conditioned-train-envs.jsonl"))

    monkeypatch.setenv("NPA_SIM2REAL_POLICY_VARIANT", "explore")
    write_action_conditioned_envs(
        EnvGenConfig(run_id="r", output_uri="s3://bucket/run/", env_count=1, seed=7),
        tmp_path / "alt",
        policy_image="cr.example/npa-sim2real-explore-policy:0.1.1",
        limit=1,
        train_envs_uri="s3://bucket/run/envs/train/envs.jsonl",
    )
    alt_amp = _max_abs(_read_actions(tmp_path / "alt" / "action-conditioned-train-envs.jsonl"))

    assert ref["policy_image"].endswith("reference-policy:0.1.1")
    assert alt_amp > ref_amp


def test_isaac_heldout_script_requires_source_tarball() -> None:
    script = _component_job_script("heldout_eval", sim_backend="isaac")
    assert "NPA_SIM2REAL_SOURCE_TARBALL_URI" in script
    assert "missing NPA_SIM2REAL_SOURCE_TARBALL_URI" in script
    assert "heldout_entry" in script
    assert "sim2real.cli" not in script
    assert "git clone" not in script
