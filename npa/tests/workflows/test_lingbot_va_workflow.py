"""Contract tests for the quarantined LingBot-VA LIBERO-Long workflow."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from npa.solutions import lingbot_va as L
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/lingbot-va-libero-long.yaml"
DOCKERFILE = ROOT / "npa/docker/workbench/lingbot-va/Dockerfile"


def _dataset(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"codebase_version":"v2.1"}\n')
    records = []
    for index in range(2):
        record = {
            "episode_index": index,
            "length": 4,
            "tasks": [f"long-horizon task {index}"],
            "action_config": [
                {"start_frame": 0, "end_frame": 4, "action_text": "move then place"}
            ],
        }
        records.append(json.dumps(record))
        for camera in L.CAMERAS:
            latent = (
                root / "latents/chunk-000" / camera / f"episode_{index:06d}_0_4.pth"
            )
            latent.parent.mkdir(parents=True, exist_ok=True)
            latent.write_bytes(b"not-read-by-prepare")
    (root / "meta/episodes.jsonl").write_text("\n".join(records) + "\n")
    (root / "empty_emb.pt").write_bytes(b"runtime-loads-real-tensor")
    return root


def test_prepare_requires_real_lerobot_metadata_action_segments_and_latents(
    tmp_path: Path,
) -> None:
    source = _dataset(tmp_path / "source")
    destination = tmp_path / "prepared.json"

    result = L.prepare(f"file://{source}", f"file://{destination}")

    persisted = json.loads(destination.read_text())
    assert result["prepared_manifest_uri"] == f"file://{destination}"
    assert persisted["attention"] == {"training": "flex", "inference": "torch"}
    assert persisted["action_contract"] == L.ACTION_CONTRACT
    assert persisted["train_episode_indices"]
    assert persisted["heldout_episode_indices"]
    assert set(persisted["train_episode_indices"]).isdisjoint(
        persisted["heldout_episode_indices"]
    )
    assert persisted["provenance"]["dataset"]["license"] == "CC BY-NC-SA-4.0"


def test_workflow_is_five_connected_native_stages_with_exact_artifact_handoffs() -> (
    None
):
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    plan = build_plan(spec, run_id="lingbot-va-contract")
    payload = yaml.safe_load(WORKFLOW.read_text())

    assert list(spec.states) == [
        "prepare",
        "posttrain",
        "rollout",
        "evaluate",
        "visualize",
    ]
    assert len(plan.steps) == 5
    assert payload["resources"]["train"]["accelerators"] == "B200:8"
    assert payload["resources"]["gpu"]["accelerators"] == "B200:1"
    assert (
        "REPLACE_WITH_VERIFIED_CANDIDATE_DIGEST" in payload["config"]["candidate_image"]
    )
    for state in payload["states"].values():
        argv = state["run"]["argv"]
        assert argv[:5] == [
            "lingbot-va-runtime",
            "exec",
            "python",
            "-m",
            "npa.solutions.lingbot_va",
        ]
        assert "echo" not in argv
        assert "sleep" not in argv
    assert (
        payload["states"]["posttrain"]["inputs"][0]["uri"]
        == payload["states"]["prepare"]["outputs"][0]["uri"]
    )
    assert (
        payload["states"]["rollout"]["inputs"][1]["uri"]
        == payload["states"]["posttrain"]["outputs"][0]["uri"]
    )
    assert (
        payload["states"]["evaluate"]["inputs"][0]["uri"]
        == payload["states"]["rollout"]["outputs"][1]["uri"]
    )
    assert (
        payload["states"]["visualize"]["inputs"][2]["uri"]
        == payload["states"]["evaluate"]["outputs"][0]["uri"]
    )


def test_source_only_image_pins_the_distinct_cuda_contract_without_extra_acceptance() -> (
    None
):
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    requirements = DOCKERFILE.parent.joinpath("runtime-requirements.txt").read_text(
        encoding="utf-8"
    )
    runtime_script = DOCKERFILE.parent.joinpath("lingbot_va_runtime.sh").read_text(
        encoding="utf-8"
    )

    assert L.SOURCE_REF in dockerfile
    assert L.POSTTRAIN_MODEL_REF in dockerfile
    assert "torch==2.9.0+cu126" in requirements
    assert "lerobot==0.3.3" in runtime_script
    assert "--no-deps" in runtime_script
    assert "ACCEPT_" not in dockerfile
    assert "robbyant/libero-long-lerobot" not in dockerfile
    assert "npa-wan2-2@sha256:" in dockerfile
    assert "install -d -m 0755 /opt/npa-native/npa/solutions" in dockerfile
    assert "/usr/share/doc/npa-lingbot-va" in dockerfile


def test_visualization_binds_the_recording_to_the_workflow_run() -> None:
    """Require the factual RRD to retain its exact workflow run identity."""
    source = Path(L.__file__).read_text(encoding="utf-8")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert "NPA_WORKFLOW_RUN_ID" in source
    assert 'recording_id=run_id' in source
    assert workflow["states"]["visualize"]["outputs"][0]["schema"] == (
        "application/vnd.rerun.rrd"
    )
