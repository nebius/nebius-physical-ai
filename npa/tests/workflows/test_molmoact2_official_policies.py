"""Contract checks for the real five-stage official MolmoAct2 LIBERO path."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.workflows.byof import molmoact2_pipeline as pipeline


REPO_ROOT = Path(__file__).parents[3]
SPEC = REPO_ROOT / "workflows" / "testing" / "molmoact2-official-policies.yaml"
DOCKERFILE = REPO_ROOT / "npa" / "docker" / "workbench" / "molmoact2" / "Dockerfile"


def test_official_libero_workflow_has_five_connected_substantive_stages() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="molmoact2-contract")

    assert [step.state for step in plan.steps] == [
        "prepare_libero_dataset",
        "finetune_official_policy",
        "rollout_closed_loop_libero",
        "evaluate_heldout_actions",
        "emit_factual_visualization",
    ]
    assert all(
        step.argv[:3] == ["python3", "-m", "npa.workflows.byof.molmoact2_pipeline"]
        for step in plan.steps
    )
    # Dataset preparation feeds training and held-out evaluation; training feeds
    # closed-loop rollout and action evaluation; rollout and evaluation feed RRD.
    assert plan.steps[1].inputs[0]["uri"] == plan.steps[0].outputs[0]["uri"]
    assert plan.steps[2].inputs[0]["uri"] == plan.steps[1].outputs[0]["uri"]
    assert plan.steps[3].inputs[0]["uri"] == plan.steps[0].outputs[0]["uri"]
    assert plan.steps[3].inputs[1]["uri"] == plan.steps[1].outputs[0]["uri"]
    assert plan.steps[3].inputs[2]["uri"] == plan.steps[2].outputs[0]["uri"]
    assert plan.steps[4].inputs[0]["uri"] == plan.steps[2].outputs[0]["uri"]
    assert plan.steps[4].inputs[1]["uri"] == plan.steps[3].outputs[0]["uri"]
    assert plan.steps[2].argv[-1] == "50"
    assert plan.steps[1].resources_profile["accelerators"] == "RTXPRO6000:1"
    assert "stub" not in SPEC.read_text(encoding="utf-8").lower()


def test_split_is_deterministic_disjoint_and_nonempty() -> None:
    train_a, heldout_a = pipeline._deterministic_split(17, "immutable-revision", 0.2)
    train_b, heldout_b = pipeline._deterministic_split(17, "immutable-revision", 0.2)

    assert (train_a, heldout_a) == (train_b, heldout_b)
    assert set(train_a).isdisjoint(heldout_a)
    assert sorted(train_a + heldout_a) == list(range(17))
    assert train_a and heldout_a


def test_dataset_contract_rejects_a_missing_official_camera(tmp_path: Path) -> None:
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "features": {
                    "action": {},
                    "observation.state": {},
                    "observation.images.image": {},
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        pipeline._dataset_contract(tmp_path)
    except pipeline.MolmoAct2PipelineError as exc:
        assert "wrist_image" in str(exc)
    else:
        raise AssertionError("the official two-camera contract must be enforced")


def test_workflow_never_passes_foundation_checkpoint_to_rollout() -> None:
    plan = build_plan(load_spec(SPEC), run_id="molmoact2-lineage")
    rollout_args = plan.steps[2].argv

    assert pipeline.BASE_CHECKPOINT not in rollout_args
    assert "--checkpoint-uri" in rollout_args
    assert plan.steps[1].outputs[0]["uri"] in rollout_args


def test_training_uses_upstream_lora_path_that_emits_an_inference_checkpoint() -> None:
    source = Path(pipeline.__file__).read_text(encoding="utf-8")

    assert '"--lora_enable=true"' in source
    assert '"--lora_rank=64"' in source
    assert 'root.glob("step*-merged")' in source


def test_finetune_materializes_the_exact_foundation_revision(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []
    checkpoint = tmp_path / "foundation-checkpoint"

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        target = Path(str(kwargs["local_dir"]))
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    assert pipeline._download_base_checkpoint(tmp_path) == checkpoint
    assert calls == [
        {
            "repo_id": pipeline.BASE_CHECKPOINT,
            "repo_type": "model",
            "revision": pipeline.BASE_CHECKPOINT_REVISION,
            "local_dir": str(checkpoint),
        }
    ]


def test_finetune_passes_the_materialized_checkpoint_to_the_native_trainer(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "prepared" / "train"
    heldout_root = tmp_path / "prepared" / "heldout"
    base_checkpoint = tmp_path / "exact-foundation"
    data_root.mkdir(parents=True)
    heldout_root.mkdir(parents=True)
    base_checkpoint.mkdir()
    commands: list[list[str]] = []

    monkeypatch.setattr(
        pipeline,
        "_download_prepared",
        lambda *_args: (
            data_root,
            heldout_root,
            {"train_episode_indices": [0], "heldout_episode_indices": [1]},
        ),
    )
    monkeypatch.setattr(pipeline, "_upstream_root", lambda: tmp_path)
    monkeypatch.setattr(
        pipeline, "_download_base_checkpoint", lambda _work: base_checkpoint
    )

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        output = Path(
            next(arg for arg in command if arg.startswith("--save_folder="))[14:]
        )
        (output / "step-0001-merged").mkdir(parents=True)

    monkeypatch.setattr(pipeline, "_run", fake_run)
    monkeypatch.setattr(pipeline, "_upload", lambda _source, target: target)

    manifest = pipeline.finetune(
        types.SimpleNamespace(
            work_root=str(tmp_path / "work"),
            prepared_dataset_uri="prepared-input",
            checkpoint_uri="checkpoint-output",
        )
    )

    assert commands[0][4] == str(base_checkpoint)
    assert manifest["training_command"][4] == str(base_checkpoint)
    assert manifest["base_checkpoint"]["revision"] == pipeline.BASE_CHECKPOINT_REVISION


def test_live_matrix_does_not_require_an_optional_hub_token() -> None:
    case = next(
        case
        for case in SUBMIT_LIVE_MATRIX
        if case.spec == "molmoact2-official-policies.yaml"
    )

    assert "HF_TOKEN" not in case.secret_envs
    assert case.secret_envs == ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


def test_runtime_image_removes_nonruntime_payloads_in_their_creating_layers() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    source_layer = dockerfile.split("# The two upstream dependency sets", maxsplit=1)[0]

    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in dockerfile
    assert "rm -rf /opt/molmoact2/experiments/lerobot/tests" in source_layer
    assert source_layer.index("experiments/lerobot/tests") < source_layer.index(
        "rm -rf /opt/molmoact2/.git"
    )
    assert "torchmetrics/functional/image/lpips_models" in dockerfile
    assert "torchmetrics/functional/image/dists_models" in dockerfile
    assert "imageio_ffmpeg/binaries/*" in dockerfile


def test_runtime_image_declares_the_verified_skypilot_bootstrap_contract() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert (
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"'
        in dockerfile
    )
    # The attestation is backed by the actual non-root worker prerequisites,
    # not used as a substitute for the target-cluster capability check.
    assert "openssh-server" in dockerfile
    assert "rsync sudo" in dockerfile
    assert 'USER ubuntu' in dockerfile
    assert "NOPASSWD:ALL" in dockerfile


def test_runtime_image_bakes_the_npa_console_for_skypilot_setup() -> None:
    """The worker setup clears PYTHONPATH before discovering the image CLI."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY pyproject.toml README.md /opt/npa-src/" in dockerfile
    assert "COPY src/npa /opt/npa-src/src/npa" in dockerfile
    assert "pip install --no-deps /opt/npa-src" in dockerfile
    assert "command -v npa" in dockerfile
    assert "env -u PYTHONPATH npa --help >/dev/null" in dockerfile
