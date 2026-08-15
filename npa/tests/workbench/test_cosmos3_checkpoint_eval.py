"""Contracts for the Cosmos3 still-image checkpoint evaluator (no network/GPU)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from PIL import Image

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    render_skypilot_yaml,
    tool_image_key,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.workbench.cosmos.checkpoint_eval import (
    CAMPAIGN_CONFIG_SCHEMA,
    PINNED_GUARDRAIL_POSTURE,
    Cosmos3CheckpointEvalError,
    _verify_framework_provenance,
    evict_checkpoint_cache,
    execute_phase,
    parse_nvidia_smi_inventory,
    phase_arms,
    require_b200_gpu,
    run_checkpoint_arm,
    validate_campaign_config,
)
from npa.workbench.cosmos.cosmos3 import build_cosmos3_inference_args

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "npa/workflows/workbench/configs/cosmos3-checkpoint-eval.json"
SPEC_PATH = REPO_ROOT / "npa/workflows/workbench/npa-workflows/cosmos3-checkpoint-eval.yaml"


def _config() -> dict:
    return validate_campaign_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _fake_runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "cosmos-framework"
    (repo / "cosmos_framework" / "scripts").mkdir(parents=True)
    (repo / "cosmos_framework" / "scripts" / "inference.py").write_text(
        "", encoding="utf-8"
    )
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.chmod(0o755)
    return repo, {
        "COSMOS3_REPO": str(repo),
        "HF_TOKEN": "test-token",
        "HF_HOME": str(tmp_path / "hf-home"),
    }


class _FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.json_uploads: list[tuple[str, dict]] = []

    def upload_file(self, local_file: str, uri: str) -> str:
        path = Path(local_file)
        assert path.is_file()
        self.uploads.append((local_file, uri))
        if path.suffix == ".json":
            self.json_uploads.append((uri, json.loads(path.read_text(encoding="utf-8"))))
        return uri


class _FakeSampler:
    def start(self) -> None:
        pass

    def stop(self) -> dict:
        return {
            "available": True,
            "samples": 3,
            "peak_memory_mib_by_uuid": {"GPU-B200": 123456},
            "peak_memory_mib_sum": 123456,
            "sampling_interval_seconds": 0.5,
        }


def _b200_runner(argv, **kwargs):
    return subprocess.CompletedProcess(
        argv,
        0,
        "NVIDIA B200, GPU-synthetic, 183000, 580.00.00\n",
        "",
    )


def test_campaign_is_the_required_five_by_eight_guarded_matrix() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = validate_campaign_config(raw)

    assert raw["schema"] == CAMPAIGN_CONFIG_SCHEMA
    assert len(config["checkpoint_names"]) == 5
    assert len(config["prompts"]) == 8
    assert config["guardrails_enabled"] is True
    assert "Qwen3Guard" in raw["guardrail_posture"]["prompt_input"]
    assert "fail-open" in raw["guardrail_posture"]["generated_media_content_safety"]
    assert config["campaign"] == "cosmos3-checkpoint-evaluation"
    assert "operator_acceptance" not in config
    assert {entry["agreement"] for entry in config["licenses"]} == {
        "OpenMDW License Agreement 1.1",
        "NVIDIA Open Model License Agreement",
        "Apache License 2.0",
    }


def test_campaign_rejects_framework_guardrail_posture_drift() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["framework_commit"] == PINNED_GUARDRAIL_POSTURE["framework_commit"]
    raw["framework_commit"] = "0" * 40

    with pytest.raises(Cosmos3CheckpointEvalError, match="audited guardrail posture"):
        validate_campaign_config(raw)


def test_consistency_uses_only_two_new_seeds_and_never_repeats_primary() -> None:
    config = _config()
    selected = ["Cosmos3-Nano", "Cosmos3-Super"]

    arms = phase_arms(config, phase="consistency", top_checkpoints=selected)

    assert len(arms) == 4
    assert {checkpoint for checkpoint, _seed in arms} == set(selected)
    assert {seed for _checkpoint, seed in arms} == set(config["additional_seeds"])
    assert all(seed != config["primary_seed"] for _checkpoint, seed in arms)


@pytest.mark.parametrize(
    "selected",
    [[], ["Cosmos3-Nano"], ["Cosmos3-Nano", "Cosmos3-Nano"]],
)
def test_consistency_requires_exactly_two_distinct_campaign_checkpoints(selected) -> None:
    with pytest.raises(Cosmos3CheckpointEvalError, match="exactly two distinct"):
        phase_arms(_config(), phase="consistency", top_checkpoints=selected)


def test_b200_preflight_rejects_every_fallback_accelerator() -> None:
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            "NVIDIA H200, GPU-wrong, 141000, 580.65.06\n",
            "",
        )

    with pytest.raises(Cosmos3CheckpointEvalError, match="B200-only"):
        require_b200_gpu(runner=runner)


def test_b200_preflight_translates_missing_nvidia_smi() -> None:
    def runner(argv, **kwargs):
        raise FileNotFoundError("synthetic missing executable")

    with pytest.raises(Cosmos3CheckpointEvalError, match="nvidia-smi is unavailable"):
        require_b200_gpu(runner=runner)


def test_b200_inventory_is_redacted_hardware_provenance() -> None:
    parsed = parse_nvidia_smi_inventory(
        "NVIDIA B200, GPU-abc, 183359, 580.65.06\n"
    )

    assert parsed == [
        {
            "name": "NVIDIA B200",
            "uuid": "GPU-abc",
            "memory_total_mib": 183359,
            "driver_version": "580.65.06",
        }
    ]


def test_one_arm_loads_checkpoint_once_for_all_prompts_and_publishes_metrics(
    tmp_path: Path, mocker
) -> None:
    _repo, env = _fake_runtime(tmp_path)
    config = _config()
    config["prompts"] = config["prompts"][:2]
    expected_commit = config["framework_commit"]
    mocker.patch(
        "npa.workbench.cosmos.checkpoint_eval._framework_commit",
        return_value=expected_commit,
    )
    seen: list[list[str]] = []

    def runner(argv, **kwargs):
        seen.append(list(argv))
        input_path = Path(argv[argv.index("-i") + 1])
        output_root = Path(argv[argv.index("-o") + 1])
        lines = [json.loads(line) for line in input_path.read_text().splitlines()]
        for index, spec in enumerate(lines):
            sample_dir = output_root / spec["name"]
            sample_dir.mkdir(parents=True)
            image = Image.new("RGB", (64, 64), (20 + index, 40, 60))
            image.putpixel((0, 0), (220, 200, 180))
            image.save(sample_dir / "vision.jpg")
            (sample_dir / "sample_outputs.json").write_text(
                json.dumps({"status": "success", "args": {"seed": 314159}}),
                encoding="utf-8",
            )
        (output_root / "benchmark.json").write_text(
            json.dumps(
                {
                    "all": {"OmniInference.generate_batch": [1.25, 1.5]},
                    "average": {"OmniInference.generate_batch": 1.375},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0)

    storage = _FakeStorage()
    result = run_checkpoint_arm(
        config=config,
        checkpoint="Cosmos3-Nano",
        seed=314159,
        phase="primary",
        output_uri="s3://bucket/campaign",
        work_dir=tmp_path / "work",
        run_id="unit",
        runtime_image="registry/npa-cosmos3:test",
        gpu_inventory=[{"name": "NVIDIA B200"}],
        environ=env,
        runner=runner,
        sampler_factory=_FakeSampler,
        storage_client=storage,
    )

    assert len(seen) == 1
    expected_inference_args = build_cosmos3_inference_args(
        input_json=seen[0][seen[0].index("-i") + 1],
        output_dir=seen[0][seen[0].index("-o") + 1],
        checkpoint_path="Cosmos3-Nano",
        seed=314159,
        no_guardrails=False,
        parallelism_preset=config["parallelism_preset"],
        benchmark=True,
    )
    assert seen[0][3:] == expected_inference_args
    assert "--benchmark" in expected_inference_args
    assert "--no-guardrails" not in expected_inference_args
    input_lines = Path(seen[0][seen[0].index("-i") + 1]).read_text().splitlines()
    assert len(input_lines) == 2
    assert [sample["framework_latency_seconds"] for sample in result["samples"]] == [
        1.25,
        1.5,
    ]
    assert result["gpu_memory"]["peak_memory_mib_sum"] == 123456
    assert all(sample["artifact_uri"].startswith("s3://bucket/") for sample in result["samples"])
    assert result["provenance"]["weights_baked"] is False
    assert result["provenance"]["guardrails_enabled"] is True
    posture = result["provenance"]["guardrail_posture"]
    assert posture["prompt_input"]["safety_models"] == ["Blocklist", "Qwen3Guard"]
    assert posture["generated_media"]["content_safety_models"] == []
    assert posture["generated_media"]["postprocessors"] == ["RetinaFaceFilter"]
    assert result["provenance"]["declared_runtime_assets"]
    assert len(storage.uploads) == 3  # two images and one arm manifest


def test_cache_eviction_is_scoped_to_one_checkpoint_repo(tmp_path: Path) -> None:
    target = tmp_path / "hub" / "models--nvidia--Cosmos3-Nano"
    sibling = tmp_path / "hub" / "models--nvidia--Cosmos-Guardrail1"
    target.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (target / "blob").write_bytes(b"checkpoint")

    assert evict_checkpoint_cache(tmp_path, "nvidia/Cosmos3-Nano") is True
    assert not target.exists()
    assert sibling.exists()
    with pytest.raises(Cosmos3CheckpointEvalError, match="unsafe"):
        evict_checkpoint_cache(tmp_path, "../../outside")


def test_released_image_digest_proves_framework_when_git_metadata_is_absent(
    tmp_path: Path,
) -> None:
    config = _config()
    provenance = _verify_framework_provenance(
        repo=tmp_path,
        expected_commit=config["framework_commit"],
        runtime_image=f"registry/npa-cosmos3@{config['runtime_image_digest']}",
        expected_image_digest=config["runtime_image_digest"],
    )

    assert provenance == {
        "framework_commit": config["framework_commit"],
        "framework_commit_verification": "released-image-digest-contract",
        "runtime_image_digest": config["runtime_image_digest"],
    }


def test_packaged_runtime_without_git_requires_the_expected_image_digest(
    tmp_path: Path,
) -> None:
    config = _config()
    with pytest.raises(Cosmos3CheckpointEvalError, match="run the released image"):
        _verify_framework_provenance(
            repo=tmp_path,
            expected_commit=config["framework_commit"],
            runtime_image="registry/npa-cosmos3:mutable",
            expected_image_digest=config["runtime_image_digest"],
        )


def test_s3_campaign_config_download_uses_custom_work_dir(tmp_path: Path) -> None:
    class DownloadingStorage(_FakeStorage):
        def __init__(self) -> None:
            super().__init__()
            self.downloads: list[tuple[str, str]] = []

        def download_file(self, uri: str, local_file: str) -> str:
            self.downloads.append((uri, local_file))
            target = Path(local_file)
            target.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            return local_file

    storage = DownloadingStorage()
    custom_work_dir = tmp_path / "operator-work"

    plan = execute_phase(
        campaign_config="s3://example-bucket/input/config.json",
        phase="primary",
        output_uri="s3://example-bucket/output",
        work_dir=custom_work_dir,
        dry_run=True,
        storage_client=storage,
    )

    assert plan["status"] == "planned"
    assert storage.downloads == [
        (
            "s3://example-bucket/input/config.json",
            str(custom_work_dir.resolve() / "campaign-config.json"),
        )
    ]


def test_phase_publishes_complete_plan_before_first_arm(tmp_path: Path) -> None:
    storage = _FakeStorage()

    def interrupting_arm(**kwargs):
        raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        execute_phase(
            campaign_config=str(CONFIG_PATH),
            phase="primary",
            output_uri="s3://example-bucket/output",
            work_dir=tmp_path,
            run_id="interrupted",
            runtime_image="registry.example/npa-cosmos3@sha256:" + "b" * 64,
            storage_client=storage,
            gpu_probe_runner=_b200_runner,
            arm_runner=interrupting_arm,
        )

    phase_snapshots = [
        payload for uri, payload in storage.json_uploads if uri.endswith("/primary.json")
    ]
    assert len(phase_snapshots) == 1
    assert len(phase_snapshots[0]["arms"]) == 5
    assert {arm["status"] for arm in phase_snapshots[0]["arms"]} == {"planned"}


def test_cache_eviction_failure_does_not_increment_failed_arms(
    tmp_path: Path, mocker
) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["checkpoints"] = raw["checkpoints"][:1]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    storage = _FakeStorage()
    mocker.patch(
        "npa.workbench.cosmos.checkpoint_eval.evict_checkpoint_cache",
        side_effect=OSError("synthetic cache eviction failure"),
    )

    def successful_arm(**kwargs):
        return {
            "status": "succeeded",
            "checkpoint": kwargs["checkpoint"],
            "seed": kwargs["seed"],
        }

    with pytest.raises(
        Cosmos3CheckpointEvalError,
        match="0 failed generation arm.*1 cache-eviction failure",
    ):
        execute_phase(
            campaign_config=str(config_path),
            phase="primary",
            output_uri="s3://example-bucket/output",
            work_dir=tmp_path / "work",
            run_id="cache-accounting",
            runtime_image="registry.example/npa-cosmos3@sha256:" + "b" * 64,
            storage_client=storage,
            gpu_probe_runner=_b200_runner,
            arm_runner=successful_arm,
        )

    phase_snapshots = [
        payload for uri, payload in storage.json_uploads if uri.endswith("/primary.json")
    ]
    final = phase_snapshots[-1]
    assert final["status"] == "failed"
    assert final["failed_arms"] == 0
    assert final["cache_eviction_failures"] == 1
    assert len(final["cache_eviction_errors"]) == 1


def test_workflow_and_catalog_pin_the_b200_cosmos3_runtime() -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    entry = TOOL_CATALOG["workbench.cosmos3.checkpoint_eval"]

    assert spec["resources"]["gpu"]["accelerators"] == "B200:1"
    assert spec["config"]["require_baked_npa"] == "1"
    assert spec["config"]["source_sha"] == ""
    assert spec["states"]["evaluate"]["toolRef"] == entry.name
    assert tool_image_key(entry.name) == "cosmos3"
    assert "--top-checkpoint" in entry.argv_template
    assert "--runtime-image" not in entry.argv_template
    assert "--no-guardrails" not in entry.argv_template
    assert "runtime_image" not in spec["config"]

    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == SPEC_PATH.name)
    assert case.plan_only is True
    assert case.plan_only_justification
    assert case.image_tool == "cosmos3"


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/npa-cosmos3:tag",
        "registry.example/npa-cosmos3@sha256:not-a-digest",
    ],
)
def test_workflow_rejects_mutable_or_malformed_image_before_submission(
    image: str,
) -> None:
    spec = load_spec(SPEC_PATH)
    spec.config["source_sha"] = "a" * 40
    plan = build_plan(spec, run_id="preview")
    options = SkypilotRenderOptions(
        image_overrides={
            "workbench.cosmos3.checkpoint_eval": image
        },
        materialize_registry_secrets=False,
    )

    with pytest.raises(NpaWorkflowRenderError, match="immutable image"):
        render_skypilot_yaml(spec, plan, run_id="preview", options=options)


def test_workflow_requires_source_attestation_before_submission() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="preview")
    options = SkypilotRenderOptions(
        image_overrides={
            "workbench.cosmos3.checkpoint_eval": (
                "registry.example/npa-cosmos3@sha256:" + "b" * 64
            )
        },
        materialize_registry_secrets=False,
    )

    with pytest.raises(NpaWorkflowRenderError, match="exact source SHA"):
        render_skypilot_yaml(spec, plan, run_id="preview", options=options)
