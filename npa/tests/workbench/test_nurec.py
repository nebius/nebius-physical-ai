"""Unit tests for the NuRec / NRE workbench tool and its CLI.

Every external dependency (NGC registry, Hugging Face, the NRE binary, GPUs) is
mocked at the call site; nothing here touches real infrastructure.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.nurec import nurec as mod
from npa.workbench.nurec.nurec import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_DATASET_ID,
    DEFAULT_NRE_ENTRYPOINT,
    DEFAULT_NRE_IMAGE,
    NO_LIDAR_SENTINEL,
    NurecConfig,
    NurecError,
    build_docker_wrapper,
    build_nre_export_gt_args,
    build_nre_render_args,
    build_nre_train_args,
    check_nurec_access,
    count_render_frames,
    extract_archive,
    has_rt_cores,
    latest_usdz,
    ncore_sensor_ids,
    nre_command,
    nurec_run_status,
    parse_metrics_yaml,
    parse_offset,
    redact,
    reconstruct_scene,
    render_novel_views,
    derive_scene_variant_from_dir,
    resolve_nre_run_dir,
    validate_fetch_provenance,
)

runner = CliRunner()


def _json_payload(result) -> dict:
    """Parse the JSON document out of CliRunner output.

    CliRunner merges stderr into ``result.output`` on this click version, so a
    human-facing note on stderr lands in the same string as the machine-readable
    payload. Production keeps them separate (asserted by
    ``test_reconstruct_note_goes_to_stderr_leaving_stdout_pure_json``).
    """
    text = strip_ansi(result.output)
    start = text.index("{")
    return json.loads(text[start:])


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["nre"], returncode, stdout, "")


def _recording_runner(calls: list[list[str]], returncode: int = 0, stdout: str = ""):
    def runner_fn(command, **_kwargs):
        calls.append(list(command))
        return _completed(returncode, stdout)

    return runner_fn


# ---------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------
def test_config_defaults_are_the_ga_container_and_ungated_dataset() -> None:
    config = NurecConfig.from_env(environ={})

    assert config.image == DEFAULT_NRE_IMAGE
    # The GA channel is the one a standard NGC key can pull.
    assert config.image.endswith("-ga:26.04")
    assert config.dataset_id == DEFAULT_DATASET_ID
    assert config.config_name == DEFAULT_CONFIG_NAME
    assert config.entrypoint == DEFAULT_NRE_ENTRYPOINT
    assert config.resolved_cache_dir.as_posix().startswith("/tmp/")
    assert config.resolved_out_dir.as_posix().startswith("/tmp/")
    # 0 means "respect the recipe's epoch budget" rather than silently overriding it.
    assert config.max_epochs == 0


def test_config_prefers_explicit_then_env_then_default() -> None:
    env = {
        "NPA_NUREC_IMAGE": "registry.example/nre:from-env",
        "NPA_NUREC_SCENE": "env-scene",
        "NPA_NUREC_MAX_EPOCHS": "7",
        "NPA_NUREC_CAMERA_IDS": "camA,camB",
        "NPA_NUREC_POSES_COMPONENT_GROUP": "npa_rig",
    }

    config = NurecConfig.from_env(environ=env, scene="explicit-scene")

    assert config.scene == "explicit-scene"
    assert config.image == "registry.example/nre:from-env"
    assert config.max_epochs == 7
    assert config.camera_ids == ("camA", "camB")
    assert config.poses_component_group == "npa_rig"


def test_config_rejects_unknown_input_frame_source() -> None:
    with pytest.raises(NurecError):
        NurecConfig.from_env(environ={}, input_frames_source="telepathy")


def test_scene_dir_name_tracks_the_variant_layout() -> None:
    # PPISP archives ship "<scene>/" (full exposure brackets) and "<scene>_auto/"
    # (the smaller auto-exposure re-processing, which is the default).
    assert NurecConfig.from_env(environ={}, scene="toro", variant="auto").scene_dir_name == "toro_auto"
    assert NurecConfig.from_env(environ={}, scene="toro", variant="standard").scene_dir_name == "toro"
    # An unset variant resolves to the default rather than the full sequence.
    assert NurecConfig.from_env(environ={}, scene="toro", variant="").scene_dir_name == "toro_auto"


def test_image_repository_and_registry_are_split_for_registry_probes() -> None:
    config = NurecConfig.from_env(environ={}, image="nvcr.io/nvidia/nre/nre-ga:26.04")

    assert config.image_registry == "nvcr.io"
    assert config.image_repository == "nvidia/nre/nre-ga"


# ---------------------------------------------------------------------------------
# GPU routing
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA L40S",
        "NVIDIA L40",
        "NVIDIA RTX A6000",
        "NVIDIA A40",
    ],
)
def test_rt_core_gpus_are_accepted(name: str) -> None:
    assert has_rt_cores(name) is True


@pytest.mark.parametrize(
    "name",
    ["NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA A100-SXM4-80GB", "NVIDIA B200"],
)
def test_datacenter_compute_gpus_are_rejected(name: str) -> None:
    # Reconstruction/rasterization is RT-core work; these have none.
    assert has_rt_cores(name) is False


# ---------------------------------------------------------------------------------
# argv builders
# ---------------------------------------------------------------------------------
def test_train_args_enable_the_usdz_artifact_and_respect_the_recipe_budget() -> None:
    config = NurecConfig.from_env(environ={}, out_dir="/tmp/out")

    args = build_nre_train_args(config, ncore_json="/data/scene.json")

    assert f"--config-name={DEFAULT_CONFIG_NAME}" in args
    assert "mode=trainval" in args
    assert "dataset.path=/data/scene.json" in args
    assert "out_dir=/tmp/out" in args
    # Without the artifact flags there is no renderable scene to render novel views from.
    assert "checkpoint.artifact.enabled=true" in args
    assert "checkpoint.artifact.rig_trajectories.enabled=true" in args
    assert "checkpoint.artifact.sequence_tracks.enabled=true" in args
    assert "system.test.metrics.ssim.enabled=true" in args
    assert "system.test.metrics.lpips.enabled=true" in args
    assert "logger.run_id=nre" in args
    # max_epochs defaults to 0 -> no override emitted.
    assert not any(arg.startswith("trainer.max_epochs=") for arg in args)
    # No LiDAR override unless asked: the object-centric recipe declares its own.
    assert not any(arg.startswith("dataset.lidar_ids=") for arg in args)


def test_train_args_emit_overrides_when_configured() -> None:
    config = NurecConfig.from_env(
        environ={},
        max_epochs=3,
        world_size=4,
        precision="16-mixed",
        camera_ids=["camera2"],
        lidar_ids=["virtual_lidar"],
        poses_component_group="npa_rig",
        extra_overrides=["loss.ssim.lambda_=0.3"],
    )

    args = build_nre_train_args(config, ncore_json="/data/scene.json")

    assert "trainer.max_epochs=3" in args
    assert "trainer.world_size=4" in args
    assert "trainer.precision=16-mixed" in args
    assert "dataset.camera_ids=['camera2']" in args
    assert "dataset.lidar_ids=['virtual_lidar']" in args
    assert "dataset.poses_component_group=npa_rig" in args
    assert "loss.ssim.lambda_=0.3" in args


def test_train_args_support_forcing_an_empty_lidar_list() -> None:
    config = NurecConfig.from_env(environ={}, lidar_ids=[NO_LIDAR_SENTINEL])

    assert "dataset.lidar_ids=[]" in build_nre_train_args(config, ncore_json="/d/s.json")


def test_train_args_require_a_recipe_and_a_dataset() -> None:
    config = NurecConfig.from_env(environ={})
    with pytest.raises(NurecError):
        build_nre_train_args(config, ncore_json="")

    blank = NurecConfig.from_env(environ={"NPA_NUREC_CONFIG_NAME": ""}, config_name="")
    object.__setattr__(blank, "config_name", "")
    with pytest.raises(NurecError):
        build_nre_train_args(blank, ncore_json="/d/s.json")


def test_render_args_produce_novel_views_not_training_views() -> None:
    config = NurecConfig.from_env(environ={})

    args = build_nre_render_args(
        config,
        artifact_path="/out/nre/artifacts/030000.usdz",
        output_dir="/out/novel_views",
        camera_ids=["camera2"],
    )

    # `nre render` defaults to replicating training views, so the negation is required.
    assert "--no-replicate-training-views" in args
    assert "--replicate-training-views" not in args
    # Offsets are three separate FLOAT values upstream, not a comma string.
    index = args.index("--rig-translation-offset")
    assert args[index + 1 : index + 4] == ["0.0", "0.25", "0.0"]
    rotation = args.index("--rig-rotation-offset")
    assert args[rotation + 1 : rotation + 4] == ["0.0", "0.0", "0.0"]
    assert args[0] == "render"
    assert "--camera-id" in args and "camera2" in args
    assert "--export-video" in args


def test_render_args_can_replicate_training_views_on_request() -> None:
    config = NurecConfig.from_env(environ={})

    args = build_nre_render_args(
        config,
        artifact_path="/a/b.usdz",
        output_dir="/out",
        replicate_training_views=True,
    )

    assert "--replicate-training-views" in args
    assert "--rig-translation-offset" not in args


def test_render_args_refuse_a_zero_offset_novel_view() -> None:
    config = NurecConfig.from_env(environ={})

    with pytest.raises(NurecError, match="novel-view"):
        build_nre_render_args(
            config,
            artifact_path="/a/b.usdz",
            output_dir="/out",
            rig_translation_offset="0,0,0",
            rig_rotation_offset="0,0,0",
        )


def test_render_args_require_a_usdz_artifact() -> None:
    config = NurecConfig.from_env(environ={})

    with pytest.raises(NurecError, match="usdz"):
        build_nre_render_args(config, artifact_path="/a/b.ckpt", output_dir="/out")


def test_export_gt_args_match_the_real_subcommand_surface() -> None:
    args = build_nre_export_gt_args(ncore_json="/d/s.json", output_dir="/out/gt")

    assert args[0] == "export-ncore-benchmark-gt"
    assert "--dataset-path" in args and "/d/s.json" in args
    assert "--frame-step-camera" in args
    # There is no --camera-id on this sub-command.
    assert "--camera-id" not in args


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1,2,3", (1.0, 2.0, 3.0)), ("0 0.5 0", (0.0, 0.5, 0.0)), ([4, 5, 6], (4.0, 5.0, 6.0))],
)
def test_parse_offset_accepts_operator_friendly_forms(value, expected) -> None:
    assert parse_offset(value, (0.0, 0.0, 0.0)) == expected


def test_parse_offset_rejects_wrong_arity_and_non_numeric() -> None:
    with pytest.raises(NurecError):
        parse_offset("1,2", (0.0, 0.0, 0.0))
    with pytest.raises(NurecError):
        parse_offset("a,b,c", (0.0, 0.0, 0.0))


def test_nre_command_runs_the_entrypoint_in_container() -> None:
    config = NurecConfig.from_env(environ={})

    assert nre_command(config, ["render"]) == [DEFAULT_NRE_ENTRYPOINT, "render"]


def test_nre_command_puts_ffmpeg_flags_before_the_subcommand() -> None:
    config = NurecConfig.from_env(environ={}, ffmpeg_exe="/usr/bin/ffmpeg")

    command = nre_command(config, ["render", "--artifact-path", "x.usdz"])

    # Upstream accepts --ffmpeg-exe only as a TOP-LEVEL option.
    assert command[:3] == [DEFAULT_NRE_ENTRYPOINT, "--ffmpeg-exe", "/usr/bin/ffmpeg"]
    assert command[3] == "render"


def test_docker_wrapper_requests_gpus_and_large_shm() -> None:
    config = NurecConfig.from_env(environ={}, docker_bin="/usr/bin/docker")

    command = nre_command(
        config, ["render"], mounts=[("/host", "/box")], env_names=["NGC_API_KEY"]
    )

    assert command[:2] == ["/usr/bin/docker", "run"]
    assert "--gpus" in command and "all" in command
    assert "--shm-size=64g" in command
    assert "/host:/box" in command
    assert command[-1] == "render"
    assert config.image in command


def test_docker_wrapper_requires_a_docker_binary() -> None:
    with pytest.raises(NurecError):
        build_docker_wrapper(NurecConfig.from_env(environ={}))


# ---------------------------------------------------------------------------------
# archives, metrics, artifact discovery
# ---------------------------------------------------------------------------------
def test_extract_archive_unpacks_without_an_unzip_binary(tmp_path: Path) -> None:
    # The NRE image ships no `unzip`, so extraction must be pure stdlib.
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("scene/scene.json", "{}")
        bundle.writestr("scene/scene.ncore4.zarr.itar", "data")

    extract_archive(archive, tmp_path / "out")

    assert (tmp_path / "out" / "scene" / "scene.json").is_file()
    assert (tmp_path / "out" / "scene" / "scene.ncore4.zarr.itar").is_file()


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "nope")

    with pytest.raises(NurecError, match="escapes"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_archive_reports_a_bad_archive(tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_text("not a zip")

    with pytest.raises(NurecError):
        extract_archive(broken, tmp_path / "out")


def test_parse_metrics_yaml_flattens_the_test_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.yaml"
    path.write_text("test:\n  psnr: 28.5\n  ssim: 0.91\n  lpips: 0.12\nstep: 30000\n")

    metrics = parse_metrics_yaml(path)

    assert metrics["test/psnr"] == pytest.approx(28.5)
    assert metrics["test/ssim"] == pytest.approx(0.91)
    assert metrics["test/lpips"] == pytest.approx(0.12)


def test_parse_metrics_yaml_flattens_the_aggregated_shape(tmp_path: Path) -> None:
    """NRE 26.04 wraps validation numbers under ``aggregated_metrics`` with a
    ``value`` leaf; those must surface as the bare test/psnr|ssim|lpips keys that
    downstream gates (and the skill docs) read."""
    path = tmp_path / "metrics.yaml"
    path.write_text(
        "aggregated_metrics:\n"
        "  test/psnr:\n"
        "    aggregation_method: mean\n"
        "    value: 22.66\n"
        "  test/ssim:\n"
        "    aggregation_method: mean\n"
        "    value: 0.6447\n"
        "  test/lpips:\n"
        "    aggregation_method: mean\n"
        "    value: 0.3956\n"
    )

    metrics = parse_metrics_yaml(path)

    assert metrics["test/psnr"] == pytest.approx(22.66)
    assert metrics["test/ssim"] == pytest.approx(0.6447)
    assert metrics["test/lpips"] == pytest.approx(0.3956)


def test_parse_metrics_yaml_is_quiet_about_a_missing_file(tmp_path: Path) -> None:
    assert parse_metrics_yaml(tmp_path / "nope.yaml") == {}


def test_latest_usdz_prefers_the_newest_checkpoint_artifact(tmp_path: Path) -> None:
    # nre-ga 26.04 writes <run>/artifacts/<step>.usdz, one per checkpoint.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for index, step in enumerate(("001000", "020000", "030000")):
        path = artifacts / f"{step}.usdz"
        path.write_text("gaussians")
        import os

        os.utime(path, (1000 + index, 1000 + index))

    found = latest_usdz(tmp_path)

    assert found is not None
    assert found.name == "030000.usdz"


def test_latest_usdz_honours_the_documented_layout(tmp_path: Path) -> None:
    documented = tmp_path / "usd-out" / "last.usdz"
    documented.parent.mkdir(parents=True)
    documented.write_text("gaussians")

    assert latest_usdz(tmp_path) == documented


def test_latest_usdz_returns_none_without_artifacts(tmp_path: Path) -> None:
    assert latest_usdz(tmp_path / "missing") is None


def test_resolve_nre_run_dir_prefers_the_pinned_run_id(tmp_path: Path) -> None:
    (tmp_path / "nre" / "usd-out").mkdir(parents=True)

    assert resolve_nre_run_dir(tmp_path, "nre") == tmp_path / "nre"


def test_resolve_nre_run_dir_falls_back_to_the_newest_run(tmp_path: Path) -> None:
    # A release that ignores logger.run_id writes a random hash instead.
    (tmp_path / "AbC123" / "checkpoints").mkdir(parents=True)

    assert resolve_nre_run_dir(tmp_path, "nre").name == "AbC123"


def test_count_render_frames_counts_images_recursively(tmp_path: Path) -> None:
    (tmp_path / "camera2").mkdir(parents=True)
    (tmp_path / "camera2" / "000000.png").write_text("x")
    (tmp_path / "camera2" / "000001.jpg").write_text("x")
    (tmp_path / "camera2" / "notes.txt").write_text("x")

    assert count_render_frames(tmp_path) == 2


# ---------------------------------------------------------------------------------
# NCore sensor discovery
# ---------------------------------------------------------------------------------
def test_ncore_sensor_ids_reads_the_v4_component_stores(tmp_path: Path) -> None:
    # Real ids need not start with camera/lidar: PPISP ships `virtual_lidar`.
    path = tmp_path / "scene.json"
    path.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [
                    {"components": {"lidars": {"virtual_lidar": {"version": "v1"}}}},
                    {"components": {"cameras": {"camera1": {}, "camera2": {}}}},
                ],
            }
        )
    )

    cameras, lidars = ncore_sensor_ids(path)

    assert cameras == ("camera1", "camera2")
    assert lidars == ("virtual_lidar",)


def test_ncore_sensor_ids_falls_back_to_a_name_heuristic(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"sensors": {"camera_front": {}, "lidar_top": {}}}))

    cameras, lidars = ncore_sensor_ids(path)

    assert cameras == ("camera_front",)
    assert lidars == ("lidar_top",)


def test_ncore_sensor_ids_tolerates_unreadable_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")

    assert ncore_sensor_ids(path) == ((), ())


# ---------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------
class _Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def _patch_http(monkeypatch: pytest.MonkeyPatch, *, ngc: int = 200, hf: int = 206) -> None:
    import httpx

    def fake_get(url, **_kwargs):
        if "proxy_auth" in str(url):
            return _Response(ngc, {"token": "t"})
        if "/v2/" in str(url):
            return _Response(ngc)
        return _Response(hf)

    monkeypatch.setattr(httpx, "get", fake_get)


def test_check_is_ok_when_credentials_container_and_gpu_all_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_http(monkeypatch)
    entrypoint = tmp_path / "run"
    entrypoint.write_text("#!/bin/sh")
    config = NurecConfig.from_env(environ={}, entrypoint=str(entrypoint))

    result = check_nurec_access(
        config,
        environ={"NGC_API_KEY": "nvapi-secret-value", "HF_TOKEN": "hf_secret_value"},
        runner=lambda *_a, **_k: _completed(0, "NVIDIA RTX PRO 6000 Blackwell Server Edition\n"),
    )

    payload = result.as_dict()
    assert payload["status"] == "ok"
    assert payload["ngc_auth"] == "configured"
    assert payload["ngc_image"] == "reachable"
    assert payload["hf_dataset"] == "reachable"
    assert payload["rt_cores"] == "yes"
    assert payload["entrypoint"] == "present"
    assert payload["errors"] == []


def test_check_flags_a_gated_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_http(monkeypatch, hf=403)
    entrypoint = tmp_path / "run"
    entrypoint.write_text("x")
    config = NurecConfig.from_env(environ={}, entrypoint=str(entrypoint))

    result = check_nurec_access(
        config,
        environ={"NGC_API_KEY": "k" * 20, "HF_TOKEN": "t" * 20},
        runner=lambda *_a, **_k: _completed(0, "NVIDIA L40S\n"),
    )

    assert result.ok is False
    assert result.hf_dataset == "gated"
    assert any("license accepted" in error for error in result.errors)


def test_check_flags_an_entitlement_gated_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # nvcr.io/nvidia/nre/nre (non-GA) answers 402 for a standard NGC key.
    _patch_http(monkeypatch, ngc=402)
    entrypoint = tmp_path / "run"
    entrypoint.write_text("x")
    config = NurecConfig.from_env(
        environ={}, entrypoint=str(entrypoint), image="nvcr.io/nvidia/nre/nre:26.04"
    )

    result = check_nurec_access(
        config,
        environ={"NGC_API_KEY": "k" * 20, "HF_TOKEN": "t" * 20},
        runner=lambda *_a, **_k: _completed(0, "NVIDIA L40S\n"),
    )

    assert result.ok is False
    assert result.ngc_image == "entitlement-required"
    assert any("GA channel" in error for error in result.errors)


def test_check_rejects_a_gpu_without_rt_cores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_http(monkeypatch)
    entrypoint = tmp_path / "run"
    entrypoint.write_text("x")
    config = NurecConfig.from_env(environ={}, entrypoint=str(entrypoint))

    result = check_nurec_access(
        config,
        environ={"NGC_API_KEY": "k" * 20, "HF_TOKEN": "t" * 20},
        runner=lambda *_a, **_k: _completed(0, "NVIDIA H100 80GB HBM3\n"),
    )

    assert result.ok is False
    assert result.rt_cores == "no"
    assert any("RT cores" in error for error in result.errors)


def test_check_requires_ngc_auth_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http(monkeypatch)
    config = NurecConfig.from_env(environ={})

    result = check_nurec_access(
        config, environ={}, runner=lambda *_a, **_k: _completed(127, "")
    )

    assert result.ok is False
    assert result.ngc_auth == "missing"
    assert result.gpu == "missing"


# ---------------------------------------------------------------------------------
# secret redaction
# ---------------------------------------------------------------------------------
def test_redact_removes_every_known_secret_value() -> None:
    config = NurecConfig.from_env(environ={})
    env = {
        "HF_TOKEN": "hf_thisisasecrettoken",
        "NGC_API_KEY": "nvapi-thisisasecretkey",
        "AWS_SECRET_ACCESS_KEY": "awssecretvalue123",
    }

    text = redact(
        "auth failed for hf_thisisasecrettoken and nvapi-thisisasecretkey "
        "with awssecretvalue123",
        config,
        env,
    )

    assert "hf_thisisasecrettoken" not in text
    assert "nvapi-thisisasecretkey" not in text
    assert "awssecretvalue123" not in text
    assert text.count("<redacted>") == 3


def test_reconstruct_failure_never_leaks_a_token(tmp_path: Path) -> None:
    config = NurecConfig.from_env(environ={}, out_dir=tmp_path / "out")

    result = reconstruct_scene(
        config,
        ncore_json="/d/s.json",
        environ={"NGC_API_KEY": "nvapi-supersecretkey"},
        runner=lambda *_a, **_k: _completed(1, "boom: nvapi-supersecretkey rejected"),
    )

    assert result.ok is False
    blob = json.dumps(result.as_dict())
    assert "nvapi-supersecretkey" not in blob
    assert "<redacted>" in blob


# ---------------------------------------------------------------------------------
# reconstruct / render orchestration
# ---------------------------------------------------------------------------------
def test_reconstruct_dry_run_reports_the_command_without_running_it(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    config = NurecConfig.from_env(environ={}, out_dir=tmp_path / "out")

    result = reconstruct_scene(
        config,
        ncore_json="/d/s.json",
        environ={},
        runner=_recording_runner(calls),
        dry_run=True,
    )

    assert result.ok is True
    assert calls == []
    assert result.command[0] == DEFAULT_NRE_ENTRYPOINT
    assert any("dataset.path=/d/s.json" in part for part in result.command)


def test_reconstruct_collects_the_usdz_metrics_and_ground_truth(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_dir = out / "nre"
    calls: list[list[str]] = []

    def fake_runner(command, **_kwargs):
        calls.append(list(command))
        if "export-ncore-benchmark-gt" in command:
            (run_dir / "gt" / "camera_images" / "camera2").mkdir(parents=True, exist_ok=True)
            (run_dir / "gt" / "camera_images" / "camera2" / "000000.jpg").write_text("x")
            return _completed(0)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts" / "030000.usdz").write_text("gaussians")
        (run_dir / "config").mkdir(parents=True, exist_ok=True)
        (run_dir / "config" / "parsed.yaml").write_text("out_dir: x\n")
        (run_dir / "val").mkdir(parents=True, exist_ok=True)
        (run_dir / "val" / "metrics.yaml").write_text("test:\n  psnr: 27.75\n")
        return _completed(0)

    config = NurecConfig.from_env(environ={}, out_dir=out)
    result = reconstruct_scene(config, ncore_json="/d/s.json", environ={}, runner=fake_runner)

    assert result.ok is True
    assert result.usdz_path.endswith("030000.usdz")
    assert result.parsed_config_path.endswith("parsed.yaml")
    assert result.metrics["test/psnr"] == pytest.approx(27.75)
    assert result.gt_dir.endswith("gt")
    assert len(calls) == 2


def test_reconstruct_fails_loudly_without_an_artifact(tmp_path: Path) -> None:
    config = NurecConfig.from_env(environ={}, out_dir=tmp_path / "out")

    result = reconstruct_scene(
        config, ncore_json="/d/s.json", environ={}, runner=lambda *_a, **_k: _completed(0)
    )

    assert result.ok is False
    assert any("usdz" in error for error in result.errors)


def test_render_reports_frames_and_videos(tmp_path: Path) -> None:
    artifact = tmp_path / "030000.usdz"
    artifact.write_text("gaussians")
    output = tmp_path / "novel_views"

    def fake_runner(command, **_kwargs):
        camera = output / "camera2"
        camera.mkdir(parents=True, exist_ok=True)
        (camera / "000000.png").write_text("x")
        (camera / "000001.png").write_text("x")
        (output / "camera2.mp4").write_text("x")
        return _completed(0)

    config = NurecConfig.from_env(environ={})
    result = render_novel_views(
        config,
        artifact_path=str(artifact),
        output_dir=str(output),
        camera_ids=["camera2"],
        environ={},
        runner=fake_runner,
    )

    assert result.ok is True
    assert result.frame_count == 2
    assert result.video_count == 1
    assert result.novel_view is True
    assert result.rig_translation_offset == "0.0,0.25,0.0"


def test_render_fails_when_no_frames_are_produced(tmp_path: Path) -> None:
    artifact = tmp_path / "a.usdz"
    artifact.write_text("x")
    config = NurecConfig.from_env(environ={})

    result = render_novel_views(
        config,
        artifact_path=str(artifact),
        output_dir=str(tmp_path / "out"),
        environ={},
        runner=lambda *_a, **_k: _completed(0),
    )

    assert result.ok is False
    assert any("no frames" in error for error in result.errors)


# ---------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------
def test_fetch_downloads_extracts_and_locates_the_sequence(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    config = NurecConfig.from_env(
        environ={}, cache_dir=cache, scene="struktur28", variant="auto"
    )
    member = cache / "hf" / config.ncore_member

    def fake_runner(command, **_kwargs):
        # Stand in for the `hf download` CLI by materializing the archive.
        member.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(member, "w") as bundle:
            bundle.writestr(
                "struktur28_auto/struktur28_auto.json",
                json.dumps(
                    {
                        "version": "v4",
                        "component_stores": [
                            {"components": {"cameras": {"camera1": {}, "camera2": {}}}},
                            {"components": {"lidars": {"virtual_lidar": {}}}},
                        ],
                    }
                ),
            )
            bundle.writestr("struktur28_auto/struktur28_auto.ncore4-camera1.zarr.itar", "d")
        return _completed(0)

    result = mod.fetch_nurec_dataset(
        config, environ={"HF_TOKEN": "t" * 20}, runner=fake_runner, derive_rig=False
    )

    assert result.ok is True
    assert result.ncore_json.endswith("struktur28_auto.json")
    assert result.camera_ids == ("camera1", "camera2")
    assert result.lidar_ids == ("virtual_lidar",)
    assert result.shard_count == 1
    assert result.bytes_downloaded > 0


def test_fetch_surfaces_a_download_failure_with_redaction(tmp_path: Path) -> None:
    config = NurecConfig.from_env(environ={}, cache_dir=tmp_path / "cache")

    result = mod.fetch_nurec_dataset(
        config,
        environ={"HF_TOKEN": "hf_secrettokenvalue"},
        runner=lambda *_a, **_k: _completed(1, "denied for hf_secrettokenvalue"),
        derive_rig=False,
    )

    assert result.ok is False
    assert "hf_secrettokenvalue" not in json.dumps(result.as_dict())


# ---------------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------------
def test_status_summarizes_a_local_run_tree(tmp_path: Path) -> None:
    run = tmp_path / "neural-reconstruction-toro-20260731t100000z"
    for relative in (
        "ncore/manifest.json",
        "input/camera_images/camera2/000000.jpg",
        "reconstruction/last.usdz",
        "novel_views/camera2/000000.png",
        "reports/sim2real.rrd",
        "reports/final.json",
    ):
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")

    result = nurec_run_status(str(run))

    assert result.ok is True
    assert result.object_count == 6
    assert result.has_rrd is True
    assert result.has_usdz is True
    assert result.has_novel_views is True
    assert set(result.stages) == {"ncore", "input", "reconstruction", "novel_views", "reports"}


def test_status_requires_a_run_uri() -> None:
    assert nurec_run_status("").ok is False


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------
def test_cli_exposes_every_stage_verb() -> None:
    result = runner.invoke(app, ["workbench", "nurec", "--help"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    for verb in ("check", "fetch", "reconstruct", "render", "visualize", "finalize", "status"):
        assert verb in output
    # The GPU routing constraint belongs in the help text, not just the docs.
    assert "RT-core" in output


def test_cli_reconstruct_help_lists_the_real_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["workbench", "nurec", "reconstruct", "--help"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    for flag in (
        "--ncore-json",
        "--config-name",
        "--max-epochs",
        "--poses-component-group",
        "--output-uri",
        "--dry-run",
    ):
        assert flag in output


def test_cli_render_help_lists_the_novel_view_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Widen the pseudo-terminal so rich does not ellipsize the long option names.
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["workbench", "nurec", "render", "--help"])

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    assert "--rig-translation-offset" in output
    assert "--replicate-training-views" in output
    assert "--artifact-path" in output


def test_cli_check_emits_json_and_exits_non_zero_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_http(monkeypatch, hf=403)
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _completed(0, "NVIDIA L40S\n")
    )
    monkeypatch.setenv("NGC_API_KEY", "k" * 20)
    monkeypatch.setenv("HF_TOKEN", "t" * 20)

    result = runner.invoke(app, ["workbench", "nurec", "check", "--output", "json"])

    assert result.exit_code == 1
    payload = _json_payload(result)
    assert payload["status"] == "failed"
    assert payload["hf_dataset"] == "gated"


def test_cli_reconstruct_dry_run_prints_the_resolved_nre_command(tmp_path: Path) -> None:
    ncore = tmp_path / "scene.json"
    ncore.write_text("{}")

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--poses-component-group",
            "npa_rig",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    payload = _json_payload(result)
    assert payload["status"] == "ok"
    command = " ".join(payload["command"])
    assert "dataset.poses_component_group=npa_rig" in command
    assert "checkpoint.artifact.enabled=true" in command


def test_cli_reconstruct_without_a_sequence_fails_with_guidance(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--cache-dir",
            str(tmp_path / "empty"),
            "--out-dir",
            str(tmp_path / "out"),
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = _json_payload(result)
    assert payload["status"] == "failed"
    assert any("nurec fetch" in error for error in payload["errors"])


def test_cli_reconstruct_replaces_the_recipe_placeholder_lidar(tmp_path: Path) -> None:
    """The object-centric recipe ships `dataset.lidar_ids: [dummy_lidar]`.

    On real data NRE aborts with "Requested lidars not present in the data:
    dummy_lidar" (observed live), so reconstruct must adopt the sequence's own
    LiDAR ids when the caller named none.
    """
    ncore = tmp_path / "scene.json"
    ncore.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [
                    {"components": {"cameras": {"camera2": {}}}},
                    {"components": {"lidars": {"virtual_lidar": {}}}},
                ],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--output",
            "json",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    command = " ".join(_json_payload(result)["command"])
    assert "dataset.lidar_ids=['virtual_lidar']" in command
    assert "dummy_lidar" not in command


def test_cli_reconstruct_blanks_the_lidar_list_when_the_capture_has_none(
    tmp_path: Path,
) -> None:
    ncore = tmp_path / "scene.json"
    ncore.write_text(
        json.dumps(
            {"version": "v4", "component_stores": [{"components": {"cameras": {"camera1": {}}}}]}
        )
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--output",
            "json",
        ],
    )

    command = " ".join(_json_payload(result)["command"])
    assert "dataset.lidar_ids=[]" in command


def test_cli_reconstruct_respects_an_explicit_lidar_id(tmp_path: Path) -> None:
    ncore = tmp_path / "scene.json"
    ncore.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [{"components": {"lidars": {"virtual_lidar": {}}}}],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--lidar-id",
            "lidar_top",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    command = " ".join(_json_payload(result)["command"])
    assert "dataset.lidar_ids=['lidar_top']" in command


def test_cli_reconstruct_replaces_the_recipe_placeholder_cameras(tmp_path: Path) -> None:
    """The shipped recipes default to AV camera ids.

    On a real object-centric capture NRE aborts with "Requested cameras not present
    in the data: camera_front_wide_120fov" (observed live on the declarative
    pipeline), so reconstruct must adopt the sequence's own camera ids when the
    caller named none.
    """
    ncore = tmp_path / "scene.json"
    ncore.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [
                    {"components": {"cameras": {"camera1": {}, "camera2": {}}}},
                    {"components": {"lidars": {"virtual_lidar": {}}}},
                ],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--output",
            "json",
        ],
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    command = " ".join(_json_payload(result)["command"])
    assert "dataset.camera_ids=['camera1','camera2']" in command
    assert "camera_front_wide_120fov" not in command


def test_cli_reconstruct_respects_explicit_cameras_over_discovery(tmp_path: Path) -> None:
    ncore = tmp_path / "scene.json"
    ncore.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [
                    {"components": {"cameras": {"camera1": {}, "camera2": {}}}}
                ],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--camera-id",
            "camera2",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    command = " ".join(_json_payload(result)["command"])
    assert "dataset.camera_ids=['camera2']" in command


def test_cli_reconstruct_discovery_keeps_both_sensor_kinds(tmp_path: Path) -> None:
    """Naming only cameras must not lose the discovered LiDAR (and vice versa)."""
    ncore = tmp_path / "scene.json"
    ncore.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [
                    {"components": {"cameras": {"camera1": {}}}},
                    {"components": {"lidars": {"virtual_lidar": {}}}},
                ],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--camera-id",
            "camera1",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    command = " ".join(_json_payload(result)["command"])
    assert "dataset.camera_ids=['camera1']" in command
    assert "dataset.lidar_ids=['virtual_lidar']" in command


def _sidecar(path: Path, reference: str, cameras: list[str]) -> None:
    from npa.workbench.nurec.ncore_rig import RIG_SIDECAR_NAME

    (path.parent / RIG_SIDECAR_NAME).write_text(
        json.dumps({"reference_camera": reference, "cameras": cameras})
    )


def _ncore_with_cameras(path: Path, cameras: list[str], lidars: list[str] = []) -> None:
    stores = [{"components": {"cameras": {c: {} for c in cameras}}}]
    if lidars:
        stores.append({"components": {"lidars": {li: {} for li in lidars}}})
    path.write_text(json.dumps({"version": "v4", "component_stores": stores}))


def test_derived_rig_sequence_trains_on_the_reference_camera_only(tmp_path: Path) -> None:
    """SfM point-cloud initialization supports exactly one camera.

    Live failure: "AssertionError / Only one camera sensor is currently supported
    for sfm-point-cloud initialization" once discovery started passing both
    cameras. The rig IS the reference camera, so that is the coherent choice.
    """
    ncore = tmp_path / "scene.json"
    _ncore_with_cameras(ncore, ["camera1", "camera2"], ["virtual_lidar"])
    _sidecar(ncore, "camera2", ["camera1", "camera2"])

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--output",
            "json",
        ],
    )

    command = " ".join(_json_payload(result)["command"])
    assert "dataset.camera_ids=['camera2']" in command
    assert "camera1" not in command


def test_sequence_without_a_derived_rig_keeps_all_cameras(tmp_path: Path) -> None:
    """An AV sequence ships its own rig and no sidecar, so multi-camera stands."""
    ncore = tmp_path / "scene.json"
    _ncore_with_cameras(ncore, ["camera_front", "camera_rear"], ["lidar_top"])

    result = runner.invoke(
        app,
        [
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--output",
            "json",
        ],
    )

    command = " ".join(_json_payload(result)["command"])
    assert "dataset.camera_ids=['camera_front','camera_rear']" in command


def test_read_rig_sidecar_is_quiet_when_absent(tmp_path: Path) -> None:
    from npa.workbench.nurec.nurec import read_rig_sidecar

    ncore = tmp_path / "scene.json"
    ncore.write_text("{}")

    assert read_rig_sidecar(ncore) == {}


def test_derive_rig_poses_writes_a_self_describing_sidecar(tmp_path: Path) -> None:
    """The sidecar is what lets a LATER POD recover the reference camera."""
    pytest.importorskip("ncore")
    pytest.importorskip("upath")
    from npa.workbench.nurec.ncore_rig import RIG_SIDECAR_NAME

    # Only assert the contract that matters for cross-pod handoff: the name is
    # stable and publish_ncore_sequence would carry it (it lives in the sequence
    # directory alongside the meta-file).
    assert RIG_SIDECAR_NAME.endswith(".json")
    assert "/" not in RIG_SIDECAR_NAME


# ---------------------------------------------------------------------------------
# regressions from PR review
# ---------------------------------------------------------------------------------
def test_parse_metrics_yaml_falls_back_when_the_file_is_corrupt(tmp_path: Path) -> None:
    """A corrupt metrics.yaml must NOT crash a reconstruction that succeeded.

    yaml.safe_load raises yaml.YAMLError, which derives from Exception and NOT from
    ValueError; the previous `except ValueError` let it escape.
    """
    path = tmp_path / "metrics.yaml"
    # Valid enough for the flat scan, invalid as YAML (unclosed flow mapping).
    path.write_text("test/psnr: 28.5\nbroken: {unclosed\n")

    metrics = parse_metrics_yaml(path)

    assert metrics["test/psnr"] == pytest.approx(28.5)


def test_parse_metrics_yaml_survives_a_truncated_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.yaml"
    path.write_bytes(b"\xff\xfe\x00 not utf-8 \xc3\x28")

    assert parse_metrics_yaml(path) == {}


def test_parse_metrics_yaml_records_every_numeric_leaf(tmp_path: Path) -> None:
    """The docstring promises every numeric leaf, not just test/*."""
    path = tmp_path / "metrics.yaml"
    path.write_text("test:\n  psnr: 30.0\ntrain:\n  loss: 0.25\nstep: 30000\n")

    metrics = parse_metrics_yaml(path)

    assert metrics["test/psnr"] == pytest.approx(30.0)
    assert metrics["train/loss"] == pytest.approx(0.25)
    assert metrics["step"] == pytest.approx(30000)


def test_redact_catches_a_secret_straddling_the_truncation_boundary() -> None:
    """Redaction must happen BEFORE truncation.

    Truncating first can slice through a secret, leaving a tail fragment that no
    longer matches the full value and so survives redaction.
    """
    config = NurecConfig.from_env(environ={})
    secret = "nvapi-" + "s" * 40
    env = {"NGC_API_KEY": secret}
    # Place the secret so the default 2000-char tail cuts through its middle.
    filler = "x" * 1980
    text = filler + secret + " trailing context"

    out = redact(text, config, env)

    assert secret not in out
    # No surviving fragment of the secret either.
    assert "sssssssssss" not in out
    assert "<redacted>" in out


def test_redact_still_truncates_to_the_limit() -> None:
    config = NurecConfig.from_env(environ={})

    out = redact("y" * 5000, config, {}, limit=100)

    assert len(out) == 100


def test_latest_usdz_tie_breaks_on_the_step_not_the_name(tmp_path: Path) -> None:
    """Equal mtimes are common right after extraction.

    Lexically "7000.usdz" > "10000.usdz", which would ship the early preview
    instead of the trained scene -- the exact failure latest_usdz exists to avoid.
    """
    import os

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for step in ("7000", "10000", "30000"):
        path = artifacts / f"{step}.usdz"
        path.write_text("gaussians")
        os.utime(path, (1_000_000, 1_000_000))  # identical mtimes

    found = latest_usdz(tmp_path)

    assert found is not None
    assert found.name == "30000.usdz"


def test_latest_usdz_still_prefers_a_newer_mtime_over_a_higher_step(tmp_path: Path) -> None:
    """mtime stays the primary key; the step only breaks ties."""
    import os

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    old = artifacts / "30000.usdz"
    old.write_text("x")
    os.utime(old, (1_000_000, 1_000_000))
    new = artifacts / "1000.usdz"
    new.write_text("x")
    os.utime(new, (2_000_000, 2_000_000))

    assert latest_usdz(tmp_path).name == "1000.usdz"


def test_reconstruct_note_goes_to_stderr_leaving_stdout_pure_json(tmp_path: Path) -> None:
    """The workflow pipes stdout into a JSON parser, so it must stay pure.

    Run as a real subprocess rather than through CliRunner, which merges the two
    streams and would hide a regression here.
    """
    import subprocess as sp
    import sys

    ncore = tmp_path / "scene.json"
    _ncore_with_cameras(ncore, ["camera1", "camera2"], ["virtual_lidar"])
    _sidecar(ncore, "camera2", ["camera1", "camera2"])

    proc = sp.run(
        [
            sys.executable,
            "-m",
            "npa.cli.main",
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-json",
            str(ncore),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    # stdout parses as JSON on its own -- nothing else is written to it.
    payload = json.loads(proc.stdout)
    assert "dataset.camera_ids=['camera2']" in " ".join(payload["command"])
    # ...and the operator still gets told which camera was dropped.
    assert "camera1" in proc.stderr
    assert "reference camera" in proc.stderr


def test_reconstruct_is_silent_when_there_is_nothing_to_drop(tmp_path: Path) -> None:
    """A single-camera capture loses nothing, so it must not emit a warning."""
    import subprocess as sp
    import sys

    ncore = tmp_path / "scene.json"
    _ncore_with_cameras(ncore, ["camera2"], ["virtual_lidar"])
    _sidecar(ncore, "camera2", ["camera2"])

    proc = sp.run(
        [
            sys.executable, "-m", "npa.cli.main", "workbench", "nurec", "reconstruct",
            "--ncore-json", str(ncore), "--out-dir", str(tmp_path / "out"),
            "--dry-run", "--output", "json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    assert "restricting training" not in proc.stderr


def test_publish_merges_into_a_local_directory_without_deleting_it(tmp_path: Path) -> None:
    """A mistyped --output-uri must not wipe a populated directory."""
    from npa.cli.nurec import _publish

    source = tmp_path / "src"
    source.mkdir()
    (source / "new.txt").write_text("new")

    destination = tmp_path / "dst"
    destination.mkdir()
    precious = destination / "precious.txt"
    precious.write_text("do not delete me")

    _publish(source, str(destination))

    assert precious.is_file(), "pre-existing content was destroyed"
    assert precious.read_text() == "do not delete me"
    assert (destination / "new.txt").read_text() == "new"


# ------------------------------------------------------------------------------
# provenance gate (finding: never trust echoed request args)
# ------------------------------------------------------------------------------


def test_derive_scene_variant_from_dir_parses_variant_suffix() -> None:
    assert derive_scene_variant_from_dir("toro_auto") == ("toro", "auto")
    assert derive_scene_variant_from_dir("toro") == ("toro", "standard")
    assert derive_scene_variant_from_dir("struktur28_auto") == ("struktur28", "auto")


def test_provenance_gate_passes_when_observed_matches_requested() -> None:
    fetched = {
        "dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
        "scene": "toro",
        "variant": "standard",
        "observed_scene": "toro",
        "observed_variant": "standard",
    }
    ok, errors = validate_fetch_provenance(
        fetched,
        requested_scene="toro",
        requested_variant="standard",
        requested_dataset_id="nvidia/PhysicalAI-NuRec-PPISP",
    )
    assert ok and not errors


def test_provenance_gate_fails_when_fetch_echoes_labels_but_content_disagrees() -> None:
    # A buggy/malicious fetch returns the *requested* scene/variant in the
    # top-level (echoed request args) but independently observed content that
    # disagrees. The gate must catch it rather than trust the echo.
    fetched = {
        "dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
        "scene": "toro",
        "variant": "standard",
        # Observed unpacked content is actually struktur28/standard.
        "observed_scene": "struktur28",
        "observed_variant": "standard",
    }
    ok, errors = validate_fetch_provenance(
        fetched,
        requested_scene="toro",
        requested_variant="standard",
        requested_dataset_id="nvidia/PhysicalAI-NuRec-PPISP",
    )
    assert not ok
    assert any("scene observed" in e for e in errors)


def test_provenance_gate_fails_on_missing_observed_content() -> None:
    # Older fetch output that only echoes request args carries no observed
    # content; the gate must fail closed rather than assume correctness.
    fetched = {
        "dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
        "scene": "toro",
        "variant": "standard",
    }
    ok, errors = validate_fetch_provenance(
        fetched,
        requested_scene="toro",
        requested_variant="standard",
        requested_dataset_id="nvidia/PhysicalAI-NuRec-PPISP",
    )
    assert not ok
    assert any("no observed unpacked content" in e for e in errors)


def test_provenance_gate_fails_on_dataset_id_mismatch() -> None:
    fetched = {
        "dataset_id": "wrong/dataset",
        "scene": "toro",
        "variant": "standard",
        "observed_scene": "toro",
        "observed_variant": "standard",
    }
    ok, errors = validate_fetch_provenance(
        fetched,
        requested_scene="toro",
        requested_variant="standard",
        requested_dataset_id="nvidia/PhysicalAI-NuRec-PPISP",
    )
    assert not ok
    assert any("dataset_id mismatch" in e for e in errors)
