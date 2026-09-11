"""Validate the pinned Habitat-Sim BYOF workflow and proof contract."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import optparse
from pathlib import Path
import re
import runpy
import shlex
import tempfile
import zipfile
import zlib

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = ROOT / "workflows" / "testing" / "byof-habitat-sim.yaml"
READINESS_PATH = WORKFLOW_PATH.with_suffix(".readiness.json")
PROFILE_PATH = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-habitat-sim-rtxpro-gpu.yaml"
)
GUIDE_PATH = ROOT / "docs" / "workbench" / "byof-habitat-sim.md"
LIVE_TEST_PATH = ROOT / "npa" / "tests" / "e2e" / "test_byof_onboarding_live_e2e.py"

SOURCE_REVISION = "57ee4941dc4765240f0f91f70b2c97a919bf9038"
ARCHIVE_URL = "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip"
ARCHIVE_SHA256 = "1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06"
SCENE_SHA256 = "b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56"
NAVMESH_SHA256 = "1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d"
CAPABILITY = "skokloster_castle_rgb_depth_bullet_traversal"
ARTIFACT = "habitat-sim-smoke.json"


def _payload() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _config() -> dict[str, object]:
    config = _payload().get("config")
    assert isinstance(config, dict)
    return config


def _smoke_python() -> str:
    smoke = str(_config()["smoke_command"])
    marker = "python3 - <<'PY'\n"
    assert marker in smoke
    return smoke.split(marker, 1)[1].rsplit("\nPY", 1)[0]


def _habitat_install_tokens() -> list[str]:
    build = str(_config()["build_command"])
    matches = [
        shlex.split(command.strip())
        for command in build.split("&&")
        if "python3 -m pip install --no-cache-dir ." in command
    ]
    assert len(matches) == 1
    return matches[0]


def _asset_fetch_namespace() -> dict[str, object]:
    tree = ast.parse(_smoke_python())
    imports = {"hashlib", "os", "urllib.request", "zipfile"}
    constants = {
        "ARCHIVE_URL",
        "ARCHIVE_SHA256",
        "ARCHIVE_BYTES",
        "SCENE_NAME",
        "NAVMESH_NAME",
        "SCENE_SHA256",
        "NAVMESH_SHA256",
        "MEMBER_SPECS",
    }
    functions = {
        "download_exact_archive",
        "extract_exact_member",
        "extract_exact_assets",
        "fetch_scene_assets",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Import) and node.names[0].name in imports:
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            selected.append(node)
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in constants
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    module = ast.Module(selected, type_ignores=[])
    with tempfile.TemporaryDirectory() as directory:
        module_path = Path(directory) / "habitat_assets.py"
        module_path.write_text(ast.unparse(module), encoding="utf-8")
        # run_path sees only AST-selected definitions from the fixed workflow literal.
        namespace = runpy.run_path(str(module_path))
        return namespace["fetch_scene_assets"].__globals__


def _test_archive(scene: bytes = b"scene", navmesh: bytes = b"navmesh") -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb", scene
        )
        bundle.writestr(
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh",
            navmesh,
        )
        bundle.writestr("unrelated-scene.glb", b"must-not-extract")
    return payload.getvalue()


def _configure_test_archive(namespace: dict[str, object], payload: bytes) -> None:
    scene = b"scene"
    navmesh = b"navmesh"
    namespace["ARCHIVE_BYTES"] = len(payload)
    namespace["ARCHIVE_SHA256"] = hashlib.sha256(payload).hexdigest()
    namespace["MEMBER_SPECS"] = {
        "skokloster-castle.glb": {
            "archive_member": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
            "bytes": len(scene),
            "crc32": f"{zlib.crc32(scene):08x}",
            "sha256": hashlib.sha256(scene).hexdigest(),
        },
        "skokloster-castle.navmesh": {
            "archive_member": "data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh",
            "bytes": len(navmesh),
            "crc32": f"{zlib.crc32(navmesh):08x}",
            "sha256": hashlib.sha256(navmesh).hexdigest(),
        },
    }


def test_habitat_sim_source_build_and_asset_boundary_are_immutable() -> None:
    config = _config()
    build = str(config["build_command"])
    smoke = str(config["smoke_command"])

    assert config["repo_url"] == "https://github.com/facebookresearch/habitat-sim.git"
    assert config["repo_ref"] == SOURCE_REVISION
    assert config["base_profile"] == "ubuntu"
    assert config["base_image"] == (
        "ubuntu:22.04@sha256:"
        "281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986"
    )
    assert config["apt_snapshot"] == "20260903T121500Z"
    assert config["workload"] == "solution-smoke"
    assert "git submodule update --init --jobs 8" in build
    assert "src/deps/bullet3" in build
    assert "src/deps/magnum" in build
    assert "src/deps/rlr-audio-propagation" not in build
    assert "src/deps/glfw" not in build
    assert "--recursive" not in build
    assert "HABITAT_BUILD_GUI_VIEWERS=OFF" in build
    assert "HABITAT_WITH_BULLET=ON" in build
    assert "HABITAT_WITH_CUDA=OFF" in build
    assert "SKBUILD_CMAKE_BUILD_TYPE=Release" in build
    assert "--config-settings" not in build
    locked_requirements = re.findall(r"'([^']+ --hash=sha256:[0-9a-f]{64})'", build)
    assert len(locked_requirements) == 35
    lock_sha256 = hashlib.sha256(
        ("\n".join(locked_requirements) + "\n").encode()
    ).hexdigest()
    assert (
        lock_sha256
        == "14408a8959a829803f45d8978ff3d02c9debd9ed1d2f166687924fd61a052225"
    )
    assert any(item.startswith("boto3==") for item in locked_requirements)
    assert "--require-hashes" in build
    assert "--only-binary=:all:" in build
    assert "--no-build-isolation --no-deps" in build
    assert "-r requirements.txt" not in build
    assert "pip install --no-cache-dir --upgrade pip" not in build
    assert "pip install --quiet" not in PROFILE_PATH.read_text(encoding="utf-8")
    assert "npa_habitat_requirements.lock" in build
    assert "npa_python_packages.txt" in build
    assert "npa_debian_packages.txt" in build
    assert "habitat_test_scenes" not in build
    assert "skokloster-castle" not in build

    assert ARCHIVE_URL in smoke
    assert ARCHIVE_SHA256 in smoke
    assert SCENE_SHA256 in smoke
    assert NAVMESH_SHA256 in smoke
    assert "skokloster-castle.glb" in smoke
    assert "skokloster-castle.navmesh" in smoke
    assert '"scene_license": "CC BY 4.0"' in smoke
    assert "huggingface.co" not in smoke
    assert "ai-habitat/habitat_test_scenes" not in smoke
    assert '"url_is_mutable": True' in smoke
    assert '"unrelated_members_extracted": False' in smoke
    assert "datasets_download" not in smoke
    for forbidden in (
        "matterport",
        "hm3d",
        "replica",
        "gibson",
        "mp3d",
    ):
        assert forbidden not in smoke.lower()


def test_habitat_sim_release_build_works_with_legacy_pip_frontend(capsys) -> None:
    tokens = _habitat_install_tokens()
    python_index = tokens.index("python3")
    environment = dict(token.split("=", 1) for token in tokens[:python_index])
    install_args = tokens[tokens.index("install") + 1 :]

    legacy_frontend = optparse.OptionParser(add_help_option=False)
    legacy_frontend.add_option("--no-cache-dir", action="store_true")
    legacy_frontend.add_option("--no-build-isolation", action="store_true")
    legacy_frontend.add_option("--no-deps", action="store_true")

    with pytest.raises(SystemExit) as failure:
        legacy_frontend.parse_args(
            [*install_args, "--config-settings=cmake.build-type=Release"]
        )
    assert failure.value.code == 2
    assert "no such option: --config-settings" in capsys.readouterr().err

    _options, projects = legacy_frontend.parse_args(install_args)
    assert projects == ["."]
    assert environment["SKBUILD_CMAKE_BUILD_TYPE"] == "Release"
    assert not any(token.startswith("--config-settings") for token in install_args)


def test_habitat_sim_official_archive_fetch_extracts_only_exact_members(
    tmp_path,
) -> None:
    namespace = _asset_fetch_namespace()
    payload = _test_archive()
    _configure_test_archive(namespace, payload)
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    scene, navmesh, archive, records = namespace["fetch_scene_assets"](
        opener=opener, root=tmp_path
    )
    assert calls == [(ARCHIVE_URL, 120)]
    assert scene.read_bytes() == b"scene" and navmesh.read_bytes() == b"navmesh"
    assert scene.stat().st_mode & 0o777 == 0o600
    assert navmesh.stat().st_mode & 0o777 == 0o600
    assert archive["ephemeral_copy_removed"] is True
    assert archive["unrelated_members_extracted"] is False
    assert (
        records["skokloster-castle.glb"]["sha256"]
        == hashlib.sha256(b"scene").hexdigest()
    )
    assert not (tmp_path / "unrelated-scene.glb").exists()


def test_habitat_sim_official_archive_missing_access_fails_without_files(
    tmp_path,
) -> None:
    namespace = _asset_fetch_namespace()

    def refuse(_request, timeout):
        assert timeout == 120
        raise OSError("official archive unavailable")

    with pytest.raises(OSError, match="official archive unavailable"):
        namespace["fetch_scene_assets"](opener=refuse, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_habitat_sim_official_archive_hash_mismatch_fails_closed(tmp_path) -> None:
    namespace = _asset_fetch_namespace()
    payload = _test_archive()

    with pytest.raises(RuntimeError, match="official archive identity mismatch"):
        namespace["fetch_scene_assets"](
            opener=lambda _request, timeout: io.BytesIO(payload), root=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_habitat_sim_corrupt_official_archive_fails_zip_integrity(tmp_path) -> None:
    namespace = _asset_fetch_namespace()
    payload = b"not-a-zip"
    namespace["ARCHIVE_BYTES"] = len(payload)
    namespace["ARCHIVE_SHA256"] = hashlib.sha256(payload).hexdigest()

    with pytest.raises(RuntimeError, match="official archive is not a valid ZIP"):
        namespace["fetch_scene_assets"](
            opener=lambda _request, timeout: io.BytesIO(payload), root=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_habitat_sim_official_archive_member_hash_mismatch_fails_closed(
    tmp_path,
) -> None:
    namespace = _asset_fetch_namespace()
    payload = _test_archive()
    _configure_test_archive(namespace, payload)
    namespace["MEMBER_SPECS"]["skokloster-castle.glb"]["sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="official archive member hash mismatch"):
        namespace["fetch_scene_assets"](
            opener=lambda _request, timeout: io.BytesIO(payload), root=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_habitat_sim_smoke_is_real_rgb_depth_bullet_traversal() -> None:
    config = _config()
    smoke = str(config["smoke_command"])
    script = _smoke_python()
    ast.parse(script)

    assert config["solution_name"] == "habitat-sim"
    assert config["capability_name"] == CAPABILITY
    assert config["smoke_artifact_name"] == ARTIFACT
    assert config["resource_profile_yaml"] == (
        "byof-solution-smoke-habitat-sim-rtxpro-gpu"
    )
    assert config["wait_timeout"] == -1
    assert "habitat_sim.Simulator(configuration)" in smoke
    assert "sim.make_greedy_follower" in smoke
    assert "follower.find_path" in smoke
    assert "sim.step(action, dt=1.0 / 60.0)" in smoke
    assert 'observations["color_sensor"]' in smoke
    assert 'observations["depth_sensor"]' in smoke
    assert "Image.fromarray(rgb" in smoke
    assert "np.save(depth_path" in smoke
    assert '"rgb_shape": list(rgb.shape)' in smoke
    assert '"depth_shape": list(depth.shape)' in smoke
    assert "displacement <= 0.1" in smoke
    assert "built_with_bullet" in smoke
    assert "physics_time_end <= physics_time_start" in smoke
    assert "frame_count == 0" in smoke
    assert "measured_fps" in smoke
    assert "PYTHON_LOCK_SHA256" in smoke
    assert "runtime_package_inventory" in smoke
    assert "math.isfinite(measured_fps)" in smoke
    assert "finite_depth_statistics" in smoke
    assert "capabilities_exercised" in smoke
    assert f'output_dir / "{ARTIFACT}"' in smoke
    assert '"exit_status": 0' in smoke


def test_habitat_sim_smoke_fails_closed_on_gpu_egl_and_image_identity() -> None:
    smoke = str(_config()["smoke_command"])

    assert "len(rows) != 1" in smoke
    assert "RTXPRO6000BLACKWELL" in smoke
    assert 'gpu["compute_capability"] != "12.0"' in smoke
    assert "unset DISPLAY" in smoke
    assert "sim.renderer.acquire_gl_context()" in smoke
    assert "gl.glGetString" in smoke
    assert '"nvidia" not in values["vendor"].lower()' in smoke
    assert "libEGL_nvidia.so" in smoke
    assert 'os.environ.get("BYOF_IMAGE"' in smoke
    assert 'r".+@(?P<digest>sha256:[0-9a-f]{64})"' in smoke
    assert "pod_observed_image_digest" in smoke
    assert "pod_observed_immutable_image" in smoke
    assert "image_observation_source" in smoke
    assert "NVIDIA_VISIBLE_DEVICES" not in PROFILE_PATH.read_text(encoding="utf-8")

    live_test = LIVE_TEST_PATH.read_text(encoding="utf-8")
    assert "NPA_BYOF_HABITAT_SIM_RUNTIME_RECEIPT" in live_test
    assert "NPA_BYOF_HABITAT_SIM_STRICT_CONTEXT_VERIFIED" not in live_test
    assert "DEFAULT_CONTAINER_REGISTRY" in live_test
    assert "containerStatuses" in live_test
    assert 'item.get("imageID")' in live_test
    assert "habitat-sim-live-validation.json" in live_test
    assert '"kubernetes_image_id"' in live_test
    assert '"--no-cleanup"' in live_test
    assert "_cleanup_habitat_run" in live_test


def test_habitat_sim_proof_declares_every_required_field() -> None:
    script = _smoke_python()
    required_tokens = (
        '"solution": "habitat-sim"',
        '"capability": CAPABILITY',
        '"capabilities_exercised"',
        '"source_revision"',
        '"scene_id"',
        '"scene_sha256"',
        '"scene_license"',
        '"license_url": ASSET_LICENSE_URL',
        '"original_asset"',
        '"modification_notice"',
        '"immutability_boundary"',
        '"rendered_rgb_frame_count"',
        '"rendered_depth_frame_count"',
        '"shape"',
        '"aggregate_raw_sha256"',
        '"finite_depth_statistics"',
        '"agent_start"',
        '"agent_end"',
        '"agent_displacement"',
        '"bullet_step_count"',
        '"measured_fps"',
        '"renderer_egl_evidence"',
        '"observed_rtx_gpu_model"',
        '"observed_rtx_gpu_architecture"',
        '"observed_rtx_gpu_count"',
        '"pod_observed_image_digest"',
        '"runtime_package_inventory"',
        '"exit_status": 0',
    )
    for token in required_tokens:
        assert token in script, token


def test_habitat_sim_resource_profile_is_one_exact_rtx_pro_gpu() -> None:
    docs = [
        doc
        for doc in yaml.safe_load_all(PROFILE_PATH.read_text(encoding="utf-8"))
        if doc
    ]
    tasks = [doc for doc in docs if "run" in doc]

    assert len(tasks) == 1
    task = tasks[0]
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert "B200" not in str(task["resources"]["accelerators"])
    assert task["envs"]["BYOF_IMAGE"] == ""
    assert task["envs"]["NVIDIA_DRIVER_CAPABILITIES"] == ("compute,graphics,utility")
    run = str(task["run"])
    assert '"status": "success" if smoke_exit_code == 0 else "failed"' in run
    assert '"image": os.environ.get("BYOF_IMAGE", "")' in run
    assert 'test -f "${OUTPUT_DIR}/${BYOF_SMOKE_ARTIFACT_NAME}"' in run
    assert 'proof["rendered_observations"]["frames"]' in run
    assert "s3.head_object(Bucket=parsed.netloc, Key=key)" in run


def test_habitat_sim_workflow_validates_and_plans() -> None:
    spec = load_spec(WORKFLOW_PATH)
    plan = build_plan(spec, run_id="habitat-sim-contract")

    assert spec.metadata["name"] == "byof-habitat-sim"
    assert set(spec.states) == {"byof-run"}
    assert spec.states["byof-run"].terminal
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool_ref == "workbench.byof.repo"
    rendered = "\n".join(step.argv)
    assert SOURCE_REVISION in rendered
    assert CAPABILITY in rendered
    assert ARTIFACT in rendered
    assert "byof-solution-smoke-habitat-sim-rtxpro-gpu" in rendered
    assert "20260903T121500Z" in rendered
    assert len(spec.states["byof-run"].outputs) == 3


def test_habitat_sim_readiness_hash_matches_workflow() -> None:
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    expected = hashlib.sha256(WORKFLOW_PATH.read_bytes()).hexdigest()

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == expected
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "verified"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "blocked"


def test_habitat_sim_guide_records_upstream_warning_and_deferrals() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    assert "Beyond v0.3.4" in guide
    assert "do not officially maintain releases" in normalized
    assert SOURCE_REVISION in guide
    assert ARCHIVE_SHA256 in guide
    assert SCENE_SHA256 in guide
    assert "CC BY 4.0" in guide
    assert "Hugging Face collection" not in guide
    assert "mutable" in guide
    assert "modification" in guide
    assert "35 exact wheels" in normalized
    assert "proprietary or gated datasets" in guide
    assert "semantic annotations" in guide
    assert "distributed Habitat-Lab training" in normalized
    assert "never B200" in guide
