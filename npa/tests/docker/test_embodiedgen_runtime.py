"""Unit boundaries for the EmbodiedGen operator-private runtime fetcher."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.orchestration.npa_workflow import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    render_skypilot_yaml,
    validate_image_override_selectors,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa" / "docker" / "workbench" / "embodiedgen"
WORKFLOW = ROOT / "workflows" / "testing" / "byof-embodiedgen.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
SPEC = importlib.util.spec_from_file_location(
    "embodiedgen_runtime_bootstrap", IMAGE / "runtime-bootstrap.py"
)
assert SPEC and SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)
ASSET_BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "embodiedgen_asset_bundle", IMAGE / "asset_bundle.py"
)
assert ASSET_BUNDLE_SPEC and ASSET_BUNDLE_SPEC.loader
ASSET_BUNDLE = importlib.util.module_from_spec(ASSET_BUNDLE_SPEC)
sys.modules[ASSET_BUNDLE_SPEC.name] = ASSET_BUNDLE
ASSET_BUNDLE_SPEC.loader.exec_module(ASSET_BUNDLE)


def test_manifest_pins_the_selected_upstream_components() -> None:
    manifest = IMAGE / "runtime-manifest.json"
    payload = BOOTSTRAP._read_manifest(manifest)
    assert payload["solution"]["license"] == "Apache-2.0"
    assert payload["backend"]["license"] == "MIT"
    assert payload["backend"]["model_revision"] == BOOTSTRAP.MODEL_REVISION
    assert payload["runtime"]["baked"] == {
        "source": False,
        "model": False,
        "input": False,
        "cache": False,
        "output": False,
        "credentials": False,
    }


def test_manifest_refuses_a_changed_model_revision(tmp_path: Path) -> None:
    payload = json.loads((IMAGE / "runtime-manifest.json").read_text())
    payload["backend"]["model_revision"] = "0" * 40
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model revision"):
        BOOTSTRAP._read_manifest(path)


def test_model_receipt_rejects_changed_cached_bytes(tmp_path: Path) -> None:
    model = tmp_path / "TRELLIS-image-large"
    model.mkdir()
    payload = model / "weights.safetensors"
    payload.write_bytes(b"pinned model bytes")
    receipt = model / ".npa-model-receipt.json"
    BOOTSTRAP._write_model_receipt(model)
    BOOTSTRAP._verify_model_receipt(model, receipt)
    payload.write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError, match="do not match"):
        BOOTSTRAP._verify_model_receipt(model, receipt)


def test_venv_receipt_rejects_changed_cached_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = tmp_path / "venv"
    dependency = venv / "lib" / "python3.12" / "site-packages" / "trellis.py"
    executable = venv / "bin" / "img3d-cli"
    dependency.parent.mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    dependency.write_text("pinned dependency")
    executable.write_text("pinned executable")
    marker = venv / ".npa-embodiedgen-installed.json"
    monkeypatch.setattr(BOOTSTRAP, "_installed_freeze", lambda *_: ["trellis==1"])
    BOOTSTRAP._write_install_receipt(venv, marker, tmp_path / "cache")
    BOOTSTRAP._verify_install_receipt(venv, marker, tmp_path / "cache")
    dependency.write_text("changed dependency")
    with pytest.raises(RuntimeError, match="venv bytes"):
        BOOTSTRAP._verify_install_receipt(venv, marker, tmp_path / "cache")


def test_smoke_requires_token_factory_before_creating_a_runtime_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    monkeypatch.delenv("NPA_TOKEN_FACTORY_API_KEY", raising=False)
    cache = tmp_path / "runtime-cache"
    args = SimpleNamespace(
        manifest=IMAGE / "runtime-manifest.json",
        cache_root=str(cache),
        smoke=IMAGE / "capability_smoke.py",
    )
    with pytest.raises(RuntimeError, match="Token Factory credential"):
        BOOTSTRAP.command_smoke(args)
    assert not cache.exists()


def test_trellis_only_patch_removes_unselected_sam_import(tmp_path: Path) -> None:
    inference = tmp_path / "embodied_gen" / "utils" / "inference.py"
    image_to_3d = tmp_path / "embodied_gen" / "scripts" / "imageto3d.py"
    inference.parent.mkdir(parents=True)
    image_to_3d.parent.mkdir(parents=True)
    inference.write_text(
        "from embodied_gen.models.sam3d import Sam3dInference\n"
        "def run(pipe: TrellisImageTo3DPipeline | Sam3dInference):\n"
        "    if isinstance(pipe, TrellisImageTo3DPipeline):\n        return 1\n"
        "    elif isinstance(pipe, Sam3dInference):\n        return 2\n"
        "    else:\n        raise ValueError()\n"
    )
    image_to_3d.write_text(
        "TrellisImageTo3DPipeline.from_pretrained(\n"
        '            "microsoft/TRELLIS-image-large"\n        )'
    )
    BOOTSTRAP._patch_trellis_only_import(tmp_path)
    assert "Sam3dInference" not in inference.read_text()
    assert "NPA_EMBODIEDGEN_TRELLIS_MODEL_DIR" in image_to_3d.read_text()


def test_asset_bundle_preserves_nested_generated_assets_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    collision = generated / "meshes" / "collision.obj"
    urdf = generated / "asset.urdf"
    collision.parent.mkdir(parents=True)
    urdf.write_text('<robot name="generated"/>', encoding="utf-8")
    collision.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    urdf.chmod(0o600)

    first = tmp_path / "first.tar.gz"
    receipt = ASSET_BUNDLE.archive_asset_tree(generated, first)
    assert [member["path"] for member in receipt["members"]] == [
        "generated/asset.urdf",
        "generated/meshes/collision.obj",
    ]
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == [
            "generated/asset.urdf",
            "generated/meshes/collision.obj",
        ]
        assert archive.getmember("generated/asset.urdf").mode == 0o644

    urdf.chmod(0o755)
    second = tmp_path / "second.tar.gz"
    ASSET_BUNDLE.archive_asset_tree(generated, second)
    assert first.read_bytes() == second.read_bytes()

    (generated / "unsafe-link").symlink_to(collision)
    with pytest.raises(RuntimeError, match="symbolic link"):
        ASSET_BUNDLE.archive_asset_tree(generated, tmp_path / "unsafe.tar.gz")
    assert not (tmp_path / "unsafe.tar.gz").exists()


def test_capability_smoke_executes_the_real_generation_and_simulation_chain() -> None:
    tree = ast.parse((IMAGE / "capability_smoke.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def calls(name: str) -> set[str]:
        result = set()
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    result.add(f"{node.func.value.id}.{node.func.attr}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                result.add(node.func.id)
        return result

    upstream_runs = [
        node
        for node in ast.walk(functions["run_upstream"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (node.func.value.id, node.func.attr) == ("subprocess", "run")
    ]
    assert any(
        any(
            keyword.arg == "check" and keyword.value.value is True
            for keyword in run.keywords
        )
        for run in upstream_runs
    )
    assert "cvt_embodiedgen_asset_to_anysim" in calls("convert_to_mjcf")
    assert {"bullet.loadURDF", "bullet.setGravity"} <= calls("pybullet_validation")
    assert "bullet.stepSimulation" in calls("require_rigid_body_settle")
    assert {"iio.imwrite", "iio.imiter"} <= calls("_write_decoded_view")
    assert {
        "run_upstream",
        "collision_meshes",
        "convert_to_mjcf",
        "pybullet_validation",
    } <= calls("_generate_and_validate")
    assert {"_generate_and_validate", "atomic_json"} <= calls("main")


def test_prebuilt_image_and_standalone_profile_keep_the_byof_contract() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    profile = (
        ROOT
        / "npa/src/npa/workflows/byof/profiles"
        / "byof-solution-smoke-embodiedgen-rtxpro-gpu.yaml"
    ).read_text(encoding="utf-8")
    build = (IMAGE / "build.sh").read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "/opt/byof/npa_source_metadata.json" in dockerfile
    assert "git status --porcelain --untracked-files=all" in build
    assert "NPA_REGISTRY must name the operator-controlled private registry" in build
    assert "is_public_registry" in build
    assert "npa_byof_summary.json" in profile
    assert "client.head_object" in profile
    assert "allowPrivilegeEscalation: true" in profile
    assert "SETUID" in profile
    assert "allowPrivilegeEscalation: true" in workflow
    assert "SETUID" in workflow
    assert "npa-embodiedgen@sha256:" in workflow
    assert "setpriv --no-new-privs" in (IMAGE / "entrypoint.sh").read_text(
        encoding="utf-8"
    )


def test_build_script_rejects_docker_hub_shorthand_before_building() -> None:
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = subprocess.run(
        [
            "bash",
            str(IMAGE / "build.sh"),
            f"operator/npa-embodiedgen:dev-{source_sha}",
        ],
        cwd=ROOT,
        env={**os.environ, "NPA_REGISTRY": "operator", "NPA_SOURCE_SHA": source_sha},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "fully-qualified private registry host" in result.stderr


def test_token_factory_key_is_forwarded_only_to_embodiedgen() -> None:
    runner_path = ROOT / "npa/scripts/run_byof_container_verify.py"
    spec = importlib.util.spec_from_file_location(
        "embodiedgen_byof_runner", runner_path
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    assert runner.resolve_secret_envs(
        None,
        solution_name="embodiedgen",
        environment={"NEBIUS_TOKEN_FACTORY_KEY": "value"},
    ) == ["NEBIUS_TOKEN_FACTORY_KEY"]


def test_embodiedgen_byof_renderer_rejects_mutable_or_public_images() -> None:
    runner_path = ROOT / "npa/scripts/run_byof_container_verify.py"
    spec = importlib.util.spec_from_file_location("embodiedgen_byof_image", runner_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    digest = "a" * 64
    assert runner.validate_embodiedgen_image_reference(
        f"registry.example.invalid/npa-embodiedgen@sha256:{digest}"
    ).endswith(digest)
    with pytest.raises(ValueError, match="immutable"):
        runner.validate_embodiedgen_image_reference(
            "registry.example.invalid/npa-embodiedgen:latest"
        )
    with pytest.raises(ValueError, match="anonymous/public"):
        runner.validate_embodiedgen_image_reference(
            f"docker.io/npa-embodiedgen@sha256:{digest}"
        )
    with pytest.raises(ValueError, match="fully-qualified"):
        runner.validate_embodiedgen_image_reference(
            f"operator/npa-embodiedgen@sha256:{digest}"
        )


def test_native_workflow_rejects_nonprivate_or_mutable_embodiedgen_override() -> None:
    base_spec = load_spec(WORKFLOW)
    digest = "b" * 64
    validate_image_override_selectors(base_spec, SkypilotRenderOptions())
    mutable = replace(
        base_spec,
        config={**base_spec.config, "base_image": "private/npa-embodiedgen:tag"},
    )
    with pytest.raises(NpaWorkflowRenderError, match="immutable image"):
        validate_image_override_selectors(mutable, SkypilotRenderOptions())
    public = replace(
        base_spec,
        config={
            **base_spec.config,
            "base_image": f"docker.io/npa-embodiedgen@sha256:{digest}",
        },
    )
    with pytest.raises(NpaWorkflowRenderError, match="public registry"):
        validate_image_override_selectors(public, SkypilotRenderOptions())
    shorthand = replace(
        base_spec,
        config={
            **base_spec.config,
            "base_image": f"operator/npa-embodiedgen@sha256:{digest}",
        },
    )
    with pytest.raises(NpaWorkflowRenderError, match="fully-qualified"):
        validate_image_override_selectors(shorthand, SkypilotRenderOptions())
    private = f"registry.example.invalid/npa-embodiedgen@sha256:{digest}"
    accepted = replace(
        base_spec,
        config={**base_spec.config, "base_image": private},
    )
    validate_image_override_selectors(
        accepted,
        SkypilotRenderOptions(image_overrides={"workbench.byof.repo": private}),
    )
    with pytest.raises(NpaWorkflowRenderError, match="must name the same"):
        validate_image_override_selectors(
            accepted,
            SkypilotRenderOptions(
                image_overrides={
                    "workbench.byof.repo": (
                        "registry.example.invalid/npa-embodiedgen@sha256:" + "c" * 64
                    )
                }
            ),
        )


def test_input_uri_reaches_the_byof_shell_as_literal_data(tmp_path: Path) -> None:
    """Exercise plan → SkyPilot render → BYOF bash with hostile URI text."""

    marker = tmp_path / "uri-was-executed"
    capture = tmp_path / "received-uri"
    hostile = f'https://example.invalid/object.jpg?label="$(touch {marker})"`echo bad`'
    spec = load_spec(WORKFLOW)
    spec = replace(spec, config={**spec.config, "input_uri": hostile})
    plan = build_plan(spec, run_id="literal-uri")
    smoke_command = plan.steps[0].argv[plan.steps[0].argv.index("--smoke-command") + 1]
    encoded = base64.b64encode(hostile.encode("utf-8")).decode("ascii")

    assert hostile not in smoke_command
    assert encoded in smoke_command
    rendered = render_skypilot_yaml(spec, plan, run_id="literal-uri")
    assert hostile not in rendered
    assert encoded in rendered

    worker_command = smoke_command.replace(
        "exec /usr/local/bin/npa-embodiedgen-entrypoint run-smoke",
        'printf "%s" "$NPA_EMBODIEDGEN_INPUT_URI" > "$NPA_EMBODIEDGEN_CAPTURE_PATH"',
    )
    completed = subprocess.run(
        ["/bin/bash", "-lc", '/bin/bash -lc "$BYOF_SMOKE_COMMAND"'],
        env={
            **os.environ,
            "BYOF_SMOKE_COMMAND": worker_command,
            "NPA_EMBODIEDGEN_CAPTURE_PATH": str(capture),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert capture.read_text(encoding="utf-8") == hostile


def test_workflow_declares_worker_input_and_generated_asset_outputs() -> None:
    spec = load_spec(WORKFLOW)
    state = spec.states["byof-run"]
    assert [(item.uri, item.schema) for item in state.inputs] == [
        ("{{config.input_uri}}", "image/*")
    ]
    assert {(item.uri, item.schema) for item in state.outputs} == {
        ("{{config.summary_uri}}", "npa.workbench.byof.summary.v1"),
        ("{{config.artifact_uri}}", "npa.embodiedgen.image-to-rigid-object.v1"),
        ("{{config.generated_asset_uri}}", "application/gzip"),
        ("{{config.mjcf_asset_uri}}", "application/gzip"),
        ("{{config.view_png_uri}}", "image/png"),
        ("{{config.view_mp4_uri}}", "video/mp4"),
    }


def test_readiness_record_binds_workflow_and_live_blockers() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "verified"
    assert readiness["prerequisites"]["source_image"]["status"] == "blocked"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "blocked"


def test_embodiedgen_workflow_is_explicitly_plan_only_until_gpu_accepted() -> None:
    case = next(item for item in SUBMIT_LIVE_MATRIX if item.spec == WORKFLOW.name)
    assert case.tier == "gpu"
    assert case.plan_only
    assert case.requires_token_factory
    assert "private immutable" in case.plan_only_justification
