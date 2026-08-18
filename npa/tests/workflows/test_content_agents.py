from __future__ import annotations

import json
from pathlib import Path
import subprocess
import re

import pytest
import yaml

from npa.workflows import content_agents as ca


def test_upstream_failure_preserves_private_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: list[tuple[Path, str]] = []

    monkeypatch.setattr(
        ca.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["material-agent"], 17
        ),
    )
    monkeypatch.setattr(
        ca,
        "_upload",
        lambda path, uri: uploaded.append((path, uri)) or uri,
    )

    with pytest.raises(ca.ContentAgentsError, match="private stage log was preserved"):
        ca._run(
            ["material-agent", "run"],
            cwd=tmp_path,
            log_path=tmp_path / "material-agent.log",
            failure_log_uri="s3://private/run/material-agent.failed.log",
        )

    assert uploaded == [
        (tmp_path / "material-agent.log", "s3://private/run/material-agent.failed.log")
    ]


def test_upstream_failure_remains_primary_when_log_upload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ca.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["physics-agent"], 23
        ),
    )

    def fail_upload(_path: Path, _uri: str) -> str:
        raise RuntimeError("private storage unavailable")

    monkeypatch.setattr(ca, "_upload", fail_upload)

    with pytest.raises(
        ca.ContentAgentsError,
        match="physics-agent failed with exit code 23; the private stage log could not be preserved",
    ):
        ca._run(
            ["physics-agent", "run"],
            cwd=tmp_path,
            log_path=tmp_path / "physics-agent.log",
            failure_log_uri="s3://private/run/physics-agent.failed.log",
        )


ROOT = Path(__file__).resolve().parents[3]
SPEC = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "content-agents-rigid-object.yaml"
)


def test_upstream_selection_is_immutable_and_antioch_is_review_only() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", ca.CONTENT_AGENTS_REVISION)
    assert re.fullmatch(r"[0-9a-f]{40}", ca.ANTIOCH_REVISION)
    assert ca.CONTENT_AGENTS_VERSION == "0.5.2"
    assert "NVIDIA-Omniverse/content-agents" in ca.CONTENT_AGENTS_REPOSITORY
    assert "antioch-robotics/antioch-content-agents" in ca.ANTIOCH_REPOSITORY


def test_material_config_uses_real_upstream_ovrtx_and_runtime_key(
    tmp_path: Path,
) -> None:
    config = ca.material_config(
        input_usd=tmp_path / "input.usda",
        output_usd=tmp_path / "output.usda",
        work_dir_name=".work",
        model=ca.DEFAULT_MODEL,
        base_url=ca.DEFAULT_BASE_URL,
    )

    assert config["steps"]["optimize_usd"] == {"enabled": False}
    assert config["steps"]["validate_input"]["on_failure"] == "block"
    assert config["steps"]["validate_output"]["on_failure"] == "block"
    assert config["steps"]["build_dataset_usd"]["renderer"]["backend"] == "ovrtx"
    assert config["steps"]["render"]["backend"] == "ovrtx"
    vlm = config["steps"]["predict"]["vlm"]
    assert vlm == {
        "backend": "openai",
        "model": ca.DEFAULT_MODEL,
        "base_url": ca.DEFAULT_BASE_URL,
        "api_key_env": "${NEBIUS_TOKEN_FACTORY_KEY}",
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    assert "api_key" not in json.dumps(config).lower().replace("api_key_env", "")
    assert config["steps"]["apply"]["fail_on_unknown_material"] is True
    assert config["materials"]["path"] == "materials.yaml"


def test_physics_config_requires_real_vlm_and_authors_the_rigid_contract(
    tmp_path: Path,
) -> None:
    config = ca.physics_config(
        input_usd=tmp_path / "input.usda",
        output_usd=tmp_path / "output.usda",
        work_dir_name=".work",
        model=ca.DEFAULT_MODEL,
        base_url=ca.DEFAULT_BASE_URL,
    )

    assert config["steps"]["identify_asset"]["renderer"]["backend"] == "ovrtx"
    assert config["steps"]["build_dataset_usd"]["renderer"]["backend"] == "ovrtx"
    assert config["steps"]["predict"]["enabled"] is True
    assert config["steps"]["predict"]["allow_empty_predictions"] is False
    prompts = config["steps"]["build_dataset_prepare_dataset"]["prompts"]
    assert prompts["system"] == ca.PHYSICS_SYSTEM_PROMPT
    assert "strict JSON" in prompts["system"]
    assert "never use Markdown fences, comments" in prompts["system"]
    assert prompts["user"] == ca.PHYSICS_USER_PROMPT
    apply = config["steps"]["apply_physics"]
    assert apply["enabled"] is True
    assert apply["collision_approx"] == "convexHull"
    assert apply["mass_scale_policy"] == "warn"
    assert apply["allow_empty_predictions"] is False


def test_generated_material_library_is_portable_and_asset_free(tmp_path: Path) -> None:
    manifest_path = ca._material_library(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    library = tmp_path / manifest["library_path"]

    assert library.is_file()
    assert manifest["entries"] == [
        {
            "name": "Aluminum",
            "description": (
                "Smooth silver-gray structural aluminum with metallic reflections "
                "and a moderately polished finish"
            ),
            "binding": "/World/Looks/Aluminum",
        }
    ]
    text = library.read_text(encoding="utf-8")
    assert "UsdPreviewSurface" in text
    assert "asset inputs:" not in text
    assert not list(tmp_path.rglob("*.mdl"))
    assert not list(tmp_path.rglob("*.png"))


def test_s3_contract_fails_closed() -> None:
    assert ca._s3_join("s3://bucket/run", "/physics/", "asset.usda") == (
        "s3://bucket/run/physics/asset.usda"
    )
    with pytest.raises(ca.ContentAgentsError, match="must use s3"):
        ca._s3_join("https://example.invalid/run", "asset.usda")


def test_live_stages_refuse_missing_model_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    with pytest.raises(ca.ContentAgentsError, match="NEBIUS_TOKEN_FACTORY_KEY"):
        ca.materials_stage(run_uri="s3://bucket/run", model="m", base_url="https://v1/")
    with pytest.raises(ca.ContentAgentsError, match="NEBIUS_TOKEN_FACTORY_KEY"):
        ca.physics_stage(run_uri="s3://bucket/run", model="m", base_url="https://v1/")


def test_workflow_routes_only_render_stages_to_rtx_and_never_b200() -> None:
    payload = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    resources = payload["resources"]
    assert resources["rtx-render"]["accelerators"] == "RTXPRO6000:1"
    assert all(
        "B200" not in str(resource.get("accelerators", ""))
        for resource in resources.values()
    )
    assert payload["states"]["acquire"]["resources"] == "cpu"
    assert payload["states"]["package"]["resources"] == "cpu"
    for name in ("materials", "physics", "validate"):
        assert payload["states"][name]["resources"] == "rtx-render"


def test_workflow_uses_every_real_content_agent_toolref_once() -> None:
    payload = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    assert [state["toolRef"] for state in payload["states"].values()] == [
        "workbench.content_agents.acquire",
        "workbench.content_agents.materials",
        "workbench.content_agents.physics",
        "workbench.content_agents.validate",
        "workbench.content_agents.package",
    ]
    assert "tool://" not in payload["config"]["runtime_image"]
    assert "@sha256:" in payload["config"]["runtime_image"]


def test_capability_smoke_calls_real_upstream_authoring_and_validation() -> None:
    source = Path(ca.__file__).read_text(encoding="utf-8")
    assert "from physics_agent.functions.apply_physics import apply_physics" in source
    assert '"validation-agent",\n            "validate"' in source
    assert '"--render-backend",\n            "ovrtx"' in source
    assert '"render_valid"' in source
    assert '"physics_sane"' in source
    assert "echo" not in source.lower()
