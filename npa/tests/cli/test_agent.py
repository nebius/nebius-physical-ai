from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer import Exit
from typer.testing import CliRunner

from npa.cli.agent import AGENT_UI_VERSION, _normalize_llm_models, app, build_agent_urls

runner = CliRunner()


def test_build_agent_urls_https_default() -> None:
    urls = build_agent_urls("203.0.113.50")
    assert urls["public_url"] == "https://203.0.113.50/"
    assert urls["agent_url"] == urls["public_url"]
    assert urls["rerun_url"] == "https://203.0.113.50/rerun/"
    assert urls["sim_assets_url"] == "https://203.0.113.50/assets/"
    assert urls["cameras_api_url"] == "https://203.0.113.50/assets/api/sim-assets/cameras"
    assert urls["direct_url"] == "http://203.0.113.50:8088/"


def test_build_agent_urls_http_legacy() -> None:
    urls = build_agent_urls("203.0.113.50", public_https=False)
    assert urls["public_url"] == "http://203.0.113.50:8088/"
    assert urls["agent_url"] == urls["public_url"]
    assert urls["sim_assets_url"] == "http://203.0.113.50:8088/assets/"
    assert urls["cameras_api_url"] == "http://203.0.113.50:8088/assets/api/sim-assets/cameras"


def test_ensure_terraform_state_bucket_creates_missing_bucket(monkeypatch) -> None:
    from npa.cli.agent import _ensure_terraform_state_bucket

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("npa.clients.nebius.bucket_exists", lambda _project, _bucket: False)
    monkeypatch.setattr(
        "npa.clients.nebius.ensure_bucket",
        lambda project, bucket: calls.append((project, bucket)),
    )

    _ensure_terraform_state_bucket(project_id="project-1", bucket_name="bucket-1")

    assert calls == [("project-1", "bucket-1")]


def test_ensure_terraform_state_bucket_skips_existing_bucket(monkeypatch) -> None:
    from npa.cli.agent import _ensure_terraform_state_bucket

    called = False

    monkeypatch.setattr("npa.clients.nebius.bucket_exists", lambda _project, _bucket: True)

    def _ensure(project: str, bucket: str) -> None:
        nonlocal called
        _ = (project, bucket)
        called = True

    monkeypatch.setattr("npa.clients.nebius.ensure_bucket", _ensure)

    _ensure_terraform_state_bucket(project_id="project-1", bucket_name="bucket-1")

    assert called is False


def test_resolve_deploy_storage_credentials_prefers_bootstrap_when_writable(monkeypatch) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr("npa.cli.agent._storage_credentials_allow_writes", lambda **_kwargs: True)
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="",
            s3_endpoint="",
            s3_access_key_id="",
            s3_secret_access_key="",
        ),
    )
    bootstrap = {
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(region="us-central1", bootstrap_creds=bootstrap)

    assert resolved["s3_bucket"] == "bucket-boot"
    assert resolved["nebius_api_key"] == "ak-boot"


def test_resolve_deploy_storage_credentials_prefers_shared_artifact_bucket(monkeypatch) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr("npa.cli.agent._storage_credentials_allow_writes", lambda **kwargs: kwargs["bucket"] == "shared-bucket")
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://shared-bucket/checkpoints/",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    bootstrap = {
        "s3_bucket": "npa-bucket-terraform",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(region="us-central1", bootstrap_creds=bootstrap)

    assert resolved["s3_bucket"] == "shared-bucket"
    assert resolved["s3_prefix"] == "checkpoints"
    assert resolved["nebius_api_key"] == "ak-shared"


def test_resolve_deploy_storage_credentials_falls_back_to_shared(monkeypatch) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    def _probe(**kwargs):
        return kwargs["bucket"] == "shared-bucket"

    monkeypatch.setattr("npa.cli.agent._storage_credentials_allow_writes", _probe)
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://shared-bucket/",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    bootstrap = {
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(region="us-central1", bootstrap_creds=bootstrap)

    assert resolved["s3_bucket"] == "shared-bucket"
    assert resolved["nebius_api_key"] == "ak-shared"


def test_resolve_deploy_storage_credentials_prefers_saved_project_state(monkeypatch) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    class _TfState:
        bucket = "state-bucket"
        endpoint = "https://storage.us-central1.nebius.cloud"
        access_key = "ak-state"
        secret_key = "sk-state"

    def _probe(**kwargs):
        return kwargs["bucket"] == "state-bucket"

    monkeypatch.setattr("npa.cli.agent._storage_credentials_allow_writes", _probe)
    monkeypatch.setattr("npa.cli.agent.resolve_terraform_state", lambda _project: _TfState())
    bootstrap = {
        "service_account_id": "sa-agent",
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(
        region="us-central1",
        bootstrap_creds=bootstrap,
        project_alias="fresh",
    )

    assert resolved["service_account_id"] == "sa-agent"
    assert resolved["s3_bucket"] == "state-bucket"
    assert resolved["nebius_api_key"] == "ak-state"


def test_resolve_deploy_storage_credentials_fails_without_writable_storage(monkeypatch) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr("npa.cli.agent._storage_credentials_allow_writes", lambda **_kwargs: False)
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: SimpleNamespace(
            s3_bucket="s3://shared-bucket/",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    bootstrap = {
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    with pytest.raises(Exit):
        _resolve_deploy_storage_credentials(region="us-central1", bootstrap_creds=bootstrap)


def test_deploy_persists_terraform_state_before_apply(monkeypatch, tmp_path) -> None:
    from npa.cli.agent import deploy_cmd

    events: list[tuple[str, dict]] = []
    creds = {
        "service_account_id": "sa-agent",
        "nebius_api_key": "ak-agent",
        "nebius_secret_key": "sk-agent",
        "s3_bucket": "npa-agent-state",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
    }

    def _write_config(payload: dict) -> None:
        events.append(("write_config", payload))

    def _apply_agent_terraform(**kwargs):
        assert any(
            event == "write_config"
            and payload.get("projects", {}).get("fresh", {}).get("terraform_state", {}).get("bucket")
            == "npa-agent-state"
            for event, payload in events
        )
        events.append(("apply", kwargs))
        return {
            "vm_ip": "203.0.113.55",
            "instance_id": "instance-agent",
            "ssh_key_path": str(tmp_path / "id_ed25519"),
        }

    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *_args, **kwargs: SimpleNamespace(
            project_id=kwargs.get("project_id"),
            tenant_id=kwargs.get("tenant_id"),
            region=kwargs.get("region"),
        ),
    )
    monkeypatch.setattr("npa.clients.nebius.bootstrap_agent_environment", lambda *_args, **_kwargs: creds)
    monkeypatch.setattr("npa.cli.agent._resolve_deploy_storage_credentials", lambda **_kwargs: creds)
    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam-token")
    monkeypatch.setattr("npa.cli.agent._ensure_terraform_state_bucket", lambda **_kwargs: None)
    monkeypatch.setattr("npa.cli.agent._apply_agent_terraform", _apply_agent_terraform)
    monkeypatch.setattr("npa.cli.agent._is_routable_public_ip", lambda _ip: True)
    monkeypatch.setattr("npa.cli.agent._write_auth_secret", lambda **_kwargs: tmp_path / "auth.env")
    monkeypatch.setattr("npa.cli.agent._resolve_deploy_llm_credentials", lambda: ("tf-key", "model-a"))
    monkeypatch.setattr("npa.cli.agent._resolve_operator_credentials", lambda: ("", ""))
    monkeypatch.setattr("npa.cli.agent._bootstrap_agent_stack", lambda **_kwargs: None)
    monkeypatch.setattr("npa.cli.agent.ensure_ingress", lambda **_kwargs: None)
    monkeypatch.setattr("npa.cli.agent.write_config", _write_config)

    deploy_cmd(
        project="fresh",
        name="agent",
        project_id="project-1",
        tenant_id="tenant-1",
        region="us-central1",
        ssh_user="ubuntu",
        ssh_public_key_path=str(tmp_path / "id_ed25519.pub"),
        tf_var=[],
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model="model-a",
        llm_models=[],
        no_public_https=False,
    )

    assert [event for event, _payload in events].count("write_config") >= 2
    assert any(event == "apply" for event, _payload in events)


def test_bootstrap_enables_public_https_nginx() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "ssl_certificate /etc/nginx/ssl/npa-agent.crt" in source
    assert "DEFAULT_HTTPS_PORT" in source
    assert "Customer URL: use" in source
    assert "--no-public-https" in source


def test_bootstrap_nginx_serves_public_rerun_recording() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "location /rerun/recordings/" in source
    assert "auth_basic off" in source
    assert "alias /opt/npa-agent/recordings/" in source


def test_franka_rerun_fallback_keeps_3d_outside_pinhole_projection() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_franka_demo_joint_angles" in source
    assert "frame_count = 90" in source
    assert "world/camera_frustums/{{name}}" in source
    assert 'f"{entity}/frustum"' not in source
    assert 'f"{entity}/origin"' not in source


def test_agent_artifact_discovery_requires_s3_components() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "list_runs(" in source
    assert "list_artifacts(" in source
    assert "download_s3_uri(" in source
    assert "Use this S3-backed Sim2Real run" in source
    assert "No S3 artifacts found for that run" in source
    assert '"source": "s3"' in source
    assert "local_path.resolve() != target.resolve()" in source
    assert "run artifacts to S3" in source
    assert "def _local_run_summaries" not in source


def test_agent_help_smoke() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deploy" in result.output
    assert "fresh-setup" in result.output
    assert "bootstrap" in result.output
    assert "verify-live" in result.output


def test_bootstrap_embeds_chat_endpoint() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '@app.post("/chat")' in source
    assert '@app.get("/session")' in source
    assert '@app.get("/models")' in source
    assert "Workbench Chat" in source
    assert "NEBIUS_TOKEN_FACTORY_KEY" in source
    assert "NPA_AGENT_LLM_MODELS" in source
    assert 'id="chatModel"' in source
    assert "llm.env" in source
    assert "renderInlineMarkdownLite" in source
    assert "showThinkingBubble" in source
    assert "thinking-dots" in source
    assert "font-family: Inter, system-ui" in source
    assert "font-family: monospace" not in source
    assert "quick-pill" in source
    assert "--brand: #5e43f3;" in source
    assert "--sidebar: #1e1f22;" in source
    assert "markdownLiteHtml" in source
    assert "Secure basic-auth session" in source
    assert "sparkle" in source
    assert "run_byof_repo.py" in source
    assert "For BYOF solution onboarding" in source
    assert "Always use real registry-qualified images" in source
    assert "`<your-registry-id>` placeholders" in source
    assert "sky gpus list" in source
    bootstrap_split = '        const lines = String(text || "").split(/\\r?\\n/);'
    assert "\r" not in bootstrap_split
    assert "\\r?\\n" in bootstrap_split
    assert "restoreSession" in source
    assert "bootPage()" in source
    assert "ensureFrankaRerunLoaded" in source
    assert "setTimeout(() =>" in source
    assert "startPeriodicRefresh" in source
    assert "fetchWithTimeout" in source
    assert "welcome.html" in source
    assert "login-help.html" in source
    assert "/welcome" in source
    assert "_agent_public_login_form_html" in source
    assert 'id="npa-sign-in"' in source
    assert "Sign in</button>" in source
    assert "encodeURIComponent(user)" in source
    assert "normalizedPath === \"/login-help.html\"" in source
    assert "normalizedPath === \"/welcome\"" in source
    assert "showRerunPlaceholder" in source
    assert "rerunIframeLoaded" in source
    assert "setChatModels" in source
    assert "selectedChatModel" in source
    assert "startApp()" in source
    assert "function bindClick(" in source
    assert "function wireUi()" in source
    assert "function showToast(" in source
    assert "id=\"statusBar\"" in source
    assert "id=\"toastHost\"" in source
    assert "DOMContentLoaded" in source
    assert "initNpaAgentUi" in source
    assert 'id="chatForm"' in source
    assert "mobile-agent" in source
    assert 'name="viewport" content="width=device-width' in source
    assert "mobileChatAuth" in source
    assert "npa_agent_basic_auth" in source
    assert "mobileAuthTokenCache" in source
    assert "verifyMobileChatAuth" in source
    assert 'credentials: useExplicitAuth ? "omit" : "include"' in source
    assert "activeChatSessionId" in source
    assert "/api/chat/sessions" in source
    assert "npa-agent/tenants/" in source
    assert "Send failed; your draft was restored." in source
    assert "AGENT_UI_VERSION" in source or "npa-ui-version" in source
    assert 'add_header Cache-Control "no-store, no-cache, must-revalidate"' in source
    assert "@media (max-width: 900px)" in source
    assert "safe-area-inset-bottom" in source


def test_watch_intent_uses_live_sim_viz_status() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'elif intent in {"sim2real_status", "watch_sim"}:' in source
    assert "live_status = sim_viz_status()" in source
    assert 'state["sim_viz"] = dict(live_status)' in source


def test_bootstrap_public_login_form() -> None:
    from npa.cli import agent as agent_module

    html = agent_module._agent_public_login_form_html("npa")
    assert 'id="npa-sign-in"' in html
    assert 'id="npa-sign-in-btn">Sign in</button>' in html or 'type="submit">Sign in</button>' in html
    assert 'value="npa"' in html
    assert "encodeURIComponent(user)" in html
    assert "encodeURIComponent(pass)" in html
    assert "history.replaceState" in html
    assert "persistBasicAuth" in html
    assert 'normalizedPath === "/login-help.html"' in html or '"/login-help.html"' in html


def test_bootstrap_ui_button_wiring_patterns() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    for control_id in (
        "chatActionS3",
        "chatActionCosmos",
        "chatActionWatch",
        "loadFrankaRerun",
        "openRerun",
        "applySelection",
        "submitWorkflow",
        "workflowStatus",
    ):
        assert f'bindClick("{control_id}"' in source
    assert 'id="chatForm"' in source
    assert "chatForm.addEventListener(\"submit\"" in source
    assert "await apiJson(\"/api/chat\"" in source
    assert "await apiJson(\"/api/sim-viz/load-franka-demo\"" in source
    assert "await apiJson(\"/api/sim-viz/camera-preview\"" in source
    assert "await apiJson(\"/api/sim-assets/selection\"" in source
    assert "setChatBusy(false)" in source
    assert "finally {" in source.split("async function sendChat")[1].split("async function")[0]


def test_bootstrap_embeds_cameras_panel() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "cameras-panel" in source
    assert "Preview in Rerun" in source
    assert "cameraCards" in source
    assert '@app.get("/sim-assets/cameras")' in source
    assert '@app.post("/sim-viz/camera-preview")' in source
    assert "world/cameras/" in source
    assert "The **Cameras** panel is the center column below chat" in source
    assert "stock_workspace" in source
    assert "stock_ee_mounted" in source
    assert "frustumSvg" in source


def test_bootstrap_stock_camera_defaults_match_scene_assets() -> None:
    from npa.cli import agent as agent_module
    from npa.genesis.scene_assets import (
        CAMERA_PLACEMENT_STOCK_EE_MOUNTED,
        CAMERA_PLACEMENT_STOCK_WORKSPACE,
        DEFAULT_CAMERA_NAMES,
    )

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    for name in DEFAULT_CAMERA_NAMES:
        assert f'"name": "{name}"' in source
    assert CAMERA_PLACEMENT_STOCK_WORKSPACE in source
    assert CAMERA_PLACEMENT_STOCK_EE_MOUNTED in source
    assert '"pos": [1.0, 0.0, 0.8]' in source
    assert '"pos": [0.4, 0.0, 0.4]' in source


def test_bootstrap_embeds_franka_rerun_ux() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '@app.post("/sim-viz/load-franka-demo")' in source
    assert "_wire_franka_demo" in source
    assert "_generate_franka_demo_rrd" in source
    assert "_log_franka_robot_geometry" in source
    assert "robot/franka/links" in source
    assert "Load active Sim2Real in Rerun" in source
    assert "Open in Rerun" in source
    assert "robotPreset" in source
    assert "rerunPlaceholder" in source
    assert 'id="rerunFrame" title="rerun" src="/rerun/?url=/rerun/recordings/sim2real.rrd&camera=workspace"' in source
    assert "RERUN_RECORDING_PATH" in source
    assert "location.origin + RERUN_RECORDING_PATH" in source
    assert "const rrdUrl = await resolveRerunRecordingUrl();" in source
    assert "/rerun/recordings/sim2real.rrd" in source
    assert 'rel="preload" href="/rerun/re_viewer.js"' in source
    assert "waitForRerunReady" in source
    assert "mountRerunIframe" in source
    assert "mountRerunIframeUntilSuccess" in source
    assert "simViz && (simViz.rerun_ready || simViz.rrd_uri)" in source
    assert "_wait_for_rerun_web_viewer" in source
    apply_selection_source = source.split("async function applySelection")[1].split("async function submitWorkflow")[0]
    assert "await waitForRerunSuccess" in apply_selection_source
    assert "activeArtifactRender = \"rerun\"" in apply_selection_source
    api_json_before_fetch = source.split("async function apiJson")[1].split("let resp;")[0]
    assert 'throw new Error("Unlock chat with your agent password.");' not in api_json_before_fetch
    assert "lastRerunBlobStatus" in source
    assert "lastRerunMountStatus" in source
    assert "baselineRrdUpdatedAt" in source
    assert "successStreakTarget" in source
    assert "successStreak" in source
    assert "stageAdvanced" in source
    assert "RERUN_MOUNT_SUCCESS" in source
    assert "Rerun iframe mount missing SUCCESS blob/mount state" in source
    assert "resolveRerunRrdUrl" not in source
    assert "RERUN_BLOB_SUCCESS" in source
    assert "/api/sim-viz/rrd-blob" in source
    assert "const rrdUrl = await resolveRerunRecordingUrl();" in source
    assert "?run_id=" in source
    assert '"/api/sim-viz/status?run_id="' in source
    assert "URL.createObjectURL" not in source
    assert "apis_used" in source
    assert "format_live_context_block" in source
    assert "match_chat_intent" in source
    assert "renderAssetsSummary" in source
    assert "selectionPayloadFromUi" in source


def test_bootstrap_embeds_run_switching_controls() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'id="runIdInput"' in source
    assert 'id="runIdSelect"' in source
    assert 'id="loadRunData"' in source
    assert '@app.post("/sim-viz/load-run")' in source
    assert "available_run_ids" in source
    assert "active_run_id" in source
    assert "_record_sim_viz_run" in source


def test_bootstrap_embeds_artifact_browser_and_endpoints() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'id="artifactPrefix"' in source
    assert 'id="artifactRunSelect"' in source
    assert 'id="artifactList"' in source
    assert 'id="renderedDataSummary"' in source
    assert '@app.get("/artifacts/runs")' in source
    assert '@app.get("/artifacts/run/{{run_id:path}}")' in source
    assert '@app.post("/sim-viz/load-artifact")' in source
    assert 'Select a discovered run or enter a run_id first' in source
    assert 'No S3 artifacts found for <code>' in source
    assert "updateRenderedDataSummary" in source
    assert "_wait_rerun_web_viewer_healthy" in source
    assert "await mountRerunIframeUntilSuccess(String(simViz.camera || \"workspace\"), 8, loadedRunId)" in source
    assert "EnvironmentFile=-/opt/npa-agent/s3.env" in source
    embedded = agent_module._embedded_agent_artifacts_source()
    assert "list_runs" in embedded
    assert "list_artifacts" in embedded


def test_bootstrap_run_history_uses_run_id_index() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '"sim_viz_runs": []' not in source
    assert 'if not isinstance(entries, dict):' in source
    assert 'entries[run_id] = snapshot' in source
    assert 'state["active_run_id"] = run_id' in source


def test_bootstrap_ui_strips_url_credentials() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "location.username" in source
    assert "location.password" in source
    assert "history.replaceState" in source
    assert 'location.protocol + "//" + location.host + location.pathname' in source
    assert "_agent_strip_url_credentials_js" in source
    assert "stripUrlCredentials" in source


def test_bootstrap_ui_fetch_uses_credentials_include() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'credentials: "include"' in source
    assert 'credentials: "same-origin"' not in source
    assert "setChatBusy(true)" in source
    assert "setChatBusy(false)" in source
    assert "if (btn) btn.disabled = busy;" in source
    assert "if (input) input.disabled = busy;" in source
    assert "JSON.stringify(value)" in source
    assert "JSON.stringify(assets.selection" not in source


def test_bootstrap_system_prompt_no_localhost() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "Never suggest localhost" in source
    assert "Load active Sim2Real in Rerun" in source
    assert "/api/sim-viz/load-franka-demo" in source
    assert "localhost:8080" not in source.split("_agent_system_prompt")[1].split("return")[0]


def test_resolve_deploy_llm_credentials_reads_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: type("Creds", (), {"token_factory_api_key": "tf-test-key"})(),
    )
    from npa.cli.agent import _resolve_deploy_llm_credentials

    key, model = _resolve_deploy_llm_credentials()
    assert key == "tf-test-key"
    assert model == "nvidia/Cosmos3-Super-Reasoner"


def test_normalize_llm_models_supports_repeated_and_csv_values() -> None:
    models = _normalize_llm_models(
        [
            "nvidia/Cosmos3-Super-Reasoner,meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ]
    )
    assert models[0] == "nvidia/Cosmos3-Super-Reasoner"
    assert "meta-llama/Llama-3.3-70B-Instruct" in models
    assert "Qwen/Qwen2.5-VL-72B-Instruct" in models


def test_agent_status_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "agent_url": "https://203.0.113.50/",
            "public_url": "https://203.0.113.50/",
            "public_https": True,
            "direct_url": "http://203.0.113.50:8088/",
            "rerun_url": "https://203.0.113.50/rerun/",
            "sim_viz_url": "https://203.0.113.50/rerun/",
            "sim_assets_url": "https://203.0.113.50/assets/",
            "cameras_api_url": "https://203.0.113.50/assets/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
            "llm": {"provider": "token_factory", "model": "nvidia/Cosmos3-Super-Reasoner"},
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr(
        "npa.cli.agent._health",
        lambda *_args, **_kwargs: (True, 200),
    )

    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["health"] is True
    assert payload["ui_status_code"] == 200
    assert payload["rerun_status_code"] == 200
    assert payload["sim_viz_url"].endswith("/rerun/")
    assert payload["sim_assets_url"].endswith("203.0.113.50/assets/")
    assert payload["cameras_api_url"].endswith("/assets/api/sim-assets/cameras")


def test_verify_live_runs_pytests(monkeypatch) -> None:
    class _Resp:
        def __init__(self, payload: dict[str, object] | str | bytes, *, status_code: int = 200) -> None:
            self.status_code = status_code
            self._payload = payload
            if isinstance(payload, (bytes, str)):
                self.content = payload.encode("utf-8") if isinstance(payload, str) else payload
                self.text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
            else:
                self.content = b""
                self.text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            if isinstance(self._payload, dict):
                return self._payload
            return {"ok": True}

        @property
        def headers(self) -> dict[str, str]:
            if isinstance(self._payload, (bytes, str)):
                return {"content-type": "application/octet-stream"}
            return {"content-type": "application/json"}

    class _Proc:
        def __init__(self, code: int = 0) -> None:
            self.returncode = code

    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "region": "us-central1",
            "agent_url": "https://203.0.113.50/",
            "public_url": "https://203.0.113.50/",
            "public_https": True,
            "direct_url": "http://203.0.113.50:8088/",
            "rerun_url": "https://203.0.113.50/rerun/",
            "sim_viz_url": "https://203.0.113.50/rerun/",
            "sim_assets_url": "https://203.0.113.50/assets/",
            "cameras_api_url": "https://203.0.113.50/assets/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))
    def _fake_http_get(url, *_args, **_kwargs):
        url_s = str(url)
        if url_s.endswith("/api/tools"):
            return _Resp({"tool_refs": [f"tool.{idx}" for idx in range(19)]})
        if url_s.endswith("/api/sim-assets"):
            return _Resp({"scene_spec": {"schema": "x"}, "robot_spec": {"schema": "y"}})
        if url_s.endswith("/api/sim-assets/cameras"):
            return _Resp(
                {
                    "cameras": [
                        {"name": "workspace", "placement": "stock_workspace", "fov": 60.0},
                        {"name": "wrist", "placement": "stock_ee_mounted", "fov": 90.0},
                    ],
                    "selected": ["workspace"],
                }
            )
        if url_s.endswith("/api/sim-assets/selection"):
            return _Resp(
                {
                    "scene_spec_uri": "stock://scene/default",
                    "assets_uri": "",
                    "robot_spec_uri": "stock://robot/franka",
                    "cameras_uri": "stock://cameras/default",
                    "robot_preset": "franka",
                    "sim_backend": "isaac",
                }
            )
        if url_s.endswith("/api/session"):
            return _Resp({"chat_history": [], "selection": {}})
        if url_s.endswith("/api/sim-viz/status"):
            return _Resp({"rerun_ready": True, "rrd_uri": "/api/sim-viz/rrd", "stage": "demo"})
        if url_s.endswith("/api/sim-viz/rrd") or url_s.endswith("/api/sim-viz/rrd-blob"):
            return _Resp(b"RRD" * 32, status_code=200)
        if url_s.endswith("/api/health"):
            return _Resp({"ok": True})
        if url_s.endswith("/api/infra/k8s"):
            return _Resp({"ok": True, "agent_npa_ready": True})
        if url_s.endswith("/api/workflows/sim2real/status"):
            return _Resp({"latest_submit": {"run_id": "agent-run-123"}, "sim_viz": {"stage": "demo"}})
        if url_s.endswith("/welcome"):
            return _Resp("<html>NPA Agent is running</html>", status_code=200)
        if url_s.endswith("/healthz"):
            return _Resp('{"ok":true}', status_code=200)
        if "/rerun/" in url_s:
            return _Resp(b"console.log('rerun');", status_code=200)
        if url_s.rstrip("/").endswith(("203.0.113.50", ":8088")):
            html = (
                f'<html><head><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<meta name="npa-ui-version" content="{AGENT_UI_VERSION}"></head>'
                '<body><div id="mobileChatAuth"></div><script>function wireUi(){} id="chatForm"; function sendChat(){} initNpaAgentUi; mobile-agent; '
                'history.replaceState(null, "", ""); location.username; location.password</script></body></html>'
            )
            return _Resp(html, status_code=200)
        return _Resp({"ok": True, "tool_ref": "tool.0", "argv_template": ["echo", "ok"]})

    def _fake_http_post(url, *_args, **_kwargs):
        url_s = str(url)
        if url_s.endswith("/api/chat"):
            payload = (_kwargs.get("json") or {}) if isinstance(_kwargs, dict) else {}
            messages = payload.get("messages", []) if isinstance(payload, dict) else []
            last_content = ""
            if isinstance(messages, list) and messages:
                tail = messages[-1]
                if isinstance(tail, dict):
                    last_content = str(tail.get("content") or "")
            if "create 2-step sim2real workflow" in last_content.lower():
                return _Resp(
                    {
                        "ok": True,
                        "grounded": True,
                        "reply": "**Generated npa.workflow/v0.0.1 spec**",
                        "workflow_yaml": "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: sim2real-two-step\nstates:\n  augment: {}\n  envgen: {}\n",
                        "apis_used": ["workflows/draft", "workflows/validate"],
                    }
                )
            if "add an open source repo" in last_content.lower() or "leisaac" in last_content.lower():
                from npa.cli.agent_chat import format_onboard_solution

                return _Resp(
                    {
                        "ok": True,
                        "grounded": True,
                        "reply": format_onboard_solution(),
                        "apis_used": ["tools", "workflows/validate", "workflows/plan"],
                    }
                )
            return _Resp(
                {
                    "ok": True,
                    "grounded": True,
                    "reply": "**Sim2Real status**\n- **run_id**: `agent-run-123`\n- **stage**: `demo`",
                    "apis_used": ["sim-viz/status"],
                }
            )
        if url_s.endswith("/api/sim-assets/selection"):
            return _Resp({"ok": True, "selection": {"scene_spec_uri": "stock://scene/default"}})
        if url_s.endswith("/api/workflows/sim2real/submit"):
            return _Resp({"ok": True, "run_id": "agent-run-123"})
        if url_s.endswith("/api/workflows/submit"):
            return _Resp(
                {
                    "ok": True,
                    "submit_mode": "agent-live-infra-dry-run",
                    "scheduler_plan": {"ok": True},
                    "run_id": "verify-live-agent-infra",
                }
            )
        if url_s.endswith("/api/sim-viz/load-franka-demo"):
            return _Resp({"ok": True, "sim_viz": {"rerun_ready": True, "rrd_uri": "/api/sim-viz/rrd"}})
        if url_s.endswith("/api/sim-viz/camera-preview"):
            return _Resp({"ok": True, "entity_path": "world/cameras/workspace"})
        return _Resp({"ok": True})

    monkeypatch.setattr("npa.cli.agent.httpx.get", _fake_http_get)
    monkeypatch.setattr("npa.cli.agent.httpx.post", _fake_http_post)
    calls: list[list[str]] = []

    def _fake_run(args, **_kwargs):
        calls.append(list(args))
        return _Proc(0)

    monkeypatch.setattr("npa.cli.agent.subprocess.run", _fake_run)

    result = runner.invoke(app, ["verify-live"])
    assert result.exit_code == 0, result.output
    assert "verify-live: ok" in result.output
    assert calls == [
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
            "-q",
        ],
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/cli/test_agent.py",
            "npa/tests/cli/test_agent_workflow.py",
            "-q",
        ],
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/e2e/test_agent_live.py", "-q"],
    ]


def _sample_agent_state(*, rerun_ready: bool = True, stage: str = "demo") -> dict:
    return {
        "sim_viz": {
            "run_id": "agent-run-123",
            "stage": stage,
            "camera": "workspace",
            "rerun_ready": rerun_ready,
            "rrd_updated_at": "2025-06-25T12:00:00+00:00",
        },
        "selection": {
            "robot_preset": "franka",
            "sim_backend": "isaac",
            "scene_spec_uri": "stock://scene/default",
            "robot_spec_uri": "stock://robot/franka",
            "cameras_uri": "stock://cameras/default",
            "assets_uri": "",
            "props": ["cube"],
        },
        "latest_submit": {"run_id": "agent-run-123", "submitted_at": "2025-06-25T11:00:00+00:00"},
        "camera_selection": ["workspace"],
    }


def test_match_chat_intent_status_queries() -> None:
    from npa.cli.agent_chat import match_chat_intent

    assert match_chat_intent("what is the current sim2real status") == "sim2real_status"
    assert match_chat_intent("workflow status please") == "sim2real_status"
    assert match_chat_intent("check simViz status now") == "sim2real_status"
    assert match_chat_intent("status for sim_viz run") == "sim2real_status"
    assert match_chat_intent("watch the sim in rerun") == "watch_sim"
    assert match_chat_intent("tail the simulation timeline") == "watch_sim"
    assert match_chat_intent("open the rerun iframe and show latest timeline") == "watch_sim"
    assert match_chat_intent("show stage badge overlay for this run") == "watch_sim"
    assert match_chat_intent("poll sim-viz/status and refresh rerun iframe") == "watch_sim"
    assert match_chat_intent("rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("RERUN_BLOB_IFRAME_UNTIL_SUCCESS") == "watch_sim"
    assert match_chat_intent("rerunblobiframeuntilsuccess") == "watch_sim"
    assert match_chat_intent("Rerun blob iframe;\nuntil SUCCESS.") == "watch_sim"
    assert match_chat_intent("rerun blob/iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("rerun blob + iframe until success, keep retrying mount") == "watch_sim"
    assert match_chat_intent("rerun blob iframe till successful mount") == "watch_sim"
    assert match_chat_intent("rerunblobiframetilsuccess") == "watch_sim"
    assert match_chat_intent("rerun blob iframe until successful for run-id scoped checks") == "watch_sim"
    assert match_chat_intent("blob+iframe until success") == "watch_sim"
    assert match_chat_intent("blobiframeuntilsuccess") == "watch_sim"
    assert match_chat_intent("until SUCCESS rerun blob iframe for this run") == "watch_sim"
    assert match_chat_intent("keep trying rerun iframe until both blob and mount are success") == "watch_sim"
    assert match_chat_intent("wait for RERUN_BLOB_SUCCESS and RERUN_MOUNT_SUCCESS") == "watch_sim"
    assert match_chat_intent("watch sim-viz/status until rrd_uri is non-empty") == "watch_sim"
    assert match_chat_intent("watch the sim until SUCCESS") == "watch_sim"
    assert match_chat_intent("watch the sim timeline until SUCCESS") == "watch_sim"
    assert match_chat_intent("watch sim-viz timeline until SUCCESS and keep retrying") == "watch_sim"
    assert match_chat_intent("watch sim-viz/status until rrd_uri is not empty") == "watch_sim"
    assert match_chat_intent("watch sim-viz/status until rrd_uri is populated") == "watch_sim"
    assert match_chat_intent("watch rrduri for active runid until SUCCESS") == "watch_sim"
    assert match_chat_intent("keep monitoring rerun until rrd_uri is set") == "watch_sim"
    assert match_chat_intent("watchrrduriuntilsuccess for runid agent-run-123") == "watch_sim"
    assert match_chat_intent("rrduriuntilsuccess for runid agent-run-123") == "watch_sim"
    assert match_chat_intent("watchsimuntilsuccess for runid agent-run-123") == "watch_sim"
    assert match_chat_intent("runidrrduriuntilsuccess") == "watch_sim"
    assert match_chat_intent("runid/rrduri SUCCESS for the active run") == "watch_sim"
    assert match_chat_intent("runidrrdurisuccess") == "watch_sim"
    assert match_chat_intent("runidscoped rerun blob iframe until success") == "watch_sim"
    assert match_chat_intent("runid + stage scoped rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("runid stage scoped rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("rerun blob iframe until SUCCESS with runid and stage matching") == "watch_sim"
    assert match_chat_intent("rrdurinonempty until SUCCESS for active runid") == "watch_sim"
    assert match_chat_intent("rrdurinotempty until SUCCESS for active runid") == "watch_sim"
    assert (
        match_chat_intent(
            "Enhance NPA agent chat intent routing and Rerun blob iframe until SUCCESS. "
            "Branch feat/npa-agent. Bootstrap rtxpro/agent after changes."
        )
        == "watch_sim"
    )
    assert match_chat_intent("load franka in rerun and keep blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("load franka in rerun") == "load_franka"
    assert match_chat_intent("show me the sim assets selection") == "sim_assets"
    assert match_chat_intent("list cameras") == "cameras"
    assert match_chat_intent("what tools can workbench do") == "tools_catalog"
    assert match_chat_intent("configure S3 bucket") == "configure_s3"
    assert match_chat_intent("setup cosmos3") == "cosmos3"
    assert match_chat_intent("create 2-step sim2real workflow") == "create_workflow"
    assert match_chat_intent("generate two-step sim2real workflow yaml") == "create_workflow"
    assert match_chat_intent("generate an example simple workflow YAML") == "create_workflow"
    assert match_chat_intent("camera angle inspector with frustum preview") == "cameras"
    assert match_chat_intent("specify scene robot cameras props selection") == "sim_assets"
    assert match_chat_intent("hello there") is None


def test_build_grounded_status_reply_unpacks_fields() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state = _sample_agent_state()
    reply = build_grounded_reply("sim2real_status", state, ["tool.a"], rerun_ready=True)
    assert "**run_id**" in reply
    assert "`agent-run-123`" in reply
    assert "**stage**" in reply
    assert "`demo`" in reply
    assert "GET /api" not in reply


def test_build_grounded_watch_sim_reply_mentions_status_polling_and_success() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state = _sample_agent_state()
    reply = build_grounded_reply("watch_sim", state, ["tool.a"], rerun_ready=True)
    assert "/api/sim-viz/status" in reply
    assert "rrd_uri" in reply
    assert "SUCCESS" in reply
    assert "**watch_stage**" in reply
    assert "**watch_mode**" in reply
    assert "run_id` + `stage`" in reply


def test_format_live_context_block_redacts_secrets() -> None:
    from npa.cli.agent_chat import format_live_context_block

    block = format_live_context_block(_sample_agent_state())
    assert "agent-run-123" in block
    assert "password" not in block.lower()
    assert "credentials" not in block.lower()


def test_apis_for_intent_includes_status_paths() -> None:
    from npa.cli.agent_chat import apis_for_intent

    apis = apis_for_intent("sim2real_status")
    assert "sim-viz/status" in apis
    assert "workflows/sim2real/status" in apis
    watch_apis = apis_for_intent("watch_sim")
    assert "sim-viz/rrd-blob" in watch_apis


def test_bootstrap_embeds_recordings_endpoint() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '@app.get("/sim-viz/recordings")' in source
    assert "sim_viz_recordings" in source
    assert '"/opt/npa-agent/recordings"' in source
    assert '"recordings"' in source
    assert '"count"' in source
    assert '"size_bytes"' in source
    assert '"updated_at"' in source


def test_bootstrap_chat_copy_yaml_support_present() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "msg-copy-btn" in source
    assert "extractFencedCode" in source
    assert "copyTextToClipboard" in source


def test_bootstrap_emitted_ui_script_is_valid_javascript(monkeypatch) -> None:
    if not shutil.which("node"):
        return
    from npa.cli import agent as agent_module

    captured: dict[str, str] = {}

    class _DummySsh:
        def upload_file(self, local_path: str, _remote_path: str) -> None:
            try:
                text = Path(local_path).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return
            if "npa-agent-bootstrap" in _remote_path:
                captured["setup_script"] = text

        def run_or_raise(self, _command: str) -> None:
            return None

        def run(self, _command: str) -> None:
            return None

    monkeypatch.setattr(agent_module, "SSHClient", lambda config: _DummySsh())
    monkeypatch.setattr(agent_module, "resolve_ssh_config", lambda **_kwargs: SimpleNamespace(ssh={}))

    agent_module._bootstrap_agent_stack(
        host="203.0.113.50",
        ssh_user="ubuntu",
        ssh_key_path="/tmp/key",
        project_alias="smoke",
        project_id="project-id",
        tenant_id="tenant-id",
        region="us-central1",
        auth_user="npa",
        auth_password="password",
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model="nvidia/Cosmos3-Super-Reasoner",
        llm_models=["nvidia/Cosmos3-Super-Reasoner", "meta-llama/Llama-3.3-70B-Instruct"],
        tf_api_key="",
        nebius_ai_key="",
        public_https=True,
    )

    setup_script = captured["setup_script"]
    html_match = re.search(
        r"cat <<'HTML' \| sudo tee /opt/npa-agent/ui\.html >/dev/null\n(?P<html>.*?)\nHTML",
        setup_script,
        flags=re.DOTALL,
    )
    assert html_match, "bootstrap setup script must emit ui.html"
    scripts = re.findall(r"<script>(.*?)</script>", html_match.group("html"), flags=re.DOTALL)
    assert scripts, "ui.html must include browser JavaScript"
    proc = subprocess.run(
        ["node", "--check", "-"],
        input="\n".join(scripts),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_bootstrap_recordings_api_in_system_prompt() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "sim-viz/recordings" in source
    assert "available .rrd recording" in source


def test_bootstrap_uses_unique_remote_setup_script_path() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "npa-agent-bootstrap-{secrets.token_hex" in source


def test_bootstrap_installs_boto3_for_artifact_endpoints() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "pip install fastapi uvicorn httpx pyyaml boto3" in source


def test_bootstrap_installs_nebius_cli_and_sa_profile() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "storage.eu-north1.nebius.cloud/cli/install.sh" in source
    assert "--token-file /mnt/cloud-metadata/token" in source
    assert 'nebius_profile = "cursor-sa"' in source
    assert "--profile {nebius_profile}" in source
    assert '"$NEBIUS_BIN" --profile {nebius_profile} iam get-access-token >/dev/null' in source
    assert "nebius CLI binary not found after install" in source
    assert "--parent-id" in source


def test_list_recordings_intent_routing() -> None:
    from npa.cli.agent_chat import apis_for_intent, match_chat_intent

    assert match_chat_intent("list recordings") == "list_recordings"
    assert match_chat_intent("show run history") == "list_recordings"
    assert match_chat_intent("browse available .rrd files") == "list_recordings"
    assert match_chat_intent("switch to a different run recording") == "list_recordings"
    apis = apis_for_intent("list_recordings")
    assert "sim-viz/recordings" in apis
    assert "sim-viz/runs" in apis


def test_list_recordings_grounded_reply() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state: dict = {}
    reply = build_grounded_reply("list_recordings", state, [])
    assert "recordings" in reply.lower() or "run history" in reply.lower()
    assert "sim-viz/recordings" in reply or "sim-viz/runs" in reply


def test_agent_config_persists_ssh_and_credentials() -> None:
    from npa.cli.agent import AgentConfig

    record = AgentConfig(
        project_alias="rtxpro",
        name="agent",
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        public_ip="203.0.113.50",
        instance_id="instance-1",
        agent_url="https://203.0.113.50/",
        rerun_url="https://203.0.113.50/rerun/",
        sim_viz_url="https://203.0.113.50/rerun/",
        sim_assets_url="https://203.0.113.50/assets/",
        cameras_api_url="https://203.0.113.50/assets/api/sim-assets/cameras",
        auth_user="npa",
        auth_secret_path="/tmp/auth.env",
        llm_provider="token_factory",
        llm_model="nvidia/Cosmos3-Super-Reasoner",
        ssh_key_path="~/.ssh/id_ed25519",
        service_account_id="serviceaccount-abc",
        credentials={
            "service_account_id": "serviceaccount-abc",
            "s3_bucket": "npa-bucket-test",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
            "access_key": "key",
            "secret_key": "secret",
        },
    )
    payload = record.to_dict()
    assert payload["ssh_key_path"] == "~/.ssh/id_ed25519"
    assert payload["service_account_id"] == "serviceaccount-abc"
    assert payload["credentials"]["access_key"] == "key"


def test_resolve_agent_ssh_key_prefers_record_and_cli() -> None:
    from npa.cli.agent import _resolve_agent_ssh_key

    record = {"ssh_key_path": "/record/key"}
    assert _resolve_agent_ssh_key(record, cli_ssh_key="/cli/key") == "/cli/key"
    assert _resolve_agent_ssh_key(record) == "/record/key"


def test_resolve_agent_storage_credentials_prefers_record() -> None:
    from npa.cli.agent import _resolve_agent_storage_credentials

    record = {
        "service_account_id": "serviceaccount-abc",
        "credentials": {
            "service_account_id": "serviceaccount-abc",
            "s3_bucket": "bucket",
            "s3_prefix": "runs",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
            "access_key": "key",
            "secret_key": "secret",
        },
    }
    bucket, prefix, endpoint, access_key, secret_key, sa_id = _resolve_agent_storage_credentials(
        "rtxpro",
        record,
    )
    assert bucket == "bucket"
    assert prefix == "runs"
    assert endpoint.endswith("nebius.cloud")
    assert access_key == "key"
    assert secret_key == "secret"
    assert sa_id == "serviceaccount-abc"


def test_bootstrap_stages_nebius_env_and_record_ssh_key() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "EnvironmentFile=-/opt/npa-agent/nebius.env" in source
    assert "_write_agent_nebius_env" in source
    assert "bootstrap_agent_environment" in source
    assert "--refresh-credentials" in source
    assert "--ssh-key" in source
    assert "_resolve_agent_ssh_key" in source
    assert "_creds_from_terraform_state" in source


def test_creds_from_terraform_state(monkeypatch) -> None:
    from npa.cli.agent import _creds_from_terraform_state

    class _Tf:
        bucket = "npa-bucket-test"
        endpoint = "https://storage.us-central1.nebius.cloud"
        access_key = "AKIA"
        secret_key = "SECRET"

    monkeypatch.setattr("npa.cli.agent.resolve_terraform_state", lambda _p: _Tf())
    monkeypatch.setattr(
        "npa.cli.agent._resolve_agent_service_account_id",
        lambda _project, _record: "serviceaccount-abc",
    )
    record = {
        "project_id": "project-1",
        "tenant_id": "tenant-1",
        "region": "us-central1",
    }
    creds = _creds_from_terraform_state("rtxpro", record)
    assert creds is not None
    assert creds["nebius_api_key"] == "AKIA"
    assert creds["s3_bucket"] == "npa-bucket-test"
    assert creds["service_account_id"] == "serviceaccount-abc"


def test_bootstrap_embed_uses_placeholder_for_agent_chat() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_AGENT_CHAT_EMBED" in source
    assert '.replace(_AGENT_CHAT_EMBED, agent_chat_source)' in source
    raw = agent_module._embedded_agent_chat_source()
    assert '"onboard_solution"' in raw
    assert "{0,140}" in raw
    rendered = source.split("_AGENT_CHAT_EMBED = ", 1)[0]  # sanity: module loads
    assert rendered


def test_bootstrap_embeds_skill_context_and_api_accounting() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_resolve_skill_context" in source
    assert "_skill_index_candidates" in source
    assert "apis_suggested" in source
    assert "skills_used" in source
    assert "_dedupe(apis_used)" in source


def test_bootstrap_embeds_scoped_state_s3_persistence() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_state_s3_key" in source
    assert "NPA_AGENT_STATE_S3_PREFIX" in source
    assert "NPA_AGENT_SESSION_SCOPE" in source
    assert "_save_state_to_s3" in source
    assert "_load_state_from_s3" in source


def test_bootstrap_embeds_provider_resilience_fallback() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_chat_with_resilience" in source
    assert "_provider_chat" in source
    assert "NPA_AGENT_LLM_PROVIDER" in source
    assert "NPA_AGENT_LLM_PROVIDERS" in source
    assert "default_provider" in source


def test_resolve_agent_service_account_id_from_nebius(mocker) -> None:
    from npa.cli.agent import _resolve_agent_service_account_id

    mocker.patch(
        "npa.clients.nebius.resolve_service_account_id",
        return_value="serviceaccount-u00s24wzj2wk8z9tqq",
    )
    record = {"project_id": "project-u00zhx4tpr00xh99b28n52"}
    assert _resolve_agent_service_account_id("rtxpro", record) == "serviceaccount-u00s24wzj2wk8z9tqq"
