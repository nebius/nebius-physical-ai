"""Run the endpoint and pytest gates behind ``npa agent verify-live``."""

from __future__ import annotations

import os
import ssl
import subprocess
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, NoReturn

import httpx
import typer

from npa.cli import agent_artifact_sources as agent_artifact_options
from npa.cli import agent_resources
from npa.cli.agent_artifact_sources import AgentStorageCredentialError
from npa.cli.agent_public import (
    record_customer_url,
    record_public_https,
    record_tls_verify,
)
from npa.cli.agent_verification_ui import verify_agent_ui
from npa.clients.nebius import NebiusError
from npa.clients.network import NetworkIngressError
from npa.clients.ssh import SSHError


@dataclass(frozen=True)
class _VerificationContext:
    record: dict[str, Any]
    auth: tuple[str, str]
    ui_version: str
    health_check: Callable[..., tuple[bool, int]]
    fail: Callable[[str], NoReturn]

    @property
    def agent_url(self) -> str:
        return str(self.record.get("agent_url", ""))

    @property
    def agent_base(self) -> str:
        return self.agent_url.rstrip("/")

    @property
    def sim_assets_base(self) -> str:
        return str(self.record.get("sim_assets_url", self.agent_url)).rstrip("/")

    @cached_property
    def tls_verify(self) -> bool | ssl.SSLContext:
        try:
            return record_tls_verify(self.record)
        except (ValueError, SSHError, NetworkIngressError, NebiusError, OSError) as exc:
            self.fail(
                f"Agent HTTPS identity verification failed ({type(exc).__name__}); "
                "check provider access and the recorded SSH key before retrying."
            )


def _verify_public_access(context: _VerificationContext) -> None:
    customer_url = record_customer_url(context.record)
    if not customer_url:
        return
    public_ip = str(context.record.get("public_ip", "")).strip()
    if record_public_https(context.record) and customer_url != f"https://{public_ip}/":
        context.fail(
            "public customer URL is not the canonical HTTPS public-IP endpoint"
        )
    try:
        _verify_public_pages(context, customer_url)
        _verify_public_authentication(context, customer_url)
    except httpx.HTTPError as exc:
        context.fail(f"public customer URL unreachable: {exc}")


def _verify_public_pages(context: _VerificationContext, customer_url: str) -> None:
    welcome_response = httpx.get(
        f"{customer_url.rstrip('/')}/welcome",
        timeout=5.0,
        verify=context.tls_verify,
    )
    if welcome_response.status_code != 200:
        context.fail(
            f"public welcome page unhealthy (status={welcome_response.status_code})"
        )
    healthz_response = httpx.get(
        f"{customer_url.rstrip('/')}/healthz",
        timeout=5.0,
        verify=context.tls_verify,
    )
    if healthz_response.status_code != 200:
        context.fail(
            f"public healthz unhealthy (status={healthz_response.status_code})"
        )


def _verify_public_authentication(
    context: _VerificationContext, customer_url: str
) -> None:
    unauthenticated_ui = httpx.get(
        customer_url,
        timeout=5.0,
        verify=context.tls_verify,
        follow_redirects=False,
    )
    if unauthenticated_ui.status_code != 401:
        context.fail(
            "public UI did not enforce basic authentication "
            f"(status={unauthenticated_ui.status_code})"
        )
    api_health = httpx.get(
        f"{customer_url.rstrip('/')}/api/health",
        auth=context.auth,
        timeout=5.0,
        verify=context.tls_verify,
    )
    if api_health.status_code != 200:
        context.fail(
            f"authenticated public API unhealthy (status={api_health.status_code})"
        )


def _verify_viewer_health(context: _VerificationContext) -> None:
    ui_ok, ui_code = context.health_check(
        context.agent_url,
        user=context.auth[0],
        password=context.auth[1],
        verify=context.tls_verify,
    )
    if not ui_ok:
        context.fail(f"UI health failed behind basic auth (status={ui_code})")
    sim_viz_url = str(
        context.record.get("sim_viz_url", context.record.get("rerun_url", ""))
    )
    rerun_ok, rerun_code = context.health_check(
        sim_viz_url,
        user=context.auth[0],
        password=context.auth[1],
        verify=context.tls_verify,
    )
    if not rerun_ok:
        context.fail(f"embedded rerun iframe endpoint unhealthy (status={rerun_code})")


def _verify_sim_viz_status(context: _VerificationContext) -> None:
    try:
        sim_viz_status_response = httpx.get(
            f"{context.agent_base}/api/sim-viz/status",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        sim_viz_status_response.raise_for_status()
        sim_viz_status = sim_viz_status_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"sim viz status endpoint unhealthy: {exc}")
    if not isinstance(sim_viz_status, dict):
        context.fail("sim viz status endpoint did not return JSON object")


def _verify_sim_assets(context: _VerificationContext) -> None:
    if not context.sim_assets_base:
        context.fail("sim_assets_url missing from agent config")
    try:
        sim_assets_response = httpx.get(
            f"{context.sim_assets_base}/api/sim-assets",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        sim_assets_response.raise_for_status()
        sim_assets_payload = sim_assets_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"sim assets endpoint unhealthy: {exc}")
    if (
        not isinstance(sim_assets_payload, dict)
        or "scene_spec" not in sim_assets_payload
        or "robot_spec" not in sim_assets_payload
    ):
        context.fail("sim assets endpoint missing scene_spec/robot_spec payload")


def _verify_cameras(context: _VerificationContext) -> None:
    try:
        cameras_response = httpx.get(
            f"{context.sim_assets_base}/api/sim-assets/cameras",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        cameras_response.raise_for_status()
        cameras_payload = cameras_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"cameras endpoint unhealthy: {exc}")
    cameras = (
        cameras_payload.get("cameras", []) if isinstance(cameras_payload, dict) else []
    )
    if not isinstance(cameras, list) or not cameras:
        context.fail("cameras endpoint returned no cameras")


def _roundtrip_selection(context: _VerificationContext) -> tuple[dict, dict]:
    selection_body = {
        "robot_preset": "franka",
        "sim_backend": "isaac",
        "scene_spec_uri": "stock://scene/default",
        "assets_uri": "",
        "robot_spec_uri": "stock://robot/franka",
        "cameras_uri": "stock://cameras/default",
        "props": ["cube"],
    }
    try:
        selection_set = httpx.post(
            f"{context.sim_assets_base}/api/sim-assets/selection",
            auth=context.auth,
            json=selection_body,
            timeout=5.0,
            verify=context.tls_verify,
        )
        selection_set.raise_for_status()
        selection_get = httpx.get(
            f"{context.sim_assets_base}/api/sim-assets/selection",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        selection_get.raise_for_status()
        selected_payload = selection_get.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"sim asset selection round-trip failed: {exc}")
    if not isinstance(selected_payload, dict):
        context.fail("sim asset selection GET did not return JSON object")
    return selection_body, selected_payload


def _verify_selection(context: _VerificationContext) -> None:
    selection_body, selected_payload = _roundtrip_selection(context)
    for key in (
        "scene_spec_uri",
        "assets_uri",
        "robot_spec_uri",
        "cameras_uri",
        "robot_preset",
        "sim_backend",
    ):
        if selected_payload.get(key) != selection_body[key]:
            context.fail(f"sim asset selection round-trip did not persist {key}")


def _submit_sim2real(context: _VerificationContext) -> str:
    try:
        submit_response = httpx.post(
            f"{context.agent_base}/api/workflows/sim2real/submit",
            auth=context.auth,
            json={},
            timeout=5.0,
            verify=context.tls_verify,
        )
        submit_response.raise_for_status()
        submit_payload = submit_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"workflow submit endpoint failed: {exc}")
    if not isinstance(submit_payload, dict) or not submit_payload.get("run_id"):
        context.fail("workflow submit endpoint did not return run_id")
    submit_run_id = str(submit_payload.get("run_id") or "").strip()
    submit_viz = submit_payload.get("sim_viz", {})
    if not isinstance(submit_viz, dict) or submit_viz.get("run_id") != submit_run_id:
        context.fail("workflow submit endpoint did not return run-scoped sim_viz")
    if not (submit_viz.get("rrd_uri") or submit_viz.get("rerun_ready")):
        context.fail(
            "workflow submit endpoint did not attach a visualizable .rrd to the run"
        )
    return submit_run_id


def _verify_submitted_status(context: _VerificationContext, submit_run_id: str) -> None:
    try:
        submitted_status_response = httpx.get(
            f"{context.agent_base}/api/sim-viz/status",
            auth=context.auth,
            params={"run_id": submit_run_id},
            timeout=15.0,
            verify=context.tls_verify,
        )
        submitted_status_response.raise_for_status()
        submitted_status = submitted_status_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"submitted sim2real run status endpoint failed: {exc}")
    if (
        not isinstance(submitted_status, dict)
        or submitted_status.get("run_id") != submit_run_id
    ):
        context.fail("submitted sim2real run status did not preserve run_id")
    if not submitted_status.get("rrd_uri"):
        context.fail("submitted sim2real run status did not include rrd_uri")


def _verify_submitted_recording(
    context: _VerificationContext, submit_run_id: str
) -> None:
    try:
        submitted_rrd_blob = httpx.get(
            f"{context.agent_base}/api/sim-viz/rrd-blob",
            auth=context.auth,
            params={"run_id": submit_run_id},
            timeout=15.0,
            verify=context.tls_verify,
        )
        submitted_rrd_blob.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"submitted sim2real run rrd-blob endpoint failed: {exc}")
    if len(submitted_rrd_blob.content) < 64:
        context.fail(
            "submitted sim2real run rrd-blob endpoint returned unexpectedly small payload"
        )


def _verify_demo(context: _VerificationContext) -> None:
    try:
        load_demo_response = httpx.post(
            f"{context.agent_base}/api/sim-viz/load-franka-demo",
            auth=context.auth,
            json={"camera": "workspace"},
            timeout=30.0,
            verify=context.tls_verify,
        )
        load_demo_response.raise_for_status()
        load_demo_payload = load_demo_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"load-franka-demo endpoint failed: {exc}")
    if not isinstance(load_demo_payload, dict) or not load_demo_payload.get("ok"):
        context.fail("load-franka-demo endpoint did not return ok=true")
    sim_viz_after_demo = load_demo_payload.get("sim_viz", {})
    if not isinstance(sim_viz_after_demo, dict) or not (
        sim_viz_after_demo.get("rerun_ready") or sim_viz_after_demo.get("rrd_uri")
    ):
        context.fail("load-franka-demo did not mark rerun_ready/rrd_uri")


def _verify_camera_preview(context: _VerificationContext) -> None:
    try:
        preview_response = httpx.post(
            f"{context.agent_base}/api/sim-viz/camera-preview",
            auth=context.auth,
            json={"camera": "workspace"},
            timeout=15.0,
            verify=context.tls_verify,
        )
        preview_response.raise_for_status()
        preview_payload = preview_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"camera-preview endpoint failed: {exc}")
    if not isinstance(preview_payload, dict) or not preview_payload.get("ok"):
        context.fail("camera-preview endpoint did not return ok=true")


def _verify_recording(context: _VerificationContext) -> None:
    try:
        rrd_response = httpx.get(
            f"{context.agent_base}/api/sim-viz/rrd",
            auth=context.auth,
            timeout=15.0,
            verify=context.tls_verify,
        )
        rrd_response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"sim-viz rrd endpoint failed after load-franka-demo: {exc}")
    recording_content_type = str(rrd_response.headers.get("content-type", ""))
    if "application/json" in recording_content_type:
        if not isinstance(rrd_response.json(), dict):
            context.fail("sim-viz rrd JSON response was not an object")
    elif len(rrd_response.content) < 64:
        context.fail("sim-viz rrd endpoint returned unexpectedly small payload")


def _verify_recording_blob(context: _VerificationContext) -> None:
    try:
        rrd_blob_response = httpx.get(
            f"{context.agent_base}/api/sim-viz/rrd-blob",
            auth=context.auth,
            timeout=15.0,
            verify=context.tls_verify,
        )
        rrd_blob_response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"sim-viz rrd-blob endpoint failed after load-franka-demo: {exc}")
    recording_blob_content_type = str(rrd_blob_response.headers.get("content-type", ""))
    if "application/json" in recording_blob_content_type:
        if not isinstance(rrd_blob_response.json(), dict):
            context.fail("sim-viz rrd-blob JSON response was not an object")
    elif len(rrd_blob_response.content) < 64:
        context.fail("sim-viz rrd-blob endpoint returned unexpectedly small payload")


def _verify_rerun_static_assets(context: _VerificationContext) -> None:
    rerun_static_ok = False
    for static_path in (
        "/rerun/index.js",
        "/rerun/re_viewer.js",
        "/rerun/favicon.ico",
        "/rerun/version",
    ):
        try:
            static_response = httpx.get(
                f"{context.agent_base}{static_path}",
                auth=context.auth,
                timeout=15.0,
                verify=context.tls_verify,
            )
            if static_response.status_code == 200 and static_response.content:
                rerun_static_ok = True
                break
        except httpx.HTTPError:
            continue
    if not rerun_static_ok:
        context.fail(
            "rerun static asset probe failed (no /rerun/*.js|ico|version responded 200)"
        )


def _report_lichtblick(context: _VerificationContext) -> None:
    # Lichtblick embed probe (informational): the recordings alias serves the co-served
    # MCAP same-origin, and /lichtblick/ proxies the viewer sidecar. The sidecar is
    # best-effort (docker/image), so this never fails the run — it reports embed status.
    for lichtblick_path in ("/lichtblick/recordings/sim2real.mcap", "/lichtblick/"):
        try:
            lichtblick_response = httpx.get(
                f"{context.agent_base}{lichtblick_path}",
                auth=context.auth,
                timeout=15.0,
                verify=context.tls_verify,
            )
            typer.echo(
                f"lichtblick embed probe {lichtblick_path} -> {lichtblick_response.status_code}"
            )
        except httpx.HTTPError as exc:
            typer.echo(f"lichtblick embed probe {lichtblick_path} -> error: {exc}")


def _verify_rerun_bundle(context: _VerificationContext) -> None:
    from npa.agent_rerun_bundle_check import (
        check_rerun_bundle_load_budget,
        format_bundle_budget_report,
    )

    bundle_result = check_rerun_bundle_load_budget(
        context.agent_base,
        auth=context.auth,
        verify=context.tls_verify,
    )
    typer.echo(format_bundle_budget_report(bundle_result))
    if not bundle_result.ok:
        context.fail(
            "rerun bundle load budget failed: " + "; ".join(bundle_result.errors[:4])
        )


def _verify_backend_health(context: _VerificationContext) -> None:
    try:
        health_response = httpx.get(
            f"{context.agent_base}/api/health",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        health_response.raise_for_status()
        health_payload = health_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"health endpoint failed: {exc}")
    if not isinstance(health_payload, dict) or not health_payload.get("ok"):
        context.fail("health endpoint did not return ok=true")


def _verify_leisaac(context: _VerificationContext) -> None:
    try:
        leisaac_status_response = httpx.get(
            f"{context.agent_base}/api/leisaac/status",
            auth=context.auth,
            timeout=10.0,
            verify=context.tls_verify,
        )
        leisaac_status_response.raise_for_status()
        leisaac_status_payload = leisaac_status_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"LeIsaac status endpoint failed: {exc}")
    if not isinstance(leisaac_status_payload, dict) or not isinstance(
        leisaac_status_payload.get("available"), bool
    ):
        context.fail("LeIsaac status endpoint did not return an availability object")


def _verify_workflow_status(context: _VerificationContext) -> None:
    try:
        workflow_status_response = httpx.get(
            f"{context.agent_base}/api/workflows/sim2real/status",
            auth=context.auth,
            timeout=30.0,
            verify=context.tls_verify,
        )
        workflow_status_response.raise_for_status()
        workflow_status_payload = workflow_status_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"workflow status endpoint failed: {exc}")
    if not isinstance(workflow_status_payload, dict):
        context.fail("workflow status endpoint did not return JSON object")


def _verify_ui(context: _VerificationContext) -> None:
    ui_response = httpx.get(
        context.agent_url,
        auth=context.auth,
        timeout=10.0,
        verify=context.tls_verify,
    )
    if ui_response.status_code != 200:
        context.fail(f"UI html fetch failed (status={ui_response.status_code})")
    ui_html = ui_response.text
    verify_agent_ui(ui_html, ui_version=context.ui_version, fail=context.fail)


def _verify_session(context: _VerificationContext) -> None:
    try:
        session_response = httpx.get(
            f"{context.agent_base}/api/session",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        session_response.raise_for_status()
        session_payload = session_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"session endpoint failed: {exc}")
    if not isinstance(session_payload, dict):
        context.fail("session endpoint did not return JSON object")


def _verify_access(context: _VerificationContext) -> None:
    try:
        access_response = httpx.get(
            f"{context.agent_base}/api/access",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        access_response.raise_for_status()
        access_payload = access_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"agent access endpoint failed: {exc}")
    try:
        agent_artifact_options.validate_live_access_payload(
            context.record, access_payload
        )
    except AgentStorageCredentialError as exc:
        context.fail(str(exc))


def _verify_tools(context: _VerificationContext) -> None:
    try:
        tools_response = httpx.get(
            f"{context.agent_base}/api/tools",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        tools_response.raise_for_status()
        tool_refs = tools_response.json().get("tool_refs", [])
    except Exception as exc:  # noqa: BLE001
        context.fail(f"agent toolRef catalog request failed: {exc}")
    if len(tool_refs) < 19:
        context.fail(f"toolRef catalog too small: expected >=19, got {len(tool_refs)}")
    try:
        resolve_response = httpx.get(
            f"{context.agent_base}/api/tools/{tool_refs[0]}",
            auth=context.auth,
            timeout=5.0,
            verify=context.tls_verify,
        )
        resolve_response.raise_for_status()
        resolved = resolve_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"agent toolRef resolve failed: {exc}")
    if not resolved.get("ok"):
        context.fail("agent failed to resolve toolRef catalog entry")
    if not isinstance(resolved.get("argv_template"), list):
        context.fail("resolved toolRef entry missing argv_template list")


def _request_status_chat(context: _VerificationContext) -> Any:
    try:
        chat_smoke = httpx.post(
            f"{context.agent_base}/api/chat",
            auth=context.auth,
            json={
                "messages": [
                    {"role": "user", "content": "what is the current sim2real status"}
                ]
            },
            timeout=30.0,
            verify=context.tls_verify,
        )
        chat_smoke.raise_for_status()
        chat_payload = chat_smoke.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"chat endpoint smoke failed: {exc}")
    return chat_payload


def _verify_status_chat(context: _VerificationContext) -> None:
    chat_payload = _request_status_chat(context)
    if not isinstance(chat_payload, dict) or not chat_payload.get("ok"):
        context.fail("chat endpoint did not return ok=true")
    reply = str(chat_payload.get("reply") or "")
    if "run_id" not in reply and "stage" not in reply:
        context.fail("chat status reply missing run_id/stage fields")
    if (
        reply.strip().startswith("GET /api")
        or reply.strip() == "GET /api/sim-viz/status"
    ):
        context.fail(
            "chat status reply returned raw GET path instead of unpacked status"
        )
    if not chat_payload.get("grounded"):
        context.fail("chat status reply expected grounded=true from intent router")
    apis_used = chat_payload.get("apis_used")
    if not isinstance(apis_used, list) or not apis_used:
        context.fail("chat status reply expected non-empty apis_used list")


def _request_workflow(context: _VerificationContext) -> str:
    try:
        workflow_chat = httpx.post(
            f"{context.agent_base}/api/chat",
            auth=context.auth,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "create 2-step sim2real workflow with 5000 environments, seed 9, an RTX PRO 6000 accelerator, and 1 GPU",
                    }
                ]
            },
            timeout=30.0,
            verify=context.tls_verify,
        )
        workflow_chat.raise_for_status()
        workflow_payload = workflow_chat.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"create-workflow chat smoke failed: {exc}")
    if not isinstance(workflow_payload, dict) or not workflow_payload.get(
        "workflow_yaml"
    ):
        context.fail("create-workflow chat did not return workflow_yaml")
    workflow_yaml = str(workflow_payload.get("workflow_yaml") or "")
    # Assert the chat authored a real npa.workflow spec, not specific stage names:
    # the author's stage naming is not a stable contract (a valid 2-step sim2real
    # spec may use finalize/heldout-eval rather than augment/envgen), and the
    # validate + submit dry-run below already prove the spec is runnable.
    if "apiVersion" not in workflow_yaml or "npa.workflow" not in workflow_yaml:
        context.fail("create-workflow chat yaml is not an npa.workflow spec")
    return workflow_yaml


def _verify_workflow_yaml(context: _VerificationContext, workflow_yaml: str) -> None:
    try:
        workflow_validation = httpx.post(
            f"{context.agent_base}/api/workflows/validate",
            auth=context.auth,
            json={"yaml": workflow_yaml},
            timeout=15.0,
            verify=context.tls_verify,
        )
        workflow_validation.raise_for_status()
        workflow_validation_payload = workflow_validation.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"workflow validate endpoint failed: {exc}")
    if not isinstance(
        workflow_validation_payload, dict
    ) or not workflow_validation_payload.get("ok"):
        context.fail("workflow validate endpoint did not return ok=true")


def _verify_infra(context: _VerificationContext) -> None:
    try:
        infra_response = httpx.get(
            f"{context.agent_base}/api/infra/k8s",
            auth=context.auth,
            timeout=15.0,
            verify=context.tls_verify,
        )
        infra_response.raise_for_status()
        infra_payload = infra_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"infra discovery endpoint failed: {exc}")
    if not isinstance(infra_payload, dict) or not infra_payload.get("ok"):
        context.fail("infra discovery endpoint did not return ok=true")
    if not infra_payload.get("agent_npa_ready"):
        context.fail(
            f"agent NPA runtime is not ready: {infra_payload.get('agent_npa_error')}"
        )


def _verify_resources(context: _VerificationContext) -> None:
    try:
        resources_response = httpx.get(
            f"{context.agent_base}/api/resources?refresh=true",
            auth=context.auth,
            timeout=60.0,
            verify=context.tls_verify,
        )
        resources_response.raise_for_status()
        resources_payload = resources_response.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"tenant resource inventory endpoint failed: {exc}")
    try:
        agent_resources.validate_resource_inventory(resources_payload)
    except ValueError as exc:
        context.fail(str(exc))


def _verify_workflow_submission(
    context: _VerificationContext, workflow_yaml: str
) -> None:
    try:
        workflow_submission = httpx.post(
            f"{context.agent_base}/api/workflows/submit",
            auth=context.auth,
            json={
                "yaml": workflow_yaml,
                "run_id": "verify-live-agent-infra",
                "dry_run": True,
                "allow_provision": True,
                "validate_infra": False,
            },
            timeout=120.0,
            verify=context.tls_verify,
        )
        workflow_submission.raise_for_status()
        workflow_submission_payload = workflow_submission.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"workflow submit dry-run endpoint failed: {exc}")
    if not isinstance(
        workflow_submission_payload, dict
    ) or not workflow_submission_payload.get("ok"):
        context.fail("workflow submit dry-run endpoint did not return ok=true")
    if "scheduler_plan" not in workflow_submission_payload:
        context.fail("workflow submit dry-run missing scheduler_plan")
    if (
        str(workflow_submission_payload.get("submit_mode") or "")
        != "agent-live-infra-dry-run"
    ):
        context.fail("workflow submit dry-run did not report agent-live-infra-dry-run")


def _request_onboarding_chat(context: _VerificationContext) -> Any:
    try:
        onboard_chat = httpx.post(
            f"{context.agent_base}/api/chat",
            auth=context.auth,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "add an open source repo, containerize, push to registry, and run LeIsaac on live infra"
                        ),
                    }
                ]
            },
            timeout=30.0,
            verify=context.tls_verify,
        )
        onboard_chat.raise_for_status()
        onboard_payload = onboard_chat.json()
    except Exception as exc:  # noqa: BLE001
        context.fail(f"onboard_solution chat smoke failed: {exc}")
    return onboard_payload


def _verify_onboarding_chat(context: _VerificationContext) -> None:
    onboard_payload = _request_onboarding_chat(context)
    if not isinstance(onboard_payload, dict) or not onboard_payload.get("ok"):
        context.fail("onboard_solution chat did not return ok=true")
    onboard_reply = str(onboard_payload.get("reply") or "")
    if (
        "npa workbench byof run" not in onboard_reply
        and "run_byof_repo.py" not in onboard_reply
    ):
        context.fail(
            "onboard_solution chat reply missing byof CLI or run_byof_repo.py command"
        )
    if (
        "byof-onboard" not in onboard_reply
        and "skills/workflows/byof-onboard" not in onboard_reply
    ):
        context.fail("onboard_solution chat reply missing byof-onboard skill reference")
    if "--base-profile" not in onboard_reply and "--base-image" not in onboard_reply:
        context.fail("onboard_solution chat reply missing base image guidance")
    if "<repo-url>" not in onboard_reply:
        context.fail("onboard_solution chat reply missing runnable placeholders")
    if onboard_reply.strip().startswith("GET /api"):
        context.fail("onboard_solution chat returned raw GET path instead of guidance")
    if not onboard_payload.get("grounded"):
        context.fail("onboard_solution chat expected grounded=true")
    onboard_apis = onboard_payload.get("apis_used")
    if not isinstance(onboard_apis, list) or "tools" not in onboard_apis:
        context.fail("onboard_solution chat expected tools in apis_used")


def verify_agent_endpoints(
    record: dict[str, Any],
    *,
    auth: tuple[str, str],
    ui_version: str,
    health_check: Callable[..., tuple[bool, int]],
    fail: Callable[[str], NoReturn],
) -> None:
    """Check the deployed agent's access, viewers, workflows, and grounded chat.

    Args:
        record: Saved deployment endpoints and storage identity.
        auth: HTTP Basic Auth username and password.
        ui_version: Required rendered UI version.
        health_check: Existing authenticated endpoint health probe.
        fail: Callback that reports a failed check and terminates verification.

    Returns:
        None.

    Raises:
        typer.Exit: Via ``fail`` when a required check fails.
    """
    context = _VerificationContext(record, auth, ui_version, health_check, fail)
    _verify_public_access(context)
    _verify_viewers_and_assets(context)
    _verify_backend_and_ui(context)
    _verify_chat_and_workflows(context)


def _verify_viewers_and_assets(context: _VerificationContext) -> None:
    _verify_viewer_health(context)
    _verify_sim_viz_status(context)
    _verify_sim_assets(context)
    _verify_cameras(context)
    _verify_selection(context)
    submit_run_id = _submit_sim2real(context)
    _verify_submitted_status(context, submit_run_id)
    _verify_submitted_recording(context, submit_run_id)
    _verify_demo(context)
    _verify_camera_preview(context)
    _verify_recording(context)
    _verify_recording_blob(context)
    _verify_rerun_static_assets(context)
    _report_lichtblick(context)
    _verify_rerun_bundle(context)


def _verify_backend_and_ui(context: _VerificationContext) -> None:
    _verify_backend_health(context)
    _verify_leisaac(context)
    _verify_workflow_status(context)
    _verify_ui(context)
    _verify_session(context)
    _verify_access(context)
    _verify_tools(context)


def _verify_chat_and_workflows(context: _VerificationContext) -> None:
    _verify_status_chat(context)
    workflow_yaml = _request_workflow(context)
    _verify_workflow_yaml(context, workflow_yaml)
    _verify_infra(context)
    _verify_resources(context)
    _verify_workflow_submission(context, workflow_yaml)
    _verify_onboarding_chat(context)


def run_agent_verification_tests(
    project: str, name: str, *, fail: Callable[[str], NoReturn]
) -> None:
    """Run the smoke, unit, and live pytest gates for a named deployment.

    Args:
        project: Configured project alias.
        name: Agent deployment name.
        fail: Callback that reports a failed suite and terminates verification.

    Returns:
        None.

    Raises:
        typer.Exit: Via ``fail`` when a pytest suite fails.
        OSError: If the test interpreter cannot start.
    """
    test_env = {
        **dict(os.environ),
        "NPA_INTEGRATION_E2E": "1",
        "NPA_AGENT_LIVE": "1",
        "NPA_AGENT_PROJECT": project,
        "NPA_AGENT_NAME": name,
    }
    repo_root = Path(__file__).resolve().parents[4]
    for paths, failure in _VERIFICATION_SUITES:
        result = subprocess.run(
            [sys.executable or "python3", "-m", "pytest", *paths, "-q"],
            check=False,
            env=test_env,
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            fail(failure)


_VERIFICATION_SUITES = (
    (
        (
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
        ),
        "pytest npa/tests/smoke/test_agent_smoke.py test_agent_chat_smoke.py failed",
    ),
    (
        ("npa/tests/cli/test_agent.py", "npa/tests/cli/test_agent_workflow.py"),
        "pytest npa/tests/cli/test_agent.py failed",
    ),
    (
        ("npa/tests/e2e/test_agent_live.py",),
        "pytest npa/tests/e2e/test_agent_live.py failed",
    ),
)
