#!/usr/bin/env python
"""Audit which agent-backend capabilities are real on the code under test.

Renders the exact ``backend.py`` that ``npa agent bootstrap`` installs, runs it
in-process against a sandbox state root, and exercises every registered route
plus the grounded chat router. The report separates capabilities that answer
from ones that only exist as a route, so "the agent supports X" can be checked
instead of assumed.

Zero-cost and offline by default: no Token Factory call, no cluster, no VM. Any
route that needs cloud credentials is reported as its real outcome rather than
being skipped, because "this needs credentials" is itself an audit finding.

Usage:
    npa/.venv/bin/python npa/scripts/audit_agent_capabilities.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

SANDBOX_MARKER = "/opt/npa-agent"


def render_backend_body() -> str:
    """Return the rendered backend source bootstrap would upload."""

    from npa.cli import agent as agent_module

    captured: dict[str, str] = {}

    class _DummySsh:
        def upload_file(self, local_path: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                try:
                    captured["setup_script"] = Path(local_path).read_text(
                        encoding="utf-8"
                    )
                except UnicodeDecodeError:
                    pass

        def upload_private_text(self, content: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                captured["setup_script"] = content

        def run_or_raise(self, _command: str, **_kwargs) -> None:
            return None

        def run(self, _command: str) -> None:
            return None

    original_ssh = agent_module.SSHClient
    original_resolve = agent_module.resolve_ssh_config
    agent_module.SSHClient = lambda config: _DummySsh()  # type: ignore[assignment]
    agent_module.resolve_ssh_config = lambda **_kwargs: SimpleNamespace(ssh={})  # type: ignore[assignment]
    try:
        agent_module._bootstrap_agent_stack(
            host="203.0.113.50",
            ssh_user="ubuntu",
            ssh_key_path="/tmp/audit-key",
            project_alias="audit",
            project_id="project-audit",
            tenant_id="tenant-audit",
            region="us-central1",
            auth_user="npa",
            auth_password="audit-password",
            agent_port=8088,
            backend_port=8787,
            rerun_port=9090,
            llm_model="nvidia/Cosmos3-Super-Reasoner",
            llm_models=["nvidia/Cosmos3-Super-Reasoner"],
            tf_api_key="",
            nebius_ai_key="",
            public_https=True,
        )
    finally:
        agent_module.SSHClient = original_ssh  # type: ignore[assignment]
        agent_module.resolve_ssh_config = original_resolve  # type: ignore[assignment]

    setup_script = captured.get("setup_script", "")
    match = re.search(
        r"cat <<'PY' \| sudo tee /opt/npa-agent/backend\.py >/dev/null\n(?P<body>.*?)\nPY\n",
        setup_script,
        flags=re.DOTALL,
    )
    if not match:
        raise SystemExit("bootstrap setup script did not emit a backend.py heredoc")
    return match.group("body")


def load_backend_app(body: str, sandbox: Path) -> Any:
    """Exec the rendered backend against a sandbox state root and return its app."""

    # The rendered backend hardcodes /opt/npa-agent. Redirect it so the audit
    # never touches a real deployment's state on a shared machine.
    sandboxed = body.replace(SANDBOX_MARKER, str(sandbox))
    for child in ("recordings", "runs", "reports", "retrieval", "trace", "foxglove"):
        (sandbox / child).mkdir(parents=True, exist_ok=True)

    # On the VM, backend.py sits in /opt/npa-agent next to the shipped
    # agent_backend package, so its own directory supplies the import root.
    # Reproduce that layout from the repo copy.
    import npa.agent_backend as shipped

    shipped_root = Path(shipped.__file__).resolve().parent
    link = sandbox / "agent_backend"
    if not link.exists():
        link.symlink_to(shipped_root, target_is_directory=True)
    if str(sandbox) not in sys.path:
        sys.path.insert(0, str(sandbox))

    # Register a real module before exec: dataclasses resolve their annotations
    # through sys.modules[cls.__module__].
    module = ModuleType("npa_audit_backend")
    module.__file__ = str(sandbox / "backend.py")
    sys.modules[module.__name__] = module
    exec(compile(sandboxed, "backend.py", "exec"), module.__dict__)  # noqa: S102
    app = getattr(module, "app", None)
    if app is None:
        raise SystemExit("rendered backend exposed no FastAPI app")
    return app, module.__dict__


def iter_routes(app: Any) -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []
    for route in app.routes:
        path = getattr(route, "path", "")
        for method in sorted(getattr(route, "methods", set()) or set()):
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.append((method, path))
    return sorted(set(routes))


def probe_routes(app: Any) -> list[dict[str, Any]]:
    """GET every parameterless read route and record its real outcome."""

    from fastapi.testclient import TestClient

    results: list[dict[str, Any]] = []
    with TestClient(app, raise_server_exceptions=False) as client:
        for method, path in iter_routes(app):
            if method != "GET" or "{" in path:
                continue
            entry: dict[str, Any] = {"method": method, "path": path}
            try:
                response = client.get(path)
                entry["status"] = response.status_code
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    entry["keys"] = sorted(payload.keys())[:12]
                    # Namespace body signals so a payload `status` cannot be
                    # mistaken for the HTTP status this probe recorded.
                    for signal in ("ok", "error", "status", "scope", "grounded"):
                        if signal in payload:
                            entry[f"body_{signal}"] = payload[signal]
            except Exception as exc:  # noqa: BLE001 - the failure is the finding
                entry["status"] = "exception"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
    return results


CHAT_PROBES: tuple[tuple[str, str], ...] = (
    ("sim2real_status", "what is the current sim2real status"),
    ("sim_assets", "show me the sim assets"),
    ("cameras", "which cameras are selected"),
    ("tools_catalog", "list the available tools"),
    ("find_artifacts", "what can I view?"),
    ("load_franka", "load the franka demo"),
    ("watch_sim", "watch the sim until the blob and iframe both report success"),
    ("list_recordings", "list the available recordings"),
    ("foxglove_viewer", "open foxglove"),
    ("start_sim2real", "start the sim2real pipeline"),
    ("drive_sim2real", "autonomously drive the sim2real outer loop"),
    ("onboard_solution", "containerize this github repo as a workbench solution"),
    ("create_data_factory_workflow", "create a PAIDF workflow yaml"),
    ("create_vlm_rl_workflow", "create a sim-to-real workflow yaml"),
    ("create_workflow", "create a 2-step sim2real npa.workflow"),
    # "quality gate" alone is claimed by the earlier VLM-RL rule by design, so
    # probe this template through its own Token Factory / Cosmos-gate wording.
    ("create_gate_workflow", "create a token factory gate workflow"),
    ("create_loop_gate_workflow", "create a sim2real workflow with a loop gate"),
    ("create_rl_policy_workflow", "create an RL policy training workflow"),
    ("workflow_execute_guidance", "how do I actually run this workflow"),
    ("infra_backends", "which infra backends are available"),
    ("mk8s_provision", "provision an mk8s cluster"),
    ("live_infra_loop", "run the live infra loop"),
    ("tenant_resources", "what tenant resources do I have"),
    ("configure_s3", "configure S3 bucket access"),
    ("cosmos3", "set up cosmos3"),
    ("soperator", "deploy a slurm cluster"),
    ("cosmos_capabilities", "what can cosmos do"),
    ("lancedb_capabilities", "what can lancedb do"),
    ("sonic_capabilities", "what can sonic do"),
    ("lerobot_capabilities", "what can lerobot do"),
    ("groot_capabilities", "what can groot do"),
    ("genesis_capabilities", "what can genesis do"),
    ("mjlab_capabilities", "what can mjlab do"),
    ("isaac_lab_capabilities", "what can isaac lab do"),
    ("component_capabilities", "what components are available"),
)


def classify_outcome(probe: dict[str, Any]) -> str:
    """Separate a real defect from a route that simply needs arguments.

    A bare GET against a route with required query parameters answers 422/400 by
    design, and a disabled feature answers 403. Counting those as failures would
    hide the outcomes that matter: a 5xx or an unhandled exception.
    """

    status = probe.get("status")
    if not isinstance(status, int):
        return "error"
    if status < 400:
        return "answered"
    if status in {400, 422}:
        return "needs_arguments"
    if status in {401, 403}:
        return "gated"
    if status == 404:
        return "absent_in_sandbox"
    return "error"


def probe_chat_router() -> list[dict[str, Any]]:
    """Check every advertised intent matches and produces a grounded reply."""

    from npa.cli.agent_chat import build_grounded_reply, match_chat_intent

    state = {
        "selection": {
            "robot_preset": "franka",
            "sim_backend": "isaac",
            "scene_spec_uri": "stock://scene/default",
        },
        "sim_viz": {
            "run_id": "audit-run",
            "stage": "demo",
            "camera": "workspace",
            "rerun_ready": True,
            "rrd_uri": "file:///tmp/audit.rrd",
        },
        "latest_submit": {},
        "camera_selection": ["workspace"],
        "chat_history": [],
    }

    tool_refs = ["workbench.genesis.train", "workbench.lerobot.train", "workbench.vlm-eval.run"]
    results: list[dict[str, Any]] = []
    for expected, prompt in CHAT_PROBES:
        entry: dict[str, Any] = {"expected_intent": expected, "prompt": prompt}
        matched = match_chat_intent(prompt)
        entry["matched_intent"] = matched
        entry["intent_ok"] = matched == expected
        try:
            reply = build_grounded_reply(matched or expected, state, tool_refs)
            entry["reply_chars"] = len(reply or "")
            entry["reply_ok"] = bool(reply and reply.strip())
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            entry["reply_ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default="", help="Write the report as JSON.")
    args = parser.parse_args()

    report: dict[str, Any] = {"apiVersion": "npa.agent.capability-audit/v1"}

    body = render_backend_body()
    report["backend_render"] = {
        "chars": len(body),
        "unsubstituted_placeholder": "__NPA_AGENT_" in body,
    }

    with tempfile.TemporaryDirectory(prefix="npa-agent-audit-") as tmp:
        sandbox = Path(tmp)
        app, _globals = load_backend_app(body, sandbox)
        routes = iter_routes(app)
        report["routes"] = {
            "total": len(routes),
            "parameterless_get": sum(
                1 for method, path in routes if method == "GET" and "{" not in path
            ),
            "all": [f"{method} {path}" for method, path in routes],
        }
        report["route_probes"] = probe_routes(app)

    report["chat_probes"] = probe_chat_router()

    probes = report["route_probes"]
    for probe in probes:
        probe["outcome"] = classify_outcome(probe)
    counts: dict[str, int] = {}
    for probe in probes:
        counts[probe["outcome"]] = counts.get(probe["outcome"], 0) + 1
    report["summary"] = {
        "routes_total": report["routes"]["total"],
        "routes_probed": len(probes),
        "route_outcomes": dict(sorted(counts.items())),
        "chat_intents_probed": len(report["chat_probes"]),
        "chat_intents_matched": sum(
            1 for p in report["chat_probes"] if p.get("intent_ok")
        ),
        "chat_replies_ok": sum(1 for p in report["chat_probes"] if p.get("reply_ok")),
    }

    print(json.dumps(report["summary"], indent=2))
    print("\n-- route probes --")
    for probe in probes:
        print(
            f"  [{probe['outcome']:17s}] {probe['method']:4s} {probe['path']:44s} "
            f"{probe.get('status')} {probe.get('error', '')[:80]}"
        )
    print("\n-- chat intents --")
    for probe in report["chat_probes"]:
        flag = "ok " if probe.get("intent_ok") and probe.get("reply_ok") else "BAD"
        print(
            f"  [{flag}] {probe['expected_intent']:32s} matched={str(probe.get('matched_intent')):32s} "
            f"reply={probe.get('reply_chars', 0)}ch {probe.get('error', '')[:70]}"
        )

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - print the real traceback for triage
        traceback.print_exc()
        sys.exit(2)
