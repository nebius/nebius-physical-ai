#!/usr/bin/env python
"""Audit which agent-backend capabilities are real on the code under test.

Renders the exact ``backend.py`` that ``npa agent bootstrap`` installs, runs it
against a sandbox state root, and exercises every registered route plus the
grounded chat router. The report separates capabilities that answer from ones
that only exist as a route, so "the agent supports X" can be checked instead of
assumed.

Three tiers, same probes:

* in-process (default) drives the ASGI app directly -- fastest, no ports;
* ``--serve-live`` starts ``uvicorn backend:app`` from the rendered runtime
  directory with the same arguments as the deployed ``npa-agent-backend``
  systemd unit and probes it over real HTTP on loopback. This catches what
  in-process probing cannot: import-time and lifespan failures under the real
  server, and any route that only works because the test client bypasses it.
* ``--base-url`` probes a **deployed** agent VM through its authenticated
  HTTPS ingress. Route enumeration then comes from the deployment's own
  ``/openapi.json``, and the report includes the drift against the locally
  rendered backend -- which is how you learn the VM is running older code.

The first two tiers are offline and free: no Token Factory call, no cluster, no
VM, no public port. Any route that needs cloud credentials is reported as its
real outcome rather than being skipped, because "this needs credentials" is
itself an audit finding.

The live tier talks to a real deployment, so the capability probes -- which POST
to ``/chat``, run memory, and retrieval -- are skipped unless
``--allow-mutations`` is passed. Route probing is read-only and always runs.

Usage:
    npa/.venv/bin/python npa/scripts/audit_agent_capabilities.py [--json out.json]
    npa/.venv/bin/python npa/scripts/audit_agent_capabilities.py --serve-live
    npa/.venv/bin/python npa/scripts/audit_agent_capabilities.py \\
        --base-url https://<agent-ip>/api --auth-env ~/.npa/agents/<p>/<n>/auth.env \\
        --insecure --allow-mutations
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
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


def materialize_runtime(body: str, sandbox: Path) -> str:
    """Lay out the sandbox the way ``/opt/npa-agent`` looks on a deployed VM.

    Returns the sandboxed backend source. The rendered backend hardcodes
    ``/opt/npa-agent``; redirecting it is what keeps the audit from touching a
    real deployment's state on a shared machine.
    """

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
    (sandbox / "backend.py").write_text(sandboxed, encoding="utf-8")
    return sandboxed


def load_backend_app(body: str, sandbox: Path) -> Any:
    """Exec the rendered backend against a sandbox state root and return its app."""

    sandboxed = materialize_runtime(body, sandbox)
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


def read_auth_env(path: Path) -> tuple[str, str]:
    """Return ``(user, password)`` from an agent's ``auth.env``.

    `npa agent deploy` writes the shell-style file this parses. The values are
    never printed by this script; only their presence is reported.
    """

    user = password = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if key.strip() == "AGENT_USER":
            user = value
        elif key.strip() == "AGENT_PASSWORD":
            password = value
    if not user or not password:
        raise SystemExit(f"{path} did not define AGENT_USER and AGENT_PASSWORD")
    return user, password


# FastAPI serves these itself and never lists them in the OpenAPI document, so
# comparing a rendered routing table against a deployment's /openapi.json would
# always report them as missing.
_OPENAPI_UNLISTED_PATHS = frozenset(
    {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
)

_PATH_CONVERTER = re.compile(r"\{([^{}:]+):[^{}]+\}")


def comparable_route(method: str, path: str) -> tuple[str, str]:
    """Normalize a route for cross-source comparison.

    Starlette keeps the converter in ``route.path`` (``{run_id:path}``) while
    OpenAPI emits the bare name (``{run_id}``). Comparing the two raw forms
    reports every path-converter route as drift in both directions.
    """

    return method, _PATH_CONVERTER.sub(r"{\1}", path)


def live_routes(client: Any) -> list[tuple[str, str]]:
    """Enumerate routes from the deployment's own OpenAPI document.

    Auditing a VM against the local checkout's routing table would report the
    code under test, not the code deployed. Ask the deployment instead.
    """

    response = client.get("/openapi.json")
    if response.status_code != 200:
        raise SystemExit(
            f"GET /openapi.json returned {response.status_code}; cannot enumerate "
            "the deployment's routes"
        )
    paths = (response.json() or {}).get("paths") or {}
    routes: list[tuple[str, str]] = []
    for path, operations in paths.items():
        for method in operations or {}:
            upper = str(method).upper()
            if upper in {"HEAD", "OPTIONS", "PARAMETERS"}:
                continue
            routes.append((upper, str(path)))
    return sorted(set(routes))


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def serve_live(sandbox: Path, *, timeout: float = 90.0):
    """Run the rendered backend under uvicorn as the systemd unit does.

    Yields an ``httpx.Client`` bound to the loopback port. The argument list
    mirrors ``ExecStart`` of ``npa-agent-backend.service`` so a flag that breaks
    the real service also breaks here. Binding to 127.0.0.1 keeps the audit off
    the network on a shared machine.
    """

    import httpx

    port = _free_loopback_port()
    log_path = sandbox / "uvicorn.log"
    argv = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        "--no-access-log",
        "--ws",
        "websockets",
        "--ws-max-size",
        "4194304",
        "--ws-max-queue",
        "4",
        "--ws-ping-interval",
        "10",
        "--ws-ping-timeout",
        "10",
        "--ws-per-message-deflate",
        "false",
    ]
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv, cwd=str(sandbox), stdout=log_file, stderr=subprocess.STDOUT
        )
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
        try:
            deadline = time.monotonic() + timeout
            while True:
                if process.poll() is not None:
                    raise SystemExit(
                        "uvicorn exited before serving; log:\n"
                        + log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                    )
                try:
                    client.get("/health")
                    break
                except Exception:  # noqa: BLE001 - not up yet
                    if time.monotonic() >= deadline:
                        raise SystemExit(
                            f"uvicorn did not answer /health within {timeout:.0f}s; log:\n"
                            + log_path.read_text(encoding="utf-8", errors="replace")[
                                -4000:
                            ]
                        ) from None
                    time.sleep(0.5)
            yield client
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def iter_routes(app: Any) -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []
    for route in app.routes:
        path = getattr(route, "path", "")
        for method in sorted(getattr(route, "methods", set()) or set()):
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.append((method, path))
    return sorted(set(routes))


def shadowed_routes(app: Any) -> list[str]:
    """Return method+path pairs registered more than once.

    Starlette resolves the first match, so every later registration is
    unreachable. Probing cannot find these -- both copies answer the same URL --
    and de-duplicating the route list hides them, so report them explicitly.
    """

    counts: dict[tuple[str, str], int] = {}
    for route in app.routes:
        path = getattr(route, "path", "")
        for method in sorted(getattr(route, "methods", set()) or set()):
            key = (method, path)
            counts[key] = counts.get(key, 0) + 1
    return sorted(
        f"{method} {path} (x{count})"
        for (method, path), count in counts.items()
        if count > 1
    )


def probe_routes(client: Any, routes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """GET every parameterless read route and record its real outcome."""

    results: list[dict[str, Any]] = []
    for method, path in routes:
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
                # The FastAPI error text says *why* a route declined, which
                # is the difference between "needs an argument" and "this
                # deployment is missing a dependency".
                if "detail" in payload:
                    entry["detail"] = str(payload["detail"])[:200]
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


def probe_capabilities(client: Any) -> list[dict[str, Any]]:
    """Exercise the advertised capabilities end to end, not just their routes.

    A registered route proves nothing about whether the capability works. Each
    probe here drives a real request/response cycle that needs no cloud
    credentials and no model call, so the result is evidence rather than a
    reachability check. Anything requiring Token Factory is deliberately absent:
    it would cost tokens and turn the audit into a network test.
    """

    probes: list[dict[str, Any]] = []

    def record(name: str, expectation: str, call) -> None:
        entry: dict[str, Any] = {"capability": name, "expectation": expectation}
        try:
            status, detail, ok = call()
            entry.update({"status": status, "detail": detail, "works": bool(ok)})
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            entry.update(
                {"status": "exception", "detail": f"{type(exc).__name__}: {exc}", "works": False}
            )
        probes.append(entry)

    def grounded_chat():
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "what is the current sim2real status"}
                ]
            },
        )
        body = response.json() if response.status_code == 200 else {}
        grounded = bool(body.get("grounded"))
        apis = body.get("apis_used") or []
        return (
            response.status_code,
            f"grounded={grounded} apis_used={apis} reply_chars={len(str(body.get('reply') or ''))}",
            response.status_code == 200 and grounded and bool(apis),
        )

    record(
        "grounded chat (zero tokens)",
        "200, grounded=true, non-empty apis_used, no model call",
        grounded_chat,
    )

    def retrieval_corpus_discovery():
        # Indexing for real needs a Token Factory embedding call, so assert
        # the invariant that costs nothing and matters most: the corpus
        # scanner reports an empty corpus honestly instead of claiming a
        # successful index over zero documents.
        with tempfile.TemporaryDirectory(prefix="npa-audit-corpus-") as empty:
            response = client.post("/agent/retrieval/index", json={"roots": [empty]})
        body = response.json() if response.status_code == 200 else {}
        declined = body.get("ok") is False and "no corpus documents" in str(
            body.get("error", "")
        )
        return (
            response.status_code,
            f"empty_corpus_declined={declined}",
            declined,
        )

    record(
        "retrieval corpus discovery",
        "an empty corpus is refused, never reported as a successful index",
        retrieval_corpus_discovery,
    )

    def memory_roundtrip():
        record_response = client.post(
            "/agent/memory/record",
            json={"run_id": "audit-run", "metrics": {"success_rate": 0.5}},
        )
        listing = client.get("/agent/memory/runs")
        body = listing.json() if listing.status_code == 200 else {}
        runs = body.get("runs") or []
        return (
            f"record={record_response.status_code} list={listing.status_code}",
            f"runs={len(runs)}",
            listing.status_code == 200,
        )

    record(
        "run memory record + list",
        "a recorded run is listed back",
        memory_roundtrip,
    )

    def trace_analyze():
        response = client.post("/agent/trace/analyze", json={"spans": []})
        return (
            response.status_code,
            str(response.json())[:120],
            response.status_code < 500,
        )

    record(
        "trace analyze",
        "analyzes spans without a tracer backend installed",
        trace_analyze,
    )

    def action_loop_requires_a_goal():
        missing = client.post("/agent/act", json={})
        detail = str((missing.json() or {}).get("detail", ""))
        return (
            missing.status_code,
            f"detail={detail!r}",
            missing.status_code == 400 and "goal" in detail,
        )

    record(
        "bounded action loop contract",
        "refuses a goal-less request instead of planning against nothing",
        action_loop_requires_a_goal,
    )

    def chat_workflow_authoring():
        # Chat emits YAML only after validation *and* planning succeed, so
        # without a staged bucket/accelerator the correct outcome is a named
        # placeholder refusal. Either branch is a pass; emitting YAML that
        # does not validate is the failure this guards.
        chat = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "create 2-step sim2real workflow with 5000 "
                            "environments, seed 9, an RTX PRO 6000 "
                            "accelerator, and 1 GPU"
                        ),
                    }
                ]
            },
        )
        body = chat.json() if chat.status_code == 200 else {}
        yaml_text = str(body.get("workflow_yaml") or "")
        reply = str(body.get("reply") or "")
        if not yaml_text:
            refused = "could not generate runnable workflow yaml" in reply.lower()
            named = "placeholder" in reply.lower() or "configure-" in reply
            return (
                chat.status_code,
                f"declined_with_reason={refused and named}",
                refused and named,
            )
        validate = client.post("/workflows/validate", json={"yaml": yaml_text})
        result = validate.json() if validate.status_code == 200 else {}
        return (
            f"chat={chat.status_code} validate={validate.status_code}",
            f"yaml_chars={len(yaml_text)} validate_ok={result.get('ok')} "
            f"runnable={result.get('runnable')}",
            bool(result.get("ok")),
        )

    record(
        "chat workflow authoring",
        "emits YAML only when it validates and plans; otherwise names the "
        "unresolved placeholders",
        chat_workflow_authoring,
    )

    return probes


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
    parser.add_argument(
        "--serve-live",
        action="store_true",
        help=(
            "Probe a real uvicorn process on loopback, started with the same "
            "arguments as the deployed npa-agent-backend systemd unit, instead "
            "of driving the ASGI app in-process."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="",
        help=(
            "Probe a deployed agent through its authenticated ingress, e.g. "
            "https://<agent-ip>/api. Routes are enumerated from the "
            "deployment's own /openapi.json."
        ),
    )
    parser.add_argument(
        "--auth-env",
        default="",
        help="Path to the agent's auth.env (AGENT_USER / AGENT_PASSWORD).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS verification (agent ingress uses a self-signed cert).",
    )
    parser.add_argument(
        "--allow-mutations",
        action="store_true",
        help=(
            "Run the capability probes against a deployed agent. They POST to "
            "/chat, run memory, and retrieval, so they touch its state."
        ),
    )
    args = parser.parse_args()

    if args.base_url and args.serve_live:
        raise SystemExit("--base-url and --serve-live are mutually exclusive")

    report: dict[str, Any] = {"apiVersion": "npa.agent.capability-audit/v1"}
    if args.base_url:
        report["tier"] = "deployed-agent-https"
    elif args.serve_live:
        report["tier"] = "served-uvicorn-loopback"
    else:
        report["tier"] = "in-process-asgi"

    body = render_backend_body()
    report["backend_render"] = {
        "chars": len(body),
        "unsubstituted_placeholder": "__NPA_AGENT_" in body,
    }

    with tempfile.TemporaryDirectory(prefix="npa-agent-audit-") as tmp:
        sandbox = Path(tmp)
        # Route enumeration needs the app object either way; it only reads the
        # routing table, so it stays valid for the served tier too.
        app, _globals = load_backend_app(body, sandbox)
        routes = iter_routes(app)
        shadowed = shadowed_routes(app)
        report["routes"] = {
            "total": len(routes),
            "parameterless_get": sum(
                1 for method, path in routes if method == "GET" and "{" not in path
            ),
            "shadowed": shadowed,
            "all": [f"{method} {path}" for method, path in routes],
        }

        if args.base_url:
            import httpx

            auth = None
            if args.auth_env:
                auth = read_auth_env(Path(args.auth_env).expanduser())
            with httpx.Client(
                base_url=args.base_url.rstrip("/"),
                auth=auth,
                verify=not args.insecure,
                timeout=60.0,
                follow_redirects=True,
            ) as client:
                deployed = live_routes(client)
                deployed_keys = {comparable_route(m, p) for m, p in deployed}
                rendered_keys = {
                    comparable_route(m, p)
                    for m, p in routes
                    if p not in _OPENAPI_UNLISTED_PATHS
                }
                report["routes"]["deployed_total"] = len(deployed)
                report["routes"]["missing_on_deployment"] = [
                    f"{m} {p}" for m, p in sorted(rendered_keys - deployed_keys)
                ]
                report["routes"]["absent_from_render"] = [
                    f"{m} {p}" for m, p in sorted(deployed_keys - rendered_keys)
                ]
                report["route_probes"] = probe_routes(client, deployed)
                report["capability_probes"] = (
                    probe_capabilities(client) if args.allow_mutations else []
                )
        elif args.serve_live:
            # A separate root keeps the served process's state off the one the
            # in-process import already touched.
            served = Path(tmp) / "served"
            served.mkdir()
            materialize_runtime(body, served)
            with serve_live(served) as client:
                report["route_probes"] = probe_routes(client, routes)
                report["capability_probes"] = probe_capabilities(client)
        else:
            from fastapi.testclient import TestClient

            with TestClient(app, raise_server_exceptions=False) as client:
                report["route_probes"] = probe_routes(client, routes)
                report["capability_probes"] = probe_capabilities(client)

    report["chat_probes"] = probe_chat_router()

    probes = report["route_probes"]
    for probe in probes:
        probe["outcome"] = classify_outcome(probe)
    counts: dict[str, int] = {}
    for probe in probes:
        counts[probe["outcome"]] = counts.get(probe["outcome"], 0) + 1
    report["summary"] = {
        "tier": report["tier"],
        "routes_total": report["routes"]["total"],
        "routes_shadowed": len(report["routes"]["shadowed"]),
        "routes_probed": len(probes),
        "route_outcomes": dict(sorted(counts.items())),
        "chat_intents_probed": len(report["chat_probes"]),
        "chat_intents_matched": sum(
            1 for p in report["chat_probes"] if p.get("intent_ok")
        ),
        "chat_replies_ok": sum(1 for p in report["chat_probes"] if p.get("reply_ok")),
        "capabilities_probed": len(report["capability_probes"]),
        "capabilities_working": sum(
            1 for p in report["capability_probes"] if p.get("works")
        ),
    }

    if "deployed_total" in report["routes"]:
        report["summary"]["routes_deployed"] = report["routes"]["deployed_total"]
        report["summary"]["routes_missing_on_deployment"] = len(
            report["routes"]["missing_on_deployment"]
        )
        report["summary"]["routes_absent_from_render"] = len(
            report["routes"]["absent_from_render"]
        )

    print(json.dumps(report["summary"], indent=2))
    for label in ("missing_on_deployment", "absent_from_render"):
        for entry in report["routes"].get(label, []):
            print(f"  [{label}] {entry}")
    if report["routes"]["shadowed"]:
        print("\n-- shadowed routes (registered twice; only the first serves) --")
        for entry in report["routes"]["shadowed"]:
            print(f"  [SHADOWED] {entry}")
    print("\n-- route probes --")
    for probe in probes:
        note = probe.get("error") or probe.get("detail") or ""
        print(
            f"  [{probe['outcome']:17s}] {probe['method']:4s} {probe['path']:44s} "
            f"{probe.get('status')} {note[:80]}"
        )
    print("\n-- capabilities --")
    for probe in report["capability_probes"]:
        flag = "works" if probe.get("works") else "CHECK"
        print(
            f"  [{flag:5s}] {probe['capability']:30s} {str(probe.get('status')):26s} "
            f"{str(probe.get('detail'))[:90]}"
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
