#!/usr/bin/env python
"""Run the NPA agent locally: render the embedded stack, then serve it on localhost.

``npa agent deploy`` provisions a VM; there is no localhost mode. This script
gives one for development and screenshots by reusing the *same* embedded-backend
render the bootstrap performs, so the code under test is byte-identical to what
the agent VM runs:

1. Render ``backend.py``, the shipped ``agent_backend`` modules, and ``ui.html``
   out of the bootstrap script (via a mocked SSH client).
2. Serve them the way nginx does on the VM: ``ui.html`` at ``/`` and the backend
   mounted under ``/api``.

Artifact loading needs S3 credentials and the bucket the artifacts live in::

    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_ENDPOINT_URL=...
    export NPA_AGENT_S3_BUCKET=<bucket>
    # optional: LLM chat instead of grounded-only replies
    export NEBIUS_TOKEN_FACTORY_KEY=...
    npa/.venv/bin/python npa/scripts/run_agent_local.py

Then open http://127.0.0.1:8088/. The backend downloads artifacts under
``/opt/npa-agent``, so create it once and make it writable by your user.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

DEFAULT_ROOT = Path("/tmp/npa-agent-local")
HEREDOC = re.compile(
    r"cat <<'(?:PY|HTML|JS)' \| sudo tee {path} >/dev/null\n(?P<body>.*?)\n(?:PY|HTML|JS)\n",
    re.DOTALL,
)
SHIPPED_MODULES = ("memory", "retrieval", "trace", "foxglove", "foxglove_routes")


def render_stack(out_dir: Path) -> Path:
    """Render backend + shipped modules + UI into ``out_dir``; return the dir."""

    from npa.cli import agent as agent_module

    captured: dict[str, str] = {}

    class _CapturingSsh:
        def upload_file(self, local_path: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                try:
                    captured["setup"] = Path(local_path).read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass

        def run_or_raise(self, _command: str) -> None:
            return None

        def run(self, _command: str) -> None:
            return None

    agent_module.SSHClient = lambda config: _CapturingSsh()
    agent_module.resolve_ssh_config = lambda **_kwargs: SimpleNamespace(ssh={})
    agent_module._bootstrap_agent_stack(
        host="203.0.113.50",
        ssh_user="ubuntu",
        ssh_key_path="/tmp/key",
        project_alias="local",
        project_id="local-project",
        tenant_id="local-tenant",
        region="eu-north1",
        auth_user="npa",
        auth_password="local",
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model="nvidia/Cosmos3-Super-Reasoner",
        llm_models=["nvidia/Cosmos3-Super-Reasoner"],
        tf_api_key="",
        nebius_ai_key="",
        public_https=False,
    )
    setup = captured.get("setup", "")
    if not setup:
        raise SystemExit("bootstrap did not produce a setup script")

    def extract(remote_path: str) -> str:
        match = re.search(
            HEREDOC.pattern.format(path=re.escape(remote_path)), setup, re.DOTALL
        )
        if not match:
            raise SystemExit(f"bootstrap does not write {remote_path}")
        return match.group("body")

    out_dir.mkdir(parents=True, exist_ok=True)
    package = out_dir / "agent_backend"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in SHIPPED_MODULES:
        (package / f"{name}.py").write_text(
            extract(f"/opt/npa-agent/agent_backend/{name}.py"), encoding="utf-8"
        )
    (out_dir / "backend.py").write_text(
        extract("/opt/npa-agent/backend.py"), encoding="utf-8"
    )
    (out_dir / "ui.html").write_text(
        agent_module.rendered_agent_ui_html(), encoding="utf-8"
    )
    return out_dir


def build_app(root: Path):
    """Return the local front door: UI at ``/`` and the backend under ``/api``."""

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    sys.path.insert(0, str(root))
    import backend as agent_backend  # noqa: PLC0415 - rendered at runtime

    ui_html = (root / "ui.html").read_text(encoding="utf-8")
    app = FastAPI(title="npa-agent-local")

    @app.get("/", response_class=HTMLResponse)
    def _index() -> HTMLResponse:
        return HTMLResponse(ui_html)

    @app.get("/healthz")
    def _healthz() -> dict:
        return {"ok": True, "mode": "local"}

    # nginx strips /api before proxying, so the backend keeps its own root paths.
    app.mount("/api", agent_backend.app)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render the stack and exit without serving.",
    )
    args = parser.parse_args(argv)

    root = render_stack(args.root)
    print(f"rendered agent stack -> {root}")
    if args.render_only:
        return 0

    import uvicorn

    print(f"serving NPA agent UI on http://{args.host}:{args.port}/")
    uvicorn.run(build_app(root), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
