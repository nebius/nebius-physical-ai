"""Launch the team service without inherited operator credentials or source overlays."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .models import load_config


def serve(config: Path, *, host: str, port: int):
    """Replace the CLI process with a clean, single-process authenticated supervisor.

    Args:
        config: Administrator-owned service policy.
        host, port: Private listener behind HTTPS ingress.
    Returns:
        Does not return after a successful exec.
    Raises:
        TeamError, OSError: Configuration or process startup fails.
    """
    selected = load_config(config)
    home = selected.state_dir / "home"
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    allowed = ("PATH", "LANG", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        HOME=str(home),
        NPA_CONFIG_DIR=str(home / "npa"),
        PYTHONPATH=str(Path(__file__).resolve().parents[3]),
    )
    os.chdir(home)
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-m",
            "npa.workbench.team.server",
            str(config),
            host,
            str(port),
        ],
        environment,
    )


def _main():
    import uvicorn

    from .api import create_app

    os.umask(0o077)
    uvicorn.run(
        create_app(Path(sys.argv[1])),
        host=sys.argv[2],
        port=int(sys.argv[3]),
        workers=1,
        proxy_headers=False,
        access_log=False,
    )


if __name__ == "__main__":
    _main()
