"""Execution backends: thin translators into existing machinery.

Backends do not implement tool logic. They take (pinned image, argv,
resources, outputs) and use existing launch/storage machinery.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from .runtime import BackendResult


class LocalDockerBackend:
    """Runs the container with `docker run --rm`, capturing output files.

    Used for local validation. GPU pass-through via --gpus when requested.
    """

    name = "local"

    def __init__(self, docker_bin: str = "docker"):
        self.docker_bin = docker_bin

    def run(
        self,
        image_pinned: str,
        argv: list[str],
        gpu: int,
        outputs: list[tuple[str, str]],
        payload: dict[str, str] | None = None,
    ) -> BackendResult:
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="manifest-out-") as tmpd:
            outdir = Path(tmpd)
            payload = payload or {}
            # Bind-mount a host dir at /work/out and rewrite container output
            # paths to land there, so artifacts survive --rm.
            remap = {cpath: f"/work/out/{oname}.dat" for oname, cpath in outputs}
            rewritten = []
            for token in argv:
                for cpath, mpath in remap.items():
                    token = token.replace(cpath, mpath)
                rewritten.append(token)

            cmd = [self.docker_bin, "run", "--rm", "-v", f"{outdir}:/work/out"]
            # Payload files land under /work/out/payload/<basename>; argv
            # tokens referencing their container paths are rewritten.
            paydir = outdir / "payload"
            paydir.mkdir()
            for cpath, content in (payload or {}).items():
                dest = paydir / Path(cpath).name
                dest.write_text(content)
                mpath = f"/work/out/payload/{dest.name}"
                rewritten = [t.replace(cpath, mpath) for t in rewritten]
            if gpu > 0:
                cmd += ["--gpus", "all"]
            cmd += [image_pinned] + rewritten
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            artifacts: dict[str, str] = {}
            for oname, _cpath in outputs:
                host_file = outdir / f"{oname}.dat"
                if host_file.exists():
                    artifacts[oname] = host_file.read_text()
            return BackendResult(
                exit_code=proc.returncode,
                logs=proc.stdout + proc.stderr,
                artifacts_raw=artifacts,
                elapsed_s=time.time() - t0,
                stdout=proc.stdout,
            )
