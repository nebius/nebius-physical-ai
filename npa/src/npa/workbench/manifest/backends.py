"""Execution backends: thin translators into existing machinery.

Backends do not implement tool logic. They take (pinned image, argv,
resources, outputs) and use existing launch/storage machinery.
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
import time
from pathlib import Path

from .runtime import BackendResult


class LocalDockerBackend:
    """Runs the container with `docker run --rm`, capturing output files.

    Used for local validation. GPU pass-through via --gpus when requested.
    Binary outputs are base64-captured (no S3 here); s3 inputs are not
    supported on this backend.
    """

    name = "local"
    s3_upload = False

    def __init__(self, docker_bin: str = "docker", timeout_s: int | None = None):
        # No default time limit: a run is bounded only when the operator
        # passes timeout_s explicitly (repo convention: no time/cost/job-count
        # limits unless the operator asks for them).
        self.docker_bin = docker_bin
        self.timeout_s = timeout_s

    def run(
        self,
        image_pinned: str,
        argv: list[str],
        gpu: int,
        outputs: list[tuple[str, str, str, str]],
        payload: dict[str, str] | None = None,
        memory_gb: int = 0,
        s3_inputs: list[tuple[str, str]] | None = None,
        s3_output_names: list[str] | None = None,
        s3_prefix: str = "",
        s3_env: dict[str, str] | None = None,
        pip_packages: list[str] | None = None,
        apt_packages: list[str] | None = None,
    ) -> BackendResult:
        if s3_inputs:
            raise ValueError("s3 inputs require the nebius backend")
        if pip_packages:
            raise ValueError(
                "environment.pip requires the nebius backend "
                "(local docker run cannot install into the image)"
            )
        if apt_packages:
            raise ValueError("environment.apt requires the nebius backend")
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="manifest-out-") as tmpd:
            outdir = Path(tmpd)
            # Bind-mount a host dir at /work/out and rewrite container output
            # paths to land there, so artifacts survive --rm. Stdout-source
            # outputs have no path and are skipped (no empty str.replace).
            remap = {
                cpath: f"/work/out/{oname}.dat"
                for oname, cpath, _fmt, source in outputs
                if source == "file" and cpath
            }
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
            if memory_gb > 0:
                cmd += ["--memory", f"{memory_gb}g"]
            cmd += [image_pinned] + rewritten
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )

            by_name = {oname: (fmt, source) for oname, _, fmt, source in outputs}
            artifacts: dict[str, str] = {}
            for oname, (fmt, source) in by_name.items():
                if source != "file":
                    continue
                host_file = outdir / f"{oname}.dat"
                if not host_file.exists():
                    continue
                if fmt == "binary":
                    artifacts[oname] = base64.b64encode(host_file.read_bytes()).decode()
                else:
                    artifacts[oname] = host_file.read_text()
            return BackendResult(
                exit_code=proc.returncode,
                logs=proc.stdout + proc.stderr,
                artifacts_raw=artifacts,
                elapsed_s=time.time() - t0,
                stdout=proc.stdout,
            )
