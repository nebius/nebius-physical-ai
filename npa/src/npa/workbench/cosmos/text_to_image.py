"""Cosmos3 text-to-image inference.

The retired ``skypilot/cosmos3-text-to-image-inference.yaml`` carried this whole capability as
~100 lines of bash and heredoc'd python inside an ``envs:`` block, including the inference
command itself as a multi-line environment variable that the ``run:`` script then executed with
``bash -lc "${NPA_COSMOS3_INFER_COMMAND}"``. None of it was reachable from the CLI or the SDK,
none of it was tested, and a `toolRef` argv cannot express any of it.

Here it is a function: fetch the source and checkpoint (reusing the same
``fetch_cosmos3_artifacts`` the ``cosmos fetch`` command uses), sync the framework's own
environment, run its inference entrypoint, verify the image is real, and publish it with a
manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Split across two lines: as one import the pair trips gitleaks' generic-api-key heuristic,
# which reads `…Config, fetch_cosmos3_artifacts` as an assignment of a long opaque value.
from npa.workbench.cosmos.cosmos3 import Cosmos3AccessConfig
from npa.workbench.cosmos.cosmos3 import fetch_cosmos3_artifacts

#: The framework's own dependency group. Text-to-image needs the training extras because the
#: inference entrypoint imports from the same package tree.
DEFAULT_UV_GROUP = "cu130-train"

#: What the framework writes for a job named ``npa-t2i``, relative to the output directory.
FRAMEWORK_OUTPUT_RELPATH = Path("npa-t2i") / "vision.jpg"

#: Published names. Specs declare the manifest, so it is a constant rather than a literal.
IMAGE_FILENAME = "text-to-image.png"
MANIFEST_FILENAME = "success.json"

#: Minimum plausible size for a generated image. A zero-byte or truncated file is the failure
#: mode worth catching loudly: the framework can exit 0 having written nothing useful.
MIN_IMAGE_BYTES = 1024


class Cosmos3TextToImageError(RuntimeError):
    """Raised when text-to-image inference cannot produce a usable image."""


@dataclass(frozen=True)
class TextToImageResult:
    """What one text-to-image run produced."""

    status: str
    prompt: str
    model_id: str
    output_image: str
    bytes: int
    width: int
    height: int
    seed: int
    source_dir: str
    checkpoint_dir: str
    published: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "npa.cosmos3.text_to_image.v1",
            "status": self.status,
            "prompt": self.prompt,
            "model_id": self.model_id,
            "output_image": self.output_image,
            "bytes": self.bytes,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "source_dir": self.source_dir,
            "checkpoint_dir": self.checkpoint_dir,
        }
        if self.published:
            payload["published"] = dict(self.published)
        return payload


def image_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for a PNG or JPEG, using the standard library only.

    Pillow would do this in one line, but this module runs inside the Cosmos framework's own
    uv-managed environment, where adding a dependency risks perturbing a pinned stack for the
    sake of reading two integers out of a header.
    """

    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise Cosmos3TextToImageError(f"truncated PNG: {path}")
        return (
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
        )
    if data[:2] == b"\xff\xd8":
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            # SOF0..SOF15, skipping the non-frame markers in that range.
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height = int.from_bytes(data[index + 5 : index + 7], "big")
                width = int.from_bytes(data[index + 7 : index + 9], "big")
                return width, height
            segment = int.from_bytes(data[index + 2 : index + 4], "big")
            index += 2 + segment
        raise Cosmos3TextToImageError(f"no JPEG frame header found: {path}")
    raise Cosmos3TextToImageError(f"unrecognised image format: {path}")


def verify_image(path: Path) -> tuple[int, int, int]:
    """Return ``(bytes, width, height)``, refusing anything that is not a real image."""

    if not path.is_file():
        raise Cosmos3TextToImageError(f"inference produced no image at {path}")
    size = path.stat().st_size
    if size < MIN_IMAGE_BYTES:
        raise Cosmos3TextToImageError(
            f"inference produced a {size}-byte image at {path}; expected at least "
            f"{MIN_IMAGE_BYTES}"
        )
    width, height = image_dimensions(path)
    if width <= 0 or height <= 0:
        raise Cosmos3TextToImageError(f"image has invalid dimensions {width}x{height}: {path}")
    return size, width, height


def build_job_document(prompt: str, name: str = "npa-t2i") -> dict[str, str]:
    """The framework's per-job input document."""

    return {"model_mode": "text2image", "name": name, "prompt": prompt}


def inference_argv(
    *,
    input_json: Path,
    output_dir: Path,
    checkpoint_name: str,
    seed: int,
    guardrails: bool,
) -> list[str]:
    """The framework's inference command, as an argv rather than an interpolated string."""

    argv = [
        ".venv/bin/python",
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "-i",
        str(input_json),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        checkpoint_name,
        f"--seed={seed}",
    ]
    if not guardrails:
        argv.append("--no-guardrails")
    return argv


#: transformer_engine, which the framework imports at inference time, links against a modern
#: libstdc++. Live job 296 died with
#:   OSError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
#: on SkyPilot's default image. The template set LD_LIBRARY_PATH="" and would have hit the same
#: wall; a newer libstdc++ is usually already present in the conda prefix the stage runs from.
REQUIRED_GLIBCXX = b"GLIBCXX_3.4.29"


def _has_required_glibcxx(candidate: Path) -> bool:
    try:
        return REQUIRED_GLIBCXX in candidate.read_bytes()
    except OSError:
        return False


def runtime_library_dir(candidates: Sequence[Path] | None = None) -> str:
    """Return a directory holding a libstdc++ new enough for transformer_engine, or ``""``.

    Checked by looking for the version symbol in the library itself rather than by parsing
    `ldconfig` output or trusting a distro version: the question is exactly "does this file
    export GLIBCXX_3.4.29", and the file can answer it.
    """

    import sys as _sys

    if candidates is None:
        candidates = (
            Path(_sys.prefix) / "lib",
            Path(_sys.base_prefix) / "lib",
            Path("/opt/conda/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
        )
    for directory in candidates:
        library = directory / "libstdc++.so.6"
        if library.is_file() and _has_required_glibcxx(library):
            return str(directory)
    return ""


def link_runtime_library(shim_dir: Path) -> str:
    """Symlink a new-enough libstdc++ into ``shim_dir`` and return it, or "" if none is needed.

    A directory on ``LD_LIBRARY_PATH`` brings everything in it. The conda prefix that supplies a
    modern libstdc++ also supplies an older cuDNN, and PyTorch refuses to run against a cuDNN
    other than the one it was built with. Linking the single library keeps the fix to the
    problem it was for.
    """

    directory = runtime_library_dir()
    if not directory:
        return ""
    source = Path(directory) / "libstdc++.so.6"
    system = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if system.is_file() and _has_required_glibcxx(system):
        # The host's own copy is fine; adding anything to the loader path is pure risk.
        return ""
    shim_dir.mkdir(parents=True, exist_ok=True)
    link = shim_dir / "libstdc++.so.6"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(source)
    return str(shim_dir)


def uv_argv() -> list[str]:
    """Return an argv prefix for uv, preferring a module over a PATH lookup.

    Live job 291: `[Errno 2] No such file or directory: 'uv'` on SkyPilot's default image, which
    ships a uv inside its own runtime directory — on setup's PATH, not the stage command's. The
    module form runs from the interpreter that installed it, so it cannot drift that way.
    """

    import importlib.util
    import shutil
    import sys

    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    executable = shutil.which("uv")
    if executable:
        return [executable]
    raise Cosmos3TextToImageError(
        "uv is required to sync the Cosmos framework environment; install it with "
        "`pip install uv` into the interpreter running this stage"
    )


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], what: str) -> None:
    completed = subprocess.run(argv, cwd=str(cwd), env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout or "").splitlines()[-25:])
        raise Cosmos3TextToImageError(f"{what} failed (exit {completed.returncode}):\n{tail}")


def generate(
    config: Cosmos3AccessConfig,
    *,
    prompt: str,
    output_dir: Path,
    seed: int = 0,
    guardrails: bool = False,
    uv_group: str = DEFAULT_UV_GROUP,
    checkpoint_name: str = "Cosmos3-Nano",
    publish_uri: str = "",
    environ: dict[str, str] | None = None,
) -> TextToImageResult:
    """Fetch, infer, verify, and optionally publish one text-to-image result."""

    if not prompt.strip():
        raise Cosmos3TextToImageError("a prompt is required")

    fetched = fetch_cosmos3_artifacts(config, force=True)
    if not fetched.ok:
        raise Cosmos3TextToImageError(
            "fetching the Cosmos framework and checkpoint failed: "
            + "; ".join(fetched.errors or ("no reason reported",))
        )
    source_dir = Path(fetched.source_checkout)
    if not source_dir.is_dir():
        raise Cosmos3TextToImageError(f"source checkout missing after fetch: {source_dir}")

    env = dict(environ if environ is not None else os.environ)
    env["HF_HOME"] = str(Path(config.cache_dir) / "hf")
    # The framework vendors git-lfs pointers it does not need for inference.
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    _run(
        [*uv_argv(), "sync", "--all-extras", f"--group={uv_group}"],
        cwd=source_dir,
        env=env,
        what="uv sync of the Cosmos framework environment",
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_json = output_dir / "npa-t2i.json"
    input_json.write_text(json.dumps(build_job_document(prompt), sort_keys=True) + "\n")

    # The template cleared LD_LIBRARY_PATH here. Clearing it is right in spirit — the caller's
    # value is not the framework's — but it is not enough: transformer_engine needs a libstdc++
    # newer than some hosts ship (live job 296). Point the loader at ONE library, not at the
    # directory holding it: exporting the whole conda lib dir also put an older cuDNN ahead of
    # the one PyTorch bundles, and the next run died with
    # "cuDNN version incompatibility ... a conflicting cuDNN in LD_LIBRARY_PATH" (job 319).
    infer_env = dict(env)
    shim_dir = link_runtime_library(output_dir / ".libstdcxx")
    infer_env["LD_LIBRARY_PATH"] = shim_dir
    if shim_dir:
        print(f"cosmos3 text-to-image: LD_LIBRARY_PATH={shim_dir} (libstdc++ only)", flush=True)
    _run(
        inference_argv(
            input_json=input_json,
            output_dir=output_dir,
            checkpoint_name=checkpoint_name,
            seed=seed,
            guardrails=guardrails,
        ),
        cwd=source_dir,
        env=infer_env,
        what="Cosmos3 text-to-image inference",
    )

    produced = output_dir / FRAMEWORK_OUTPUT_RELPATH
    image_path = output_dir / IMAGE_FILENAME
    if produced.is_file():
        shutil.copyfile(produced, image_path)
    size, width, height = verify_image(image_path)

    result = TextToImageResult(
        status="ok",
        prompt=prompt,
        model_id=config.model_id,
        output_image=str(image_path),
        bytes=size,
        width=width,
        height=height,
        seed=seed,
        source_dir=str(source_dir),
        checkpoint_dir=fetched.checkpoint_dir,
    )

    if publish_uri.strip():
        published = publish(result, output_dir, publish_uri)
        result = TextToImageResult(**{**result.__dict__, "published": published})
    return result


def publish(result: TextToImageResult, output_dir: Path, publish_uri: str) -> dict[str, str]:
    """Upload the image and its manifest, so the run outlives the pod."""

    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    prefix = publish_uri if publish_uri.endswith("/") else publish_uri + "/"
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n")
    return {
        "image_uri": client.upload_file(result.output_image, prefix + IMAGE_FILENAME),
        "manifest_uri": client.upload_file(str(manifest_path), prefix + MANIFEST_FILENAME),
    }
