"""Locate an NVIDIA Cosmos Curator checkout and check what it can run here.

The upstream project (https://github.com/nvidia-cosmos/cosmos-curate,
Apache-2.0) publishes no wheel of its pipeline code — the released package is
the launcher client, and the pipelines ship inside the curator container. Its
stage classes are, however, plain Python objects with a ``process_data(tasks)``
method, so a source checkout on ``sys.path`` is enough to drive the real stages
in-process (see :mod:`.pipeline`).

This module answers three questions for callers:

- where is the checkout (``NPA_COSMOS_CURATE_SRC``, else conventional image
  paths)?
- can its stages be imported here (Ray is pulled in transitively, but never
  started)?
- can ``ffmpeg`` encode H.264 the way the transcoding stage demands?

Nothing here downloads code or models.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

UPSTREAM_REPO = "https://github.com/nvidia-cosmos/cosmos-curate"
UPSTREAM_LICENSE = "Apache-2.0"
SRC_ENV = "NPA_COSMOS_CURATE_SRC"

# Conventional locations a workbench image can bake an upstream checkout into.
# ``/opt/cosmos-curator`` is upstream's own in-container code directory
# (``CONTAINER_PATHS_CODE_DIR``), so a checkout there also satisfies the paths
# upstream resolves internally, such as its model registry.
DEFAULT_SRC_CANDIDATES = (
    "/opt/cosmos-curator",
    "/opt/cosmos-curate",
    "/opt/nvidia/cosmos-curate",
)

# Upstream's own switch for "not running as a cloud job", which redirects its
# scratch directories from the container's /config workspace to $TMPDIR. Setting
# it is what lets the stages run outside the curator container.
LOCAL_JOB_ENV = "COSMOS_CURATOR_LOCAL_DOCKER_JOB"

# The encoders upstream's ClipTranscodingStage accepts.
CPU_ENCODER = "libopenh264"
GPU_ENCODER = "h264_nvenc"

# Upstream declares requires-python >=3.12,<3.13 and its modules use 3.12 typing
# features, so an older interpreter fails deep in an import with a confusing
# message ("cannot import name 'Self' from 'typing'") rather than at the boundary.
MIN_PYTHON = (3, 12)

# The upstream commit :mod:`.pipeline` was written against. Upstream's stages are
# constructed with explicit keyword arguments, and it publishes no wheel and makes no
# stability promise about them, so a checkout from another commit can differ in ways
# that only surface as a TypeError from inside a stage constructor. The image records
# what it checked out (see REVISION_STAMP_FILE) and the probe compares the two, so a
# bumped Dockerfile pin fails at the boundary with something actionable.
PINNED_REVISION = "de89c5e4f76219c3f2985dc8fda790a74ab8bab0"

# Written by the workbench image after it checks out the pinned commit, because the
# build drops .git. A checkout without this file is somebody's own clone; that is a
# supported way to work, so it is reported as unknown rather than refused.
REVISION_STAMP_FILE = ".npa-upstream-revision"


class CosmosCurateError(RuntimeError):
    """Raised when a Cosmos Curator request cannot be satisfied."""


class CosmosCurateUnavailable(CosmosCurateError):
    """Raised when the upstream curator cannot run in this environment."""


@dataclass(frozen=True)
class CuratorAvailability:
    """What this environment can do with Cosmos Curator."""

    source: str = ""
    importable: bool = False
    import_error: str = ""
    ffmpeg: str = ""
    encoders: tuple[str, ...] = field(default_factory=tuple)
    pipeline_cli: str = ""
    python_version: str = ""
    revision: str = ""

    @property
    def revision_ok(self) -> bool:
        """False only when the checkout states a revision and it is the wrong one."""

        return not self.revision or self.revision == PINNED_REVISION

    @property
    def python_ok(self) -> bool:
        if not self.python_version:
            return True
        try:
            parts = tuple(int(part) for part in self.python_version.split(".")[:2])
        except ValueError:
            return True
        return parts >= MIN_PYTHON

    @property
    def encoder(self) -> str:
        """Preferred encoder for the transcoding stage, or ``""`` if none works."""

        if GPU_ENCODER in self.encoders and _has_gpu():
            return GPU_ENCODER
        if CPU_ENCODER in self.encoders:
            return CPU_ENCODER
        return ""

    @property
    def can_run_in_process(self) -> bool:
        return (
            bool(self.source)
            and self.python_ok
            and self.revision_ok
            and self.importable
            and bool(self.encoder)
        )

    def reason(self) -> str:
        """Human-readable explanation of why in-process curation cannot run."""

        if not self.source:
            return (
                f"no Cosmos Curator checkout found; set {SRC_ENV} to a clone of {UPSTREAM_REPO} "
                f"(or bake one at {DEFAULT_SRC_CANDIDATES[0]})"
            )
        if not self.python_ok:
            wanted = ".".join(str(part) for part in MIN_PYTHON)
            return (
                f"Cosmos Curator needs Python >= {wanted} (upstream declares "
                f"requires-python >=3.12,<3.13); this interpreter is {self.python_version}"
            )
        if not self.revision_ok:
            return (
                f"Cosmos Curator at {self.source} is checked out at {self.revision}, but this "
                f"code drives upstream's stages at {PINNED_REVISION}; upstream publishes no "
                "wheel and no stability promise for those constructors, so re-check "
                "cosmos_curate/pipeline.py against the new commit before bumping the pin"
            )
        if not self.importable:
            return f"Cosmos Curator at {self.source} is not importable: {self.import_error}"
        if not self.ffmpeg:
            return "ffmpeg is not on PATH"
        if not self.encoder:
            return (
                f"ffmpeg at {self.ffmpeg} supports neither {CPU_ENCODER} nor {GPU_ENCODER}; "
                "the curator's transcoding stage requires one of them"
            )
        return ""

    def to_dict(self) -> dict[str, object]:
        return {
            "upstream_repo": UPSTREAM_REPO,
            "upstream_license": UPSTREAM_LICENSE,
            "source": self.source,
            "importable": self.importable,
            "import_error": self.import_error,
            "ffmpeg": self.ffmpeg,
            "encoders": list(self.encoders),
            "encoder": self.encoder,
            "pipeline_cli": self.pipeline_cli,
            "python_version": self.python_version,
            "revision": self.revision,
            "pinned_revision": PINNED_REVISION,
            "can_run_in_process": self.can_run_in_process,
            "reason": self.reason(),
        }


def upstream_source_dir(*, environ: dict[str, str] | None = None) -> Path | None:
    """Return a Cosmos Curator checkout root, or ``None`` when there is none."""

    env = os.environ if environ is None else environ
    explicit = str(env.get(SRC_ENV, "") or "").strip()
    candidates = [explicit] if explicit else list(DEFAULT_SRC_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if (root / "cosmos_curator" / "pipelines").is_dir():
            return root
    return None


def upstream_revision(root: Path) -> str:
    """Return the commit the checkout records, or ``""`` when it records none."""

    try:
        return (root / REVISION_STAMP_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def ensure_upstream_importable(*, environ: dict[str, str] | None = None) -> Path:
    """Put the checkout on ``sys.path`` and enable upstream's local-job mode."""

    root = upstream_source_dir(environ=environ)
    if root is None:
        raise CosmosCurateUnavailable(CuratorAvailability().reason())
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    # Upstream writes scratch under /config when it believes it is a cloud job;
    # outside the curator container that path does not exist.
    os.environ.setdefault(LOCAL_JOB_ENV, "1")
    return root


def ffmpeg_encoders(*, ffmpeg: str = "") -> tuple[str, ...]:
    """Return the H.264 encoder names the local ffmpeg advertises."""

    exe = ffmpeg or shutil.which("ffmpeg") or ""
    if not exe:
        return ()
    try:
        proc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ()
    found = [name for name in (CPU_ENCODER, GPU_ENCODER) if name in (proc.stdout or "")]
    return tuple(found)


@functools.lru_cache(maxsize=1)
def _has_gpu() -> bool:
    """True only when a GPU is actually usable here, not merely tooled for.

    `nvidia-smi` on PATH is not enough: a CPU-tier pod on a GPU node has the binary
    and no device allocated to it. Selecting ``h264_nvenc`` there makes every clip
    encode fail with ``CUDA_ERROR_NO_DEVICE`` (observed live), which upstream logs
    per clip rather than raising, so the run reports success having dropped work.
    """

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "-L"], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "GPU 0" in (proc.stdout or "")


def probe_availability(*, environ: dict[str, str] | None = None) -> CuratorAvailability:
    """Report whether and how Cosmos Curator can run in this environment."""

    root = upstream_source_dir(environ=environ)
    ffmpeg = shutil.which("ffmpeg") or ""
    encoders = ffmpeg_encoders(ffmpeg=ffmpeg)
    pipeline_cli = shutil.which("video-pipeline") or ""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if root is None:
        return CuratorAvailability(
            ffmpeg=ffmpeg,
            encoders=encoders,
            pipeline_cli=pipeline_cli,
            python_version=python_version,
        )
    revision = upstream_revision(root)
    if sys.version_info[:2] < MIN_PYTHON:
        # Report the version gap instead of the confusing import error it causes.
        return CuratorAvailability(
            source=str(root),
            ffmpeg=ffmpeg,
            encoders=encoders,
            pipeline_cli=pipeline_cli,
            python_version=python_version,
            revision=revision,
        )
    if revision and revision != PINNED_REVISION:
        # Importing would succeed and the mismatch would only surface later, from
        # inside a stage constructor, as a TypeError with no hint of the cause.
        return CuratorAvailability(
            source=str(root),
            ffmpeg=ffmpeg,
            encoders=encoders,
            pipeline_cli=pipeline_cli,
            python_version=python_version,
            revision=revision,
        )

    importable = True
    import_error = ""
    try:
        ensure_upstream_importable(environ=environ)
        import importlib

        importlib.import_module("cosmos_curator.pipelines.video.clipping.clip_extraction_stages")
        importlib.import_module("cosmos_curator.pipelines.video.read_write.metadata_writer_stage")
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot run here"
        importable = False
        import_error = f"{type(exc).__name__}: {exc}"[:300]
    return CuratorAvailability(
        source=str(root),
        importable=importable,
        import_error=import_error,
        ffmpeg=ffmpeg,
        encoders=encoders,
        pipeline_cli=pipeline_cli,
        python_version=python_version,
        revision=revision,
    )
