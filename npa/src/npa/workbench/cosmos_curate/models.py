"""Fetch the model weights Cosmos Curator's GPU stages need, at run time.

The curator's GPU stages load weights for TransNetV2 shot detection, InternVideo2
or Cosmos-Embed1 embeddings, CLIP + an aesthetic head for filtering, and a Qwen
VLM for captioning. Those are third-party and NVIDIA models under their own
licenses, so the workbench image must not carry them: it ships upstream's code and
this module downloads weights on demand with the operator's own Hugging Face
token.

Everything about *which* weights and *which revision* comes from upstream's own
registry (``cosmos_curator/configs/all_models.json``), and the download itself is
upstream's ``download_model_weights_from_huggingface_to_workspace``, so a model or
a pinned revision changes when the checkout changes — never because NPA hardcoded
one. Weights land in upstream's own cache layout under :func:`weights_dir`, which
is where its stages look for them.

NGC: the curator's models are all Hugging Face repos, so ``HF_TOKEN`` is the
credential this needs. ``NGC_API_KEY`` is what pulls NVIDIA *containers* (and is
what upstream's own NVCF deployment path uses); it is reported here so an operator
can see both, but it is not used to fetch these weights.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from npa.workbench.cosmos_curate.upstream import (
    UPSTREAM_LICENSE,
    UPSTREAM_REPO,
    CosmosCurateError,
    ensure_upstream_importable,
    upstream_source_dir,
)

_log = logging.getLogger(__name__)

# Upstream's registry of every model it knows, with a pinned revision each.
REGISTRY_RELATIVE_PATH = Path("cosmos_curator") / "configs" / "all_models.json"

# Where upstream's stages look for weights inside its container workspace.
DEFAULT_WEIGHTS_DIR = "/config/models"
WEIGHTS_DIR_ENV = "NPA_COSMOS_CURATE_WEIGHTS_DIR"
# Written into a model directory once its download returns. Its absence is what
# distinguishes an interrupted fetch from a finished one, and the revision it
# records is what makes a pinned request re-fetch weights from another commit.
COMPLETION_STAMP = ".npa-fetch-complete.json"

HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")

# Named sets of upstream model keys, one per curator capability. The membership of
# each set mirrors the ``model_id_names`` property of the upstream model class that
# the stage instantiates, so a stage's weights are requested as a unit.
MODEL_SETS: dict[str, tuple[str, ...]] = {
    # TransNetV2 shot-boundary splitting (--splitting-algorithm transnetv2).
    "split-transnetv2": ("transnetv2",),
    # InternVideo2 clip embeddings (--embedding-algorithm internvideo2); its model
    # class loads a BERT text tower alongside the video tower.
    "embed-internvideo2": ("internvideo2_mm", "bert"),
    # Cosmos-Embed1 clip embeddings (--embedding-algorithm cosmos-embed1-336p).
    "embed-cosmos-embed1": ("cosmos_embed1_336p",),
    # Motion/aesthetic filtering (--aesthetic-filter): a CLIP tower plus the
    # linear aesthetic head scored on its embeddings.
    "filter-aesthetic": ("clip_vit", "aesthetic_scorer"),
    # VLM captioning (--captioning-algorithm qwen).
    "caption-qwen": ("qwen2.5_vl",),
    # T5 text embeddings for the Cosmos-Predict2 dataset export.
    "dataset-t5": ("t5_xxl",),
}

# What upstream's `video-pipeline split` needs with its default flags.
DEFAULT_SET = "split-annotate"
MODEL_SETS[DEFAULT_SET] = (
    MODEL_SETS["split-transnetv2"]
    + MODEL_SETS["embed-internvideo2"]
    + MODEL_SETS["caption-qwen"]
)

# The stages NPA's in-process curation path runs need no weights at all: clipping,
# transcoding, motion scoring, and metadata writing are ffmpeg and OpenCV work.
CPU_STAGES_NEED_NO_WEIGHTS = True


@dataclass(frozen=True)
class ModelSpec:
    """One model to fetch, as upstream's registry describes it."""

    key: str
    model_id: str
    revision: str = ""
    files: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelStatus:
    """Whether a model's weights are already present locally."""

    key: str
    model_id: str
    revision: str
    local_dir: str
    present: bool
    file_count: int = 0
    bytes: int = 0
    #: Why the cache was not accepted, when ``present`` is false but files exist.
    stale_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a fetch request."""

    status: str
    weights_dir: str
    requested: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    models: list[ModelStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["models"] = [model.to_dict() for model in self.models]
        payload["upstream"] = {"repo": UPSTREAM_REPO, "license": UPSTREAM_LICENSE}
        return payload


def weights_dir(*, environ: dict[str, str] | None = None) -> Path:
    """Directory upstream's stages read weights from."""

    env = os.environ if environ is None else environ
    return Path(str(env.get(WEIGHTS_DIR_ENV, "") or DEFAULT_WEIGHTS_DIR)).expanduser()


def hf_token(*, environ: dict[str, str] | None = None) -> str:
    """The Hugging Face token to download with, from any of its usual names."""

    env = os.environ if environ is None else environ
    for name in HF_TOKEN_ENVS:
        value = str(env.get(name, "") or "").strip()
        if value:
            return value
    return ""


def load_registry(*, environ: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Read upstream's model registry from the checkout."""

    root = upstream_source_dir(environ=environ)
    if root is None:
        raise CosmosCurateError(
            f"no Cosmos Curator checkout found; the model registry lives in it ({UPSTREAM_REPO})"
        )
    path = root / REGISTRY_RELATIVE_PATH
    if not path.is_file():
        raise CosmosCurateError(f"upstream model registry not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CosmosCurateError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CosmosCurateError(f"{path} is not a model registry object")
    return payload


def resolve_models(
    names: Sequence[str],
    *,
    environ: dict[str, str] | None = None,
) -> list[ModelSpec]:
    """Expand set names and/or model keys into deduplicated specs.

    ``names`` accepts a :data:`MODEL_SETS` name (``caption-qwen``) or a raw
    upstream model key (``transnetv2``); an empty list means :data:`DEFAULT_SET`.
    """

    registry = load_registry(environ=environ)
    wanted: list[str] = []
    for name in names or [DEFAULT_SET]:
        key = str(name).strip()
        if not key:
            continue
        if key in MODEL_SETS:
            wanted.extend(MODEL_SETS[key])
        elif key in registry:
            wanted.append(key)
        else:
            raise CosmosCurateError(
                f"unknown model or set {key!r}; sets: {', '.join(sorted(MODEL_SETS))}; "
                f"models: {', '.join(sorted(registry))}"
            )

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for key in wanted:
        if key in seen:
            continue
        seen.add(key)
        entry = registry.get(key) or {}
        model_id = str(entry.get("model_id") or "")
        if not model_id:
            raise CosmosCurateError(f"upstream registry entry {key!r} has no model_id")
        files = entry.get("filelist")
        specs.append(
            ModelSpec(
                key=key,
                model_id=model_id,
                revision=str(entry.get("version") or ""),
                files=tuple(str(name) for name in files) if isinstance(files, list) else (),
            )
        )
    return specs


def read_completion_stamp(local_dir: Path) -> dict[str, Any]:
    """Return the stamp a finished fetch left behind, or ``{}`` when there is none."""

    try:
        loaded = json.loads((local_dir / COMPLETION_STAMP).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_completion_stamp(local_dir: Path, spec: ModelSpec) -> None:
    """Record that ``spec`` downloaded completely, and at which revision."""

    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / COMPLETION_STAMP).write_text(
        json.dumps(
            {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "files": list(spec.files),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _cache_state(spec: ModelSpec, local: Path, names: set[str]) -> tuple[bool, str]:
    """Decide whether a local directory is a complete, current copy of ``spec``.

    Presence used to mean "some file is here", which silently accepts an interrupted
    download — the next run skips it and the missing shard surfaces as a load error —
    and never looks at the revision, so weights from a different commit satisfy a
    pinned request. A stamp written only after a fetch returns answers both.
    """

    stamp = read_completion_stamp(local)
    if stamp:
        stamped_revision = str(stamp.get("revision") or "")
        if spec.revision and stamped_revision != spec.revision:
            return False, (
                f"cached at revision {stamped_revision or 'unknown'}, wanted {spec.revision}"
            )
        if spec.files and not set(spec.files) <= names:
            return False, "stamped complete but files are missing from the directory"
        return True, ""
    if spec.files and set(spec.files) <= names:
        # Downloaded before stamps existed, and the registry's own file list vouches
        # for it. The revision cannot be verified, so the fetch re-stamps it.
        return True, ""
    if names:
        return False, "no completion stamp; treating the partial directory as incomplete"
    return False, ""


def model_status(
    specs: Iterable[ModelSpec],
    *,
    environ: dict[str, str] | None = None,
) -> list[ModelStatus]:
    """Report which of ``specs`` already have complete, current weights on disk."""

    root = weights_dir(environ=environ)
    out: list[ModelStatus] = []
    for spec in specs:
        local = root / spec.model_id
        files = [
            path
            for path in (local.rglob("*") if local.is_dir() else [])
            if path.is_file() and path.name != COMPLETION_STAMP
        ]
        names = {path.name for path in files}
        present, stale_reason = _cache_state(spec, local, names)
        out.append(
            ModelStatus(
                key=spec.key,
                model_id=spec.model_id,
                revision=spec.revision,
                local_dir=str(local),
                present=present,
                file_count=len(files),
                bytes=sum(path.stat().st_size for path in files),
                stale_reason=stale_reason,
            )
        )
    return out


def fetch_models(
    names: Sequence[str] = (),
    *,
    force: bool = False,
    environ: dict[str, str] | None = None,
) -> FetchResult:
    """Download the requested weights with upstream's own downloader.

    Skips anything already complete unless ``force``. Requires a Hugging Face
    token in the environment: these are third-party and NVIDIA models under their
    own licenses, fetched with the operator's credentials rather than shipped.
    """

    env = os.environ if environ is None else environ
    specs = resolve_models(names, environ=env)
    root = weights_dir(environ=env)
    token = hf_token(environ=env)
    if not token:
        raise CosmosCurateError(
            "no Hugging Face token found; set one of "
            f"{', '.join(HF_TOKEN_ENVS)} so weights can be downloaded with your own credentials "
            "(the image deliberately ships no model weights)"
        )

    ensure_upstream_importable(environ=env)
    # Upstream resolves its weights cache from its own environment module, so point
    # that module at the configured directory instead of patching call sites.
    _point_upstream_cache_at(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        from cosmos_curator.core.utils.model.model_utils import (  # type: ignore
            download_model_weights_from_huggingface_to_workspace,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable error
        raise CosmosCurateError(f"upstream model downloader is not importable: {exc}") from exc

    before = {status.key: status for status in model_status(specs, environ=env)}
    fetched: list[str] = []
    already: list[str] = []
    failed: dict[str, str] = {}
    # huggingface_hub reads the token from the environment under either name, but the
    # caller's process outlives this fetch, so both are scoped to it.
    with _hf_token_env(token), _upstream_hf_config(token):
        for spec in specs:
            if not force and before[spec.key].present:
                already.append(spec.key)
                continue
            try:
                download_model_weights_from_huggingface_to_workspace(
                    spec.model_id,
                    spec.revision or None,
                    list(spec.files) or None,
                )
            except Exception as exc:  # noqa: BLE001 - keep fetching the rest
                message = f"{type(exc).__name__}: {exc}"[:300]
                _log.warning(
                    "could not fetch %s (%s): %s", spec.key, spec.model_id, message, exc_info=True
                )
                failed[spec.key] = message
                continue
            # Only now, after the download returned, is the directory known complete.
            write_completion_stamp(root / spec.model_id, spec)
            fetched.append(spec.key)

    return FetchResult(
        status="completed" if not failed else "partial",
        weights_dir=str(root),
        requested=[spec.key for spec in specs],
        fetched=fetched,
        already_present=already,
        failed=failed,
        models=model_status(specs, environ=env),
    )


@contextmanager
def _hf_token_env(token: str) -> Iterator[None]:
    """Expose the token to ``huggingface_hub`` for the duration of the fetch only.

    Leaving it in ``os.environ`` would hand the operator's credential to every later
    stage and child process in the run, which no other part of the fetch requires.
    """

    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = token
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _upstream_hf_config(token: str) -> Iterator[None]:
    """Give upstream's downloader the config file it reads its HF token from.

    Upstream does not read ``HF_TOKEN``: it loads ``huggingface.api_key`` from a
    config file at its in-container path, which its own launcher mounts from
    ``~/.config/cosmos_curator/config.yaml``. Rather than ask operators to maintain
    a second credential file, this writes a minimal one into a private temporary
    directory for the duration of the call and points upstream's config loader at
    it, so the token stays out of any persistent location and out of the image.
    """

    try:
        from cosmos_curator.core.utils.config import config as upstream_config  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise CosmosCurateError(f"upstream config module is not importable: {exc}") from exc

    attribute = "CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE"
    previous = getattr(upstream_config, attribute, None)
    with tempfile.TemporaryDirectory(prefix="npa-cosmos-curate-cfg-") as tmp:
        path = Path(tmp) / "cosmos_curator.yaml"
        path.write_text(f'huggingface:\n  api_key: "{token}"\n', encoding="utf-8")
        path.chmod(0o600)
        setattr(upstream_config, attribute, path)
        try:
            yield
        finally:
            if previous is not None:
                setattr(upstream_config, attribute, previous)


def _point_upstream_cache_at(root: Path) -> None:
    """Make upstream's weights-cache constant resolve to ``root``.

    Upstream hardcodes ``/config/models`` for its in-container cache. Rebinding the
    constant keeps both the downloader and the stages that later load the weights
    reading the same directory, which is what an operator controls via
    ``NPA_COSMOS_CURATE_WEIGHTS_DIR``.
    """

    try:
        from cosmos_curator.core.utils import environment as upstream_env  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise CosmosCurateError(f"upstream environment module is not importable: {exc}") from exc
    if Path(upstream_env.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR) != root:
        upstream_env.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR = root


def describe_models(*, environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Describe the model sets, their upstream pins, and what is present locally."""

    env = os.environ if environ is None else environ
    payload: dict[str, Any] = {
        "weights_dir": str(weights_dir(environ=env)),
        "hf_token_present": bool(hf_token(environ=env)),
        "ngc_key_present": bool(str(env.get("NGC_API_KEY", "") or "").strip()),
        "cpu_stages_need_no_weights": CPU_STAGES_NEED_NO_WEIGHTS,
        "upstream": {"repo": UPSTREAM_REPO, "license": UPSTREAM_LICENSE},
        "sets": {},
    }
    try:
        registry = load_registry(environ=env)
    except CosmosCurateError as exc:
        payload["error"] = str(exc)
        return payload
    payload["registry_size"] = len(registry)
    for name, keys in sorted(MODEL_SETS.items()):
        specs = resolve_models([name], environ=env)
        payload["sets"][name] = {
            "keys": list(keys),
            "models": [status.to_dict() for status in model_status(specs, environ=env)],
        }
    return payload
