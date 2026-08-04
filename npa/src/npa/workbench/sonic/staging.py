"""Object-storage staging for the SONIC export tool.

Why this exists
---------------
``npa workbench sonic export`` loads its checkpoint with ``Path(checkpoint).exists()``
and writes the ONNX to a local path. That is fine for a laptop, but the raw SkyPilot
template ``sonic-export.yaml`` had to carry ~60 lines of inline bash + boto3 to download
``SONIC_CHECKPOINT`` from S3 and upload the results back afterwards.

An ``npa.workflow`` spec has no such escape hatch: a ``toolRef`` argv template passes
``{{config.checkpoint_uri}}`` straight through. So the twin looked equivalent, planned
and rendered cleanly, and then failed live with

    Error: checkpoint not found: s3://<bucket>/.../sonic-export/checkpoint.pt

(run ``npa-wf-gpu-sonic-export-45b108b8``, SkyPilot job 184). Teaching the tool to speak
``s3://`` — which its sibling ``sonic eval`` already does through ``StorageClient`` —
makes the spec genuinely equivalent to the template it replaces and deletes the bash.

The staging concern lives in this module (not inline in ``export_onnx``) so it is unit
testable with an injected storage client and no infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from npa.clients.storage import StorageClient

DEFAULT_ONNX_NAME = "sonic_policy.onnx"


def is_object_uri(value: Any) -> bool:
    """True when a value is an ``s3://`` URI (and therefore needs staging)."""

    return isinstance(value, str) and value.strip().startswith("s3://")


def resolve_object_onnx_uri(output: str) -> str:
    """Return the ONNX object URI for an ``s3://`` output prefix or object.

    Mirrors the local behaviour: an explicit ``*.onnx`` key is used as-is, anything
    else is treated as a prefix that receives ``sonic_policy.onnx``.
    """

    text = output.strip()
    if text.lower().endswith(".onnx"):
        return text
    return text.rstrip("/") + "/" + DEFAULT_ONNX_NAME


@dataclass
class ExportStaging:
    """Local paths for one export, plus the object URIs to publish back to."""

    workdir: Path
    #: Local path passed to the exporter for each staged input.
    inputs: dict[str, str] = field(default_factory=dict)
    #: Local ONNX path the exporter writes to.
    local_output: str = ""
    #: Object URI the ONNX is uploaded to (empty when the output is local).
    onnx_uri: str = ""

    @property
    def stages_output(self) -> bool:
        return bool(self.onnx_uri)


def _client(storage_client: "StorageClient | None") -> Any:
    if storage_client is not None:
        return storage_client
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def stage_inputs(
    values: dict[str, Any],
    *,
    workdir: Path,
    storage_client: "StorageClient | None" = None,
) -> dict[str, str]:
    """Download every ``s3://`` value in ``values`` into ``workdir``.

    ``values`` maps a logical name (``checkpoint``, ``obs_spec``, ...) to the argument
    the caller was given. Non-object values are returned unchanged, so a local run is
    untouched and no storage client is constructed.
    """

    staged: dict[str, str] = {}
    client: Any | None = None
    workdir = Path(workdir)
    for name, value in values.items():
        if not is_object_uri(value):
            continue
        if client is None:
            client = _client(storage_client)
        suffix = Path(str(value).split("?", 1)[0]).suffix or ""
        local = workdir / f"{name}{suffix}"
        local.parent.mkdir(parents=True, exist_ok=True)
        client.download_path(str(value), str(local))
        staged[name] = str(local)
    return staged


def publish_outputs(
    staging: ExportStaging,
    *,
    storage_client: "StorageClient | None" = None,
) -> dict[str, str]:
    """Upload every file the exporter produced next to the ONNX.

    Returns a map of local path -> object URI. The sidecar metadata (and anything else
    the exporter drops in the workdir, e.g. a parity report) is published alongside, so
    ``sonic eval`` can consume the pair straight from the run prefix.
    """

    if not staging.stages_output:
        return {}
    client = _client(storage_client)
    onnx_local = Path(staging.local_output)
    prefix = staging.onnx_uri.rsplit("/", 1)[0] + "/"
    published: dict[str, str] = {}
    for path in sorted(onnx_local.parent.iterdir()):
        if not path.is_file():
            continue
        uri = staging.onnx_uri if path == onnx_local else prefix + path.name
        published[str(path)] = client.upload_file(str(path), uri)
    return published


def sidecar_uri_candidates(onnx_uri: str) -> tuple[str, ...]:
    """Return the sidecar metadata URIs to try for an ONNX object URI.

    Mirrors the local resolution order in ``eval._resolve_metadata_path``: the
    exporter writes ``<stem>.metadata.json`` (``with_suffix``), and the appended
    ``<name>.metadata.json`` form is accepted too.
    """

    text = onnx_uri.strip()
    stem = text[: -len(".onnx")] if text.lower().endswith(".onnx") else text
    return (f"{stem}.metadata.json", f"{text}.metadata.json")


def external_data_uri_candidates(onnx_uri: str) -> tuple[str, ...]:
    """Return sibling URIs that may hold the ONNX's external weights.

    ``torch.onnx.export`` writes tensors larger than the protobuf limit to a separate
    ``<name>.onnx.data`` file, and onnxruntime resolves it *relative to the model
    file*. Staging the ``.onnx`` alone therefore produces a model that loads and then
    fails on a missing initializer — the live ``sonic-export`` run published
    ``sonic_policy.onnx`` (1.6 KB) plus ``sonic_policy.onnx.data`` (11.5 KB).
    """

    return (onnx_uri.strip() + ".data",)


def stage_eval_inputs(
    *,
    onnx: str,
    metadata: str | None,
    workdir: Path,
    storage_client: "StorageClient | None" = None,
) -> tuple[str, str | None]:
    """Download an ``s3://`` ONNX policy and its sidecar metadata.

    Returns ``(local_onnx, local_metadata_or_None)``. Non-object inputs pass through
    untouched, so a local evaluation never constructs a storage client.

    ``sonic eval`` could already *write* its result to S3 but only ever *read* local
    files, which is why the ``sonic-eval`` / ``sonic-export-eval`` twins could not
    consume the ONNX their own run prefix holds.
    """

    if not is_object_uri(onnx):
        return onnx, metadata

    client = _client(storage_client)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    local_onnx = workdir / (onnx.rstrip("/").rsplit("/", 1)[-1] or DEFAULT_ONNX_NAME)
    client.download_path(onnx, str(local_onnx))

    # External weights must land NEXT TO the model file, under the exact name the
    # model references. Absent for small graphs, required for large ones.
    for candidate in external_data_uri_candidates(onnx):
        target = local_onnx.parent / candidate.rsplit("/", 1)[-1]
        try:
            client.download_path(candidate, str(target))
        except Exception:  # noqa: BLE001 - no external data is the common case
            continue

    if metadata and is_object_uri(metadata):
        # An explicit sidecar URI must exist; failing loudly beats a confusing
        # "metadata not found" for a path the caller never asked for.
        local_metadata = workdir / Path(metadata.rsplit("/", 1)[-1]).name
        client.download_path(metadata, str(local_metadata))
        return str(local_onnx), str(local_metadata)
    if metadata:
        return str(local_onnx), metadata

    # Land the sidecar under the name the local resolver looks for first.
    expected = local_onnx.with_suffix(".metadata.json")
    for candidate in sidecar_uri_candidates(onnx):
        try:
            client.download_path(candidate, str(expected))
        except Exception:  # noqa: BLE001 - any miss just means "try the next name"
            continue
        return str(local_onnx), str(expected)
    return str(local_onnx), None


def plan_export_staging(
    *,
    workdir: Path,
    output: str,
    inputs: dict[str, Any],
    storage_client: "StorageClient | None" = None,
) -> ExportStaging:
    """Stage inputs and decide where the exporter should write."""

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    staged_inputs = stage_inputs(inputs, workdir=workdir, storage_client=storage_client)
    if is_object_uri(output):
        onnx_uri = resolve_object_onnx_uri(output)
        local_output = str(workdir / "export" / onnx_uri.rsplit("/", 1)[-1])
        Path(local_output).parent.mkdir(parents=True, exist_ok=True)
        return ExportStaging(
            workdir=workdir,
            inputs=staged_inputs,
            local_output=local_output,
            onnx_uri=onnx_uri,
        )
    return ExportStaging(workdir=workdir, inputs=staged_inputs, local_output=output)


__all__ = [
    "DEFAULT_ONNX_NAME",
    "ExportStaging",
    "external_data_uri_candidates",
    "is_object_uri",
    "plan_export_staging",
    "publish_outputs",
    "resolve_object_onnx_uri",
    "sidecar_uri_candidates",
    "stage_eval_inputs",
    "stage_inputs",
]
