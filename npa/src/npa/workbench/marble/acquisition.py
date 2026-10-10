"""Materialize generated or explicitly attributed example worlds into durable bundles."""

import hashlib
import json

import httpx
from botocore.exceptions import ClientError

from npa.clients.storage import LazyStorageClient, StoragePreconditionFailed
from npa.workbench.dataset.storage import read_bytes_uri, uri_join, write_bytes_uri
from npa.workbench.storage_scope import authorize_uri

from .api import MarbleError, api_request, await_world, require_api_key

SAMPLE_REVISION = "308fbf8d0d9a336112c57697a0b0499d16b31504"
SAMPLE_REPO = "https://github.com/bmild/spark-physics"
SAMPLE_BASE = f"https://raw.githubusercontent.com/bmild/spark-physics/{SAMPLE_REVISION}"


def _download(url):
    try:
        response = httpx.get(url, timeout=180, follow_redirects=True)
        response.raise_for_status()
        return response.content
    except httpx.HTTPError:
        raise MarbleError(
            "World asset download failed; provider URLs were not logged"
        ) from None


def _sample():
    return {
        "display_name": "Hobbit interior — upstream Marble example",
        "source_kind": "upstream-example",
        "generated_this_run": False,
        "source": SAMPLE_REPO,
        "source_revision": SAMPLE_REVISION,
        "license": "MIT; upstream notice included in bundle",
        "units": "upstream scene units; not calibrated meters",
        "splat_transform": {"scale": [3, -3, 3], "translation": [0, 0, 0]},
        "mesh_transform": {"scale": [-1, -1, 1], "translation": [0, 0, 0]},
        "assets": {
            "world.spz": f"{SAMPLE_BASE}/public/hobbit_stitched.spz",
            "collider.glb": f"{SAMPLE_BASE}/public/hobbit_stitched.glb",
            "license.txt": f"{SAMPLE_BASE}/LICENSE",
        },
    }


def _existing_operation(uri, fingerprint, run_id):
    try:
        journal = json.loads(read_bytes_uri(uri))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
            raise
        return None
    if journal.get("request_sha256") != fingerprint or journal.get("run_id") != run_id:
        raise MarbleError(
            "Existing generation belongs to a different request or run; use a new output prefix"
        )
    if not journal.get("operation_id"):
        raise MarbleError(
            "Prior generation acceptance is uncertain; reconcile it in World Labs before retrying"
        )
    return journal["operation_id"]


def _persist_journal(uri, journal, etag=""):
    authorize_uri(uri, operation="write")
    try:
        return LazyStorageClient().put_bytes_conditional(
            json.dumps(journal).encode(),
            uri,
            if_none_match=not etag,
            if_match=etag,
            content_type="application/json",
        )
    except StoragePreconditionFailed:
        raise MarbleError(
            "Generation journal changed concurrently; inspect the saved operation before retrying"
        ) from None


def _generation_operation(request):
    require_api_key()
    uri = uri_join(request.output_path, "operation.json")
    fingerprint = hashlib.sha256(
        f"{request.model}\n{request.prompt}".encode()
    ).hexdigest()
    existing = _existing_operation(uri, fingerprint, request.run_id)
    if existing:
        return existing
    journal = {
        "request_sha256": fingerprint,
        "run_id": request.run_id,
        "state": "request-intent",
    }
    etag = _persist_journal(uri, journal)
    operation = api_request(
        "POST",
        "worlds:generate",
        {
            "display_name": request.run_id,
            "model": request.model,
            "world_prompt": {"type": "text", "text_prompt": request.prompt},
        },
    )
    journal.update(operation_id=operation["operation_id"], state="accepted")
    _persist_journal(uri, journal, etag)
    return operation["operation_id"]


def _generated(request):
    world = await_world(_generation_operation(request))
    splats = world["assets"]["splats"]
    semantics = splats.get("semantics_metadata")
    if not semantics:
        raise MarbleError(
            "World API omitted splat scale metadata; refusing guessed metric geometry"
        )
    scale = semantics["metric_scale_factor"]
    offset = semantics["ground_plane_offset"]
    return {
        "display_name": world.get("display_name") or request.run_id,
        "source_kind": "world-api",
        "generated_this_run": True,
        "source": "https://docs.worldlabs.ai/api",
        "model": request.model,
        "world_id": world["id"],
        "prompt": request.prompt,
        "units": "provider-estimated meters; not surveyed site calibration",
        "splat_transform": {
            "scale": [scale, -scale, -scale],
            "translation": [0, offset, 0],
        },
        "mesh_transform": {
            "scale": [scale, -scale, -scale],
            "translation": [0, offset, 0],
        },
        "assets": {
            "world.spz": splats["spz_urls"]["500k"],
            "collider.glb": world["assets"]["mesh"]["collider_mesh_url"],
        },
    }


def acquire_bundle(request):
    """Acquire real source assets and publish their byte hashes and attribution.

    Args: An AcquireRequest with a validated S3 destination.
    Returns: The durable world manifest.
    Raises: MarbleError on provider or byte verification failure.
    """
    world = _sample() if request.source == "sample-hobbit" else _generated(request)
    assets = world.pop("assets")
    records = {}
    for filename, url in assets.items():
        payload = _download(url)
        uri = uri_join(request.output_path, filename)
        write_bytes_uri(uri, payload)
        digest = hashlib.sha256(payload).hexdigest()
        if hashlib.sha256(read_bytes_uri(uri)).hexdigest() != digest:
            raise MarbleError("World asset read-after-write hash mismatch")
        records[filename] = {"sha256": digest, "bytes": len(payload)}
    world.update(
        schema_version="npa.marble.world.v1", run_id=request.run_id, files=records
    )
    write_bytes_uri(
        uri_join(request.output_path, "world.json"), json.dumps(world).encode()
    )
    return world
