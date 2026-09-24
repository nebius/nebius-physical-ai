"""Living-lab parameterized neural-reconstruction digital twin (fan-out + join).

Pure, framework-free logic for the ``living-lab-nurec-fanout.yaml`` workflow: a
multi-zone research-space digital twin built from real NVIDIA NuRec / NRE
neural reconstructions, one per independent RTX PRO 6000 shard, joined into a
composite twin with a contact-sheet panorama.

Zone model
----------
Topology size is derived from explicit capture and sector inputs, not a magic
fixed count. The shipped blueprint defaults to the operator's reserved
16-device capacity: 16 deterministic zones = the 8 real PPISP NCore sequences
(4 scenes x 2 variants) x 2 view sectors (novel-view azimuth offsets). The same
generator also supports a 24-zone topology (8 capture pairs x 3 distinct
sectors). Each zone is a fully independent reconstruction shard that runs the
complete real NRE pipeline (``npa workbench nurec
check|fetch|reconstruct|render|visualize``) on its own RTX PRO 6000, then
publishes ``zone_manifest.json`` with objective GPU participation evidence (GPU
name, timing, val PSNR/SSIM, USDZ presence).

Each view sector over the same sequence carries a genuinely distinct rig offset:
the reconstruction itself is deterministic, while the sectors render distinct
novel-view sweeps (different azimuth offsets) so each shard contributes a
distinct zone to the twin.

Objectivity contract
--------------------
The join (`join_living_lab_zones`) fails loudly unless every expected zone
manifest is present and each records a real GPU identity and a real USDZ. It
never invents success from a submitted job: it reads the per-zone published
evidence from S3, so the composite report is only as good as the real per-shard
artifacts. The expected device count is derived from the generated zone list and
cross-checked against the fan-out's explicit parallel member count (the workflow
``parallelCount`` contract), so an operator cannot weaken the proof by
understating the device count.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

ZONE_MANIFEST_FILENAME = "zone_manifest.json"
DIGITAL_TWIN_SCHEMA = "npa.living_lab.digital_twin.v1"
PANORAMA_FILENAME = "panorama.png"
PANORAMA_JSON_FILENAME = "panorama.json"

#: The 8 real, ungated (CC-BY-4.0) NCore V4 sequences from
#: ``nvidia/PhysicalAI-NuRec-PPISP``. Each is a genuine object-centric capture.
SCENES = (
    ("huerstholz", "auto"),
    ("huerstholz", "standard"),
    ("struktur28", "auto"),
    ("struktur28", "standard"),
    ("toro", "auto"),
    ("toro", "standard"),
    ("valiant", "auto"),
    ("valiant", "standard"),
)

#: Novel-view rig offsets (rig-rotation-offset yaw,-roll,-pitch deg; then
#: rig-translation-offset tx,ty,tz m) for each supported view sector. Each
#: sector carries a genuinely distinct requested rig offset, so two shards of the
#: same capture always render distinct novel-view sweeps. The shipped default
#: uses sectors ``a`` and ``b`` (16 zones); sector ``c`` exists so the same
#: generator can also emit a 24-zone topology (8 capture pairs x 3 sectors).
VIEW_SECTORS = {
    "a": ("0,0,0", "0,0.25,0"),
    "b": ("120,0,0", "0,0,0.45"),
    "c": ("240,0,0", "0,0,-0.45"),
}

#: Default capture set: the 8 real NCore V4 pairs drive the shipped topology.
DEFAULT_CAPTURES: tuple[tuple[str, str], ...] = SCENES
#: Default view-sector set: two sectors -> the shipped 8 x 2 = 16-zone topology.
DEFAULT_SECTORS: tuple[str, ...] = ("a", "b")
#: Default topology's expected device count (8 captures x 2 sectors = 16 zones).
DEFAULT_DEVICE_COUNT = 16

#: Per-zone NNRE run id is deterministic from the zone name so re-runs are
#: reproducible and the join can re-derive URIs without a registry.


def living_lab_zones(
    *,
    captures: Sequence[tuple[str, str]] = DEFAULT_CAPTURES,
    sectors: Sequence[str] = DEFAULT_SECTORS,
) -> list[dict[str, Any]]:
    """Deterministic zone definitions for the living-lab digital twin.

    Topology size is derived from the explicit ``captures`` x ``sectors`` inputs:
    the default 8 capture pairs x 2 sectors yields 16 zones, while 8 capture
    pairs x 3 sectors yields 24 zones with three genuinely distinct sector rig
    offsets per capture. Rejects an empty or unrepresentable topology.
    """
    if not captures or not sectors:
        raise ValueError(
            "living-lab topology requires at least one capture pair and one view sector"
        )
    captures_list = list(captures)
    zones: list[dict[str, Any]] = []
    for scene, variant in captures_list:
        for sector in sectors:
            try:
                rot, trans = VIEW_SECTORS[sector]
            except KeyError as exc:  # noqa: BLE001
                raise ValueError(
                    f"unknown living-lab view sector {sector!r}; "
                    f"supported sectors: {sorted(VIEW_SECTORS)}"
                ) from exc
            zones.append(
                {
                    "sequence_index": captures_list.index((scene, variant)),
                    "scene": scene,
                    "variant": variant,
                    "view_sector": sector,
                    "zone_name": f"{scene}-{variant}-{sector}",
                    "rig_rotation_offset": str(rot),
                    "rig_translation_offset": str(trans),
                }
            )
    return zones


def zone_uris(*, run_uri: str, zone_name: str) -> dict[str, str]:
    """Per-zone S3 locations under a run prefix."""
    base = f"{run_uri.rstrip('/')}/zones/{zone_name}"
    return {
        "zone_uri": f"{base}/",
        "ncore_uri": f"{base}/ncore/",
        "reconstruction_uri": f"{base}/reconstruction/",
        "novel_views_uri": f"{base}/novel_views/",
        "rrd_uri": f"{base}/reports/sim2real.rrd",
        "manifest_uri": f"{base}/{ZONE_MANIFEST_FILENAME}",
    }


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def zone_names(
    shards: str | Sequence[str] = "",
    *,
    topologies: Sequence[str] | None = None,
) -> list[str]:
    """Explicit zone list, or discover canonical zone names for a topology.

    ``topologies`` is a list of zone names (already-expanded topology) to use as
    a default when ``shards`` is empty; it lets callers pin the expected zone set
    without re-running the generator on a differently-sized topology.
    """
    if isinstance(shards, str):
        names = [p.strip() for p in shards.split(",") if p.strip()]
    else:
        names = [str(v).strip() for v in (shards or ())]
    if names:
        return names
    if topologies:
        return [str(v).strip() for v in topologies if str(v).strip()]
    return [z["zone_name"] for z in living_lab_zones()]


def _download_json(uri: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-living-lab-") as tmp:
        out = Path(_storage().download_path(uri, tmp))
        if out.is_dir():
            want = Path(urlparse(uri).path).name or ZONE_MANIFEST_FILENAME
            matches = sorted(out.rglob(want))
            if not matches:
                raise FileNotFoundError(uri)
            out = matches[0]
        return json.loads(out.read_text(encoding="utf-8"))


def _upload_file(local: Path, uri: str) -> str:
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment().upload_file(str(local), uri)


def _upload_bytes(payload: bytes, uri: str) -> str:
    with tempfile.TemporaryDirectory(prefix="npa-living-lab-") as tmp:
        path = Path(tmp) / "out"
        path.write_bytes(payload)
        return _upload_file(path, uri)


def _download_path(uri: str, tmp: str) -> Path:
    out = Path(_storage().download_path(uri, tmp))
    if out.is_dir():
        matches = sorted(
            (p for p in out.rglob("*") if p.suffix in (".png", ".jpg", ".jpeg"))
        )
        if not matches:
            raise FileNotFoundError(f"no image under {uri}")
        out = matches[0]
    return out


def build_panorama(
    *,
    zone_uris_map: dict[str, str],
    output_uri: str,
    cols: int = 4,
) -> dict[str, Any]:
    """Stitch one representative render per zone into a contact-sheet panorama.

    Real, human-viewable artifact: downloads each zone's first novel-view frame
    and lays the 16 frames onto a grid with PIL. Fails if any zone has no image,
    so the panorama is only as complete as the real per-zone renders.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - cpu pod provides pillow
        raise RuntimeError("Pillow is required to build the panorama") from exc

    frame = 320
    thumb = (frame, frame)
    cells: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="npa-living-lab-") as tmp:
        for zone, uri in sorted(zone_uris_map.items()):
            img_path = _download_path(uri, tmp)
            cells.append(img_path)
        if not cells:
            raise RuntimeError("no zone renders to assemble into a panorama")
        rows = (len(cells) + cols - 1) // cols
        canvas = Image.new("RGB", (frame * cols, frame * rows), (18, 18, 26))
        for idx, cell in enumerate(cells):
            r, c = divmod(idx, cols)
            with Image.open(cell) as im:
                im = im.convert("RGB").resize(thumb)
                canvas.paste(im, (c * frame, r * frame))
        with tempfile.TemporaryDirectory(prefix="npa-living-lab-") as out:
            local = Path(out) / "panorama.png"
            canvas.save(local)
            uploaded = _upload_file(local, output_uri)
    return {"panorama_uri": uploaded, "cells": len(cells), "cols": cols}


def join_living_lab_zones(
    *,
    zones_uri: str,
    report_uri: str,
    panorama_uri: str,
    shards: str | Sequence[str] = "",
    run_id: str = "",
) -> dict[str, Any]:
    """Barrier join: read every expected zone manifest, assert completeness.

    Reads each zone's ``zone_manifest.json`` (published by the shard after the
    real NRE pipeline ran on its own RTX PRO 6000). The expected zone set is the
    explicit topology passed via ``shards`` (the workflow's comma-joined zone
    list), so the join is topology-agnostic: it works identically for the
    shipped 16-zone (8 x 2) default and a 24-zone (8 x 3) topology. It fails
    unless every expected zone records a real USDZ, a real GPU UUID + node,
    fail-closed input provenance, and real (non-missing) NRE validation metrics.
    Success additionally requires the device-count proof to hold exactly: one
    distinct, non-empty GPU UUID per required device (never inferred from model
    names) and a material all-required-device temporal overlap. It aggregates
    objective metrics, distinct GPU UUID/node participation, and the
    all-required-overlap proof, then publishes a composite digital-twin report
    plus a contact-sheet panorama.
    """
    base = zones_uri.rstrip("/") + "/"
    expected = zone_names(shards)
    if not expected:
        raise ValueError(
            "living-lab join requires a non-empty expected zone set; refusing to "
            "succeed with zero expected devices"
        )
    # Derive the expected scene/variant for each zone from its own name
    # (``{scene}-{variant}-{sector}``) so the provenance gate stays accurate for
    # any topology size instead of hard-coding the default 16-zone map.
    expected_prov: dict[str, tuple[str, str]] = {}
    for zone in expected:
        parts = zone.split("-")
        expected_prov[zone] = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    gpu_names: set[str] = set()
    gpu_uuids: set[str] = set()
    nodes: set[str] = set()
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    lpips_vals: list[float] = []
    windows: list[tuple[int, int]] = []
    zones_uri_map: dict[str, str] = {}

    def _finite(val: Any) -> float | None:
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if f == f else None

    for zone in expected:
        uri = f"{base}{zone}/{ZONE_MANIFEST_FILENAME}"
        try:
            payload = _download_json(uri)
        except Exception as exc:  # noqa: BLE001 - surfaced below as a hard failure
            missing.append(zone)
            entries.append({"zone": zone, "status": "missing", "error": str(exc)[:240]})
            continue
        status = payload.get("status")
        usdz = bool(payload.get("usdz_path"))
        gpu_uuid = str(payload.get("gpu_uuid") or "").strip()
        gpu_name = str(payload.get("gpu_name") or "").strip()
        node_name = str(payload.get("node_name") or "").strip()
        metrics_path = str(payload.get("metrics_path") or "").strip()
        metrics = payload.get("metrics") or {}
        psnr = _finite(metrics.get("test/psnr"))
        ssim = _finite(metrics.get("test/ssim"))
        lpips = _finite(metrics.get("test/lpips"))
        # Real validation metrics must be measured and parseable; a missing or
        # 0.0-placeholder metric is never accepted (PSNR/SSIM/LPIPS of exactly
        # 0.0 are not genuine NRE validation values).
        metrics_ok = (
            bool(metrics_path)
            and psnr not in (None, 0.0)
            and ssim not in (None, 0.0)
            and lpips not in (None, 0.0)
        )
        prov = payload.get("provenance") or {}
        prov_scene = str(prov.get("scene") or payload.get("scene") or "").strip()
        prov_variant = str(prov.get("variant") or payload.get("variant") or "").strip()
        exp_scene, exp_variant = expected_prov.get(zone, ("", ""))
        provenance_ok = prov_scene == exp_scene and prov_variant == exp_variant
        started = _finite(payload.get("started_epoch")) or 0
        ended = _finite(payload.get("ended_epoch")) or 0
        ok = (
            status == "ok"
            and usdz
            and bool(gpu_uuid)
            and bool(node_name)
            and metrics_ok
            and provenance_ok
        )
        if not ok:
            missing.append(zone)
        entries.append(
            {
                "zone": zone,
                "status": "ok" if ok else payload.get("status", "unknown"),
                "gpu_uuid": gpu_uuid,
                "gpu_name": gpu_name,
                "node_name": node_name,
                "pod_name": payload.get("pod_name", ""),
                "usdz_present": usdz,
                "reconstruction_uri": payload.get("reconstruction_uri", ""),
                "novel_views_uri": payload.get("novel_views_uri", ""),
                "metrics_path": metrics_path,
                "metrics": metrics,
                "provenance": {"scene": prov_scene, "variant": prov_variant},
                "started_epoch": started,
                "ended_epoch": ended,
                "elapsed_seconds": payload.get("elapsed_seconds"),
            }
        )
        if gpu_uuid:
            gpu_uuids.add(gpu_uuid)
        if gpu_name:
            gpu_names.add(gpu_name)
        if node_name:
            nodes.add(node_name)
        if psnr is not None:
            psnr_vals.append(psnr)
        if ssim is not None:
            ssim_vals.append(ssim)
        if lpips is not None:
            lpips_vals.append(lpips)
        if started and ended:
            windows.append((int(started), int(ended)))
        if ok:
            zones_uri_map[zone] = f"{base}{zone}/novel_views/"

    panorama: dict[str, Any] = {}
    if zones_uri_map:
        try:
            panorama = build_panorama(
                zone_uris_map=zones_uri_map,
                output_uri=panorama_uri,
            )
        except Exception as exc:  # noqa: BLE001 - non-fatal; report it
            panorama = {"error": str(exc)[:240]}

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    # All-required-overlap proof: every expected device's execution window must
    # materially overlap at a common instant (max start < min end). Fail closed
    # when any expected zone is missing its timestamps (windows shorter than the
    # expected set).
    overlap_start = max((w[0] for w in windows), default=None)
    overlap_end = min((w[1] for w in windows), default=None)
    concurrent = (
        len(windows) == len(expected)
        and overlap_start is not None
        and overlap_end is not None
        and overlap_start < overlap_end
    )

    # Device-count proof is never inferred from model names: it is derived from
    # the distinct, non-empty GPU UUIDs the shards actually recorded, and must
    # equal the required device count exactly.
    required_device_count = len(expected)
    distinct_uuids = {u for u in gpu_uuids if u}
    device_proof_ok = len(distinct_uuids) == required_device_count

    report = {
        "schema": DIGITAL_TWIN_SCHEMA,
        "run_id": run_id,
        "zones_uri": base,
        "zone_count": len(expected),
        "joined_zones": len(expected) - len(missing),
        "missing_zones": missing,
        # Informational model-name participation; never used for the device-proof gate.
        "gpu_participation": sorted(gpu_names),
        # Device participation (UUID-based, not model-name based).
        "distinct_gpu_count": len(distinct_uuids),
        "distinct_gpu_uuid_count": len(distinct_uuids),
        "gpu_uuids": sorted(distinct_uuids),
        "distinct_node_count": len(nodes),
        "nodes": sorted(nodes),
        "concurrency": {
            "required_device_count": required_device_count,
            "all_required_overlap": concurrent,
            "overlapping_zones": len(windows),
            "overlap_start_epoch": overlap_start,
            "overlap_end_epoch": overlap_end,
        },
        "aggregate_metrics": {
            "test/psnr_mean": _avg(psnr_vals),
            "test/ssim_mean": _avg(ssim_vals),
            "test/lpips_mean": _avg(lpips_vals),
        },
        "panorama": panorama,
        "zones": entries,
    }
    target = (
        report_uri
        if report_uri.endswith(".json")
        else f"{report_uri.rstrip('/')}/digital_twin.json"
    )
    report["report_uri"] = _upload_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(), target
    )
    print(json.dumps(report))

    # Fail closed: a report is never success by itself. The join must also prove
    # exactly one distinct, non-empty GPU UUID per required device and a material
    # all-required-device temporal overlap. Error text is diagnostic (counts /
    # reasons only) and never exposes live resource identifiers.
    reasons: list[str] = []
    if missing:
        reasons.append(
            f"{len(missing)} of {len(expected)} zones missing/invalid: {missing}"
        )
    if not device_proof_ok:
        reasons.append(
            f"distinct GPU UUID proof failed: expected {required_device_count} "
            f"distinct non-empty GPU UUIDs, observed {len(distinct_uuids)}"
        )
    if not concurrent:
        reasons.append(
            f"all-required-overlap proof failed: material temporal overlap across "
            f"all {required_device_count} expected devices not demonstrated "
            f"(overlapping windows: {len(windows)})"
        )
    if reasons:
        raise RuntimeError("living-lab join incomplete: " + "; ".join(reasons))
    return report


__all__ = [
    "DEFAULT_CAPTURES",
    "DEFAULT_DEVICE_COUNT",
    "DEFAULT_SECTORS",
    "DIGITAL_TWIN_SCHEMA",
    "PANORAMA_FILENAME",
    "SCENES",
    "ZONE_MANIFEST_FILENAME",
    "VIEW_SECTORS",
    "build_living_lab_workflow_spec",
    "build_panorama",
    "join_living_lab_zones",
    "living_lab_zones",
    "living_lab_workflow_yaml",
    "zone_uris",
    "zone_names",
]


# ---------------------------------------------------------------------------
# Workflow spec builder (generates the parameterized fan-out YAML).
# ---------------------------------------------------------------------------

_GPU_RESOURCE = {
    "cloud": "kubernetes",
    "accelerators": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1",
    "cpus": 8,
    "memory": "64Gi",
    "image": "{{config.nurec_image}}",
    "kubernetes": {
        "provision_timeout": 2400,
        "pod_config": {
            "spec": {
                "imagePullSecrets": [{"name": "ngc-nvcr-imagepullsecret"}],
                "initContainers": [
                    {
                        "name": "npa-sudo-shim",
                        "image": "{{config.nurec_image}}",
                        "command": [
                            "/bin/sh",
                            "-c",
                            "printf '#!/bin/sh\\nexec \"$@\"\\n' > /shim/sudo\nchmod 0755 /shim/sudo\n",
                        ],
                        "volumeMounts": [
                            {"name": "npa-sudo-shim", "mountPath": "/shim"}
                        ],
                    }
                ],
                "containers": [
                    {
                        "name": "ray-node",
                        "env": [
                            {
                                "name": "NODE_NAME",
                                "valueFrom": {
                                    "fieldRef": {"fieldPath": "spec.nodeName"}
                                },
                            }
                        ],
                        "volumeMounts": [
                            {"name": "npa-sudo-shim", "mountPath": "/usr/local/sbin"},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "npa-sudo-shim", "emptyDir": {}},
                    {
                        "name": "dshm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
                    },
                ],
            }
        },
    },
}

_ZONE_SHARD_TEMPLATE = """set -euo pipefail
ZONE="{{config.zone_name}}"
RUN_URI="{{config.run_prefix_uri}}"
DATASET_ID="{{config.dataset_id}}"
SCENE="{{config.scene}}"
VARIANT="{{config.variant}}"
ZU="${RUN_URI}zones/${ZONE}/"
NC="${ZU}ncore/"
RC="${ZU}reconstruction/"
NV="${ZU}novel_views/"
INPUT="${ZU}input/"
RRD="${ZU}reports/sim2real.rrd"
FINAL="${ZU}reports/final.json"
MF="${ZU}zone_manifest.json"

# ---- runtime setup inside the NRE container --------------------------------
# The NGC NRE image ships no npa, no ffmpeg, and none of the workbench's Python
# deps. The runtime setup installs npa from $NPA_SRC_S3_URI; this step fills in
# the NRE-runtime Python deps (nvidia-ncore for fetch, rerun-sdk for the Rerun
# recording, pillow/pyyaml for the join inputs) and ffmpeg (backs render
# --export-video). It is idempotent and matches the shipped single-pod reference.
export DEBIAN_FRONTEND=noninteractive
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y -qq --no-install-recommends ffmpeg || true
fi
npa_pip() {
  python3 -m pip install -q "$@" || python3 -m pip install -q "$@" --break-system-packages || python3 -m pip install -q "$@" --user
}
npa_pip "boto3>=1.34" "awscli>=1.32" "huggingface_hub>=0.30" "nvidia-ncore" "rerun-sdk" "pillow>=10.0" "pyyaml>=6.0"
command -v npa >/dev/null 2>&1 || { echo "npa not found; set NPA_SRC_S3_URI" >&2; exit 1; }

# ---- objective per-shard GPU / scheduler identity ---------------------------
# Captured once at the top of the shard so the manifest carries real, per-GPU
# provenance (UUID + model + node + pod + wave), not just a model-name count.
# HOSTNAME inside a Kubernetes pod is the pod name; NODE_NAME comes from the
# downward-API env the workflow's pod_config injects.
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 || echo unknown)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo unknown)
POD_NAME="${HOSTNAME:-unknown}"
NODE_NAME="${NODE_NAME:-unknown}"
START=$(date +%s)
export ZONE RUN_URI DATASET_ID SCENE VARIANT ZU NC RC NV INPUT RRD FINAL MF
export GPU_UUID GPU_NAME POD_NAME NODE_NAME START

echo "=== zone $ZONE: check (container + dataset rights + RT-core GPU) ==="
npa workbench nurec check --require-gpu --output json

echo "=== zone $ZONE: fetch real NCore V4 shards + derived rig pose edge ==="
npa workbench nurec fetch --dataset "${DATASET_ID}" --scene "${SCENE}" --variant "${VARIANT}" --output-uri "${NC}" --output json >/tmp/nurec-fetch.json

# ---- fail-closed provenance gate -------------------------------------------
# Provenance is validated against the *independently observed* unpacked content,
# not merely the echoed request arguments: the fetch result carries
# observed_scene/observed_variant derived from the scene directory that actually
# landed in the extracted archive. A shard must never reconstruct a capture it
# did not ask for; if the observed content disagrees with the requested scene or
# variant, or the fetch did not record observed content at all, fail the shard
# (and therefore the join).
python3 - "$SCENE" "$VARIANT" <<'PY'
import json, os, sys
requested = {"scene": sys.argv[1], "variant": sys.argv[2]}
with open(os.environ.get("NUREC_FETCH_JSON", "/tmp/nurec-fetch.json")) as fh:
    fetched = json.load(fh)
# Observed content is authoritative. When a fetch does not record it (older
# output), fall back only for diagnostics -- the gate then fails closed because
# observed content is missing.
observed_scene = str(fetched.get("observed_scene") or "")
observed_variant = str(fetched.get("observed_variant") or "")
if not observed_scene and not observed_variant:
    print("PROVENANCE MISMATCH for zone", os.environ["ZONE"], ":",
          "fetch result carries no observed unpacked content", file=sys.stderr)
    sys.exit(1)
mismatch = {
    k: (v, requested[k])
    for k, v in (("scene", observed_scene), ("variant", observed_variant))
    if v != requested[k]
}
if mismatch:
    print("PROVENANCE MISMATCH for zone", os.environ["ZONE"], ":",
          json.dumps({"observed": mismatch, "requested": requested}), file=sys.stderr)
    sys.exit(1)
if str(fetched.get("dataset_id") or "") != os.environ.get("DATASET_ID", ""):
    print("PROVENANCE MISMATCH for zone", os.environ["ZONE"], ": dataset_id",
          file=sys.stderr)
    sys.exit(1)
print("provenance-ok", json.dumps({"scene": observed_scene, "variant": observed_variant}))
PY
test "$(python3 -c "import json;print(json.load(open('/tmp/nurec-fetch.json'))['status'] or '')")" = "ok"

NCORE_JSON=$(python3 -c "import json;print(json.load(open('/tmp/nurec-fetch.json'))['ncore_json'])")
POSES_GROUP=$(python3 -c "import json;print(json.load(open('/tmp/nurec-fetch.json'))['poses_component_group'])")
CAMERA=$(python3 -c "import json;print(json.load(open('/tmp/nurec-fetch.json'))['reference_camera'])")
test -n "${NCORE_JSON}" && test -n "${POSES_GROUP}" && test -n "${CAMERA}"
export NCORE_JSON POSES_GROUP CAMERA

# ---- reconstruct + real, non-missing quality metrics ------------------------
# The reconstruct stage writes NRE's val metrics.yaml. metrics_path preserves
# where the numbers came from; metrics are recorded only when actually measured.
# A missing/unparseable metrics payload makes the shard (and the join) FAIL
# rather than substituting a numeric zero.
echo "=== zone $ZONE: reconstruct (3DGUT Gaussians -> renderable USDZ) ==="
npa workbench nurec reconstruct --ncore-json "${NCORE_JSON}" --poses-component-group "${POSES_GROUP}" --camera-id "${CAMERA}" --world-size 1 --export-gt --output-uri "${RC}" --input-uri "${INPUT}" --output json >/tmp/nurec-reconstruct.json
USDZ=$(python3 -c "import json;print(json.load(open('/tmp/nurec-reconstruct.json'))['usdz_path'])")
METRICS_PATH=$(python3 -c "import json;print(json.load(open('/tmp/nurec-reconstruct.json')).get('metrics_path') or '')")
export USDZ METRICS_PATH
test -n "${USDZ}"

echo "=== zone $ZONE: render novel views (rig-offset, not training views) ==="
npa workbench nurec render --artifact-path "${USDZ}" --output-dir /tmp/render-out --camera-id "${CAMERA}" --renderer default --rig-translation-offset "{{config.rig_translation_offset}}" --rig-rotation-offset "{{config.rig_rotation_offset}}" --no-replicate-training-views --output-uri "${NV}" --output json >/tmp/nurec-render.json

echo "=== zone $ZONE: visualize (reports/sim2real.rrd for the agent panel) ==="
npa workbench nurec visualize --input-uri "${ZU}" --output-uri "${RRD}" --output json >/tmp/nurec-viz.json

echo "=== zone $ZONE: finalize (reports/final.json) ==="
npa workbench nurec finalize --input-uri "${ZU}" --output-uri "${FINAL}" --run-id "${ZONE}" --output json >/tmp/nurec-final.json

END=$(date +%s)
export END

python3 - <<'PY'
import json, os
def _load_json(path, default=None):
    default = default if default is not None else {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default
recon = _load_json("/tmp/nurec-reconstruct.json")
final = _load_json("/tmp/nurec-final.json")
raw_metrics = recon.get("metrics") or {}
metrics_path = os.environ.get("METRICS_PATH", "")
# Keep an unavailable metric distinct from a measured value: None means no
# value was produced, never substitute 0.0.
def _m(key):
    val = raw_metrics.get(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
metrics = {"test/psnr": _m("test/psnr"), "test/ssim": _m("test/ssim"), "test/lpips": _m("test/lpips")}
metrics_present = bool(metrics_path) and all(metrics[k] is not None for k in metrics)
if not metrics_present:
    print("zone", os.environ["ZONE"], ": NRE validation metrics missing/unparseable (metrics_path=%r); failing shard" % metrics_path, flush=True)
    raise SystemExit(1)
payload = {
  "schema": "npa.living_lab.zone_manifest.v1",
  "zone_name": os.environ["ZONE"],
  "status": "ok",
  "provenance": {
    "dataset_id": os.environ["DATASET_ID"],
    "scene": os.environ["SCENE"],
    "variant": os.environ["VARIANT"],
    "requested": {k: os.environ[k] for k in ("DATASET_ID", "SCENE", "VARIANT")},
  },
  "gpu": {
    "gpu_uuid": os.environ.get("GPU_UUID", ""),
    "gpu_name": os.environ.get("GPU_NAME", ""),
    "node_name": os.environ.get("NODE_NAME", ""),
    "pod_name": os.environ.get("POD_NAME", ""),
    "wave_id": os.environ.get("RUN_URI", ""),
  },
  "gpu_uuid": os.environ.get("GPU_UUID", ""),
  "gpu_name": os.environ.get("GPU_NAME", ""),
  "node_name": os.environ.get("NODE_NAME", ""),
  "pod_name": os.environ.get("POD_NAME", ""),
  "wave_id": os.environ.get("RUN_URI", ""),
  "usdz_path": os.environ.get("USDZ", ""),
  "reconstruction_uri": os.environ.get("RC", ""),
  "novel_views_uri": os.environ.get("NV", ""),
  "rrd_uri": os.environ.get("RRD", ""),
  "final_report_uri": os.environ.get("FINAL", ""),
  "started_epoch": int(os.environ.get("START", 0)),
  "ended_epoch": int(os.environ.get("END", 0)),
  "elapsed_seconds": int(os.environ.get("END", 0)) - int(os.environ.get("START", 0)),
  "metrics_path": metrics_path,
  "metrics": metrics,
  "finalize": {
    "status": final.get("status", ""),
    "has_usdz": bool(final.get("has_usdz", False)),
    "has_rrd": bool(final.get("has_rrd", False)),
    "artifact_count": final.get("artifact_count", 0),
  },
}
json.dump(payload, open("/tmp/zone_manifest.json", "w"))
from npa.clients.storage import StorageClient
StorageClient.from_environment().upload_file("/tmp/zone_manifest.json", os.environ["MF"])
print(json.dumps(payload))
PY
"""


def build_living_lab_workflow_spec(
    *,
    captures: Sequence[tuple[str, str]] = DEFAULT_CAPTURES,
    sectors: Sequence[str] = DEFAULT_SECTORS,
) -> dict[str, Any]:
    """Return the ``living-lab-nurec-fanout`` npa.workflow spec as a dict.

    Fan-out size is derived from the explicit ``captures`` x ``sectors`` inputs
    rather than a magic fixed count. The default 8 capture pairs x 2 sectors
    yields the shipped 16-zone topology; the same generator with 3 sectors
    yields a 24-zone topology. The expected device count is derived from the
    generated zone list, exposed in ``config.expected_device_count``, and
    cross-checked against the explicit parallel member list via the workflow
    ``parallelCount`` contract so an operator cannot understate the required
    devices without validate failing.
    """
    zones = living_lab_zones(captures=captures, sectors=sectors)
    zone_names_list = [z["zone_name"] for z in zones]
    expected_device_count = len(zone_names_list)
    if expected_device_count < 1:
        raise ValueError("living-lab topology must contain at least one zone")

    capture_count = len(captures)
    sector_count = len(sectors)

    shard_states: dict[str, Any] = {}
    parallel_members: list[str] = []
    for z in zones:
        name = f"zone-{z['zone_name']}"
        parallel_members.append(name)
        shard_states[name] = {
            "description": (
                f"Zone {z['zone_name']}: full real NuRec/NRE reconstruction of the "
                f"{z['scene']} {z['variant']} capture (view sector {z['view_sector']}) "
                f"on its own RTX PRO 6000, then novel-view render + Rerun recording."
            ),
            "resources": "gpu",
            "params": {
                "zone_name": z["zone_name"],
                "scene": z["scene"],
                "variant": z["variant"],
                "view_sector": z["view_sector"],
                "rig_rotation_offset": str(z["rig_rotation_offset"]),
                "rig_translation_offset": str(z["rig_translation_offset"]),
            },
            "run": {"shell": _ZONE_SHARD_TEMPLATE},
            "outputs": [
                {
                    "uri": "{{config.zones_uri}}{{config.zone_name}}/zone_manifest.json",
                    "schema": "npa.living_lab.zone_manifest.v1",
                }
            ],
        }

    zone_description = (
        f"{expected_device_count} living-lab zone reconstructions as one SkyPilot "
        f"JobGroup (one RTX PRO 6000 each). Each member is a fully "
        f"independent, real NRE reconstruction of its zone, so all "
        f"{expected_device_count} reserved RTX PRO 6000 GPUs are materially busy "
        f"at once."
    )

    states: dict[str, Any] = {
        "living-lab-zones": {
            "description": zone_description,
            "parallel": parallel_members,
            "parallelCount": "{{config.expected_device_count}}",
            "maxConcurrency": "{{config.max_concurrency}}",
            "next": "join",
        },
        **shard_states,
        "join": {
            "description": (
                "Barrier: read every expected zone manifest, require every zone "
                "present with a real GPU identity and a real USDZ, aggregate "
                "objective metrics and GPU participation, publish the composite "
                "digital-twin report plus a contact-sheet panorama, and fail "
                "closed unless the proof demonstrates exactly the required number "
                "of distinct non-empty GPU UUIDs with material all-required-device "
                "temporal overlap."
            ),
            "needs": ["living-lab-zones"],
            "resources": "cpu",
            "run": {
                "shell": (
                    'python3 -c "from npa.workflows.living_lab import '
                    "join_living_lab_zones; join_living_lab_zones("
                    "zones_uri='{{config.zones_uri}}', "
                    "report_uri='{{config.report_uri}}', "
                    "panorama_uri='{{config.panorama_uri}}', "
                    "shards='{{config.zones}}', run_id='{{run.id}}')\""
                )
            },
            "outputs": [
                {
                    "uri": "{{config.report_uri}}",
                    "schema": "npa.living_lab.digital_twin.v1",
                },
                {
                    "uri": "{{config.panorama_uri}}",
                    "schema": "npa.living_lab.panorama.v1",
                },
            ],
            "terminal": True,
        },
    }

    return {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {
            "name": "living-lab-nurec-fanout",
            "description": (
                f"{expected_device_count}-zone living-lab digital twin from real "
                f"NVIDIA NuRec / NRE neural reconstructions: {expected_device_count} "
                f"independent RTX PRO 6000 shards ({capture_count} real NCore "
                f"capture pairs x {sector_count} view sectors) each run the full "
                f"nurec pipeline, then a barrier join composes a composite "
                f"digital-twin report and contact-sheet panorama with objective "
                f"per-zone GPU participation evidence. Topology size is derived "
                f"from the capture/sector inputs."
            ),
        },
        "config": {
            "bucket": "example-bucket",
            "prefix": "living-lab/{{run.id}}",
            "nurec_image": "nvcr.io/nvidia/nre/nre-ga:26.04",
            "dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
            "capture_count": str(capture_count),
            "sector_count": str(sector_count),
            "expected_device_count": str(expected_device_count),
            "max_concurrency": str(expected_device_count),
            "zones": ",".join(zone_names_list),
            "run_prefix_uri": "s3://{{config.bucket}}/{{config.prefix}}/",
            "zones_uri": "s3://{{config.bucket}}/{{config.prefix}}/zones/",
            "report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/digital_twin.json",
            "panorama_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/panorama.png",
            "zone_name": "",
        },
        "resources": {
            "gpu": _GPU_RESOURCE,
            "cpu": {"cloud": "kubernetes", "cpus": 4, "memory": "16Gi"},
        },
        "initial": "living-lab-zones",
        "states": states,
    }


def living_lab_workflow_yaml() -> str:
    """Render the living-lab workflow spec to YAML text."""
    import yaml

    return yaml.safe_dump(
        build_living_lab_workflow_spec(),
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
