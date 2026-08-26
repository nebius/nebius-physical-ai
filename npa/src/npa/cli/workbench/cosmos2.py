"""Workbench Cosmos2 commands."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import tempfile
import threading
import time
from typing import Any, Optional

import typer

from npa.workflows.cosmos_split import (
    Cosmos2TransferConfig,
    build_cosmos2_transfer_manifest,
    write_manifest,
)
from npa.workbench.cosmos.transfer import (
    REFERENCE_AUGMENT_MODE,
    REFERENCE_AUGMENT_STATUS,
    TRANSFER_MANIFEST_FILENAME,
    TRANSFER_MANIFEST_MODE,
    TRANSFER_MANIFEST_STATUS,
    transfer_manifest_uri_for,
)

app = typer.Typer(
    name="cosmos2",
    help="Cosmos2 transfer workflow contracts.",
    no_args_is_help=True,
)


#: Compatibility alias; the workbench implementation owns the canonical name.
MANIFEST_FILENAME = TRANSFER_MANIFEST_FILENAME


def _publish_manifest(client: Any, payload: dict, output_uri: str) -> str:
    """Upload the stage manifest next to the augmented clip and return its URI."""

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory(prefix="npa-cosmos2-") as tmp:
        local = Path(tmp) / MANIFEST_FILENAME
        local.write_bytes(_manifest_bytes(payload))
        return client.upload_file(str(local), transfer_manifest_uri_for(output_uri))


def _manifest_bytes(payload: dict) -> bytes:
    """Return the canonical manifest serialization used by every backend."""

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_output_manifest(payload: dict, output_uri: str) -> str:
    """Publish a canonical transfer manifest for an S3 or local output prefix."""

    manifest_uri = transfer_manifest_uri_for(output_uri)
    if output_uri.strip().startswith("s3://"):
        from npa.clients.storage import StorageClient

        return _publish_manifest(StorageClient.from_environment(), payload, output_uri)

    local_output = output_uri.removeprefix("local://").removeprefix("file://")
    manifest_path = Path(local_output) / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_manifest_bytes(payload))
    return manifest_uri


def _all_augmentations(configs_uri: str) -> list[dict]:
    """Read the Config-Gen manifest and return every sampled appearance combo.

    Each combo drives one Cosmos Transfer 2.5 inference ("multiply"), so a config
    manifest with N augmentations yields N scenario variants.  A configured
    manifest is authoritative: an unreadable or empty manifest must not silently
    collapse a requested multi-variant/gang run into one default render.
    """
    try:
        from npa.workflows.data_factory_stages import _download_json

        uri = (
            configs_uri
            if configs_uri.endswith(".json")
            else configs_uri.rstrip("/") + "/manifest.json"
        )
        manifest = _download_json(uri)
    except Exception:  # noqa: BLE001 - sanitized operator boundary
        raise typer.BadParameter(
            "could not read the configured augmentation manifest"
        ) from None
    if not isinstance(manifest, dict):
        raise typer.BadParameter(
            "configured augmentation manifest must be a JSON object"
        )
    raw_combos = manifest.get("augmentations")
    if not isinstance(raw_combos, list) or not raw_combos:
        raise typer.BadParameter(
            "configured augmentation manifest contains no augmentation variants"
        )
    if not all(isinstance(combo, dict) for combo in raw_combos):
        raise typer.BadParameter(
            "configured augmentation manifest contains an invalid variant"
        )
    return list(raw_combos)


def _first_augmentation(configs_uri: str) -> dict:
    """Read the Config-Gen manifest and return its first sampled combo."""
    combos = _all_augmentations(configs_uri)
    return combos[0]


def _load_refinement(refinement_uri: str) -> dict[str, Any]:
    """Load only a commit-marked run-scoped adaptive refinement policy."""

    if not refinement_uri:
        return {}
    from npa.workflows.data_factory_stages import (
        RefinementStateError,
        _download_json,
        _verify_committed_refinement,
    )

    try:
        payload = _download_json(refinement_uri)
    except Exception as exc:  # noqa: BLE001 - sanitized CLI boundary
        raise typer.BadParameter(
            "could not read the committed refinement artifact"
        ) from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("refinement artifact must be a JSON object")
    try:
        immutable = _verify_committed_refinement(payload)
    except RefinementStateError as exc:
        raise typer.BadParameter(str(exc)) from exc
    settings = immutable.get("settings")
    if not isinstance(settings, dict):
        raise typer.BadParameter("refinement artifact has no settings object")
    try:
        control_weight = float(settings["control_weight"])
        guidance_number = float(settings["guidance"])
        attempt = int(immutable.get("attempt", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "refinement artifact settings must contain numeric control_weight and guidance"
        ) from exc
    if not 0.0 <= control_weight <= 1.0:
        raise typer.BadParameter(
            "refinement control_weight must be between 0 and 1"
        )
    if guidance_number < 0.0 or not guidance_number.is_integer():
        raise typer.BadParameter(
            "refinement guidance must be a non-negative integer"
        )
    guidance = int(guidance_number)
    if attempt < 0:
        raise typer.BadParameter("refinement artifact settings cannot be negative")
    return {
        "schema": str(immutable.get("schema") or ""),
        "attempt": attempt,
        "adapted_from_prior_evaluation": bool(
            immutable.get("adapted_from_prior_evaluation")
        ),
        "failed_checks": [
            str(item)
            for item in immutable.get("failed_checks", [])
            if isinstance(item, str)
        ],
        "failed_attributes": [
            str(item)
            for item in immutable.get("failed_attributes", [])
            if isinstance(item, str)
        ],
        "settings": {
            "control_weight": control_weight,
            "guidance": guidance,
        },
    }


def _apply_refinement_prompt(
    combo: dict[str, Any], refinement: dict[str, Any]
) -> dict[str, Any]:
    """Emphasize exactly the attributes that failed the preceding evaluation."""

    failed = [
        str(value).strip()
        for value in refinement.get("failed_attributes", [])
        if str(value).strip() and str(value).strip() in combo
    ]
    if not failed:
        return dict(combo)
    adapted = dict(combo)
    requirements = "; ".join(
        f"{variable.replace('_', ' ')} is unambiguously {combo[variable]}"
        for variable in failed
    )
    adapted["prompt"] = (
        f"{str(combo.get('prompt') or '').strip()} "
        f"Retry emphasis based only on failed attribute checks: {requirements}. "
        "Keep every other object, geometry, camera, timing, motion, and appearance "
        "property unchanged."
    ).strip()
    return adapted


_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _frames_to_conditioning_clip(frames: list[Path], output_dir: Path) -> str:
    """Encode PAIDF input frames as the short clip Cosmos Transfer consumes.

    The PAIDF first-run path intentionally seeds still frames because the
    preceding caption stage consumes images.  Cosmos Transfer consumes video,
    so its runner assembles those same frames into an ephemeral conditioning
    clip.  The clip matches the qualified procedural fixture's dimensions,
    frame rate, and frame count without copying or packaging any source media.
    """
    import shutil
    import subprocess

    if not frames:
        return ""

    sequence_dir = output_dir / "conditioning-frames"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    sequence: list[Path] = []
    for index, frame in enumerate(frames):
        suffix = frame.suffix.lower()
        if suffix not in _IMAGE_EXTS:
            continue
        normalized = sequence_dir / f"frame-{index:05d}{suffix}"
        shutil.copyfile(frame, normalized)
        sequence.append(normalized)
    if not sequence:
        return ""

    # Concat accepts mixed PNG/JPEG inputs.  All list entries are paths authored
    # above (not object-key text), and duplicating the final frame makes its
    # duration effective under the concat demuxer.
    concat_file = output_dir / "conditioning-frames.ffconcat"
    lines = ["ffconcat version 1.0"]
    for frame in sequence:
        lines.extend((f"file '{frame}'", "duration 0.5"))
    lines.append(f"file '{sequence[-1]}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = output_dir / "npa-paidf-conditioning.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vf",
            (
                "fps=16,tpad=stop_mode=clone:stop_duration=8,"
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
            ),
            "-frames:v",
            "93",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("FFmpeg did not produce a PAIDF conditioning clip")
    typer.echo(
        "PAIDF conditioning: encoded "
        f"{len(sequence)} input frame(s) as a 1280x720, 93-frame clip",
        err=True,
    )
    return str(output)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _detect_gpu_count() -> int:
    """Best-effort count of GPUs visible to this process (>=1).

    Prefers an explicit ``CUDA_VISIBLE_DEVICES`` list, then ``nvidia-smi -L``.
    Used to auto-parallelize the multiply fan-out (one variant per GPU) so a
    workflow that requests ``RTXPRO6000:4`` actually drives all four GPUs.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        ids = [x for x in cvd.split(",") if x.strip() != ""]
        return max(1, len(ids))
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
        n = len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")])
        return max(1, n)
    except Exception:  # noqa: BLE001 - detection is advisory; default to 1
        return 1


def _gang_shard() -> tuple[int, int]:
    """Return this worker's ``(rank, node_count)`` in the augment block.

    A spec asks for a multi-node augment with ``resources.gpu.num_nodes``; SkyPilot
    then gang-schedules that many identical pods for the one task and exports
    ``SKYPILOT_NODE_RANK`` / ``SKYPILOT_NUM_NODES`` into each. Every pod runs this
    same command, so without a shard the gang would render every variant N times.
    ``NPA_COSMOS_NODE_RANK`` / ``NPA_COSMOS_NODE_COUNT`` override for local runs.

    An inconsistent identity fails closed: silently collapsing to one node would
    duplicate GPU work and leave the run manifest reporting a fan-out that never
    happened.
    """

    rank, nodes, _metadata = _gang_environment()
    return rank, nodes


def _gang_contract_required() -> bool:
    """Return whether runtime evidence indicates the sharded contract applies.

    SkyPilot exports rank/count/IP variables for ordinary one-node tasks too.
    Such a generic Cosmos transfer has no renderer-owned NPA identity and must
    remain valid.  Conversely, an authoritative NPA count, a SkyPilot count
    above one, a nonzero rank, malformed numeric evidence, or multiple member
    IPs is enough to require the complete fail-closed gang validation.
    """

    npa_evidence = any(
        str(os.environ.get(name, "")).strip()
        for name in (
            "NPA_COSMOS_NODE_COUNT",
            "NPA_COSMOS_NODE_RANK",
            "NPA_COSMOS_ATTEMPT_ID",
        )
    )
    if npa_evidence:
        return True
    sky_nodes = str(os.environ.get("SKYPILOT_NUM_NODES", "")).strip()
    sky_rank = str(os.environ.get("SKYPILOT_NODE_RANK", "")).strip()
    sky_internal_job = str(os.environ.get("SKYPILOT_INTERNAL_JOB_ID", "")).strip()
    sky_managed_job = str(os.environ.get("SKYPILOT_MANAGED_JOB_ID", "")).strip()
    node_ips = [
        line.strip()
        for line in str(os.environ.get("SKYPILOT_NODE_IPS", "")).splitlines()
        if line.strip()
    ]
    sky_evidence = any((sky_nodes, sky_rank, node_ips, sky_internal_job, sky_managed_job))
    if not sky_evidence:
        return False
    return not (sky_nodes == "1" and sky_rank == "0" and len(node_ips) == 1)


def _gang_environment() -> tuple[int, int, dict[str, Any]]:
    """Validate scheduler identity and return its immutable gang evidence.

    ``NPA_COSMOS_NODE_COUNT`` comes from the workflow renderer and is the source
    of truth.  SkyPilot 0.12.2 independently supplies count, rank, ordered node
    IPs, internal job id, and managed-job id. All are checked before a worker can
    claim or publish. ``SKYPILOT_TASK_ID`` is deliberately excluded: SkyPilot
    preserves it across managed-job recoveries.
    """

    def _read(name: str) -> str:
        return str(os.environ.get(name, "")).strip()

    raw_nodes = _read("NPA_COSMOS_NODE_COUNT")
    raw_local_rank = _read("NPA_COSMOS_NODE_RANK")
    sky_nodes = _read("SKYPILOT_NUM_NODES")
    sky_rank = _read("SKYPILOT_NODE_RANK")
    sky_ips = _read("SKYPILOT_NODE_IPS")
    sky_internal_job = _read("SKYPILOT_INTERNAL_JOB_ID")
    sky_managed_job = _read("SKYPILOT_MANAGED_JOB_ID")
    base_attempt = _read("NPA_WORKFLOW_ATTEMPT_ID")
    fence_sequence = _read("NPA_WORKFLOW_FENCE_SEQUENCE")
    fence_attempt = _read("NPA_WORKFLOW_FENCE_ATTEMPT")
    local_attempt = _read("NPA_COSMOS_ATTEMPT_ID")
    identity_evidence = any(
        (
            raw_local_rank,
            sky_nodes,
            sky_rank,
            sky_ips,
            sky_internal_job,
            sky_managed_job,
            local_attempt,
        )
    )
    if not raw_nodes:
        if identity_evidence:
            raise typer.BadParameter(
                "multi-node augment identity is missing authoritative "
                "NPA_COSMOS_NODE_COUNT"
            )
        return 0, 1, {"scheduler": "single", "attempt_id": ""}
    try:
        nodes = int(raw_nodes)
    except ValueError as exc:
        raise typer.BadParameter(
            "multi-node augment identity is not numeric "
            f"(authoritative node count {raw_nodes!r})"
        ) from exc
    sky_evidence = any(
        (sky_nodes, sky_rank, sky_ips, sky_internal_job, sky_managed_job)
    )
    if sky_evidence:
        missing = [
            name
            for name, value in (
                ("SKYPILOT_NUM_NODES", sky_nodes),
                ("SKYPILOT_NODE_RANK", sky_rank),
                ("SKYPILOT_NODE_IPS", sky_ips),
                ("SKYPILOT_INTERNAL_JOB_ID", sky_internal_job),
                ("SKYPILOT_MANAGED_JOB_ID", sky_managed_job),
                ("NPA_WORKFLOW_ATTEMPT_ID", base_attempt),
                ("NPA_WORKFLOW_FENCE_SEQUENCE", fence_sequence),
                ("NPA_WORKFLOW_FENCE_ATTEMPT", fence_attempt),
            )
            if not value
        ]
        if missing:
            raise typer.BadParameter(
                "multi-node augment identity is incomplete: missing "
                + ", ".join(missing)
            )
        try:
            observed_nodes = int(sky_nodes)
            rank = int(sky_rank)
            scheduler_sequence = int(fence_sequence)
            scheduler_attempt = int(fence_attempt)
        except ValueError as exc:
            raise typer.BadParameter(
                "multi-node augment identity is not numeric "
                f"(SkyPilot node count {sky_nodes!r}, rank {sky_rank!r})"
            ) from exc
        if observed_nodes != nodes:
            raise typer.BadParameter(
                "multi-node augment identity is contradictory: renderer requested "
                f"{nodes} node(s), SkyPilot reported {observed_nodes}"
            )
        if scheduler_sequence < 1 or scheduler_attempt < 1:
            raise typer.BadParameter(
                "multi-node augment scheduler publication fence must be positive"
            )
        if raw_local_rank and raw_local_rank != sky_rank:
            raise typer.BadParameter(
                "multi-node augment identity is contradictory: "
                f"NPA rank {raw_local_rank!r}, SkyPilot rank {sky_rank!r}"
            )
        node_ips = [line.strip() for line in sky_ips.splitlines() if line.strip()]
        if len(node_ips) != nodes or len(set(node_ips)) != nodes:
            raise typer.BadParameter(
                "multi-node augment identity is contradictory: "
                f"SKYPILOT_NODE_IPS has {len(node_ips)} unique member(s), "
                f"expected {nodes}"
            )
        membership_digest = hashlib.sha256(
            json.dumps(
                {
                    "internal_job": sky_internal_job,
                    "managed_job": sky_managed_job,
                    "scheduler_fence_attempt": scheduler_attempt,
                    "scheduler_fence_sequence": scheduler_sequence,
                    "node_count": nodes,
                    "node_ips": node_ips,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        metadata = {
            "scheduler": "skypilot-0.12.2-managed",
            "logical_wave_id": base_attempt,
            "internal_job_id": sky_internal_job,
            "managed_job_id": sky_managed_job,
            "node_ips": node_ips,
            "membership_digest": membership_digest,
            "scheduler_fence_sequence": scheduler_sequence,
            "scheduler_fence_attempt": scheduler_attempt,
        }
    else:
        raw_rank = raw_local_rank or "0"
        try:
            rank = int(raw_rank)
        except ValueError as exc:
            raise typer.BadParameter(
                "multi-node augment identity is not numeric "
                f"(node count {raw_nodes!r}, rank {raw_rank!r})"
            ) from exc
        if nodes > 1 and not local_attempt:
            raise typer.BadParameter(
                "multi-node augment identity is incomplete: a local gang requires "
                "NPA_COSMOS_ATTEMPT_ID"
            )
        metadata = {"scheduler": "local", "attempt_id": local_attempt}
    if nodes < 1 or not 0 <= rank < nodes:
        raise typer.BadParameter(
            f"multi-node augment identity is inconsistent: rank {rank} of {nodes} node(s)"
        )
    return rank, nodes, metadata


def _rendezvous_port(logical_wave_id: str, membership_digest: str) -> int:
    material = f"{logical_wave_id}\0{membership_digest}".encode("utf-8")
    return 30000 + (int(hashlib.sha256(material).hexdigest()[:8], 16) % 20000)


def _recv_json_line(connection: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while len(chunks) <= 16384:
        part = connection.recv(4096)
        if not part:
            break
        chunks.extend(part)
        if b"\n" in part:
            break
    if len(chunks) > 16384:
        raise ValueError("gang identity rendezvous message exceeds 16 KiB")
    payload = json.loads(bytes(chunks).split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gang identity rendezvous message is not an object")
    return payload


def _identity_timeout() -> float | None:
    raw = str(os.environ.get("NPA_COSMOS_IDENTITY_TIMEOUT_S", "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise typer.BadParameter(
            "NPA_COSMOS_IDENTITY_TIMEOUT_S must be a non-negative number"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise typer.BadParameter(
            "NPA_COSMOS_IDENTITY_TIMEOUT_S must be finite and non-negative"
        )
    return value


def _sky_gang_rendezvous(
    *,
    rank: int,
    node_count: int,
    node_ips: list[str],
    logical_wave_id: str,
    membership_digest: str,
    internal_job_id: str,
    offered: dict[str, Any] | None,
) -> dict[str, Any]:
    """Share rank 0's scheduler-fenced attempt with the exact gang members.

    SkyPilot 0.12.2 exposes no ordered recovery epoch to the workload. Rank 0
    therefore claims only the NPA runtime's pre-issued ``(sequence, attempt)``
    fence and serves that claim to the exact ordered members. An inner SkyPilot
    recovery retains the fence and cannot supersede an existing same-token claim;
    if no worker claimed it before failing, the replacement may safely be its first
    claimant. Only an explicit later NPA retry may carry a higher scheduler token.
    """

    port = _rendezvous_port(logical_wave_id, membership_digest)
    protocol = "npa.cosmos.gang-attempt/v1"
    if rank == 0:
        if not isinstance(offered, dict):
            raise typer.BadParameter("rank 0 has no publication generation to share")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((node_ips[0], port))
            server.listen(max(1, node_count - 1))
        except OSError as exc:
            server.close()
            raise typer.BadParameter(
                "rank 0 could not open the gang-attempt rendezvous on its "
                "authoritative SkyPilot node IP"
            ) from exc

        def serve() -> None:
            committed_nonces: dict[int, str] = {}
            delivered: set[int] = set()
            in_flight: dict[int, str] = {}
            received_lock = threading.Lock()

            def handle(connection: socket.socket, peer: tuple[str, int]) -> None:
                reserved_rank: int | None = None
                claimant_nonce = ""
                with connection:
                    try:
                        member_rank = -1
                        valid = False
                        replay_committed = False
                        retryable = False
                        try:
                            request = _recv_json_line(connection)
                            member_rank = int(request.get("rank", -1))
                            claimant_nonce = str(
                                request.get("claimant_nonce") or ""
                            )
                            identity_matches = bool(
                                request.get("protocol") == protocol
                                and request.get("logical_wave_id")
                                == logical_wave_id
                                and request.get("membership_digest")
                                == membership_digest
                                and request.get("internal_job_id") == internal_job_id
                                and int(request.get("node_count", 0)) == node_count
                                and 0 < member_rank < node_count
                                and peer[0] == node_ips[member_rank]
                                and len(claimant_nonce) == 64
                                and all(
                                    character in "0123456789abcdef"
                                    for character in claimant_nonce
                                )
                            )
                            with received_lock:
                                committed_nonce = committed_nonces.get(member_rank)
                                in_flight_nonce = in_flight.get(member_rank)
                                if identity_matches and committed_nonce == claimant_nonce:
                                    valid = True
                                    replay_committed = True
                                elif (
                                    identity_matches
                                    and committed_nonce is None
                                    and in_flight_nonce is None
                                ):
                                    valid = True
                                    in_flight[member_rank] = claimant_nonce
                                    reserved_rank = member_rank
                                elif (
                                    identity_matches
                                    and committed_nonce is None
                                    and in_flight_nonce == claimant_nonce
                                ):
                                    retryable = True
                            if not valid:
                                response: dict[str, Any] = {
                                    "protocol": protocol,
                                    "error": (
                                        "gang identity exchange is still in flight"
                                        if retryable
                                        else "contradictory gang identity"
                                    ),
                                    "retryable": retryable,
                                }
                            else:
                                response = {"protocol": protocol, **offered}
                        except (
                            OSError,
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ):
                            response = {
                                "protocol": protocol,
                                "error": "invalid gang identity request",
                            }
                        try:
                            connection.sendall(
                                json.dumps(response, sort_keys=True).encode("utf-8")
                                + b"\n"
                            )
                        except OSError:
                            # The member retries if the response was interrupted;
                            # it has not acknowledged this generation yet.
                            return
                        if not valid:
                            return
                        try:
                            acknowledgement = _recv_json_line(connection)
                        except (
                            OSError,
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ):
                            # ``sendall`` only proves the kernel accepted bytes,
                            # not that the member application received them. Keep
                            # the rank retryable until it acknowledges the exact
                            # response on the same authenticated connection. Each
                            # connection has its own daemon handler, so a hung peer
                            # cannot prevent another rank receiving the generation.
                            return
                        if not (
                            acknowledgement.get("protocol") == protocol
                            and acknowledgement.get("acknowledged") is True
                            and int(acknowledgement.get("rank", -1)) == member_rank
                            and acknowledgement.get("claimant_nonce")
                            == claimant_nonce
                        ):
                            return
                        with received_lock:
                            if replay_committed:
                                committed = (
                                    committed_nonces.get(member_rank)
                                    == claimant_nonce
                                )
                            else:
                                committed = bool(
                                    in_flight.get(member_rank) == claimant_nonce
                                    and member_rank not in committed_nonces
                                )
                                if committed:
                                    in_flight.pop(member_rank, None)
                                    committed_nonces[member_rank] = claimant_nonce
                                    reserved_rank = None
                        if not committed:
                            return
                        connection.sendall(
                            json.dumps(
                                {
                                    "protocol": protocol,
                                    "rank": member_rank,
                                    "claimant_nonce": claimant_nonce,
                                    "committed": True,
                                },
                                sort_keys=True,
                            ).encode("utf-8")
                            + b"\n"
                        )
                        try:
                            receipt = _recv_json_line(connection)
                        except (
                            OSError,
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ):
                            return
                        if (
                            receipt.get("protocol") == protocol
                            and receipt.get("commitment_received") is True
                            and int(receipt.get("rank", -1)) == member_rank
                            and receipt.get("claimant_nonce") == claimant_nonce
                        ):
                            with received_lock:
                                if (
                                    committed_nonces.get(member_rank)
                                    == claimant_nonce
                                ):
                                    delivered.add(member_rank)
                    finally:
                        if reserved_rank is not None:
                            with received_lock:
                                if in_flight.get(reserved_rank) == claimant_nonce:
                                    in_flight.pop(reserved_rank, None)

            try:
                server.settimeout(0.2)
                while True:
                    with received_lock:
                        if len(delivered) >= node_count - 1:
                            break
                    try:
                        connection, peer = server.accept()
                    except TimeoutError:
                        continue
                    threading.Thread(
                        target=handle,
                        args=(connection, peer),
                        name="npa-cosmos-gang-attempt-member",
                        daemon=True,
                    ).start()
            finally:
                server.close()

        threading.Thread(
            target=serve,
            name="npa-cosmos-gang-attempt",
            daemon=True,
        ).start()
        return offered

    claimant_nonce = secrets.token_hex(32)
    request = {
        "protocol": protocol,
        "rank": rank,
        "claimant_nonce": claimant_nonce,
        "node_count": node_count,
        "logical_wave_id": logical_wave_id,
        "membership_digest": membership_digest,
        "internal_job_id": internal_job_id,
    }
    started = time.monotonic()
    limit = _identity_timeout()
    last_report = -60.0
    while True:
        elapsed = time.monotonic() - started
        if elapsed - last_report >= 60.0:
            typer.echo(
                "multi-node augment identity rendezvous waiting: "
                f"rank={rank} leader={node_ips[0]} elapsed={elapsed:.1f}s "
                f"timeout={'disabled' if limit is None else f'{limit:g}s'}",
                err=True,
            )
            last_report = elapsed
        if limit is not None and elapsed >= limit:
            raise typer.BadParameter(
                "multi-node augment identity rendezvous timed out for rank "
                f"{rank} after {limit:g}s"
            )
        try:
            remaining = None if limit is None else max(0.1, limit - elapsed)
            connect_timeout = 5.0 if remaining is None else min(5.0, remaining)
            with socket.create_connection(
                (node_ips[0], port),
                timeout=connect_timeout,
                source_address=(node_ips[rank], 0),
            ) as connection:
                connection.sendall(
                    json.dumps(request, sort_keys=True).encode("utf-8") + b"\n"
                )
                response = _recv_json_line(connection)
                if response.get("protocol") != protocol:
                    raise typer.BadParameter(
                        "multi-node augment identity rendezvous returned an invalid "
                        "protocol"
                    )
                if response.get("error"):
                    if response.get("retryable") is True:
                        raise ConnectionError(str(response["error"]))
                    raise typer.BadParameter(
                        "multi-node augment identity is contradictory: "
                        f"{response['error']}"
                    )
                connection.sendall(
                    json.dumps(
                        {
                            "protocol": protocol,
                            "rank": rank,
                            "claimant_nonce": claimant_nonce,
                            "acknowledged": True,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                commitment = _recv_json_line(connection)
                if not (
                    commitment.get("protocol") == protocol
                    and commitment.get("committed") is True
                    and int(commitment.get("rank", -1)) == rank
                    and commitment.get("claimant_nonce") == claimant_nonce
                ):
                    raise typer.BadParameter(
                        "multi-node augment identity rendezvous did not commit the "
                        "rank reservation"
                    )
                connection.sendall(
                    json.dumps(
                        {
                            "protocol": protocol,
                            "rank": rank,
                            "claimant_nonce": claimant_nonce,
                            "commitment_received": True,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                return {
                    key: value for key, value in response.items() if key != "protocol"
                }
        except typer.BadParameter:
            raise
        except (OSError, ValueError, json.JSONDecodeError):
            remaining = None if limit is None else max(0.0, limit - elapsed)
            time.sleep(1.0 if remaining is None else min(1.0, remaining))


def _gang_identity(
    *,
    output_uri: str,
    run_id: str,
    storage_client: Any = None,
    rendezvous: Any = None,
) -> tuple[int, int, str, str, int, dict[str, Any]]:
    """Establish the gang's shared publication identity and durable fence."""

    rank, nodes, metadata = _gang_environment()
    if metadata.get("scheduler") == "single":
        return rank, nodes, "", "", 0, {}
    if metadata.get("scheduler") == "local":
        raise typer.BadParameter(
            "NPA workflow Cosmos publication requires SkyPilot managed-job "
            "identity and its rank-0 publication fence; local rank/count "
            "overrides are shard-planning aids only"
        )

    logical_wave_id = str(metadata["logical_wave_id"])
    membership_digest = str(metadata["membership_digest"])
    internal_job_id = str(metadata["internal_job_id"])
    scheduler_sequence = int(metadata["scheduler_fence_sequence"])
    scheduler_attempt = int(metadata["scheduler_fence_attempt"])
    offered: dict[str, Any] | None = None
    claim_etag = ""
    if rank == 0:
        from npa.workbench.cosmos.transfer import claim_run_publication

        attempt_id, claim_etag, generation = claim_run_publication(
            output_uri,
            run_id=run_id,
            logical_wave_id=logical_wave_id,
            node_count=nodes,
            membership_digest=membership_digest,
            scheduler_fence_sequence=scheduler_sequence,
            scheduler_fence_attempt=scheduler_attempt,
            scheduler_launch_id=internal_job_id,
            storage_client=storage_client,
        )
        offered = {
            "attempt_id": attempt_id,
            "publication_generation": generation,
            "logical_wave_id": logical_wave_id,
            "membership_digest": membership_digest,
            "internal_job_id": internal_job_id,
            "scheduler_fence_sequence": scheduler_sequence,
            "scheduler_fence_attempt": scheduler_attempt,
            "node_count": nodes,
        }
    if nodes == 1:
        shared = offered
    else:
        exchange = rendezvous or _sky_gang_rendezvous
        shared = exchange(
            rank=rank,
            node_count=nodes,
            node_ips=list(metadata["node_ips"]),
            logical_wave_id=logical_wave_id,
            membership_digest=membership_digest,
            internal_job_id=internal_job_id,
            offered=offered,
        )
    try:
        attempt_id = str(shared.get("attempt_id") or "").strip()
        generation = int(shared.get("publication_generation", 0))
    except (AttributeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "multi-node augment identity rendezvous returned an invalid generation"
        ) from exc
    if not (
        len(attempt_id) == 64
        and all(char in "0123456789abcdef" for char in attempt_id)
        and generation > 0
        and shared.get("logical_wave_id") == logical_wave_id
        and shared.get("membership_digest") == membership_digest
        and shared.get("internal_job_id") == internal_job_id
        and int(shared.get("scheduler_fence_sequence", 0)) == scheduler_sequence
        and int(shared.get("scheduler_fence_attempt", 0)) == scheduler_attempt
        and int(shared.get("node_count", 0)) == nodes
    ):
        raise typer.BadParameter(
            "multi-node augment identity rendezvous returned contradictory identity"
        )
    publication_identity = {
        "logical_wave_id": logical_wave_id,
        "scheduler_fence_sequence": scheduler_sequence,
        "scheduler_fence_attempt": scheduler_attempt,
        "scheduler_launch_id": internal_job_id,
    }
    return rank, nodes, attempt_id, claim_etag, generation, publication_identity


def _apply_validation_fault(
    *, run_id: str, rank: int, generation: int, phase: str
) -> None:
    """Apply an explicitly scoped live-validation delay or failure.

    These hooks are inert unless the caller repeats the exact run id in
    ``NPA_COSMOS_VALIDATION_SCOPE``. They let an operator exercise delayed-rank
    and managed-recovery behavior in a task-owned job without deleting pods or
    touching shared workloads.
    """

    names = {
        "NPA_COSMOS_VALIDATION_DELAY_S",
        "NPA_COSMOS_VALIDATION_FAIL_PHASE",
    }
    if not any(str(os.environ.get(name, "")).strip() for name in names):
        return
    scope = str(os.environ.get("NPA_COSMOS_VALIDATION_SCOPE", "")).strip()
    if not run_id or scope != run_id:
        raise typer.BadParameter(
            "Cosmos validation fault hooks require NPA_COSMOS_VALIDATION_SCOPE "
            "to equal the exact non-empty --run-id"
        )

    def selected(prefix: str) -> bool:
        raw_rank = str(os.environ.get(f"{prefix}_RANK", "")).strip()
        raw_generation = str(
            os.environ.get(f"{prefix}_GENERATION", "")
        ).strip()
        try:
            return bool(
                (not raw_rank or int(raw_rank) == rank)
                and (not raw_generation or int(raw_generation) == generation)
            )
        except ValueError as exc:
            raise typer.BadParameter(
                f"{prefix}_RANK and {prefix}_GENERATION must be integers"
            ) from exc

    delay_raw = str(os.environ.get("NPA_COSMOS_VALIDATION_DELAY_S", "")).strip()
    delay_phase = str(
        os.environ.get("NPA_COSMOS_VALIDATION_DELAY_PHASE", "before-render")
    ).strip()
    if delay_raw and delay_phase == phase and selected("NPA_COSMOS_VALIDATION_DELAY"):
        try:
            delay = float(delay_raw)
        except ValueError as exc:
            raise typer.BadParameter(
                "NPA_COSMOS_VALIDATION_DELAY_S must be a finite non-negative number"
            ) from exc
        if not math.isfinite(delay) or delay < 0:
            raise typer.BadParameter(
                "NPA_COSMOS_VALIDATION_DELAY_S must be a finite non-negative number"
            )
        typer.echo(
            "task-scoped Cosmos validation delay: "
            f"run={run_id} rank={rank} generation={generation} "
            f"phase={phase} delay={delay:g}s",
            err=True,
        )
        time.sleep(delay)

    fail_phase = str(
        os.environ.get("NPA_COSMOS_VALIDATION_FAIL_PHASE", "")
    ).strip()
    if fail_phase == phase and selected("NPA_COSMOS_VALIDATION_FAIL"):
        raise RuntimeError(
            "task-scoped Cosmos validation fault: "
            f"run={run_id} rank={rank} generation={generation} phase={phase}"
        )


def _shard_indices(count: int, *, rank: int, nodes: int) -> list[int]:
    """Variant indices this node renders, striding so the load stays balanced.

    Striding (rank, rank+nodes, ...) rather than contiguous blocks keeps every
    node within one variant of the others when the count does not divide evenly.
    """

    if nodes <= 1:
        return list(range(max(0, count)))
    return list(range(rank, max(0, count), nodes))


def _variant_parallelism(num_variants: int) -> int:
    """Resolve how many variant inferences to run concurrently (>=1).

    ``NPA_COSMOS_VARIANT_PARALLELISM`` overrides; otherwise auto-detect the GPU
    count. Capped at the number of variants so we never spawn idle workers.
    """
    override = os.environ.get("NPA_COSMOS_VARIANT_PARALLELISM", "").strip()
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 1
    else:
        requested = _detect_gpu_count()
    return max(1, min(requested, max(1, int(num_variants))))


def _inference_seed(combo: dict[str, Any]) -> int | None:
    """Return a validated deterministic seed from one sampled candidate."""

    raw = combo.get("inference_seed")
    if raw in (None, ""):
        return None
    if isinstance(raw, bool):
        raise typer.BadParameter("candidate inference_seed must be an integer")
    try:
        seed = int(raw)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "candidate inference_seed must be an integer"
        ) from exc
    if not 0 <= seed < 2**31:
        raise typer.BadParameter("candidate inference_seed must be within 0..2147483647")
    return seed


def _materialize_input_clip(src: str, *, allow_frame_sequence: bool = False) -> str:
    """Resolve a local path or ``s3://`` URI to a local conditioning video.

    Returns an empty string only when the source was successfully inspected and no
    supported input exists. In the PAIDF path only, ``allow_frame_sequence`` turns
    the captionable input frames into a temporary video. Storage setup, listing,
    authentication, download, and encoding failures propagate so the CLI can
    report them separately from an empty prefix.
    """
    import glob as _glob
    import shutil
    import tempfile
    from urllib.parse import urlsplit

    s = str(src or "").strip()
    if not s:
        return ""
    if not s.startswith("s3://"):
        return s if Path(s).is_file() else ""
    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    tmp = tempfile.mkdtemp(prefix="npa-cosmos-input-")
    keep_tmp = False
    try:
        source_path = urlsplit(s).path
        if source_path.lower().endswith(_VIDEO_EXTS):
            downloaded = client.download_path(s, str(Path(tmp) / Path(source_path).name))
            keep_tmp = True
            return downloaded
        client.download_directory(s, tmp)
        vids = sorted(
            f for f in _glob.glob(str(Path(tmp) / "**" / "*"), recursive=True)
            if f.lower().endswith(_VIDEO_EXTS) and Path(f).is_file()
        )
        if vids:
            keep_tmp = True
            # PAIDF prepares the exact normalized model input under this name.
            return next(
                (video for video in vids if Path(video).name == "conditioning.mp4"),
                vids[0],
            )
        if allow_frame_sequence:
            frames = sorted(
                Path(f)
                for f in _glob.glob(str(Path(tmp) / "**" / "*"), recursive=True)
                if f.lower().endswith(_IMAGE_EXTS) and Path(f).is_file()
            )
            clip = _frames_to_conditioning_clip(frames, Path(tmp))
            if clip:
                keep_tmp = True
                return clip
        return ""
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _materialize_control_asset(src: str, *, label: str) -> str:
    """Resolve a precomputed control/mask video to a local path.

    Unlike the conditioning input, an asset the operator explicitly named must
    exist: silently continuing would compute the control on-the-fly instead and
    the run would look like it honoured the asset.
    """

    value = str(src or "").strip()
    if not value:
        return ""
    if not value.lower().endswith(_VIDEO_EXTS):
        raise typer.BadParameter(
            f"{label} must be an mp4/video file, got: {value!r}"
        )
    if not value.startswith("s3://"):
        if not Path(value).is_file():
            raise typer.BadParameter(f"{label} does not exist: {value!r}")
        return value
    import atexit
    import shutil
    import tempfile
    from urllib.parse import urlsplit

    from npa.clients.storage import StorageClient

    tmp = tempfile.mkdtemp(prefix="npa-cosmos-control-")

    def cleanup() -> None:
        shutil.rmtree(tmp, ignore_errors=True)

    # The downloaded asset must remain available for every variant in this CLI
    # process, including concurrent renders. Remove it when the process exits;
    # also clean it immediately when the download itself fails.
    atexit.register(cleanup)
    name = Path(urlsplit(value).path).name or "control.mp4"
    try:
        return StorageClient.from_environment().download_path(value, str(Path(tmp) / name))
    except Exception as exc:  # noqa: BLE001 - sanitize storage failures
        cleanup()
        atexit.unregister(cleanup)
        raise typer.BadParameter(
            f"could not download {label} from {value!r}; verify the object-storage "
            "endpoint, credentials, permissions, and that the object exists"
        ) from exc


def _materialize_conditioning_input(
    src: str, *, allow_frame_sequence: bool = False
) -> str:
    """Adapt storage failures to a sanitized, actionable CLI error."""
    try:
        if allow_frame_sequence:
            return _materialize_input_clip(src, allow_frame_sequence=True)
        return _materialize_input_clip(src)
    except Exception as exc:
        raise typer.BadParameter(
            "could not inspect or download the configured conditioning input; "
            "verify the object-storage endpoint, credentials, permissions, and availability"
        ) from exc


def _persist_generated_conditioning_clip(
    local_input: str, input_uri: str, *, publish: bool = True
) -> str:
    """Persist PAIDF's frame-derived clip so evaluation uses the exact source.

    Operator-side preparation already persists ``conditioning.mp4``. The legacy
    fixture path still creates ``npa-paidf-conditioning.mp4`` in the worker and
    needs it published. In both cases return the canonical URI so evaluation
    records the exact clip Cosmos consumed.

    ``publish=False`` resolves the URI without writing: in a multi-node augment
    every node derives the same clip, so only one of them uploads it.
    """

    path = Path(str(local_input or ""))
    if not input_uri.startswith("s3://"):
        return ""
    uri = input_uri.rstrip("/") + "/conditioning.mp4"
    if path.name == "conditioning.mp4":
        return uri
    if path.name != "npa-paidf-conditioning.mp4":
        return ""
    if not publish:
        return uri
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment().upload_file(str(path), uri)


def _safe_paidf_cli_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate stage evidence without prompts, paths, IDs, or hashes."""

    segmentation = payload.get("segmentation")
    safe_segmentation: dict[str, Any] = {"mode": "off"}
    if isinstance(segmentation, dict) and segmentation.get("mode") != "off":
        safe_segmentation = {
            key: segmentation[key]
            for key in (
                "mode",
                "engine",
                "component",
                "component_version",
                "frame_count",
                "object_count",
                "mask_coverage",
                "runtime",
                "cache_status",
            )
            if key in segmentation
        }
    safe = {
        key: payload[key]
        for key in (
            "status",
            "mode",
            "output_kind",
            "frame_count",
            "variant_count",
            "attempted_variant_count",
            "failed_variant_count",
            "multiply_mode",
            "variant_parallelism",
            "node_count",
            "node_rank",
            "shard_variant_count",
            "shard_attempted_variant_count",
            "shard_failed_variant_count",
            "input_conditioned",
        )
        if key in payload
    }
    safe["segmentation"] = safe_segmentation
    return safe


@app.command("transfer")
def transfer_cmd(
    input_uri: str = typer.Option(..., "--input-uri", help="Input frames, assets, or rollout URI."),
    output_uri: str = typer.Option(..., "--output-uri", help="Output prefix for transferred frames."),
    assets_uri: str = typer.Option("", "--assets-uri", help="Optional sim asset source path."),
    scene_spec_uri: str = typer.Option("", "--scene-spec-uri", help="Optional SceneSpec path."),
    image: str = typer.Option("", "--image", help="BYO Cosmos2 transfer image."),
    run_id: str = typer.Option("", "--run-id", help="Run id carried into the manifest."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Write manifest JSON locally."),
    execute: bool = typer.Option(
        False,
        "--execute",
        help=(
            "Force the real Cosmos-Transfer2.5 model (requires the transfer image/GPU). "
            "Note: when that runtime is already present on the host the real model runs "
            "even without --execute; --execute only makes its absence a hard error "
            "instead of falling back to reference augmentation."
        ),
    ),
    spec: str = typer.Option(
        "", "--spec", help="controlnet_spec path (relative to the transfer repo) for --execute."
    ),
    configs_uri: str = typer.Option(
        "",
        "--configs-uri",
        help="Config-Gen manifest URI; the first sampled augmentation combo is "
        "recorded as the clip's appearance variables (drives the Rerun label).",
    ),
    input_video: str = typer.Option(
        "",
        "--input-video",
        help="Local path or s3:// URI of an input clip to CONDITION the augmentation "
        "on. When set (with --execute), the output is a real augmentation of THIS "
        "clip (edge control computed on-the-fly; prompt drives the new appearance).",
    ),
    condition_on_input: bool = typer.Option(
        False,
        "--condition-on-input",
        help="Condition on the first video under --input-uri. Also enabled by "
        "NPA_COSMOS_CONDITION_ON_INPUT=1.",
    ),
    control: str = typer.Option(
        "edge",
        "--control",
        help="Control modality for input-conditioning: edge, vis, depth, or seg. "
        "Edge/vis/seg can be derived from the input; depth requires an "
        "operator-owned precomputed control and never invokes Video Depth Anything.",
    ),
    control_weight: float = typer.Option(1.0, "--control-weight", help="Control weight for input-conditioning."),
    control_asset: str = typer.Option(
        "",
        "--control-asset",
        help="Local path or s3:// URI of a PRECOMPUTED control video (e.g. a "
        "segmentation map) to condition on instead of computing the modality "
        "on-the-fly.",
    ),
    control_prompt: str = typer.Option(
        "",
        "--control-prompt",
        help="Objects on-the-fly 'seg' should segment (e.g. 'robot arm, conveyor, "
        "bin'). Passed to GroundingDINO to seed SAM2 tracking; upstream defaults to "
        "the first 128 words of the appearance prompt when unset.",
    ),
    mask_asset: str = typer.Option(
        "",
        "--mask-asset",
        help="Local path or s3:// URI of a PRECOMPUTED binary spatiotemporal region "
        "mask. The control applies only where the mask is white. Mutually exclusive "
        "with --mask-prompt.",
    ),
    mask_prompt: str = typer.Option(
        "",
        "--mask-prompt",
        help="Objects SAM2 should segment into a region mask, restricting the control "
        "to those pixels (e.g. 'robot arm'). Mutually exclusive with --mask-asset.",
    ),
    control_output_uri: str = typer.Option(
        "",
        "--control-output-uri",
        help="s3:// prefix to publish the control map and region mask that "
        "conditioned each variant, as <prefix>/<clip>/control_<modality>.mp4 plus "
        "extracted frames. Sibling of --output-uri, never nested inside it.",
    ),
    guidance: float = typer.Option(3.0, "--guidance", help="Classifier-free guidance for input-conditioning."),
    refinement_uri: str = typer.Option(
        "",
        "--refinement-uri",
        help=(
            "Run-scoped adaptive-refinement JSON; its validated settings override "
            "both CLI and NPA_COSMOS_* control/guidance values."
        ),
    ),
    protected_chroma_mode: str = typer.Option(
        "off",
        "--protected-chroma-mode",
        help="Optional protected-region color policy: off or source-chroma.",
    ),
    protected_regions_json: str = typer.Option(
        "",
        "--protected-regions-json",
        help="JSON normalized rectangles used only when protected chroma mode is source-chroma.",
    ),
    protected_luma_max_delta: int = typer.Option(
        32,
        "--protected-luma-max-delta",
        help="Maximum per-pixel protected-region luma change from source (0..255).",
    ),
    protected_feather_pixels: int = typer.Option(
        12,
        "--protected-feather-pixels",
        help="Inward feather width for protected rectangle boundaries.",
    ),
    segmentation_mode: str = typer.Option(
        "off",
        "--segmentation-mode",
        help=(
            "Optional real protected-content segmentation: off or sam2-auto. "
            "SAM2 generates frame-aligned masks once, then reuses them across "
            "augmentation variants."
        ),
    ),
    segmentation_uri: str = typer.Option(
        "",
        "--segmentation-uri",
        help="Versioned S3 prefix for SAM2 masks and lineage evidence.",
    ),
    sam2_model: str = typer.Option(
        "facebook/sam2.1-hiera-tiny",
        "--sam2-model",
        help="Official upstream Meta SAM2 Hugging Face model id.",
    ),
    sam2_model_revision: str = typer.Option(
        "de431c4043854a71d8101e17995dfe596bf101a5",
        "--sam2-model-revision",
        help="Immutable Hugging Face revision for the SAM2 checkpoint.",
    ),
    sam2_points_per_side: int = typer.Option(
        16,
        "--sam2-points-per-side",
        help="SAM2 auto-mask sampling density (4..64).",
    ),
    sam2_predicted_iou_threshold: float = typer.Option(
        0.86,
        "--sam2-predicted-iou-threshold",
        help="SAM2 automatic-mask predicted-IoU threshold (0..1).",
    ),
    sam2_stability_threshold: float = typer.Option(
        0.92,
        "--sam2-stability-threshold",
        help="SAM2 automatic-mask stability threshold (0..1).",
    ),
    sam2_min_area_fraction: float = typer.Option(
        0.002,
        "--sam2-min-area-fraction",
        help="Minimum eligible automatic-mask area as a frame fraction.",
    ),
    sam2_max_area_fraction: float = typer.Option(
        0.65,
        "--sam2-max-area-fraction",
        help="Maximum eligible automatic-mask area as a frame fraction.",
    ),
    sam2_max_objects: int = typer.Option(
        6,
        "--sam2-max-objects",
        help="Maximum first-frame SAM2 objects to propagate (1..32).",
    ),
) -> None:
    """Build a transfer manifest; pass --execute for real vendor output.

    Mode is chosen by runtime availability, not just the flag: if the
    Cosmos-Transfer2.5 runtime is present (or ``--execute`` is passed) the real
    world-transfer model runs and publishes a video; otherwise a genuine
    reference augmentation writes real augmented image frames. Inspect
    ``output_kind`` in the manifest ("video" vs "frames") to disambiguate.
    """

    # Resolve every deterministic control knob before probing the runtime or
    # touching input/control storage.  The same import-light validator is used
    # by workflow validate/plan/submit.
    control = os.environ.get("NPA_COSMOS_CONTROL", "").strip() or control
    control_asset = (
        os.environ.get("NPA_COSMOS_CONTROL_ASSET", "").strip() or control_asset
    )
    control_prompt = (
        os.environ.get("NPA_COSMOS_CONTROL_PROMPT", "").strip() or control_prompt
    )
    mask_asset = os.environ.get("NPA_COSMOS_MASK_ASSET", "").strip() or mask_asset
    mask_prompt = os.environ.get("NPA_COSMOS_MASK_PROMPT", "").strip() or mask_prompt
    raw_control_weight = os.environ.get("NPA_COSMOS_CONTROL_WEIGHT", "").strip()
    raw_guidance = os.environ.get("NPA_COSMOS_GUIDANCE", "").strip()
    requested_control_weight: object = control_weight
    if raw_control_weight:
        requested_control_weight = raw_control_weight
    if raw_guidance:
        try:
            guidance = float(raw_guidance)
        except ValueError as exc:
            raise typer.BadParameter(
                "NPA_COSMOS_GUIDANCE must be a finite number"
            ) from exc
        if not math.isfinite(guidance):
            raise typer.BadParameter("NPA_COSMOS_GUIDANCE must be a finite number")
    from npa.workbench.cosmos.control_contract import (
        ControlContractError,
        validate_control_request,
    )

    try:
        checkpoint, normalized_weight = validate_control_request(
            modality=control,
            weight=requested_control_weight,
            control_asset=control_asset,
            control_prompt=control_prompt,
            mask_asset=mask_asset,
            mask_prompt=mask_prompt,
        )
    except ControlContractError as exc:
        raise typer.BadParameter(str(exc)) from exc
    control = checkpoint.modality
    control_weight = normalized_weight

    # The renderer sets the authoritative count even for one-node tasks.  Check
    # the complete scheduler identity and prove the shard-producing mode is
    # active before probing the model runtime or touching input/model storage.
    # Otherwise a reused multi-node profile would run the generic writer once
    # per worker against the same output.
    requested_nodes = 1
    if _gang_contract_required():
        _rank_hint, requested_nodes, _gang_metadata = _gang_environment()
    if requested_nodes > 1 and not configs_uri.strip():
        raise typer.BadParameter(
            "multi-node Cosmos transfer requires a non-empty --configs-uri so "
            "each worker executes a rank-local augmentation stride"
        )
    if requested_nodes > 1 and not output_uri.strip().startswith("s3://"):
        raise typer.BadParameter(
            "multi-node Cosmos transfer requires an s3:// --output-uri so "
            "workers can publish and join attempt-fenced shards"
        )

    payload = build_cosmos2_transfer_manifest(
        Cosmos2TransferConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            assets_uri=assets_uri,
            scene_spec_uri=scene_spec_uri,
            image=image,
            run_id=run_id,
        )
    )
    from npa.workbench.cosmos.transfer import (
        cosmos_transfer_available,
        reference_augment_frames,
        run_cosmos_transfer,
    )

    runtime_available = cosmos_transfer_available()
    if execute and not runtime_available:
        raise typer.BadParameter(
            "--execute needs the cosmos-transfer2.5 runtime "
            "(run inside the npa-cosmos2-transfer image on a GPU)."
        )

    if execute or runtime_available:
        # Real Cosmos-Transfer2.5 world-transfer model.
        #
        # Data Factory context (`transfer_execute` passes --configs-uri and always
        # enables input conditioning): the sampled appearance combo drives the prompt,
        # and the augment CONDITIONS on the run's real input clip (edge control
        # computed on-the-fly — a genuine augmentation of that footage),
        # and the result is published in the per-clip layout
        # that data_factory curate / build_run_rrd / provenance consume. Generic
        # callers opt in via --input-video, --condition-on-input, or
        # NPA_COSMOS_CONDITION_ON_INPUT=1.
        #
        # Otherwise (generic `transfer` for sim2real / cosmos-gate / fanout), publish
        # the generated video, flat extracted frames, and durable manifest together.
        condition_requested = bool(
            input_video or condition_on_input or _env_truthy("NPA_COSMOS_CONDITION_ON_INPUT")
        )
        data_factory_mode = bool(configs_uri)
        local_input = ""
        if condition_requested:
            local_input = _materialize_conditioning_input(
                input_video or input_uri,
                # PAIDF Config-Gen produces/captions image frames. If its input
                # prefix has no video, condition Cosmos on a temporary clip made
                # from those frames. Generic/standalone transfer remains strict.
                allow_frame_sequence=bool(configs_uri),
            )
            if not local_input:
                expected = (
                    "supported video or PAIDF PNG/JPEG input frames"
                    if configs_uri
                    else "supported video"
                )
                raise typer.BadParameter(
                    f"input conditioning was requested, but no {expected} "
                    f"({', '.join(_VIDEO_EXTS)}) was found at the configured input"
                )
        control_asset = _materialize_control_asset(control_asset, label="--control-asset")
        mask_asset = _materialize_control_asset(mask_asset, label="--mask-asset")
        refinement = _load_refinement(refinement_uri)
        if refinement:
            settings = refinement["settings"]
            control_weight = float(settings["control_weight"])
            guidance = float(settings["guidance"])
        protected_chroma_mode = protected_chroma_mode.strip().lower()
        segmentation_mode = segmentation_mode.strip().lower()
        from npa.workbench.cosmos.sam2_masks import Sam2MaskConfig, Sam2MaskError

        sam2_config = Sam2MaskConfig(
            mode=segmentation_mode,
            model_id=sam2_model,
            model_revision=sam2_model_revision,
            points_per_side=sam2_points_per_side,
            predicted_iou_threshold=sam2_predicted_iou_threshold,
            stability_threshold=sam2_stability_threshold,
            min_area_fraction=sam2_min_area_fraction,
            max_area_fraction=sam2_max_area_fraction,
            max_objects=sam2_max_objects,
        )
        if segmentation_mode != "off":
            try:
                sam2_config.validate()
            except Sam2MaskError as exc:
                raise typer.BadParameter(str(exc)) from exc
            if not segmentation_uri.startswith("s3://"):
                raise typer.BadParameter(
                    "SAM2 segmentation requires --segmentation-uri as an s3:// prefix"
                )
            if not run_id.strip():
                raise typer.BadParameter(
                    "SAM2 segmentation requires --run-id to bind mask reuse to one run"
                )
            if protected_regions_json:
                raise typer.BadParameter(
                    "SAM2 masks and --protected-regions-json are mutually exclusive"
                )
            if requested_nodes > 1:
                raise typer.BadParameter(
                    "SAM2 auto segmentation requires one augment node; use the "
                    "supported multi-GPU variant fan-out on that node"
                )
            # SAM2 masks protect their selected pixels with the same chroma
            # compositor as rectangular regions, but are frame aligned.
            protected_chroma_mode = "source-chroma"
        if protected_chroma_mode not in {"off", "source-chroma"}:
            raise typer.BadParameter(
                "--protected-chroma-mode must be off or source-chroma"
            )
        if (
            protected_chroma_mode == "source-chroma"
            and segmentation_mode == "off"
            and not protected_regions_json
        ):
            raise typer.BadParameter(
                "--protected-chroma-mode source-chroma requires --protected-regions-json"
            )
        if not 0 <= protected_luma_max_delta <= 255:
            raise typer.BadParameter("--protected-luma-max-delta must be within 0..255")
        if protected_feather_pixels < 1:
            raise typer.BadParameter("--protected-feather-pixels must be positive")

        sam2_temp: tempfile.TemporaryDirectory[str] | None = None
        sam2_masks_dir = ""
        segmentation_evidence: dict[str, Any] = {"mode": "off"}
        if segmentation_mode != "off":
            if not local_input:
                raise typer.BadParameter(
                    "SAM2 segmentation requires an input-conditioned source video"
                )
            from npa.clients.storage import StorageClient
            from npa.workbench.cosmos.sam2_masks import (
                generate_sam2_video_masks,
                load_published_sam2_masks,
                publish_sam2_masks,
            )

            sam2_temp = tempfile.TemporaryDirectory(prefix="npa-paidf-sam2-")
            try:
                sam2_storage = StorageClient.from_environment()
                sam2_result = load_published_sam2_masks(
                    segmentation_uri,
                    sam2_temp.name,
                    config=sam2_config,
                    run_id=run_id,
                    storage_client=sam2_storage,
                )
                segmentation_cache_status = "reused"
                if sam2_result is None:
                    sam2_result = generate_sam2_video_masks(
                        local_input,
                        sam2_temp.name,
                        config=sam2_config,
                        run_id=run_id,
                    )
                    published_segmentation = publish_sam2_masks(
                        sam2_result,
                        segmentation_uri,
                        storage_client=sam2_storage,
                    )
                    segmentation_cache_status = "generated"
                else:
                    published_segmentation = sam2_result.manifest
            except Exception as exc:  # noqa: BLE001 - sanitized fail-closed boundary
                sam2_temp.cleanup()
                detail = str(exc) if isinstance(exc, Sam2MaskError) else "runtime error"
                raise typer.BadParameter(
                    f"configured SAM2 segmentation failed closed: {detail}"
                ) from exc
            sam2_masks_dir = str(sam2_result.masks_dir)
            segmentation_evidence = {
                "mode": segmentation_mode,
                "engine": published_segmentation["engine"],
                "component": published_segmentation["component"],
                "component_version": published_segmentation["component_version"],
                "component_source": published_segmentation["component_source"],
                "component_revision": published_segmentation["component_revision"],
                "license": published_segmentation["license"],
                "config": published_segmentation["config"],
                "frame_count": published_segmentation["frame_count"],
                "object_count": published_segmentation["object_count"],
                "mask_coverage": published_segmentation["mask_coverage"],
                "runtime": published_segmentation["runtime"],
                "cache_status": segmentation_cache_status,
                "manifest_uri": published_segmentation["manifest_uri"],
                "masks_uri": published_segmentation["masks_uri"],
            }

        if data_factory_mode and output_uri.strip().startswith("s3://"):
            # Augment & MULTIPLY. Run one REAL Cosmos Transfer 2.5 inference per
            # sampled appearance combo (each with its own prompt), publishing each
            # as its own per-clip dir under the cosmos_augmented/ prefix, then write
            # a single run-level manifest.json listing them all. A config manifest
            # with N augmentations therefore yields N scenario variants (not one
            # image). The per-clip layout is what data_factory curate /
            # build_run_rrd / provenance consume.
            from npa.workbench.cosmos.transfer import (
                attempt_output_uri_for,
                merge_shard_manifests,
                preserve_source_chroma,
                publish_transfer_clip,
                publish_transfer_failure,
                write_run_manifest,
                write_shard_manifest,
            )

            combos = [
                _apply_refinement_prompt(combo, refinement)
                for combo in _all_augmentations(configs_uri)
            ]

            # Multi-node fan-out: this node renders only its stride of the sampled
            # combos. Variant indices stay GLOBAL, so clip names remain disjoint
            # across the gang and the merged manifest keeps the sampled order.
            (
                rank,
                node_count,
                attempt_id,
                publication_claim_etag,
                publication_generation,
                publication_identity,
            ) = _gang_identity(output_uri=output_uri, run_id=run_id)
            _apply_validation_fault(
                run_id=run_id,
                rank=rank,
                generation=publication_generation,
                phase="before-render",
            )
            shard = [(i, combos[i]) for i in _shard_indices(len(combos), rank=rank, nodes=node_count)]
            fenced_publication = bool(attempt_id)
            publish_output_uri = (
                attempt_output_uri_for(output_uri, attempt_id)
                if fenced_publication
                else output_uri
            )
            publish_control_output_uri = (
                attempt_output_uri_for(control_output_uri, attempt_id)
                if fenced_publication and control_output_uri
                else control_output_uri
            )

            conditioning_clip_uri = _persist_generated_conditioning_clip(
                local_input,
                input_uri,
                # One writer for a key the whole gang would otherwise race on.
                publish=rank == 0,
            )

            parallelism = _variant_parallelism(len(shard))

            def _render_variant(slot: int, i: int, combo: dict) -> dict:
                variant_run = f"{run_id}-v{i}" if run_id else f"v{i}"
                # Pin each concurrent variant to a distinct GPU so an N-GPU pod
                # runs N diffusions at once (sequential when parallelism == 1).
                # The device comes from the node-local slot, never the global
                # variant index: rank 1 of a gang must still start at GPU 0.
                device = str(slot % parallelism) if parallelism > 1 else None
                result = run_cosmos_transfer(
                    run_id=variant_run,
                    spec=spec or None,
                    prompt=str(combo.get("prompt") or "") or None,
                    input_video=local_input or None,
                    control=control,
                    control_weight=control_weight,
                    control_asset=control_asset,
                    control_prompt=control_prompt,
                    mask_asset=mask_asset,
                    mask_prompt=mask_prompt,
                    guidance=guidance,
                    seed=_inference_seed(combo),
                    cuda_visible_devices=device,
                    variant_tag=variant_run,
                )
                if protected_chroma_mode == "source-chroma":
                    result = preserve_source_chroma(
                        result,
                        source_video=local_input,
                        regions_json=protected_regions_json,
                        masks_dir=sam2_masks_dir,
                        segmentation=segmentation_evidence,
                        feather_pixels=protected_feather_pixels,
                        luma_max_delta=protected_luma_max_delta,
                    )
                result["conditioning_clip_uri"] = conditioning_clip_uri
                result["refinement"] = refinement
                result["effective_control_weight"] = control_weight
                result["effective_guidance"] = guidance
                return result

            # Fan the GPU-bound diffusions out across this pod's GPUs, then publish
            # sequentially in combo order (publish/S3 upload stays single-threaded).
            transfers: dict[int, dict] = {}
            failures: dict[int, dict] = {}

            def _record_failure(i: int, combo: dict, exc: Exception) -> None:
                # Vendor stderr can contain the effective prompt and input path,
                # so preserve only typed, actionable provenance in the durable
                # failure record. The private combo manifest already holds the
                # exact requested variables for a targeted retry.
                failures[i] = {
                    "run_id": run_id,
                    "variant_index": i,
                    "variables": combo,
                    "error_type": type(exc).__name__,
                    "failure_category": "inference-or-output",
                    "retryable": True,
                    "refinement": refinement,
                    "effective_control_weight": control_weight,
                    "effective_guidance": guidance,
                    "inference_seed": _inference_seed(combo),
                }

            if parallelism > 1 and len(shard) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=parallelism) as pool:
                    futures = {
                        pool.submit(_render_variant, slot, i, combo): i
                        for slot, (i, combo) in enumerate(shard)
                    }
                    for future in futures:
                        index = futures[future]
                        combo = combos[index]
                        try:
                            transfers[index] = future.result()
                        except Exception as exc:  # noqa: BLE001 - per-candidate boundary
                            _record_failure(index, combo, exc)
            else:
                for slot, (i, combo) in enumerate(shard):
                    try:
                        transfers[i] = _render_variant(slot, i, combo)
                    except Exception as exc:  # noqa: BLE001 - per-candidate boundary
                        _record_failure(i, combo, exc)

            published_failures = [
                publish_transfer_failure(
                    failures[i],
                    publish_output_uri,
                )
                for i in sorted(failures)
            ]

            clips: list[dict] = []
            for i, combo in shard:
                if i not in transfers:
                    continue
                clip_name = f"aug-{run_id}-{i}" if run_id else f"aug{i}"
                clips.append(
                    publish_transfer_clip(
                        transfers[i],
                        publish_output_uri,
                        run_id=run_id,
                        clip_name=clip_name,
                        variables=combo,
                        variant_index=i,
                        control_output_uri=publish_control_output_uri,
                        require_frames=True,
                    )
                )
            if node_count == 1 and not clips:
                raise RuntimeError(
                    "all Cosmos Transfer variants failed; typed attempt evidence "
                    "was preserved and no empty candidate batch was published"
                )
            if node_count > 1:
                # Each node publishes its own shard manifest; rank 0 joins them into
                # the single run manifest the downstream stages read. A worker's
                # payload describes its own shard -- it never claims the run's total.
                from npa.workbench.cosmos.transfer import build_run_manifest

                write_shard_manifest(
                    clips,
                    output_uri,
                    run_id=run_id,
                    rank=rank,
                    node_count=node_count,
                    variant_parallelism=parallelism,
                    variant_total=len(combos),
                    attempt_id=attempt_id,
                    scheduler_fence_sequence=int(
                        publication_identity["scheduler_fence_sequence"]
                    ),
                    scheduler_fence_attempt=int(
                        publication_identity["scheduler_fence_attempt"]
                    ),
                    scheduler_launch_id=str(
                        publication_identity["scheduler_launch_id"]
                    ),
                    logical_wave_id=str(publication_identity["logical_wave_id"]),
                    publication_generation=publication_generation,
                    failures=published_failures,
                )
                _apply_validation_fault(
                    run_id=run_id,
                    rank=rank,
                    generation=publication_generation,
                    phase="after-shard",
                )
                manifest = (
                    merge_shard_manifests(
                        output_uri,
                        run_id=run_id,
                        node_count=node_count,
                        attempt_id=attempt_id,
                        publication_claim_etag=publication_claim_etag,
                        publication_generation=publication_generation,
                    )
                    if rank == 0
                    else build_run_manifest(
                        clips,
                        run_id=run_id,
                        variant_parallelism=parallelism,
                        node_count=node_count,
                        attempt_id=attempt_id,
                        failures=published_failures,
                    )
                )
            else:
                publication_kwargs = (
                    {
                        "attempt_id": attempt_id,
                        "publication_claim_etag": publication_claim_etag,
                        "publication_generation": publication_generation,
                    }
                    if fenced_publication
                    else {}
                )
                manifest = write_run_manifest(
                    clips,
                    output_uri,
                    run_id=run_id,
                    variant_parallelism=parallelism,
                    failures=published_failures,
                    **publication_kwargs,
                )
            payload["status"] = TRANSFER_MANIFEST_STATUS
            payload["output_kind"] = "video"
            payload["mode"] = TRANSFER_MANIFEST_MODE
            payload["augmented_video_uri"] = manifest["augmented_video_uri"]
            payload["augmented_videos"] = manifest["augmented_videos"]
            payload["frame_count"] = manifest["frame_count"]
            payload["variant_count"] = manifest["variant_count"]
            payload["attempted_variant_count"] = int(
                manifest.get(
                    "attempted_variant_count",
                    int(manifest["variant_count"]) + len(published_failures),
                )
            )
            payload["failed_variant_count"] = int(
                manifest.get("failed_variant_count", len(published_failures))
            )
            payload["multiply_mode"] = manifest["multiply_mode"]
            payload["variant_parallelism"] = manifest["variant_parallelism"]
            payload["node_count"] = node_count
            payload["node_rank"] = rank
            payload["shard_variant_count"] = len(clips)
            payload["shard_attempted_variant_count"] = len(shard)
            payload["shard_failed_variant_count"] = len(published_failures)
            payload["clips"] = manifest["clips"]
            local_variables = [combo for _index, combo in shard]
            payload["augmentation_variables"] = local_variables
            local_prompts = [str(combo.get("prompt") or "") for combo in local_variables]
            payload["prompts"] = local_prompts
            # Retain the legacy singular field as the first prompt this worker
            # actually executed; an empty stride reports no prompt.
            payload["prompt"] = local_prompts[0] if local_prompts else ""
            payload["attempt_id"] = attempt_id
            payload["input_conditioned"] = bool(local_input)
            payload["conditioning_clip_uri"] = manifest.get("conditioning_clip_uri", "")
            payload["control_spec"] = manifest["control_spec"]
            payload["control_weight"] = manifest["control_weight"]
            payload["control_prompt"] = manifest["control_prompt"]
            payload["mask_prompt"] = manifest["mask_prompt"]
            payload["control_uris"] = manifest["control_uris"]
            payload["segmentation"] = segmentation_evidence
            if control_output_uri:
                payload["control_output_uri"] = control_output_uri
            if local_input:
                payload["input_video"] = local_input
                payload["control"] = manifest["control"]
            # attribute-verify reads --input-path {{augmented_frames_uri}} (the prefix).
            payload["augmented_frames_uri"] = output_uri
            # Keep the temporary directory alive until every variant has consumed
            # its frame-aligned masks, then remove all source-derived local data.
            if sam2_temp is not None:
                sam2_temp.cleanup()
        else:
            # Single inference: generic transfer (sim2real / cosmos-gate / fanout)
            # or a non-S3 output. Unchanged field convention.
            variables = _first_augmentation(configs_uri) if configs_uri else {}
            transfer = run_cosmos_transfer(
                run_id=run_id,
                spec=spec or None,
                prompt=str(variables.get("prompt") or "") or None,
                input_video=local_input or None,
                control=control,
                control_weight=control_weight,
                control_asset=control_asset,
                control_prompt=control_prompt,
                mask_asset=mask_asset,
                mask_prompt=mask_prompt,
                guidance=guidance,
                seed=_inference_seed(variables),
            )
            if protected_chroma_mode == "source-chroma":
                from npa.workbench.cosmos.transfer import preserve_source_chroma

                transfer = preserve_source_chroma(
                    transfer,
                    source_video=local_input,
                    regions_json=protected_regions_json,
                    masks_dir=sam2_masks_dir,
                    segmentation=segmentation_evidence,
                    feather_pixels=protected_feather_pixels,
                    luma_max_delta=protected_luma_max_delta,
                )
            transfer["refinement"] = refinement
            transfer["effective_control_weight"] = control_weight
            transfer["effective_guidance"] = guidance
            payload["status"] = TRANSFER_MANIFEST_STATUS
            payload["output_kind"] = "video"
            payload["output_video"] = transfer["video_path"]
            payload["video_bytes"] = transfer["video_bytes"]
            payload["control_spec"] = transfer["spec"]
            payload["prompt"] = str(variables.get("prompt") or "")
            payload["input_conditioned"] = bool(local_input)
            payload["segmentation"] = segmentation_evidence
            if local_input:
                payload["input_video"] = local_input
                payload["control"] = transfer.get("control", control)
            if output_uri.strip().startswith("s3://"):
                # Generic single-video publish + sim2real-engine field convention.
                # Frame objects are deliberately flat under output_uri because envgen
                # constructs exactly <augment_uri>/frame-NNNNN.png references.
                from npa.workbench.cosmos.transfer import publish_transfer_to_s3

                manifest = publish_transfer_to_s3(
                    transfer,
                    output_uri,
                    run_id=run_id,
                    variables=variables,
                    frames_output_uri=output_uri,
                    control_output_uri=control_output_uri,
                    require_frames=True,
                )
                payload["mode"] = TRANSFER_MANIFEST_MODE
                payload["output_video"] = manifest["augmented_video_uri"]
                payload["augmented_video_uri"] = manifest["augmented_video_uri"]
                payload["augmented_frames_uri"] = manifest["augmented_frames_uri"]
                payload["frame_count"] = manifest["frame_count"]
                payload["control_uris"] = manifest.get("control_uris", {})
                payload["manifest_uri"] = transfer_manifest_uri_for(output_uri)
            else:
                payload["mode"] = TRANSFER_MANIFEST_MODE
                payload["augmented_video_uri"] = transfer["video_path"]
                payload["augmented_frames_uri"] = output_uri
            if sam2_temp is not None:
                sam2_temp.cleanup()
    else:
        # No heavy model runtime: run a genuine reference augmentation that
        # writes real augmented image frames to output_uri (not a descriptor stub).
        augment = reference_augment_frames(input_uri, output_uri, run_id=run_id)
        payload["status"] = REFERENCE_AUGMENT_STATUS
        payload["mode"] = REFERENCE_AUGMENT_MODE
        payload["output_kind"] = "frames"
        payload["augmented_frames_uri"] = augment["augmented_frames_uri"]
        payload["frames"] = augment["frames"]
        payload["frame_count"] = augment["frame_count"]
        payload["index_uri"] = augment["index_uri"]
        payload["manifest_uri"] = transfer_manifest_uri_for(output_uri)
        _publish_output_manifest(payload, output_uri)

    if output_json is not None:
        payload = write_manifest(payload, output_json)
    cli_payload = _safe_paidf_cli_result(payload) if configs_uri else payload
    typer.echo(json.dumps(cli_payload, indent=2, sort_keys=True))
