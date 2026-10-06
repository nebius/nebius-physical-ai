"""Recover only immutable, fully verified same-run native Cosmos 3 variants."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.paidf_cosmos3_media import video_sha256, verify_pair

_BATCH_SCHEMA = "npa.paidf.cosmos3.recovery-batch.v1"
_VARIANT_SCHEMA = "npa.paidf.cosmos3.recovery-variant.v1"
_IMAGE = re.compile(r"([^@\s]+)@sha256:([0-9a-f]{64})\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class VariantRecoveryError(ValueError):
    """Reject incomplete, changed, corrupted, or ambiguous recovery evidence.

    Args:
        args: Standard exception details.
    Returns:
        An exception describing an invalid recovery contract.
    Raises:
        None.
    """


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _source_binding() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): video_sha256(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    }


def _image_binding(value: str) -> str:
    match = _IMAGE.fullmatch(value)
    if match is None:
        raise VariantRecoveryError(
            "variant recovery requires an immutable runtime image"
        )
    repository = match[1]
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    return repository + "@sha256:" + match[2]


def _hardware_binding(environ: Mapping[str, str]) -> list[str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command, env=dict(environ), capture_output=True, text=True, check=False
    )
    rows = sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )
    if completed.returncode or not rows:
        raise VariantRecoveryError(
            "variant recovery requires measured GPU runtime identity"
        )
    return rows


def _read_document(storage: Any, uri: str) -> dict[str, Any] | None:
    current = storage.read_bytes_with_etag(uri)
    if current is None:
        return None
    try:
        value = json.loads(current[0])
    except (ValueError, UnicodeDecodeError) as error:
        raise VariantRecoveryError("recovery receipt is not readable JSON") from error
    if not isinstance(value, dict):
        raise VariantRecoveryError("recovery receipt is not a JSON object")
    return value


def _create_document(
    storage: Any, uri: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        storage.put_bytes_conditional(
            _json_bytes(value), uri, if_none_match=True, content_type="application/json"
        )
    except StoragePreconditionFailed:
        pass
    current = _read_document(storage, uri)
    if current is None:
        raise VariantRecoveryError("conditional recovery publication has no readback")
    return current


def _snapshot_command(
    mode: str, revision: str, environ: Mapping[str, str]
) -> dict[str, Any]:
    from npa.workbench.cosmos.generate import cosmos3_repo

    repo = cosmos3_repo(environ)
    env = dict(environ)
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source, env.get("PYTHONPATH", "")))
    )
    with tempfile.TemporaryDirectory(prefix="npa-paidf-model-receipt-") as tmp:
        receipt = Path(tmp) / "snapshot.json"
        command = [
            str(repo / ".venv/bin/python"),
            "-m",
            "npa.workbench.cosmos.nano_checkpoint_snapshot",
            "--mode",
            mode,
            "--receipt-path",
            str(receipt),
        ]
        if revision:
            command.extend(["--revision", revision])
        completed = subprocess.run(
            command, cwd=repo, env=env, capture_output=True, check=False
        )
        if completed.returncode or not receipt.is_file():
            detail = _snapshot_failure_detail(receipt, completed.returncode)
            raise VariantRecoveryError(
                f"immutable Nano snapshot {mode} failed: {detail}"
            )
        return json.loads(receipt.read_bytes())


def _snapshot_failure_detail(receipt: Path, returncode: int) -> str:
    detail = f"native exit code {returncode}"
    if not receipt.is_file():
        return detail + "; no receipt"
    document = json.loads(receipt.read_bytes())
    if document.get("schema") == "npa.cosmos3.nano-failure.v1":
        # This reason comes only from this module's literal validation errors;
        # subprocess stderr may contain authenticated URLs or private paths.
        return detail + "; " + document["reason"]
    return detail + "; no validation failure receipt"


class ImmutableVariantPublication:
    """Atomically publish variant objects and retain their complete byte inventory.

    Args:
        storage: The existing object-storage client with conditional writes.
        prefix: Exact immutable directory for this generated video.
    Returns:
        A publication adapter accepted by the ordinary PAIDF variant publisher.
    Raises:
        None; errors arise when an object is published.
    """

    def __init__(self, storage: Any, prefix: str):
        self.storage = storage
        self.prefix = prefix
        self.objects: dict[str, dict[str, Any]] = {}

    def upload_file(self, source: str, uri: str) -> str:
        """Create an immutable object and verify its complete byte readback.

        Args:
            source: The actual generated media or evidence file.
            uri: Destination confined to this publication directory.
        Returns:
            The verified destination URI.
        Raises:
            VariantRecoveryError: Bytes changed or the destination escaped its scope.
            OSError: The source cannot be read.
        """
        relative = _relative_object(uri, self.prefix)
        payload = Path(source).read_bytes()
        try:
            self.storage.put_bytes_conditional(payload, uri, if_none_match=True)
        except StoragePreconditionFailed:
            pass
        current = self.storage.read_bytes_with_etag(uri)
        if current is None or current[0] != payload:
            raise VariantRecoveryError("immutable variant object failed byte readback")
        self.objects[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        return uri


def _relative_object(uri: str, prefix: str) -> str:
    if not uri.startswith(prefix):
        raise VariantRecoveryError("variant object is outside its immutable directory")
    relative = uri[len(prefix) :]
    parts = PurePosixPath(relative).parts
    if not relative or relative.startswith("/") or ".." in parts or "\\" in relative:
        raise VariantRecoveryError("variant object has an invalid relative path")
    return relative


def _verify_native_result(result: Mapping[str, Any], batch: Mapping[str, Any]) -> None:
    from npa.workbench.cosmos.generate import _require_effective_guardrails
    from npa.workbench.cosmos.guarded_inference import GUARDRAIL_STATE_SCHEMA

    selection = result.get("native_model_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("schema") != "npa.cosmos3.native-model-selection.v1"
        or selection.get("model_binding") != batch["model_binding"]
    ):
        raise VariantRecoveryError(
            "native model selection does not match the immutable snapshot"
        )
    config = selection.get("effective_config")
    if not isinstance(config, dict) or selection.get(
        "effective_config_sha256"
    ) != _digest(config):
        raise VariantRecoveryError("native effective model configuration is not bound")
    guard = result.get("guardrail_state")
    if (
        not isinstance(guard, dict)
        or guard.get("schema") != GUARDRAIL_STATE_SCHEMA
        or guard.get("requested") is not True
    ):
        raise VariantRecoveryError("native guardrail execution receipt is absent")
    _require_effective_guardrails(guard)


def _verify_sample(
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    frames: int,
    source_sha256: str,
) -> None:
    args = (result.get("sample_outputs") or {}).get("args")
    if (result.get("sample_outputs") or {}).get("status") != "success":
        raise VariantRecoveryError("native sample did not report successful generation")
    transfer = result.get("structural_transfer")
    if not isinstance(args, dict) or not isinstance(transfer, dict):
        raise VariantRecoveryError("native sample and control evidence are absent")
    expected = {**request["sample"], "max_frames": frames}
    if any(args.get(key) != value for key, value in expected.items()):
        raise VariantRecoveryError(
            "native sample differs from the bound generation request"
        )
    flags = (
        "control_loader_verified",
        "text_guardrail_passed",
        "video_guardrail_passed",
        "guardrail_postprocessing_applied",
    )
    if (
        any(transfer.get(key) is not True for key in flags)
        or transfer.get("source_frames") != frames
    ):
        raise VariantRecoveryError(
            "native structural or guardrail evidence is incomplete"
        )
    if transfer.get("source_video_sha256") != source_sha256:
        raise VariantRecoveryError("native sample used different source video bytes")
    if (
        not isinstance(args.get("edge"), dict)
        or args["edge"].get("preset_edge_threshold") != request["edge_threshold"]
    ):
        raise VariantRecoveryError("native Canny preset differs from the bound request")
    _verify_rgb_control(transfer, request)
    _verify_control_sampling(transfer, request)


def _verify_rgb_control(transfer, request):
    if not request["rgb_weight"]:
        if transfer.get("rgb_conditioning") is not None:
            raise VariantRecoveryError("native RGB conditioning was not requested")
        return
    rgb = transfer.get("rgb_conditioning")
    if not isinstance(rgb, dict) or rgb.get("control_loader_verified") is not True:
        raise VariantRecoveryError("native RGB control evidence is incomplete")
    if rgb.get("weight") != request["rgb_weight"] or rgb.get("preset") != "none":
        raise VariantRecoveryError(
            "native RGB weight or preset differs from the bound request"
        )
    if not _DIGEST.fullmatch(str(rgb.get("control_pixels_sha256", ""))):
        raise VariantRecoveryError("native RGB pixels are not hash bound")


def _verify_control_sampling(transfer, request):
    sample = request["sample"]
    expected = {
        "output_fps": sample["fps"],
        "native_torch_compile": False,
        "first_chunk_conditional_frames": sample["num_first_chunk_conditional_frames"],
        "overlap_conditional_frames": sample["num_conditional_frames"],
        "normalize_cfg": sample["normalize_cfg"],
    }
    if any(transfer.get(key) != value for key, value in expected.items()):
        raise VariantRecoveryError(
            "native transfer sampling differs from the bound request"
        )
    prompts = transfer.get("effective_prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(value, str) and value for value in prompts)
        or transfer.get("native_chunks") != len(prompts)
    ):
        raise VariantRecoveryError(
            "native transfer has no complete guarded prompt receipt"
        )
    if not _DIGEST.fullmatch(str(transfer.get("control_pixels_sha256", ""))):
        raise VariantRecoveryError("native edge pixels are not hash bound")


class VariantRecovery:
    """Bind one PAIDF attempt and recover only its fully verified native variants.

    Args:
        storage: Object-storage client supporting conditional creation.
        output_uri: The same run's ordinary augment directory.
        attempt: Existing refinement attempt number, unchanged by worker recovery.
        identity: Exact input, prompts, configurations and sampling contract.
        environ: Actual worker environment, including immutable NPA_TASK_IMAGE.
        snapshot_command: Optional hermetic seam for native snapshot resolution.
    Returns:
        A coordinator which never creates the canonical batch manifest itself.
    Raises:
        VariantRecoveryError: Same-run evidence changed or cannot be verified.
    """

    def __init__(
        self,
        *,
        storage: Any,
        output_uri: str,
        attempt: int,
        identity: Mapping[str, Any],
        environ: Mapping[str, str],
        snapshot_command: Callable[..., dict[str, Any]] | None = None,
    ):
        self.storage, self.output_uri = storage, output_uri.rstrip("/") + "/"
        self.prefix = self.output_uri + f"_generation-recovery/attempt-{attempt:02d}/"
        self.environ = dict(environ)
        bound = {
            **identity,
            "runtime_image": _image_binding(environ.get("NPA_TASK_IMAGE", "")),
            "npa_sources": _source_binding(),
            "gpu_runtime": _hardware_binding(environ),
        }
        if (identity.get("input_provenance") or {}).get("sha256") != identity.get(
            "input_sha256"
        ):
            raise VariantRecoveryError(
                "prepared-source bytes differ from their provenance"
            )
        self.batch = self._bind_batch(bound, snapshot_command or _snapshot_command)
        self.batch_sha256 = _digest(self.batch)

    def _bind_batch(self, identity, snapshot_command):
        uri = self.prefix + "batch.json"
        current = _read_document(self.storage, uri)
        if current is not None:
            _verify_batch_identity(current, identity)
            revision = current["model_binding"]["revision"]
        else:
            revision = snapshot_command("resolve", "", self.environ)["revision"]
        materialized = snapshot_command("materialize", revision, self.environ)
        proposed = {
            "schema": _BATCH_SCHEMA,
            "identity": identity,
            "model_binding": materialized["model_binding"],
        }
        winner = current or _create_document(self.storage, uri, proposed)
        if winner != proposed:
            raise VariantRecoveryError("immutable batch model or request changed")
        return winner

    def generation_environ(self) -> dict[str, str]:
        """Return the pinned model handoff without changing inference controls.

        Args:
            None.
        Returns:
            An environment containing the batch's full Nano revision.
        Raises:
            None.
        """
        return {
            **self.environ,
            "NPA_COSMOS3_NANO_REVISION": self.batch["model_binding"]["revision"],
        }

    @classmethod
    def for_audit(
        cls, storage: Any, output_uri: str, attempt: int, batch: Mapping[str, Any]
    ) -> VariantRecovery:
        """Read existing receipts without resolving models, creating objects or inferring.

        Args:
            storage: A read-only exact-object storage client.
            output_uri: Existing run's augment directory.
            attempt: Existing refinement attempt number.
            batch: Actual durable batch receipt.
        Returns:
            An uninitialized coordinator for reading and verifying existing receipts.
        Raises:
            VariantRecoveryError: The stored batch identity is malformed.
        """
        identity = batch.get("identity")
        if not isinstance(identity, dict):
            raise VariantRecoveryError("stored batch has no source identity")
        _verify_batch_identity(batch, identity)
        instance = cls.__new__(cls)
        instance.storage = storage
        instance.output_uri = output_uri.rstrip("/") + "/"
        instance.prefix = (
            instance.output_uri + f"_generation-recovery/attempt-{attempt:02d}/"
        )
        instance.batch = dict(batch)
        instance.batch_sha256 = _digest(batch)
        return instance

    def recover(
        self, index: int, request: Mapping[str, Any], source: Path
    ) -> dict[str, Any] | None:
        """Verify all completed objects and the full video before skipping work.

        Args:
            index: The original variant position.
            request: Exact effective prompt, seed and native sample controls.
            source: Actual prepared source for full temporal alignment verification.
        Returns:
            The ordinary variant descriptor, or None when no new receipt exists.
        Raises:
            VariantRecoveryError: Existing evidence is changed, missing or corrupted.
        """
        receipt = _read_document(self.storage, self._receipt_uri(index))
        if receipt is None:
            return None
        return self._verify_receipt(receipt, index, request, source)

    def publication(self, index: int, artifact: Path) -> ImmutableVariantPublication:
        """Create a publisher scoped to actual generated bytes and this batch.

        Args:
            index: Original variant position.
            artifact: Actual native generated video.
        Returns:
            An immutable object publisher used by the existing publish path.
        Raises:
            OSError: Actual video bytes cannot be read.
        """
        prefix = (
            self.output_uri
            + f"variant-{index:04d}/_native/"
            + self.batch_sha256
            + "/"
            + video_sha256(artifact)
            + "/"
        )
        return ImmutableVariantPublication(self.storage, prefix)

    def complete(
        self,
        index: int,
        request: Mapping[str, Any],
        source: Path,
        result: Mapping[str, Any],
        variant: Mapping[str, Any],
        publication: ImmutableVariantPublication,
    ) -> dict[str, Any]:
        """Commit completion only after native proof and every object readback.

        Args:
            index: Original variant position.
            request: Exact effective native request.
            source: Actual prepared source.
            result: Actual guarded generator result.
            variant: Ordinary published variant descriptor.
            publication: Publisher containing all successfully read-back objects.
        Returns:
            The verified winning variant, including when another worker won.
        Raises:
            VariantRecoveryError: Native proof, publication or winning receipt is invalid.
        """
        receipt = _completion_receipt(
            index, self.batch_sha256, request, result, variant, publication
        )
        _verify_native_result(result, self.batch)
        _verify_sample(
            result,
            request,
            result["temporal_alignment"]["decoded_frames"],
            self.batch["identity"]["input_sha256"],
        )
        verified = self._verify_receipt(receipt, index, request, source)
        winner = _create_document(self.storage, self._receipt_uri(index), receipt)
        if winner == receipt:
            return verified
        return self._verify_receipt(winner, index, request, source)

    def _receipt_uri(self, index):
        return self.prefix + f"variant-{index:04d}.json"

    def _verify_receipt(self, receipt, index, request, source):
        _verify_completion_identity(receipt, index, request, self.batch_sha256)
        prefix = _verified_prefix(receipt, self.output_uri, index, self.batch_sha256)
        native = receipt.get("native")
        if not isinstance(native, dict):
            raise VariantRecoveryError("completed variant native evidence is absent")
        _verify_native_result(native, self.batch)
        objects = self._verify_objects(receipt, prefix, request)
        video = objects["augmented_video.mp4"]
        with tempfile.TemporaryDirectory(prefix="npa-paidf-recovered-video-") as tmp:
            path = Path(tmp) / "augmented.mp4"
            path.write_bytes(video)
            aligned = verify_pair(source, path, request["sample"]["fps"])
        _verify_sample(
            native,
            request,
            aligned["decoded_frames"],
            self.batch["identity"]["input_sha256"],
        )
        _verify_descriptor(receipt["variant"], request, prefix, video, aligned)
        _verify_publication_semantics(objects, receipt, self.batch, aligned)
        return dict(receipt["variant"])

    def _verify_objects(self, receipt, prefix, request):
        inventory = receipt.get("objects")
        required = {
            "augmented_video.mp4",
            "metadata.json",
            "transfer.json",
            "source_edges.mkv",
            "native_execution.json",
        }
        if request["rgb_weight"]:
            required.add("source_rgb.mkv")
        if not isinstance(inventory, dict) or not required <= inventory.keys():
            raise VariantRecoveryError(
                "completed variant object inventory is incomplete"
            )
        _verify_frame_sequence(inventory, receipt["variant"].get("frame_count"))
        objects = {}
        for relative, evidence in inventory.items():
            uri = prefix + relative
            _relative_object(uri, prefix)
            current = self.storage.read_bytes_with_etag(uri)
            if current is None or not isinstance(evidence, dict):
                raise VariantRecoveryError("completed variant object is absent")
            payload = current[0]
            if evidence != {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }:
                raise VariantRecoveryError("completed variant object bytes changed")
            objects[relative] = payload
        return objects


def _verify_completion_identity(receipt, index, request, batch_sha256):
    expected = {
        "schema": _VARIANT_SCHEMA,
        "batch_sha256": batch_sha256,
        "index": index,
        "request": request,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise VariantRecoveryError("completed variant belongs to a different request")
    variant = receipt.get("variant")
    if not isinstance(variant, dict) or variant.get("clip") != f"variant-{index:04d}":
        raise VariantRecoveryError("completed variant identity is invalid")


def _completion_receipt(index, batch_sha256, request, result, variant, publication):
    return {
        "schema": _VARIANT_SCHEMA,
        "batch_sha256": batch_sha256,
        "request": request,
        "index": index,
        "prefix": publication.prefix,
        "objects": publication.objects,
        "variant": dict(variant),
        "native": _native_receipt(result),
    }


def _verified_prefix(receipt, output_uri, index, batch_sha256):
    prefix = str(receipt.get("prefix") or "")
    root = output_uri + f"variant-{index:04d}/_native/{batch_sha256}/"
    if (
        not prefix.startswith(root)
        or not prefix.endswith("/")
        or not _DIGEST.fullmatch(prefix[len(root) : -1])
    ):
        raise VariantRecoveryError("completed variant publication directory is invalid")
    return prefix


def _verify_frame_sequence(inventory, count):
    if type(count) is not int or not 1 <= count <= 8:
        raise VariantRecoveryError("completed variant frame count is invalid")
    actual = {name for name in inventory if name.startswith("frame-")}
    expected = {f"frame-{index:05d}.png" for index in range(1, count + 1)}
    if actual != expected:
        raise VariantRecoveryError("completed variant frame sequence is incomplete")


def _verify_descriptor(variant, request, prefix, video, aligned):
    expected = {key: request["sample"][key] for key in ("seed", "guidance")}
    expected.update(
        {
            "steps": request["sample"]["num_steps"],
            "variables": request["variables"],
            "video_bytes": len(video),
            "temporal_alignment": aligned,
            "augmented_video_uri": prefix + "augmented_video.mp4",
            "frame_count": min(8, aligned["decoded_frames"]),
        }
    )
    if any(variant.get(key) != value for key, value in expected.items()):
        raise VariantRecoveryError(
            "completed variant descriptor differs from its native request"
        )
    digest = hashlib.sha256(video).hexdigest()
    if prefix.rsplit("/", 2)[-2] != digest:
        raise VariantRecoveryError(
            "completed variant directory does not bind the actual video bytes"
        )


def _verify_publication_semantics(objects, receipt, batch, aligned):
    native = receipt["native"]
    if json.loads(objects["native_execution.json"]) != native:
        raise VariantRecoveryError(
            "native execution object differs from its completion receipt"
        )
    transfer = native["structural_transfer"]
    if transfer.get("guarded_output_sha256") != aligned["generated_sha256"]:
        raise VariantRecoveryError(
            "published video differs from native guarded output bytes"
        )
    if hashlib.sha256(objects["source_edges.mkv"]).hexdigest() != transfer.get(
        "control_sha256"
    ):
        raise VariantRecoveryError(
            "published edge control differs from native loader evidence"
        )
    if receipt["request"]["rgb_weight"]:
        rgb = transfer.get("rgb_conditioning") or {}
        if hashlib.sha256(objects["source_rgb.mkv"]).hexdigest() != rgb.get(
            "control_sha256"
        ):
            raise VariantRecoveryError(
                "published RGB control differs from native loader evidence"
            )
    _verify_transfer_document(
        json.loads(objects["transfer.json"]), transfer, receipt["prefix"]
    )
    _verify_metadata_document(
        json.loads(objects["metadata.json"]), receipt, batch, aligned
    )


def _verify_transfer_document(document, native, prefix):
    expected = {key: value for key, value in native.items() if key != "control_path"}
    expected["control_uri"] = prefix + "source_edges.mkv"
    if native.get("rgb_conditioning"):
        expected["rgb_conditioning"] = {
            key: value
            for key, value in native["rgb_conditioning"].items()
            if key != "control_path"
        }
        expected["rgb_conditioning"]["control_uri"] = prefix + "source_rgb.mkv"
    if document != expected:
        raise VariantRecoveryError(
            "published transfer document differs from native control evidence"
        )


def _verify_metadata_document(document, receipt, batch, aligned):
    sample, variant, prefix = (
        receipt["request"]["sample"],
        receipt["variant"],
        receipt["prefix"],
    )
    expected = {
        "schema": "npa.paidf.cosmos3.augment.v1",
        "status": "executed",
        "engine": "nvidia-cosmos/cosmos-framework",
        "mode": "video2video",
        "clip": variant["clip"],
        "variables": receipt["request"]["variables"],
        "prompt": sample["prompt"],
        "model": "Cosmos3-Nano",
        "seed": sample["seed"],
        "guidance": sample["guidance"],
        "steps": sample["num_steps"],
        "guardrails": True,
        "weights_baked": False,
        "input_conditioned": True,
        "input_conditioning": "source-video",
        "conditioned_input": "source.mp4",
        "attempt": batch["identity"]["attempt"],
        "video_bytes": variant["video_bytes"],
        "frame_count": variant["frame_count"],
        "temporal_alignment": aligned,
        "structural_control": "edge",
        "published_video_sha256": aligned["generated_sha256"],
        "transfer_uri": prefix + "transfer.json",
        "native_execution_uri": prefix + "native_execution.json",
        "lineage": {"input_provenance_uri": batch["identity"]["input_provenance_uri"]},
    }
    if not isinstance(document, dict) or any(
        document.get(key) != value for key, value in expected.items()
    ):
        raise VariantRecoveryError(
            "published metadata differs from its bound native variant"
        )


def _verify_batch_identity(batch, identity):
    model = batch.get("model_binding")
    if batch.get("schema") != _BATCH_SCHEMA or batch.get("identity") != identity:
        raise VariantRecoveryError(
            "same-run batch source or generation settings changed"
        )
    if not isinstance(model, dict) or not re.fullmatch(
        r"[0-9a-f]{40}", str(model.get("revision", ""))
    ):
        raise VariantRecoveryError(
            "same-run batch has no immutable Nano model revision"
        )


def _native_receipt(result):
    return {
        key: result[key]
        for key in (
            "native_model_selection",
            "guardrail_state",
            "structural_transfer",
            "sample_outputs",
        )
    }


def recovery_enabled(
    checkpoint: str, structural_control: str, run_id: str, environ: Mapping[str, str]
) -> bool:
    """Identify the stock immutable-image Nano structural workflow recovery path.

    Args:
        checkpoint: Requested checkpoint; custom models retain fresh generation.
        structural_control: Existing structural-control selection.
        run_id: Actual workflow run identity.
        environ: Actual runtime environment.
    Returns:
        Whether complete same-run recovery is available for this invocation.
    Raises:
        VariantRecoveryError: A supplied runtime image is mutable or invalid.
    """
    if checkpoint != "Cosmos3-Nano" or structural_control != "edge" or not run_id:
        return False
    image = environ.get("NPA_TASK_IMAGE", "")
    if not image:
        return False
    _image_binding(image)
    return True
