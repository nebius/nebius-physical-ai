"""Durable, owner-bound PAIDF augmentation through Nebius Stream."""

from __future__ import annotations

import re
import hashlib
import math
from pathlib import Path
from urllib.parse import urlsplit

from .stream_client import StreamClient, StreamError, document_sha256, read_api_key

ENGINE = "nebius-stream"
RECEIPT_SCHEMA = "npa.cosmos3.stream-execution.v1"


def require_completed_evaluation(scores_uri: str) -> None:
    """Prevent infrastructure failures from triggering another paid refinement.

    Args:
        scores_uri: Existing PAIDF evaluator output directory.
    Returns:
        None when the evaluator completed its checks, regardless of quality score.
    Raises:
        StreamError: Evaluation degraded or failed; repair and rerun evaluation.
    """
    from npa.workflows.paidf_cosmos3 import _read_json

    report = _read_json(scores_uri.rstrip("/") + "/cosmos_evaluator.json")
    if report.get("status") != "completed":
        raise StreamError(
            "Hosted refinement requires a completed evaluation; repair the evaluator "
            "and resume evaluation of retained outputs before spending on new variants"
        )


def _scope(connection, uri, direction):
    parsed = urlsplit(uri)
    prefix = connection.get(direction + "_prefix", "").strip("/")
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or parsed.netloc != connection.get("bucket")
        or parsed.query
        or parsed.fragment
        or ".." in key.split("/")
        or "\\" in key
        or "//" in key
        or (prefix and key != prefix and not key.startswith(prefix + "/"))
    ):
        raise StreamError(
            "Stream storage URI is outside its registered connection scope"
        )


class StreamAugmentation:
    """Validate access and storage before submitting any paid GPU work.

    Args:
        origin: Verified HTTPS service origin supplied by the operator.
        key_file: Private bearer-key file, or empty for an NPA secret binding.
        input_connection: Owner-bound registered input connection identifier.
        output_connection: Owner-bound verified Nebius output connection identifier.
        input_uri: Prepared full-source S3 video.
        output_uri: NPA publication prefix on the verified Nebius bucket.
        checkpoint: Explicit Nano or Super checkpoint; no fallback is performed.
        storage: NPA storage client used for publication and durable recovery.
        environ: Worker environment containing only explicitly injected secrets.
        api: Optional injected client for contract tests.
    Returns:
        An adapter that can generate variants concurrently without local GPUs.
    Raises:
        StreamError: Authentication, capability, or storage verification fails.
    """

    def __init__(
        self,
        *,
        origin,
        key_file,
        input_connection,
        output_connection,
        input_uri,
        output_uri,
        checkpoint,
        storage,
        environ,
        api=None,
    ):
        if checkpoint not in {"Cosmos3-Nano", "Cosmos3-Super"}:
            raise StreamError(
                "Stream requires an explicit Cosmos3-Nano or Cosmos3-Super model"
            )
        for identifier in (input_connection, output_connection):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identifier):
                raise StreamError("Both owner-bound Stream connection IDs are required")
        self.api = api or StreamClient(origin, read_api_key(key_file, environ))
        self.storage = storage
        self.input_uri, self.output_uri = input_uri, output_uri.rstrip("/")
        self.input_connection, self.output_connection = (
            input_connection,
            output_connection,
        )
        self.model = "nvidia/" + checkpoint
        try:
            auth = self.api.request("GET", "/v1/auth/me")
            if not auth.get("owner_id") or self.model not in auth.get(
                "allowed_models", []
            ):
                raise StreamError("Stream key does not grant this model and owner")
            self.owner = auth["owner_id"]
            models = self.api.request("GET", "/v1/models").get("data", [])
            capabilities = [
                c
                for m in models
                if m.get("id") == self.model
                for c in m.get("capability_matrix", [])
            ]
            if not any(
                c.get("task") == "full_source_v2v_augmentation"
                and c.get("runtime_implemented") is True
                and c.get("endpoint_exposed") is True
                and c.get("guardrails_required") is True
                and c.get("evidence_status") == "qualified"
                for c in capabilities
            ):
                raise StreamError(
                    "Stream has no qualified guarded full-source V2V capability"
                )
            source = self.api.request(
                "GET", "/v1/storage/connections/" + input_connection
            )["connection"]
            target = self.api.request(
                "GET", "/v1/storage/connections/" + output_connection
            )["connection"]
            if source.get("purpose") not in {"input", "input_output"}:
                raise StreamError("Stream input connection does not permit reading")
            _scope(source, input_uri, "read")
            _scope(target, output_uri, "write")
            self.output_storage = {
                key: target.get(key)
                for key in (
                    "provider",
                    "endpoint_url",
                    "region",
                    "bucket",
                    "write_prefix",
                    "verification",
                    "output_eligible",
                )
            }
            endpoint = str(target.get("endpoint_url", ""))
            region = target.get("region")
            verification = target.get("verification", {})
            if (
                target.get("output_eligible") is not True
                or target.get("provider") != "nebius"
                or target.get("purpose") not in {"output", "input_output"}
                or not re.fullmatch(
                    r"https://storage\.[a-z0-9-]+\.nebius\.cloud", endpoint
                )
                or endpoint != f"https://storage.{region}.nebius.cloud"
                or verification.get("status") != "verified"
                or verification.get("endpoint_region") != region
                or verification.get("bucket_region") != region
                or storage.s3.meta.endpoint_url.rstrip("/") != endpoint
            ):
                raise StreamError(
                    "Stream and NPA publication must use the same verified Nebius storage"
                )
        except BaseException:
            self.api.close()
            raise

    def generate(
        self,
        *,
        index,
        attempt,
        run_id,
        source,
        directory,
        prompt,
        seed,
        steps,
        guidance,
        control_guidance,
    ):
        """Resume or submit one immutable logical variant, then verify its delivery.

        Args:
            index: Variant index within this fanout.
            attempt: PAIDF refinement iteration, starting at zero.
            run_id: Stable workflow run identifier.
            source: Prepared local source file used to verify input identity.
            directory: Fresh private directory for verified artifact downloads.
            prompt: Effective appearance prompt.
            seed: Effective seed after refinement.
            steps: Effective denoising step count.
            guidance: Effective guidance scale.
            control_guidance: Effective edge-control strength.
        Returns:
            Generated video path and a distinct hosted execution receipt.
        Raises:
            StreamError: Identity, runtime settings, delivery, or evidence fails.
        """
        from npa.workflows.paidf_cosmos3 import (
            _missing_object_error,
            _read_json,
            _sha256,
            _write_json,
        )

        if not 1 <= len(prompt) <= 4000:
            raise StreamError(
                "Stream appearance prompt must contain 1 to 4000 characters"
            )
        if (
            not 1 <= steps <= 200
            or not math.isfinite(guidance)
            or not 0 < guidance <= 20
            or not math.isfinite(control_guidance)
            or control_guidance <= 0
        ):
            raise StreamError(
                "Stream sampling settings are outside the supported API bounds"
            )
        if not run_id:
            raise StreamError("Hosted generation requires a stable workflow run ID")
        body = {
            "model": self.model,
            "input": {"connection_id": self.input_connection, "uri": self.input_uri},
            "output": {
                "connection_id": self.output_connection,
                "prefix": self.output_uri + "/endpoint-jobs",
            },
            "augmentation": {
                "prompt": prompt,
                "structural_control": "edge",
                "variants": 1,
            },
            "generation": {
                "seed": seed,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "control_guidance": control_guidance,
            },
            "evaluation": {
                "mode": "technical_report",
                "rubric_id": "paidf-source-timeline",
                "criteria": ["complete_decode", "timeline_coverage"],
            },
        }
        identity = document_sha256(
            {
                "origin": self.api.origin,
                "owner": self.owner,
                "run_id": run_id,
                "index": index,
                "attempt": attempt,
                "source_sha256": _sha256(source),
                "request": body,
            }
        )
        state_uri = self.output_uri + "/endpoint-state/" + identity + ".json"
        try:
            state = _read_json(state_uri, storage=self.storage)
        except Exception as exc:
            if not _missing_object_error(exc):
                raise
            state = {"schema": RECEIPT_SCHEMA, "identity": identity, "request": body}
            _write_json(state, state_uri, storage=self.storage)
        if state.get("identity") != identity or state.get("request") != body:
            raise StreamError(
                "Retained Stream state does not match this logical request"
            )
        if not state.get("job_id"):
            # An ambiguous POST is replayed only with this persisted identity.
            try:
                admitted = self.api.request(
                    "POST", "/v1/augmentations", body=body, identity=identity
                )
            except StreamError as exc:
                state["submission_error"] = str(exc)
                _write_json(state, state_uri, storage=self.storage)
                raise
            state["job_id"] = admitted["job_id"]
            _write_json(state, state_uri, storage=self.storage)
        job_id = state["job_id"]

        def observe(job):
            if job.get("job_id") != job_id:
                raise StreamError("Stream returned a different job identity")
            state["job"] = job
            _write_json(state, state_uri, storage=self.storage)

        try:
            self.api.wait(job_id, observe=observe)
        except StreamError as exc:
            # StreamError messages are fixed client diagnostics, never response
            # bodies or credentials. Retain the cause when the CLI summarizes
            # failures across a fanout, so an agent can recover the same job.
            state.setdefault("poll_errors", []).append(str(exc))
            _write_json(state, state_uri, storage=self.storage)
            raise
        manifest = self.api.request("GET", f"/v1/jobs/{job_id}/artifacts").get(
            "manifest", {}
        )
        effective = manifest.get("effective", {})
        generation = effective.get("effective_generation", effective)
        if (
            manifest.get("job_id") != job_id
            or manifest.get("prompt_sha256")
            != hashlib.sha256(prompt.encode()).hexdigest()
            or manifest.get("model") != self.model
            or effective.get("guardrails") is not True
            or effective.get("structural_control") != "edge"
            or generation.get("seed") != seed
            or generation.get("num_inference_steps", effective.get("steps")) != steps
            or generation.get("guidance_scale") != guidance
            or generation.get("control_guidance") != control_guidance
        ):
            raise StreamError(
                "Stream receipt does not prove the requested model and guarded settings"
            )
        artifacts = manifest.get("artifacts", [])
        names = [item.get("name") for item in artifacts]
        if len(set(names)) != len(names) or not {"source.mp4", "generated.mp4"} <= set(
            names
        ):
            raise StreamError(
                "Stream receipt has missing or duplicate required artifacts"
            )
        directory = Path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        for artifact in artifacts:
            self.api.download(job_id, artifact, directory)
        if _sha256(directory / "source.mp4") != _sha256(source):
            raise StreamError(
                "Stream processed different source bytes; reject this output"
            )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "engine": ENGINE,
            "job_id": job_id,
            "identity": identity,
            "model": self.model,
            "guardrails": True,
            "source_sha256": _sha256(source),
            "output_storage": self.output_storage,
            "manifest": manifest,
            "status": "verified",
            "state_uri": state_uri,
        }
        _write_json(
            receipt,
            self.output_uri + f"/endpoint-receipts/{identity}.json",
            storage=self.storage,
        )
        return {
            "output_path": str(directory / "generated.mp4"),
            "endpoint_execution": receipt,
        }
