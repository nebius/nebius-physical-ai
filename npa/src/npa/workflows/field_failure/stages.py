"""Execute single navigation workflow stages with durable S3 handoffs."""

from __future__ import annotations

import re

from botocore.exceptions import BotoCoreError, ClientError

from npa.clients.storage import StorageError, StoragePreconditionFailed
from npa.workflows.field_failure.adapters import (
    _configured_identity,
    _invoke,
    _verify_source,
)
from npa.workflows.field_failure.artifacts import _optional, _publish, _read, _root
from npa.workflows.field_failure.comparison import _decision
from npa.workflows.field_failure.contracts import _Artifact, _Bundle
from npa.workflows.field_failure.durability import _claim
from npa.workflows.field_failure.verification import (
    _check_bundle,
    _check_evaluation,
    _check_reconstruction,
    _check_training,
)


def _context(bundle_uri, bundle_sha256, root, run_id, *, validated=True):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run ID must be a safe nonempty identifier")
    _root(root, run_id)
    source = _Artifact(uri=bundle_uri, sha256=bundle_sha256)
    payload, digest = _read(source.uri, source.sha256)
    bundle = _Bundle.model_validate(payload)
    _check_bundle(bundle)
    receipt = {
        "schema_version": "npa.field-failure.validated.v1",
        "bundle": source.model_dump(),
        "run_id": run_id,
    }
    if validated and _read(root + "/validated.json")[0] != receipt:
        raise ValueError("validated bundle receipt differs from requested run")
    return bundle, digest, receipt


def _seal(uri, payload):
    try:
        _publish(uri, payload)
    except StoragePreconditionFailed:
        if _read(uri)[0] != payload:
            raise ValueError(
                "completed record conflicts with requested inputs or decision"
            ) from None


def _execute(stage, identity, request, root, validator):
    uri = root + f"/{stage}.json"
    completed = _optional(uri)
    if completed is not None:
        validator(completed)
        return
    _verify_source(identity)
    claim = _claim(
        stage, request["bundle_sha256"], identity, request["inputs_sha256"], root
    )
    request = {
        **request,
        "adapter": identity.model_dump(),
        "attempt_id": claim["attempt_id"],
        "output_prefix": root + f"/{stage}/attempt-{claim['attempt_id']}/",
    }
    payload = _invoke(identity, request)
    if _read(root + f"/claims/{stage}.json")[0] != claim:
        raise ValueError(
            "stage claim changed while adapter was running; refusing late publication"
        )
    record = validator(payload)
    if record.attempt_id != claim["attempt_id"]:
        raise ValueError("adapter returned a different attempt identity")
    # The validator rechecks the permanent claim after the adapter returns.
    _publish(uri, record.model_dump())


def _reconstruction(bundle, digest, root):
    payload, record_sha = _read(root + "/reconstruction.json")
    return _check_reconstruction(payload, bundle, digest, root), record_sha


def _training(bundle, digest, root):
    reconstruction, reconstruction_sha = _reconstruction(bundle, digest, root)
    payload, record_sha = _read(root + "/training.json")
    record = _check_training(
        payload, bundle, digest, reconstruction, reconstruction_sha, root
    )
    return record, record_sha


def _reconstruct(bundle, digest, root, adapter, runtime_image):
    identity = _configured_identity(
        bundle.adapters["reconstruct"], adapter, runtime_image
    )
    request = {
        "bundle_sha256": digest,
        "inputs_sha256": {},
        "captures": [s.model_dump() for s in bundle.captures],
        "protocol": bundle.protocol.model_dump(),
    }
    _execute(
        "reconstruction",
        identity,
        request,
        root,
        lambda payload: _check_reconstruction(payload, bundle, digest, root),
    )


def _train(bundle, digest, root, adapter, runtime_image):
    identity = _configured_identity(bundle.adapters["train"], adapter, runtime_image)
    reconstruction, reconstruction_sha = _reconstruction(bundle, digest, root)
    request = {
        "bundle_sha256": digest,
        "inputs_sha256": {"reconstruction": reconstruction_sha},
        "reconstruction_sha256": reconstruction_sha,
        "scenes": [s.model_dump() for s in reconstruction.scenes],
        "baseline": bundle.baseline.model_dump(),
        "protocol": bundle.protocol.model_dump(),
    }
    _execute(
        "training",
        identity,
        request,
        root,
        lambda payload: _check_training(
            payload, bundle, digest, reconstruction, reconstruction_sha, root
        ),
    )


def _evaluate(bundle, digest, root, adapter, runtime_image, arm):
    identity = _configured_identity(bundle.adapters["evaluate"], adapter, runtime_image)
    inputs = {}
    policy = bundle.baseline
    if arm == "candidate":
        training, training_sha = _training(bundle, digest, root)
        policy, inputs = training.candidate, {"training": training_sha}
    request = {
        "bundle_sha256": digest,
        "inputs_sha256": inputs,
        "policy": policy.model_dump(),
        "held_out": [s.model_dump() for s in bundle.held_out],
        "protocol": bundle.protocol.model_dump(),
        "metrics": [m.model_dump() for m in bundle.metrics],
    }
    _execute(
        f"{arm}-evaluation",
        identity,
        request,
        root,
        lambda payload: _check_evaluation(
            payload, bundle, digest, policy, root, arm, inputs
        ),
    )


def _compare(bundle, digest, root, run_id):
    training, training_sha = _training(bundle, digest, root)
    baseline_payload, baseline_sha = _read(root + "/baseline-evaluation.json")
    candidate_payload, candidate_sha = _read(root + "/candidate-evaluation.json")
    baseline = _check_evaluation(
        baseline_payload,
        bundle,
        digest,
        bundle.baseline,
        root,
        "baseline",
        {},
    )
    candidate = _check_evaluation(
        candidate_payload,
        bundle,
        digest,
        training.candidate,
        root,
        "candidate",
        {"training": training_sha},
    )
    hashes = {
        "training": training_sha,
        "reconstruction": training.reconstruction_sha256,
        "baseline_evaluation": baseline_sha,
        "candidate_evaluation": candidate_sha,
    }
    return _decision(bundle, digest, baseline, candidate, hashes, run_id)


def _dispatch(stage, bundle_uri, bundle_sha256, root, run_id, adapter, runtime_image):
    bundle, digest, receipt = _context(
        bundle_uri,
        bundle_sha256,
        root,
        run_id,
        validated=stage != "validate",
    )
    if stage == "validate":
        _seal(root + "/validated.json", receipt)
    elif stage == "reconstruct":
        _reconstruct(bundle, digest, root, adapter, runtime_image)
    elif stage == "train":
        _train(bundle, digest, root, adapter, runtime_image)
    elif stage in {"baseline-evaluate", "candidate-evaluate"}:
        _evaluate(bundle, digest, root, adapter, runtime_image, stage.split("-")[0])
    elif stage == "compare":
        return _compare(bundle, digest, root, run_id)
    else:
        raise ValueError("unknown field-failure stage")
    return None


def _reject(root, run_id):
    _seal(
        root + "/decision.json",
        {
            "schema_version": "npa.field-failure.decision.v1",
            "run_id": run_id,
            "status": "invalid_evidence",
            "recommendation": "retain_baseline",
            "decision": "retain_baseline",
            "promote_checkpoint": False,
            "deployment_authorized": False,
            "reason": "Evidence validation failed; inspect the failing comparison stage.",
        },
    )


def run_stage(
    stage: str,
    bundle_uri: str,
    bundle_sha256: str,
    output_root: str,
    run_id: str,
    adapter: str = "",
    runtime_image: str = "",
) -> None:
    """Execute one graph-owned stage; never submit jobs or deploy a policy.

    Args:
        stage: validate, reconstruct, train, baseline-evaluate, candidate-evaluate, or compare.
        bundle_uri: S3 object containing the sealed failure bundle.
        bundle_sha256: Operator-pinned SHA-256 of the bundle bytes.
        output_root: Fresh S3 prefix ending with run_id, without a trailing slash.
        run_id: Identifier shared by workflow state and artifacts.
        adapter: Trusted installed module:function for the requested operation.
        runtime_image: Exact image@sha256 identity supplied by the trusted workflow.
    Returns:
        None. Verifies/reuses a completed stage or writes its immutable S3 record.
    Raises:
        ValueError: Evidence, configuration, provenance, or attempt ownership is invalid.
        ImportError: The required operator adapter is not installed.
        BotoCoreError: S3 transport fails.
        ClientError: Storage access fails.
        StorageError: Conditional publication fails or a storage ETag is missing.
    """
    _run_guarded(
        stage, bundle_uri, bundle_sha256, output_root, run_id, adapter, runtime_image
    )


def _run_guarded(
    stage, bundle_uri, bundle_sha256, output_root, run_id, adapter, runtime_image
):
    _root(output_root, run_id)
    try:
        result = _dispatch(
            stage,
            bundle_uri,
            bundle_sha256,
            output_root,
            run_id,
            adapter,
            runtime_image,
        )
    except (ValueError, BotoCoreError, ClientError, StorageError):
        if stage == "compare":
            _reject(output_root, run_id)
        raise
    if stage == "compare":
        _seal(output_root + "/decision.json", result)
