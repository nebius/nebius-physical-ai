"""Reserve permanent stage attempts and bind immutable records to their claims."""

from __future__ import annotations

import uuid

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.field_failure.artifacts import _publish, _read


def _claim_payload(stage, bundle_sha, identity, inputs, attempt_id):
    return {
        "schema_version": "npa.field-failure.claim.v1",
        "stage": stage,
        "bundle_sha256": bundle_sha,
        "adapter": identity.model_dump(),
        "inputs_sha256": inputs,
        "attempt_id": attempt_id,
    }


def _claim(stage, bundle_sha, identity, inputs, root):
    attempt_id = uuid.uuid4().hex
    payload = _claim_payload(stage, bundle_sha, identity, inputs, attempt_id)
    try:
        _publish(root + f"/claims/{stage}.json", payload)
    except StoragePreconditionFailed as error:
        raise ValueError(
            "stage attempt is active, incomplete, or conflicting; use a fresh run ID; "
            "claims cannot be expired or taken over"
        ) from error
    return payload


def _check_claim(stage, record, bundle_sha, identity, inputs, root):
    expected = _claim_payload(stage, bundle_sha, identity, inputs, record.attempt_id)
    payload = _read(root + f"/claims/{stage}.json")[0]
    if (
        payload != expected
        or record.inputs_sha256 != inputs
        or record.adapter != identity
    ):
        raise ValueError(
            "stage attempt/input/adapter identity conflicts with permanent claim"
        )
    return root + f"/{stage}/attempt-{record.attempt_id}/"
