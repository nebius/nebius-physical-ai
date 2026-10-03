"""Retain failed native navigation output before adapter temporary files disappear."""

from npa.workflows.field_failure.native_artifacts import (
    _archive,
    _identity,
    _upload,
    _upload_json,
)


def retain_failure(request, stage, output):
    """Publish existing native failure artifacts under the adapter's private attempt.

    Args:
        request: Sealed adapter request with a unique attempt output prefix.
        stage: Native navigation stage that raised an exception.
        output: Locally published native output, if execution produced any.
    Returns:
        None; no success or evaluation evidence is manufactured.
    Raises:
        ValueError: Artifact containment or identity is invalid.
        OSError: Existing native output cannot be archived.
        StorageError: Conditional artifact publication fails.
    """
    if not output.is_dir():
        return
    name = output.name + "-failure"
    archive = output.parent / (name + ".tar")
    _archive(output, archive)
    prefix = request["output_prefix"]
    artifact = _upload(archive, prefix + archive.name)
    record = {
        **_identity(request, "npa.field-failure.native-failure.v1"),
        "native_stage": stage,
        "status": "failed",
        "native_runtime_verified": False,
        "protocol_sha256": request["protocol"]["sha256"],
        "artifacts": artifact,
    }
    _upload_json(record, output.parent / (name + ".json"), prefix + name + ".json")
