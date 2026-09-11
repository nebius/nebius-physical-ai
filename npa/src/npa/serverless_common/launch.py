"""Owner-only submission journal for reconnecting to one Serverless Job.

The journal is written before the non-idempotent create call. An inconclusive
response or a process crash permits observation only, including when the
provider has not made the job visible yet. It never permits a second create.
The existing S3 supervisor ledger takes over after the provider ID is known.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import tempfile

from npa.clients.serverless import (
    EndpointNotFoundError,
    JobIdentityError,
    JobInfo,
    JobSubmissionIndeterminateError,
    ServerlessClient,
    _redact_cli_args,
)
from npa.orchestration.skypilot.launch_transaction import launch_identity_lock


def _write(path: Path, record: dict[str, str]) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".submission-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def durable_create_job(
    client: ServerlessClient,
    *,
    args: list[str],
    project_id: str,
    name: str,
    create: Callable[[], JobInfo],
    allow_create: bool = True,
) -> JobInfo:
    """Create once or adopt the recorded identity with the same launch contract.

    Use the same NPA_CONFIG_DIR after reconnect. A new submission needs a new
    job name; absence following an uncertain create is not proof of rejection.
    No credentials, command, environment, or provider response enter this journal.
    Existing provider jobs may be adopted only with a matching journal. Legacy
    jobs without that evidence remain accessible through status/log/artifact
    operations, but cannot safely enter this launch/supervision transaction.
    """
    from npa.clients.config import CONFIG_PATH

    root = CONFIG_PATH.parent / "runtime" / "serverless-submissions"
    identity = hashlib.sha256(f"{project_id}\0{name}".encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(_redact_cli_args(args)).encode()).hexdigest()
    with launch_identity_lock(identity, root=root):
        path = root / f"{identity}.json"
        if path.is_symlink():
            raise JobIdentityError("Refusing a symlinked serverless submission journal")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            record = None
        except (ValueError, UnicodeError) as exc:
            raise JobIdentityError("Unreadable submission journal; restore exact launch evidence") from exc
        if record is not None:
            if not isinstance(record, dict) or any(record.get(key) != value for key, value in {
                "schema_version": "npa.serverless.submission.v1",
                "project_id": project_id, "job_name": name, "request_sha256": fingerprint,
            }.items()):
                raise JobIdentityError("The saved job name belongs to a different launch contract")
            if record.get("state") in {"creating", "confirmed"}:
                try:
                    job = client.get_job(record.get("provider_job_id") or name, project_id)
                except EndpointNotFoundError as exc:
                    raise JobSubmissionIndeterminateError(
                        "The recorded submission is not yet observable; reconnect with the same "
                        "job name and configuration. No new job was created.",
                        project_id=project_id, job_name=name,
                        provider_job_id=record.get("provider_job_id", ""),
                    ) from exc
                if not job.id or job.name != name or job.project_id != project_id or (
                    record.get("provider_job_id") and job.id != record["provider_job_id"]
                ):
                    raise JobIdentityError("The recorded submission resolved to a different job")
                record.update(state="confirmed", provider_job_id=job.id)
                _write(path, record)
                return job
            raise JobIdentityError("The submission journal has an unsupported state")
        if not allow_create:
            raise JobIdentityError(
                "An existing job has no durable launch contract in this configuration "
                "directory; restore its original submission journal before reconnecting. "
                "Inspect its exact provider status and artifacts without resubmitting."
            )
        record = {
            "schema_version": "npa.serverless.submission.v1",
            "project_id": project_id,
            "job_name": name,
            "request_sha256": fingerprint,
            "state": "creating",
            "provider_job_id": "",
        }
        _write(path, record)
        job = create()
        if not job.id or job.name != name or job.project_id != project_id:
            raise JobIdentityError("The created job has incomplete or mismatched identity")
        record.update(state="confirmed", provider_job_id=job.id)
        _write(path, record)
        return job
