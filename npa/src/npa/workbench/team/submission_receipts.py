"""Retain automatic submission identities so interrupted CLI retries cannot duplicate work."""

from contextlib import contextmanager
import hashlib
import json
import os
import stat
import uuid

from .errors import TeamError
from .models import SubmitRequest
from .session_storage import _directory, read_private_json, write_private_json
from .sessions import SessionStore, verify_session_identity


def submit_workflow(
    client,
    session,
    workflow,
    *,
    workspace=None,
    cluster=None,
    idempotency_key=None,
    new_run=False,
):
    """Submit a workflow using saved placement and durable CLI/SDK retry receipts.

    Args:
        client, session: Authenticated connection returned by open_connection.
        workflow: Parsed canonical workflow document.
        workspace, cluster: Optional overrides of saved placement.
        idempotency_key: Optional caller-managed retry identity.
        new_run: Deliberately repeat a previously acknowledged submission.
    Returns:
        Server run identity and automatic retry metadata when applicable.
    Raises:
        TeamError: Placement is ambiguous, identity changed, or submission fails.
    """
    workspace, cluster = workspace or session.workspace, cluster or session.cluster
    if not workspace or not cluster:
        raise TeamError("Choose workspace and cluster at login or submission.")
    request = SubmitRequest(
        workspace=workspace,
        cluster=cluster,
        idempotency_key=idempotency_key or "automatic",
        workflow=workflow,
    )
    return submit_saved(
        client, session, request, automatic=idempotency_key is None, new_run=new_run
    )


@contextmanager
def _receipt_lock(root, name):
    import fcntl

    with _directory(root, create=True) as directory:
        descriptor = os.open(
            name + ".lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise TeamError(
                    "submission receipt lock must be a private regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)


def _receipt(root, name, *, new_run):
    try:
        receipt = read_private_json(root, name + ".json")
    except FileNotFoundError:
        receipt = None
    if receipt is not None:
        if (
            not isinstance(receipt, dict)
            or not isinstance(receipt.get("acknowledged"), bool)
            or not isinstance(receipt.get("key"), str)
            or not receipt["key"]
        ):
            raise TeamError("invalid submission receipt; reconcile the original run")
        if not new_run:
            return receipt, True
        if not receipt["acknowledged"]:
            raise TeamError(
                "Previous submission is unconfirmed. Retry without --new-run first."
            )
    receipt = {"key": str(uuid.uuid4()), "acknowledged": False}
    write_private_json(root, name + ".json", receipt)
    return receipt, False


def submit_saved(
    client, session, request, *, automatic=True, new_run=False, store=None
):
    """Submit with a durable retry identity scoped to endpoint, person, and document.

    Args:
        client, session, request: Authenticated connection and canonical submission.
        automatic: Generate and retain a retry identity when none was supplied.
        new_run: Explicitly start a new run after an acknowledged submission.
        store: Optional private session store for isolated callers and tests.
    Returns:
        Submission response with automatic retry metadata when used.
    Raises:
        TeamError: Identity changed, receipt is unsafe, or submission fails.
    """
    if not automatic:
        if new_run:
            raise TeamError("Choose --new-run or --idempotency-key, not both.")
        return client.submit(request)
    access = client.whoami()
    verify_session_identity(session, access)
    if not isinstance(access.get("subject"), str) or not access["subject"]:
        raise TeamError("team service did not return a verified account")
    document = request.model_dump(mode="json", exclude={"idempotency_key"})
    identity = [session.endpoint, access["subject"], document]
    name = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = (store or SessionStore()).root / "requests"
    with _receipt_lock(root, name):
        receipt, reused = _receipt(root, name, new_run=new_run)
        result = client.submit(
            request.model_copy(update={"idempotency_key": receipt["key"]})
        )
        receipt["acknowledged"] = True
        write_private_json(root, name + ".json", receipt)
    return {
        **result,
        "submission": {"idempotency_key": receipt["key"], "reused": reused},
    }
