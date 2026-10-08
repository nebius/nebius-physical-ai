"""Authorize team operations and supervise durable workflow execution."""

from __future__ import annotations

import json
import threading
import tempfile
from pathlib import Path

from .authorization import authorize, bind_execution
from .engine import execute_run
from .errors import ConflictError, RunNotFoundError
from .ledger import TeamLedger
from .sky_backend import SkyBackend
from .storage import PersonalStorage
from .workflow_policy import load_bound_spec

_PUBLIC_FIELDS = ("id", "workspace", "cluster", "status", "created_at", "updated_at")
_TERMINAL = ("succeeded", "failed", "cancelled")


class TeamService:
    """Use current grants for each operation and immutable identity for each run.

    Args:
        configuration: Callable loading current administrator policy.
        backend_factory, engine, storage_factory: Execution adapters.
        enrollment_check: Required verification callback before submission.
    Returns:
        A TeamService instance.
    Raises:
        TeamError, OSError: Configuration or durable ledger initialization fails.
    """

    def __init__(
        self,
        configuration,
        *,
        backend_factory=SkyBackend,
        engine=execute_run,
        storage_factory=PersonalStorage,
        enrollment_check=None,
    ):
        self.configuration = configuration
        self.initial = configuration()
        self.ledger = TeamLedger(self.initial.state_dir)
        self.backend_factory, self.engine = backend_factory, engine
        self.storage_factory = storage_factory
        self.enrollment_check = enrollment_check
        self._workers = {}
        self._lock = threading.RLock()

    def config(self):
        """Reload grants while rejecting in-place changes to installation identity.

        Args:
            None.
        Returns:
            Current validated administrator configuration.
        Raises:
            TeamError: Configuration is invalid or installation identity changed.
        """
        config = self.configuration()
        fields = ("identity", "state_dir", "sky_endpoint", "sky_python")
        if any(getattr(config, key) != getattr(self.initial, key) for key in fields):
            raise ConflictError("installation settings changed; restart the service")
        return config

    def submit(self, actor, request):
        """Authorize and validate a submission before assigning durable ownership.

        Args:
            actor: Verified external identity.
            request: Workflow and requested workspace and cluster.
        Returns:
            Public run record, reusing an identical idempotent submission.
        Raises:
            TeamError: Authorization, workflow, enrollment, or retry validation fails.
        """
        config = self.config()
        binding = bind_execution(config, actor, request.workspace, request.cluster)
        with tempfile.TemporaryDirectory(
            dir=config.state_dir, prefix="validation-"
        ) as directory:
            load_bound_spec(request.workflow, binding, "validation", Path(directory))
        if self.enrollment_check is None:
            raise ConflictError("cluster enrollment verification is required")
        self.enrollment_check(binding)
        record, created = self.ledger.create(actor, request, binding_snapshot(binding))
        if created:
            self._start(actor, record, binding, resume=False)
        return public_run(self.ledger.get(record["id"]))

    def list(self, actor, workspace):
        """List only the authenticated actor's runs in an authorized workspace.

        Args:
            actor, workspace: Verified identity and requested workspace.
        Returns:
            Public records owned by this identity.
        Raises:
            TeamError: Current workspace access is denied.
        """
        authorize(self.config(), actor, workspace, "reader")
        return [public_run(row) for row in self.ledger.list_owned(actor, workspace)]

    def get(self, actor, run_id):
        """Read status after checking current access and immutable ownership.

        Args:
            actor, run_id: Verified identity and server-issued run ID.
        Returns:
            Public run status.
        Raises:
            TeamError: Ownership or current access cannot be established.
        """
        return public_run(self._owned(actor, run_id))

    def cancel(self, actor, run_id):
        """Stop new launches and cancel exact recorded jobs, retaining uncertainty.

        Args:
            actor, run_id: Verified identity and owned run ID.
        Returns:
            Current cancellation status; acknowledgement is not termination.
        Raises:
            TeamError: Access, execution boundary, or scheduler check fails.
        """
        record = self._owned(actor, run_id, "runner")
        if record["status"] in _TERMINAL:
            return public_run(record)
        binding = self._binding(actor, record)
        self.ledger.transition(
            run_id, ("accepted", "running", "recovery_required"), "cancelling"
        )
        self.ledger.audit(actor, "cancel", run_id)
        backend = self.backend_factory(self.config(), binding, self.ledger, run_id)
        if backend.cancel_run():
            self.ledger.transition(run_id, ("cancelling",), "cancelled")
        return public_run(self.ledger.get(run_id))

    def resume(self, actor, run_id):
        """Reconcile an interrupted run without changing its allocation or workflow.

        Args:
            actor, run_id: Verified identity and interrupted owned run ID.
        Returns:
            Current run status after starting reconciliation.
        Raises:
            TeamError: Access, enrollment, boundary, or lifecycle check fails.
        """
        record = self._owned(actor, run_id, "runner")
        binding = self._binding(actor, record)
        if record["status"] != "recovery_required":
            raise ConflictError("only an interrupted run can be resumed")
        if self.enrollment_check is None:
            raise ConflictError("cluster enrollment verification is required")
        self.enrollment_check(binding)
        self.ledger.audit(actor, "resume", run_id)
        self._start(actor, record, binding, resume=True)
        return public_run(self.ledger.get(run_id))

    def logs(self, actor, run_id):
        """Read only the exact worker logs belonging to an authorized owned run.

        Args:
            actor, run_id: Verified identity and owned run ID.
        Returns:
            Worker log text with allocated credentials redacted.
        Raises:
            TeamError: Ownership, access, or scheduler lookup fails.
        """
        record = self._owned(actor, run_id)
        binding = self._binding(actor, record)
        self.ledger.audit(actor, "logs", run_id)
        return self.backend_factory(self.config(), binding, self.ledger, run_id).logs()

    def artifacts(self, actor, run_id, relative=None):
        """List or stream owned run artifacts through its allocated cloud principal.

        Args:
            actor, run_id: Verified identity and owned run ID.
            relative: Optional relative artifact key; omit to list.
        Returns:
            Artifact metadata or a streaming object body.
        Raises:
            TeamError: Ownership or scope fails.
            ClientError: The object provider rejects the request.
        """
        record = self._owned(actor, run_id)
        binding = self._binding(actor, record)
        storage = self.storage_factory(binding.allocation.storage)
        prefix = f"{binding.allocation.storage.prefix.strip('/')}/runs/{run_id}"
        self.ledger.audit(
            actor, "artifact-read" if relative else "artifact-list", run_id
        )
        return storage.read(prefix, relative) if relative else storage.list(prefix)

    def _owned(self, actor, run_id, role="reader"):
        record = self.ledger.get(run_id)
        if (actor.issuer, actor.subject) != (record["issuer"], record["subject"]):
            raise RunNotFoundError("run not found")
        authorize(self.config(), actor, record["workspace"], role)
        return record

    def _binding(self, actor, record):
        binding = bind_execution(
            self.config(), actor, record["workspace"], record["cluster"]
        )
        if binding_snapshot(binding) != json.loads(record["binding"]):
            raise ConflictError(
                "run allocation changed; restore its original boundary before access"
            )
        return binding

    def _start(self, actor, record, binding, *, resume):
        with self._lock:
            previous = self._workers.get(record["id"])
            if previous and previous.is_alive():
                raise ConflictError("run already has an active supervisor")
            expected = ("recovery_required",) if resume else ("accepted",)
            if not self.ledger.transition(record["id"], expected, "running"):
                raise ConflictError("run lifecycle changed concurrently")
            worker = threading.Thread(
                target=self._execute, args=(actor, record, binding, resume), daemon=True
            )
            self._workers[record["id"]] = worker
            worker.start()

    def _execute(self, actor, record, binding, resume):
        run_id = record["id"]
        try:
            backend = self.backend_factory(self.config(), binding, self.ledger, run_id)
            report = self.engine(
                self.config(),
                binding,
                record,
                backend,
                lambda: self._check_running(actor, record),
                resume=resume,
            )
            status = (
                "succeeded" if report.status == "succeeded" else "recovery_required"
            )
            self.ledger.transition(run_id, ("running",), status)
        except Exception:  # noqa: BLE001 - retain uncertain jobs; never expose credentials
            self.ledger.transition(run_id, ("running",), "recovery_required")
            self.ledger.audit(actor, "execution-requires-reconciliation", run_id)
        finally:
            with self._lock:
                self._workers.pop(run_id, None)

    def _check_running(self, actor, record):
        if self.ledger.get(record["id"])["status"] != "running":
            raise ConflictError("run no longer accepts new work")
        self._binding(actor, record)


def binding_snapshot(binding):
    """Capture non-secret execution boundaries without freezing revocable GPU limits.

    Args:
        binding: Authorized personal execution allocation.
    Returns:
        Non-secret boundary values retained in the private ledger.
    Raises:
        None.
    """
    storage = binding.allocation.storage
    return {
        "namespace": binding.namespace,
        "workspace": binding.workspace,
        "cluster": binding.cluster,
        "connection": binding.connection.model_dump(mode="json"),
        "storage": storage.model_dump(mode="json"),
        "shared_inputs": list(binding.allocation.shared_inputs),
    }


def public_run(record):
    """Return stable run fields without identities, credentials, or server paths.

    Args:
        record: Server-owned run ledger record.
    Returns:
        Run identity, placement, timestamps, and status only.
    Raises:
        KeyError: A required ledger field is absent.
    """
    return {key: record[key] for key in _PUBLIC_FIELDS}
