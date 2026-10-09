"""Adapt the existing NPA runtime to a private shared SkyPilot API server."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from npa.orchestration.skypilot.workflow import ManagedJobEvidence, WorkflowResult

from .authorization import ExecutionBinding
from .deployment import scheduler_user, scheduler_workspace
from .errors import BackendError, ConflictError
from .ledger import TeamLedger
from .models import TeamConfig
from .storage import storage_credentials, workload_secrets
from .workflow_policy import enforce_rendered_tasks


class SkyBackend:
    """Keep scheduler operations bound to a server-owned person, workspace, and run.

    Args:
        config, binding, ledger, run_id: Trusted service and execution boundary.
        runner: Optional process transport for isolated SDK calls.
    Returns:
        A SkyBackend instance.
    Raises:
        OSError: Private run staging cannot be created.
    """

    def __init__(
        self,
        config: TeamConfig,
        binding: ExecutionBinding,
        ledger: TeamLedger,
        run_id: str,
        *,
        runner=subprocess.run,
    ):
        """Select the private API and immutable run boundary.

        Args:
            config, binding, ledger, run_id: Trusted service and run configuration.
            runner: Injectable process transport for offline tests.
        Returns:
            None.
        Raises:
            OSError: Private staging cannot be created.
        """
        self.config, self.binding, self.ledger, self.run_id = (
            config,
            binding,
            ledger,
            run_id,
        )
        self.runner = runner
        self.root = config.state_dir / "runs" / run_id
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def submit(self, path: Path, name: str, **_kwargs) -> WorkflowResult:
        """Submit one canonical runtime wave with an intent preceding network mutation.

        Args:
            path, name: Canonically rendered wave and engine-issued name.
            _kwargs: Existing runtime submit adapter arguments.
        Returns:
            Runtime-compatible acknowledgement bound to recorded scheduler jobs.
        Raises:
            BackendError, ConflictError: Launch failed or must be reconciled.
        """
        payload = enforce_rendered_tasks(
            path, self.binding, self._workload_credentials()
        )
        wave = self.ledger.begin_wave(self.run_id, name)
        response = self.call("launch", yaml=payload, name=name)
        self.ledger.record_wave(wave["id"], request_id=response["request_id"])
        acknowledged = self.call("result", request_id=response["request_id"])
        identifiers = _job_ids(acknowledged.get("job_ids"))
        self.ledger.record_wave(wave["id"], job_ids=identifiers)
        if self.ledger.get(self.run_id)["status"] == "cancelling":
            self.call("cancel", job_ids=list(identifiers))
        return WorkflowResult(status="SUBMITTED", job_id=wave["id"])

    def status(self, wave_id: str) -> WorkflowResult:
        """Observe only the exact scheduler jobs recorded for one wave.

        Args:
            wave_id: Server-issued wave identity.
        Returns:
            Runtime-compatible aggregate status.
        Raises:
            BackendError: Exact provider state is unavailable.
        """
        rows = self.timeline(wave_id)
        return WorkflowResult(status=_aggregate(rows), job_id=wave_id)

    def timeline(self, wave_id: str) -> list[dict]:
        """Read exact task status without exposing other scheduler users' jobs.

        Args:
            wave_id: Server-issued wave identity.
        Returns:
            Sanitized scheduler task records for this wave.
        Raises:
            BackendError: The recorded jobs cannot be observed exactly.
        """
        wave = self._wave(wave_id)
        identifiers = _job_ids(json.loads(wave["job_ids"]))
        rows = self.call("queue", job_ids=list(identifiers))["jobs"]
        if {int(row["job_id"]) for row in rows} != set(identifiers):
            raise BackendError("scheduler did not return the exact recorded jobs")
        return rows

    def cancel(self, *, job_id: str) -> dict:
        """Request cancellation of one recorded wave without name-based deletion.

        Args:
            job_id: Server-issued wave identity used by the runtime.
        Returns:
            Cancellation acknowledgement, not proof of termination.
        Raises:
            BackendError: Cancellation could not be acknowledged.
        """
        wave = self._wave(job_id)
        self.call("cancel", job_ids=list(_job_ids(json.loads(wave["job_ids"]))))
        return {"cancel_returncode": 0}

    def reconcile(self, name: str, job_id: str = "", **_kwargs) -> ManagedJobEvidence:
        """Reconcile an existing intent without automatically repeating an uncertain launch.

        Args:
            name, job_id: Engine's durable wave name and optional identity.
            _kwargs: Existing runtime reconciliation adapter arguments.
        Returns:
            Exact found state or an explicit unknown outcome.
        Raises:
            BackendError: Scheduler reconciliation is unavailable.
        """
        matches = [
            wave for wave in self.ledger.waves(self.run_id) if wave["name"] == name
        ]
        if len(matches) != 1 or (job_id and matches[0]["id"] != job_id):
            return ManagedJobEvidence(
                outcome="unknown", error="wave identity requires reconciliation"
            )
        wave = matches[0]
        if not json.loads(wave["job_ids"]):
            if not wave["request_id"]:
                return ManagedJobEvidence(
                    outcome="unknown", error="launch acknowledgement is missing"
                )
            response = self.call("result", request_id=wave["request_id"])
            self.ledger.record_wave(
                wave["id"], job_ids=_job_ids(response.get("job_ids"))
            )
        rows = self.timeline(wave["id"])
        return ManagedJobEvidence(
            outcome="found",
            job_id=wave["id"],
            status=_aggregate(rows),
            task_rows=tuple(rows),
        )

    def cancel_run(self) -> bool:
        """Cancel every acknowledged wave and report whether all are terminal.

        Args:
            None.
        Returns:
            True only when no launch is uncertain and all observed jobs terminated.
        Raises:
            BackendError: A provider operation failed.
        """
        terminal = True
        for wave in self.ledger.waves(self.run_id):
            if not json.loads(wave["job_ids"]):
                terminal = False
                continue
            self.cancel(job_id=wave["id"])
            terminal = _terminal(self.status(wave["id"]).status) and terminal
        return terminal

    def logs(self) -> str:
        """Download only this run's exact worker logs into private server storage.

        Args:
            None.
        Returns:
            Worker log text with storage credentials redacted.
        Raises:
            BackendError: Logs are unavailable or returned outside their destination.
        """
        # The pinned SDK downloads into ~/sky_logs regardless of local_dir.
        # Its process HOME is this run's private directory.
        destination = self.root / "sky_logs"
        destination.mkdir(exist_ok=True, mode=0o700)
        chunks = []
        for wave in self.ledger.waves(self.run_id):
            for job_id in json.loads(wave["job_ids"]):
                response = self.call("logs", job_id=job_id, directory=str(destination))
                chunks.extend(_read_log_paths(response["paths"], destination))
        text = "\n".join(chunks)
        secrets = {
            **storage_credentials(self.binding.allocation.storage),
            **workload_secrets(self.binding.allocation),
        }
        for secret in secrets.values():
            text = text.replace(secret, "[redacted]")
        return text

    def call(self, operation: str, **payload) -> dict:
        """Use the isolated pinned SDK without inheriting ambient cloud credentials.

        Args:
            operation, payload: Server-generated private bridge request.
        Returns:
            Decoded scheduler response.
        Raises:
            BackendError: The exact bridge operation failed.
        """
        command = [
            str(self.config.sky_python),
            str(Path(__file__).with_name("sky_bridge.py")),
        ]
        try:
            result = self.runner(
                command,
                input=json.dumps({"operation": operation, **payload}),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=self._bridge_environment(),
                cwd=self.root,
                check=False,
            )
            response = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise BackendError(
                "SkyPilot transport failed; preserve the run for reconciliation"
            ) from exc
        if result.returncode or response.get("error"):
            raise BackendError(
                "SkyPilot operation failed; preserve the run for reconciliation"
            )
        return response

    def _bridge_environment(self):
        environment = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "SSL_CERT_FILE")
            if key in os.environ
        }
        environment.update(
            HOME=str(self.root),
            SKYPILOT_GLOBAL_CONFIG=str(self._client_config()),
            SKYPILOT_API_SERVER_ENDPOINT=self.config.sky_endpoint,
            SKYPILOT_USER_ID=scheduler_user(self.binding),
            SKYPILOT_USER=scheduler_user(self.binding),
            SKYPILOT_DISABLE_USAGE_COLLECTION="1",
        )
        return environment

    def _wave(self, wave_id):
        wave = next(
            (item for item in self.ledger.waves(self.run_id) if item["id"] == wave_id),
            None,
        )
        if wave is None:
            raise ConflictError("wave does not belong to this run")
        return wave

    def _client_config(self):
        import yaml
        from .storage import PrivateRunFiles

        path = self.root / "sky-client.yaml"
        configuration = {
            "active_workspace": scheduler_workspace(self.binding),
            "api_server": {"endpoint": self.config.sky_endpoint},
        }
        PrivateRunFiles(self.root, "client").write(
            "", "client/sky-client.yaml", yaml.safe_dump(configuration).encode()
        )
        return path

    def _workload_credentials(self):
        values = storage_credentials(self.binding.allocation.storage)
        return {
            **workload_secrets(self.binding.allocation),
            "AWS_ACCESS_KEY_ID": values["aws_access_key_id"],
            "AWS_SECRET_ACCESS_KEY": values["aws_secret_access_key"],
            "AWS_SESSION_TOKEN": values.get("aws_session_token", ""),
            "AWS_ENDPOINT_URL": self.binding.allocation.storage.endpoint,
        }


def _job_ids(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise BackendError("scheduler did not acknowledge exact job identities")
    if any(type(value) is not int or value < 1 for value in values):
        raise BackendError("scheduler returned invalid job identities")
    return tuple(sorted(set(values)))


def _terminal(status):
    return status in {"SUCCEEDED", "CANCELLED"} or status.startswith("FAILED")


def _aggregate(rows):
    statuses = {str(row["status"]).upper() for row in rows}
    if not statuses:
        return "UNKNOWN"
    if any(not _terminal(status) for status in statuses):
        return "RUNNING"
    if any(status.startswith("FAILED") for status in statuses):
        return "FAILED"
    return "CANCELLED" if "CANCELLED" in statuses else "SUCCEEDED"


def _read_log_paths(paths, destination):
    result = []
    for value in paths.values():
        root = Path(value).resolve()
        if not root.is_relative_to(destination.resolve()):
            raise BackendError("scheduler logs escaped their private destination")
        for path in root.rglob("*.log"):
            if not path.resolve().is_relative_to(destination.resolve()):
                raise BackendError(
                    "scheduler log symlink escaped its private destination"
                )
            result.append(path.read_text(errors="replace"))
    return result
