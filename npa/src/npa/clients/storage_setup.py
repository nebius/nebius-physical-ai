"""Crash-safe first-run object-storage provisioning and reconciliation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import yaml

from npa.clients.storage_validation import StorageProbeResult, probe_storage_write

StatusFn = Callable[[str], None]


class StorageSetupStateError(RuntimeError):
    """Owner state cannot be read safely enough to continue provisioning."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _credentials_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from npa.clients.credentials import CREDENTIALS_PATH

    return CREDENTIALS_PATH


def _load_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise StorageSetupStateError(
            f"Cannot read storage ownership state at {path}; no cloud changes were made."
        ) from exc
    except yaml.YAMLError as exc:
        raise StorageSetupStateError(
            f"Storage ownership state at {path} is malformed; no cloud changes were made."
        ) from exc
    if not isinstance(loaded, dict):
        raise StorageSetupStateError(
            f"Storage ownership state at {path} is not a YAML mapping; "
            "no cloud changes were made."
        )
    return loaded


def _write_document(path: Path, data: Mapping[str, Any]) -> None:
    from npa.clients.credentials import write_private_yaml

    write_private_yaml(path, data)


def storage_setup_record(
    project_id: str, *, path: Path | None = None
) -> dict[str, Any]:
    """Return the secret-free storage transaction record for one project."""

    data = _load_document(_credentials_path(path))
    setup = data.get("storage_setup")
    projects = setup.get("projects") if isinstance(setup, dict) else None
    record = projects.get(project_id) if isinstance(projects, dict) else None
    return deepcopy(record) if isinstance(record, dict) else {}


@dataclass
class StorageSetupTransaction:
    project_id: str
    tenant_id: str
    region: str
    bucket_name: str
    project_alias: str = ""
    path: Path | None = None
    attempt_id: str = field(default_factory=lambda: uuid4().hex)
    _created_this_attempt: list[tuple[str, dict[str, str]]] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def credentials_path(self) -> Path:
        return _credentials_path(self.path)

    @property
    def next_command(self) -> str:
        project = f" --project {self.project_alias}" if self.project_alias else ""
        return f"npa provision-if-absent{project} --skip-k8s"

    def begin(self) -> None:
        # An interrupted attempt owns its exact bucket name. Reconcile that
        # resource even if a later invocation proposes a different default;
        # creating the new name would orphan the first bucket and overwrite its
        # provenance. Unproven/legacy records are never adopted this way.
        prior = storage_setup_record(self.project_id, path=self.credentials_path)
        prior_resources = prior.get("resources")
        prior_bucket = (
            prior_resources.get("bucket") if isinstance(prior_resources, dict) else None
        )
        if (
            prior.get("status") != "complete"
            and isinstance(prior_bucket, dict)
            and prior_bucket.get("created_by") == "npa"
            and prior_bucket.get("project_id") == self.project_id
            and str(prior_bucket.get("name", "")).strip()
        ):
            self.bucket_name = str(prior_bucket["name"]).strip()
        self._update_record(
            {
                "version": 1,
                "status": "in_progress",
                "phase": "reconciling",
                "attempt_id": self.attempt_id,
                "project_id": self.project_id,
                "tenant_id": self.tenant_id,
                "region": self.region,
                "bucket_name": self.bucket_name,
                "project_alias": self.project_alias,
                "next_command": self.next_command,
                "updated_at": _now(),
            }
        )

    def record_created(self, kind: str, metadata: dict[str, str]) -> None:
        """Persist ownership before allowing the next fallible provider step."""

        allowlists = {
            "service_account": {"id", "name"},
            "bucket": {"id", "name"},
            "access_key": {"id", "name", "service_account_id"},
        }
        if kind not in allowlists:
            raise ValueError(f"unsupported storage resource kind: {kind}")
        clean = {
            key: str(metadata.get(key, "") or "").strip()
            for key in allowlists[kind]
            if metadata.get(key)
        }
        clean.update(
            {
                "created_by": "npa",
                "project_id": self.project_id,
                "attempt_id": self.attempt_id,
                "created_at": _now(),
            }
        )
        required = "name" if kind == "bucket" else "id"
        if not clean.get(required):
            raise ValueError(
                f"created storage {kind} record is missing its {required}"
            )
        self._created_this_attempt.append((kind, clean))
        record = storage_setup_record(self.project_id, path=self.credentials_path)
        resources = record.get("resources")
        resources = deepcopy(resources) if isinstance(resources, dict) else {}
        if kind == "access_key":
            keys = resources.get("access_keys")
            keys = deepcopy(keys) if isinstance(keys, dict) else {}
            key_id = clean.get("id", "")
            if not key_id:
                raise ValueError("created access key record is missing its id")
            keys[key_id] = clean
            resources["access_keys"] = keys
        elif kind in {"bucket", "service_account"}:
            resources[kind] = clean
        self._update_record(
            {
                "status": "in_progress",
                "phase": f"created_{kind}",
                "resources": resources,
                "updated_at": _now(),
            }
        )

    def _owned_service_account(self, account_id: str) -> dict[str, str]:
        record = storage_setup_record(self.project_id, path=self.credentials_path)
        resources = record.get("resources")
        resources = resources if isinstance(resources, dict) else {}
        account = resources.get("service_account")
        if not isinstance(account, dict):
            return {}
        if (
            str(account.get("created_by", "")) != "npa"
            or str(account.get("id", "")) != account_id
            or str(account.get("name", "")) != "lerobot-training"
            or str(account.get("project_id", "")) != self.project_id
        ):
            return {}
        return {
            "service_account_id": account_id,
            "service_account_name": "lerobot-training",
            "service_account_project_id": self.project_id,
            "service_account_managed_by": "npa",
        }

    def commit(self, credentials: Mapping[str, str], probe: StorageProbeResult) -> None:
        """Atomically commit validated credentials and final provenance."""

        if not probe.ok:
            raise ValueError(
                "cannot commit storage credentials before a successful probe"
            )
        access = str(credentials.get("nebius_api_key", "") or "").strip()
        secret = str(credentials.get("nebius_secret_key", "") or "").strip()
        bucket = str(credentials.get("s3_bucket", "") or "").strip()
        endpoint = str(credentials.get("s3_endpoint", "") or "").strip()
        account_id = str(credentials.get("service_account_id", "") or "").strip()
        if not all((access, secret, bucket, endpoint)):
            raise ValueError(
                "validated storage result is missing required credential fields"
            )

        path = self.credentials_path
        document = _load_document(path)
        payload: dict[str, Any] = {
            "storage": {
                "aws_access_key_id": access,
                "aws_secret_access_key": secret,
                "endpoint_url": endpoint,
                "bucket": f"s3://{bucket.strip().removeprefix('s3://').strip('/')}/",
            },
            "storage_setup": {
                "version": 1,
                "projects": {
                    self.project_id: {
                        **storage_setup_record(self.project_id, path=path),
                        "status": "complete",
                        "phase": "credentials_committed",
                        "last_error": "",
                        "next_command": "",
                        "probe_cleanup_succeeded": probe.cleanup_succeeded,
                        "updated_at": _now(),
                    }
                },
            },
        }
        ownership = self._owned_service_account(account_id)
        if account_id:
            payload["nebius"] = {"service_account_id": account_id}
        if ownership:
            payload["storage_iam"] = ownership
        _write_document(path, _deep_merge(document, payload))

    def fail_and_rollback(self, exc: BaseException) -> list[str]:
        """Roll back only this attempt's exact creations; preserve failures."""

        from npa.clients import nebius

        rollback_errors: list[str] = []
        remaining: list[tuple[str, dict[str, str]]] = []
        for kind, metadata in reversed(self._created_this_attempt):
            try:
                if kind == "access_key":
                    nebius.delete_access_key(metadata.get("id", ""))
                elif kind == "bucket":
                    item = nebius.get_bucket_by_name(
                        self.project_id, metadata.get("name", "")
                    )
                    bucket_id = str(((item or {}).get("metadata") or {}).get("id", ""))
                    if bucket_id:
                        nebius.delete_bucket(bucket_id)
                elif kind == "service_account":
                    nebius.delete_service_account(metadata.get("id", ""))
            except Exception as rollback_exc:  # noqa: BLE001 - preserve partial state
                if nebius.is_not_found(str(rollback_exc)):
                    self._remove_resource(kind, metadata)
                else:
                    message = nebius.redact_nebius_output(str(rollback_exc))
                    rollback_errors.append(f"{kind} rollback failed: {message}")
                    remaining.append((kind, metadata))
            else:
                self._remove_resource(kind, metadata)

        safe_error = nebius.redact_nebius_output(str(exc))
        status = "partial" if rollback_errors else "rolled_back"
        self._update_record(
            {
                "status": status,
                "phase": "rollback_incomplete" if rollback_errors else "rolled_back",
                "last_error": safe_error,
                "rollback_errors": rollback_errors,
                "next_command": self.next_command,
                "updated_at": _now(),
            }
        )
        self._created_this_attempt = list(reversed(remaining))
        return rollback_errors

    def _remove_resource(self, kind: str, metadata: Mapping[str, str]) -> None:
        record = storage_setup_record(self.project_id, path=self.credentials_path)
        resources = record.get("resources")
        resources = deepcopy(resources) if isinstance(resources, dict) else {}
        if kind == "access_key":
            keys = resources.get("access_keys")
            keys = deepcopy(keys) if isinstance(keys, dict) else {}
            keys.pop(str(metadata.get("id", "")), None)
            if keys:
                resources["access_keys"] = keys
            else:
                resources.pop("access_keys", None)
        else:
            saved = resources.get(kind)
            if isinstance(saved, dict) and saved.get("attempt_id") == self.attempt_id:
                resources.pop(kind, None)
        # A deep merge cannot express deletion: merging ``resources={}`` would
        # resurrect the exact entries just removed. Replace this field in the
        # persisted project record while retaining every unrelated project.
        path = self.credentials_path
        document = _load_document(path)
        setup = document.get("storage_setup")
        setup = deepcopy(setup) if isinstance(setup, dict) else {"version": 1}
        projects = setup.get("projects")
        projects = deepcopy(projects) if isinstance(projects, dict) else {}
        project_record = projects.get(self.project_id)
        project_record = (
            deepcopy(project_record) if isinstance(project_record, dict) else {}
        )
        project_record["resources"] = resources
        project_record["updated_at"] = _now()
        projects[self.project_id] = project_record
        setup["projects"] = projects
        document["storage_setup"] = setup
        _write_document(path, document)

    def _update_record(self, patch: Mapping[str, Any]) -> None:
        path = self.credentials_path
        document = _load_document(path)
        setup = document.get("storage_setup")
        setup = deepcopy(setup) if isinstance(setup, dict) else {"version": 1}
        projects = setup.get("projects")
        projects = deepcopy(projects) if isinstance(projects, dict) else {}
        existing = projects.get(self.project_id)
        existing = existing if isinstance(existing, dict) else {}
        projects[self.project_id] = _deep_merge(existing, patch)
        setup["version"] = 1
        setup["projects"] = projects
        document["storage_setup"] = setup
        _write_document(path, document)


def provision_storage(
    *,
    project_id: str,
    tenant_id: str,
    region: str,
    bucket_name: str,
    project_alias: str = "",
    bucket_max_size_bytes: int = 0,
    bucket_storage_class: str = "standard",
    on_status: StatusFn | None = None,
) -> tuple[dict[str, str], StorageProbeResult]:
    """Reconcile, validate and atomically commit first-run storage."""

    from npa.clients import nebius

    transaction = StorageSetupTransaction(
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        bucket_name=bucket_name,
        project_alias=project_alias,
    )
    transaction.begin()
    try:
        credentials = nebius.bootstrap_environment(
            project_id,
            tenant_id,
            region,
            bucket_name=transaction.bucket_name,
            bucket_max_size_bytes=bucket_max_size_bytes,
            bucket_storage_class=bucket_storage_class,
            on_status=on_status,
            on_resource_created=transaction.record_created,
        )
        probe = probe_storage_write(
            bucket=credentials.get("s3_bucket", ""),
            endpoint_url=credentials.get("s3_endpoint", ""),
            access_key_id=credentials.get("nebius_api_key", ""),
            secret_access_key=credentials.get("nebius_secret_key", ""),
            region=region,
        )
        if not probe.ok:
            raise nebius.NebiusError(probe.summary)
        transaction.commit(credentials, probe)
        return credentials, probe
    except BaseException as exc:
        rollback_errors = transaction.fail_and_rollback(exc)
        if rollback_errors and on_status:
            on_status(
                "Storage rollback was incomplete; exact NPA ownership was preserved in "
                f"{transaction.credentials_path}. Resume with `{transaction.next_command}`."
            )
        if not isinstance(exc, Exception):
            raise
        safe_error = nebius.redact_nebius_output(str(exc))
        raise nebius.NebiusError(f"Storage provisioning failed: {safe_error}") from None
