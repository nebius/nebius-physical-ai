"""Crash-safe first-run object-storage provisioning and reconciliation."""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import os
from typing import Any, Callable, Mapping, cast
from uuid import uuid4

import yaml

from npa.clients.storage_validation import (
    StorageCapabilityProfile,
    StorageConvergencePolicy,
    StorageProbeResult,
    converge_storage_probe,
    probe_storage_write,
)
from npa.provisioning_journal import (
    ProvisioningOperation,
    current_operation,
    operation_context,
)

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
    from npa.clients.credentials import update_private_yaml

    update_private_yaml(path, lambda _current: data)


def _update_document(
    path: Path,
    updater: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> None:
    from npa.clients.credentials import update_private_yaml

    update_private_yaml(path, updater)


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
    operation: ProvisioningOperation | None = None
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

    def begin(self) -> bool:
        # An interrupted attempt owns its exact bucket name. Reconcile that
        # resource even if a later invocation proposes a different default;
        # creating the new name would orphan the first bucket and overwrite its
        # provenance. Unproven/legacy records are never adopted this way.
        resuming_owned_bucket = False
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
            resuming_owned_bucket = True
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
        if self.operation is not None:
            phase = str(self.operation.read().get("phase") or "")
            if phase != "mutating":
                self.operation.transition("mutating")
        return resuming_owned_bucket

    def record_created(self, kind: str, metadata: dict[str, str]) -> None:
        """Persist ownership before allowing the next fallible provider step."""

        allowlists = {
            "service_account": {"id", "name"},
            "bucket": {"id", "name"},
            "access_key": {"id", "name", "service_account_id"},
            "iam_group": {"id", "name"},
            "iam_permit": {"id", "group_id"},
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
        if self.operation is not None:
            resource_type = {
                "bucket": "storage_bucket",
                "service_account": "storage_service_account",
                "access_key": "storage_access_key",
                "iam_group": "storage_iam_group",
                "iam_permit": "storage_iam_access_permit",
            }[kind]
            self.operation.record_resource(
                resource_type=resource_type,
                requested_name=str(clean.get("name") or clean.get("id") or kind),
                provider_id=str(clean.get("id") or ""),
                ownership="created_by_this_operation",
                ownership_source="storage-provider-create-callback",
                project_id=self.project_id,
                creation_window_end=str(clean.get("created_at") or ""),
            )
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
        elif kind in {"bucket", "service_account", "iam_group", "iam_permit"}:
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
        account_name = str(account.get("name", "")).strip()
        if (
            str(account.get("created_by", "")) != "npa"
            or str(account.get("id", "")) != account_id
            or not account_name
            or str(account.get("project_id", "")) != self.project_id
        ):
            return {}
        return {
            "service_account_id": account_id,
            "service_account_name": account_name,
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
        saved_record = storage_setup_record(self.project_id, path=path)
        saved_resources = saved_record.get("resources")
        saved_resources = (
            deepcopy(saved_resources) if isinstance(saved_resources, dict) else {}
        )
        bucket_record = saved_resources.get("bucket")
        if not isinstance(bucket_record, dict):
            bucket_record = {
                "name": bucket.strip().removeprefix("s3://").strip("/"),
                "created_by": "pre_existing",
                "project_id": self.project_id,
                "attempt_id": self.attempt_id,
                "adopted_at": _now(),
            }
            saved_resources["bucket"] = bucket_record
        if self.operation is not None:
            self.operation.record_resource(
                resource_type="storage_bucket",
                requested_name=str(bucket_record.get("name") or self.bucket_name),
                provider_id=str(bucket_record.get("id") or ""),
                ownership=(
                    "created_by_this_operation"
                    if bucket_record.get("created_by") == "npa"
                    and bucket_record.get("attempt_id") == self.attempt_id
                    else "adopted"
                ),
                ownership_source="storage-write-probe",
                project_id=self.project_id,
            )
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
                        "resources": saved_resources,
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
        project_payload: dict[str, Any] = {
            "storage": payload.pop("storage"),
        }
        if account_id:
            project_payload["nebius"] = {"service_account_id": account_id}
            access_keys = saved_resources.get("access_keys")
            access_key_ids = (
                sorted(str(item) for item in access_keys)
                if isinstance(access_keys, dict)
                else []
            )
            generation = {
                **ownership,
                "service_account_id": account_id,
                "service_account_project_id": self.project_id,
                "access_key_ids": access_key_ids,
                "binding": {
                    "state": str(credentials.get("iam_binding_state", "")),
                    "role": str(credentials.get("iam_binding_role", "")),
                    "scope_id": str(credentials.get("iam_binding_scope_id", "")),
                    "group_id": str(credentials.get("iam_binding_group_id", "")),
                    "group_name": str(credentials.get("iam_binding_group_name", "")),
                    "access_permit_id": str(
                        credentials.get("iam_binding_access_permit_id", "")
                    ),
                    "group_managed_by": (
                        "npa"
                        if isinstance(saved_resources.get("iam_group"), dict)
                        and str(saved_resources["iam_group"].get("id") or "")
                        == str(credentials.get("iam_binding_group_id") or "")
                        else ""
                    ),
                    "access_permit_managed_by": (
                        "npa"
                        if isinstance(saved_resources.get("iam_permit"), dict)
                        and str(saved_resources["iam_permit"].get("id") or "")
                        == str(credentials.get("iam_binding_access_permit_id") or "")
                        else ""
                    ),
                },
            }
            project_payload["storage_iam"] = {
                **ownership,
                "active_service_account_id": account_id,
                "generations": [generation],
            }

        def commit_all(current: dict[str, Any]) -> dict[str, Any]:
            from npa.clients.project_credential_store import (
                merge_project_credentials_document,
            )

            document = _deep_merge(current, payload)
            return merge_project_credentials_document(
                document,
                self.project_id,
                project_payload,
                alias=self.project_alias,
                select=True,
            )

        _update_document(path, commit_all)

    def fail_and_rollback(self, exc: BaseException) -> list[str]:
        """Roll back only this attempt's exact creations; preserve failures."""

        from npa.clients import nebius

        rollback_errors: list[str] = []
        remaining: list[tuple[str, dict[str, str]]] = []
        for kind, metadata in reversed(self._created_this_attempt):
            try:
                if kind == "access_key":
                    nebius.delete_access_key(metadata.get("id", ""))
                elif kind == "iam_permit":
                    nebius.delete_access_permit(metadata.get("id", ""))
                elif kind == "iam_group":
                    nebius.delete_group(metadata.get("id", ""))
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
        def remove_from(current: dict[str, Any]) -> dict[str, Any]:
            document = deepcopy(current)
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
            return document

        _update_document(path, remove_from)

    def _update_record(self, patch: Mapping[str, Any]) -> None:
        path = self.credentials_path
        def update(current: dict[str, Any]) -> dict[str, Any]:
            document = deepcopy(current)
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
            return document

        _update_document(path, update)


def provision_storage(
    *,
    project_id: str,
    tenant_id: str,
    region: str,
    bucket_name: str,
    project_alias: str = "",
    bucket_max_size_bytes: int = 0,
    bucket_storage_class: str = "standard",
    service_account_name: str = "lerobot-training",
    access_key_name: str = "lerobot-access-key",
    on_status: StatusFn | None = None,
    convergence_policy: StorageConvergencePolicy = StorageConvergencePolicy(),
    convergence_sleep: Callable[[float], None] | None = None,
    convergence_random: Callable[[], float] | None = None,
    allow_editors_fallback: bool = False,
    allow_existing_bucket: bool = True,
) -> tuple[dict[str, str], StorageProbeResult]:
    """Reconcile, validate and atomically commit first-run storage."""

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("provision_storage")

    from npa.clients import nebius

    parent_operation = current_operation()
    owns_operation = parent_operation is None
    operation = parent_operation or ProvisioningOperation.prepare(
        command="npa provision-if-absent",
        project_alias=project_alias,
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        backend={"bucket": bucket_name, "region": region},
        resource_type="storage",
        requested_name=bucket_name,
        ownership_source="storage-setup",
        resume_command=(
            "npa provision-if-absent"
            + (f" --project {project_alias}" if project_alias else "")
            + " --skip-k8s"
        ),
    )
    operation.update_identity(
        project_alias=project_alias,
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        backend={"bucket": bucket_name, "region": region},
    )
    transaction = StorageSetupTransaction(
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        bucket_name=bucket_name,
        project_alias=project_alias,
        operation=operation,
    )
    # Promote provably owned legacy credentials before ``begin`` changes the
    # legacy storage-setup record from complete to in-progress. This preserves
    # the exact ownership proof during a deliberate bucket replacement.
    from npa.clients.project_credential_store import project_credential_record

    try:
        project_credential_record(
            project_id,
            alias=project_alias,
            path=transaction.credentials_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise StorageSetupStateError(
            f"storage owner state is unavailable or malformed: {exc}"
        ) from exc
    context = operation_context(operation) if owns_operation else nullcontext(operation)
    with context:
        resuming_owned_bucket = transaction.begin()
        try:
            identity_name_kwargs: dict[str, str] = {}
            if service_account_name != "lerobot-training":
                identity_name_kwargs["service_account_name"] = service_account_name
            if access_key_name != "lerobot-access-key":
                identity_name_kwargs["access_key_name"] = access_key_name
            fallback_enabled = (
                allow_editors_fallback
                or os.environ.get("NPA_ALLOW_EDITORS_STORAGE_FALLBACK", "")
                .strip()
                .lower()
                in {"1", "true", "yes"}
            )
            fallback_kwargs: dict[str, bool] = {}
            if fallback_enabled:
                fallback_kwargs["allow_editors_fallback"] = True
            bucket_reuse_kwargs: dict[str, bool] = {}
            if not allow_existing_bucket and not resuming_owned_bucket:
                bucket_reuse_kwargs["allow_existing_bucket"] = False
            bootstrap = cast(Callable[..., dict[str, str]], nebius.bootstrap_environment)
            credentials = bootstrap(
                project_id,
                tenant_id,
                region,
                bucket_name=transaction.bucket_name,
                bucket_max_size_bytes=bucket_max_size_bytes,
                bucket_storage_class=bucket_storage_class,
                on_status=on_status,
                on_resource_created=transaction.record_created,
                **bucket_reuse_kwargs,
                **fallback_kwargs,
                **identity_name_kwargs,
            )
            def run_probe() -> StorageProbeResult:
                return probe_storage_write(
                    bucket=credentials.get("s3_bucket", ""),
                    endpoint_url=credentials.get("s3_endpoint", ""),
                    access_key_id=credentials.get("nebius_api_key", ""),
                    secret_access_key=credentials.get("nebius_secret_key", ""),
                    region=region,
                    profile=StorageCapabilityProfile.STANDARD,
                )

            convergence_kwargs: dict[str, Any] = {}
            if convergence_sleep is not None:
                convergence_kwargs["sleep"] = convergence_sleep
            if convergence_random is not None:
                convergence_kwargs["random_value"] = convergence_random

            def report_retry(result: StorageProbeResult, delay: float) -> None:
                if on_status:
                    on_status(
                        "Waiting for newly granted S3 credentials/IAM to converge "
                        f"after typed {result.phase} failure (retry in {delay:.2f}s)..."
                    )

            probe = converge_storage_probe(
                run_probe,
                propagation_context=(
                    str(credentials.get("iam_binding_state", "")) == "created"
                    or any(
                        kind == "access_key"
                        for kind, _metadata in transaction._created_this_attempt
                    )
                ),
                policy=convergence_policy,
                on_retry=report_retry,
                **convergence_kwargs,
            )
            if not probe.ok:
                raise nebius.NebiusError(probe.summary)
            if probe.retained_object and on_status:
                on_status(probe.summary)
            transaction.commit(credentials, probe)
            if owns_operation:
                operation.transition(
                    "state-durable", details={"storage_probe": "passed"}
                )
                operation.commit()
            return credentials, probe
        except BaseException as exc:
            rollback_errors = transaction.fail_and_rollback(exc)
            phase = "rollback-incomplete" if rollback_errors else "rolled-back"
            if owns_operation:
                operation.transition(phase, error=str(exc))
            if rollback_errors and on_status:
                on_status(
                    "Storage rollback was incomplete; exact NPA ownership was preserved in "
                    f"{transaction.credentials_path}. Resume with `{transaction.next_command}`."
                )
            if not isinstance(exc, Exception):
                raise
            safe_error = nebius.redact_nebius_output(str(exc))
            raise nebius.NebiusError(
                f"Storage provisioning failed: {safe_error}. Operation "
                f"{operation.operation_id}: {operation.path}. Resume: "
                f"{transaction.next_command}"
            ) from exc
