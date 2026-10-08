"""Execute authorized team runs through the existing NPA workflow runtime."""

from __future__ import annotations

import json
import time
from urllib.parse import urlsplit

from botocore.exceptions import ClientError

from npa.orchestration.npa_workflow.run_state import RunStateStore
from npa.orchestration.npa_workflow.runtime import (
    RuntimeLedger,
    RuntimeOptions,
    SkyPilotWaveExecutor,
    run_workflow_runtime,
    s3_trigger_waiter,
)
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

from .storage import PersonalStorage, PrivateRunFiles, authorize_uri, workload_secrets
from .workflow_policy import load_bound_spec, verify_artifact_scopes


class TeamExecutor(SkyPilotWaveExecutor):
    """Check resolved artifact scopes at each canonical workflow execution boundary.

    Args:
        args, kwargs: Canonical SkyPilotWaveExecutor initialization arguments.
    Returns:
        A TeamExecutor instance.
    Raises:
        NpaWorkflowError: Canonical runtime initialization fails.
    """

    def execute(self, step):
        """Validate one resolved step before handing it to the standard runtime.

        Args:
            step: Canonically resolved workflow step.
        Returns:
            Standard runtime step result.
        Raises:
            TeamError, NpaWorkflowError: Policy or execution fails.
        """
        self.team_check()
        verify_artifact_scopes([step], self.team_binding)
        return super().execute(step)

    def execute_parallel(self, steps, *, group, max_concurrency):
        """Validate every fan-out member before launching the first member.

        Args:
            steps: Resolved fan-out members.
            group, max_concurrency: Canonical runtime scheduling parameters.
        Returns:
            Standard runtime fan-out results.
        Raises:
            TeamError, NpaWorkflowError: Policy or execution fails.
        """
        self.team_check()
        verify_artifact_scopes(steps, self.team_binding)
        return super().execute_parallel(
            steps, group=group, max_concurrency=max_concurrency
        )


class RunObjects:
    """Supply scoped artifact checks, decisions, and triggers to the standard runtime.

    Args:
        binding: Personal execution and artifact boundary.
        storage: Optional personal object transport for tests.
    Returns:
        A RunObjects instance.
    Raises:
        TeamError: Allocated cloud credentials are invalid.
    """

    def __init__(self, binding, *, storage=None):
        self.binding = binding
        self.storage = storage or PersonalStorage(binding.allocation.storage)

    def read(self, bucket, key):
        """Read a decision under the personal or explicitly shared input scope.

        Args:
            bucket, key: Resolved decision object's location.
        Returns:
            Decoded UTF-8 object contents.
        Raises:
            TeamError, ClientError: Scope or cloud access fails.
            FileNotFoundError: The decision object does not exist.
        """
        self._authorize(bucket, key)
        try:
            response = self.storage.client.get_object(Bucket=bucket, Key=key)
            with response["Body"] as body:
                return body.read().decode()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(key) from exc
            raise

    def list(self, bucket, prefix):
        """List a trigger prefix using only the allocated storage principal.

        Args:
            bucket, prefix: Resolved trigger's object scope.
        Returns:
            Matching object keys.
        Raises:
            TeamError, ClientError: Scope or cloud access fails.
        """
        self._authorize(bucket, prefix)
        pages = self.storage.client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        )
        return [item["Key"] for page in pages for item in page.get("Contents", [])]

    def exists(self, uri):
        """Verify nonempty declared output using explicit personal credentials.

        Args:
            uri: Resolved declared S3 output.
        Returns:
            Whether a nonempty object or prefix exists.
        Raises:
            TeamError, ClientError: Scope or cloud access fails.
        """
        authorize_uri(uri, self.binding.allocation.storage)
        parsed = urlsplit(uri)
        key = parsed.path.lstrip("/")
        try:
            if uri.endswith("/"):
                pages = self.storage.client.get_paginator("list_objects_v2").paginate(
                    Bucket=parsed.netloc, Prefix=key
                )
                return any(
                    item["Size"] > 0
                    for page in pages
                    for item in page.get("Contents", [])
                )
            response = self.storage.client.head_object(Bucket=parsed.netloc, Key=key)
            return response["ContentLength"] > 0
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def _authorize(self, bucket, key):
        allocation = self.binding.allocation
        authorize_uri(
            f"s3://{bucket}/{key}", allocation.storage, allocation.shared_inputs
        )

    def publish_record(self, bucket, key, body):
        """Publish a non-authoritative record copy using the personal storage principal.

        Args:
            bucket, key, body: Destination and serialized runtime record.
        Returns:
            None.
        Raises:
            TeamError, ClientError: Scope or cloud write fails.
        """
        authorize_uri(f"s3://{bucket}/{key}", self.binding.allocation.storage)
        self.storage.client.put_object(Bucket=bucket, Key=key, Body=body)


def execute_run(config, binding, record, backend, check, *, resume=False, storage=None):
    """Drive one team run through canonical planning, rendering, decisions, and resume.

    Args:
        config, binding, record, backend: Trusted installation and durable run identity.
        check: Callback checking current policy and cancellation before new work.
        resume: Reconcile the persisted runtime instead of submitting it again.
        storage: Injectable explicit object client for acceptance tests.
    Returns:
        Canonical RuntimeReport.
    Raises:
        TeamError, NpaWorkflowError: Policy or workflow execution fails.
    """
    spec, document = load_bound_spec(
        json.loads(record["workflow"]), binding, record["id"], backend.root
    )
    objects = RunObjects(binding, storage=storage)
    store = _state_store(binding, backend.root, document["config"]["prefix"], objects)
    options = _runtime_options(binding, resume)
    render = SkypilotRenderOptions(
        aws_endpoint_url=binding.allocation.storage.endpoint,
        materialize_registry_secrets=False,
    )
    ledger = RuntimeLedger(
        store,
        workflow=spec.name,
        run_id=record["id"],
        api_version=spec.api_version,
        resume=resume,
    )
    executor = _executor(spec, record, backend, objects, ledger, options, render, check)
    executor.team_binding, executor.team_check = binding, check
    return _drive(spec, record, document, executor, objects, check)


def _state_store(binding, root, prefix, objects):
    files = PrivateRunFiles(root / "records", prefix)
    store = RunStateStore(
        bucket=binding.allocation.storage.bucket,
        prefix=prefix,
        reader=files.read,
        writer=_record_writer(files, objects),
    )
    return store


def _runtime_options(binding, resume):
    secrets = workload_secrets(binding.allocation)
    return RuntimeOptions(
        max_wait_seconds=0,
        max_infrastructure_recoveries=0,
        resume=resume,
        supervise_pending=False,
        secret_envs=tuple(secrets),
        secret_env_values=secrets,
    )


def _drive(spec, record, document, executor, objects, check):
    waiter = s3_trigger_waiter(
        ledger=executor.ledger,
        lister=objects.list,
        sleeper=lambda seconds: _sleep(seconds, check),
        max_wait_seconds=0,
    )
    return run_workflow_runtime(
        spec,
        run_id=record["id"],
        executor=executor,
        options=executor.options,
        render_options=executor.render_options,
        state_store=executor.ledger.store,
        decision_reader=objects.read,
        trigger_waiter=waiter,
        workflow_yaml=json.dumps(document).encode(),
    )


def _executor(spec, record, backend, objects, ledger, options, render, check):
    return TeamExecutor(
        spec,
        run_id=record["id"],
        options=options,
        render_options=render,
        ledger=ledger,
        submitter=backend.submit,
        status_fn=backend.status,
        timeline_fn=backend.timeline,
        canceller=backend.cancel,
        reconcile_fn=backend.reconcile,
        output_checker=objects.exists,
        name_lookup_fn=lambda name: [],
        sleeper=lambda seconds: _sleep(seconds, check),
    )


def _record_writer(files, objects):
    def write(bucket, key, body):
        files.write(bucket, key, body)
        objects.publish_record(bucket, key, body)

    return write


def _sleep(seconds, check):
    check()
    time.sleep(seconds)
    check()
