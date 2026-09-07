"""One immutable execution destination, verified before workload creation.

Presence is configuration evidence; provider scope and S3 access are separate
facts. Reports deliberately contain roles and provenance, never live identities
or provider exception text. The target itself stays in owner-only runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from npa.orchestration.npa_workflow.submit_credentials import (
    STORAGE_ENDPOINT_ENV_NAMES,
    SubmitCredentialContext,
    resolve_submit_credentials,
    storage_endpoint_from_environment,
)


class ExecutionPreflightError(RuntimeError):
    """A failed or unknown prerequisite prevents creation, without secret text."""

    def __init__(self, check: str, reason: str, *, status: str = "fail") -> None:
        self.check = check
        self.status = status
        super().__init__(f"execution preflight {check}: {reason}")


@dataclass(frozen=True)
class ExecutionTarget:
    project: str = field(repr=False)
    project_id: str = field(repr=False)
    tenant_id: str = field(repr=False)
    region: str = field(repr=False)
    context: str = field(default="", repr=False)
    output_uris: tuple[str, ...] = field(default=(), repr=False)
    credentials: SubmitCredentialContext = field(default_factory=SubmitCredentialContext, repr=False)
    provenance: Mapping[str, str] = field(default_factory=dict)
    output_kinds: Mapping[str, str] = field(default_factory=dict, repr=False)


def _destination(uri: str, kind: str = "") -> tuple[str, str]:
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3" or not parsed.netloc or parsed.username
        or parsed.password or parsed.netloc != parsed.hostname or parsed.query or parsed.fragment
        or "{{" in uri or "${" in uri or "\\" in uri
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise ExecutionPreflightError("storage_target", "output must be a resolved S3 location without credentials or traversal")
    key = parsed.path.lstrip("/")
    if kind not in {"", "file", "directory"}:
        raise ExecutionPreflightError("storage_target", "output kind must be file or directory")
    # Directory semantics come from the workload contract, never an extension
    # heuristic. Legacy URI declarations keep their exact-object convention,
    # with trailing slash denoting a directory.
    directory = kind == "directory" or (not kind and uri.endswith("/"))
    prefix = key.rstrip("/") if directory else key.rpartition("/")[0]
    return parsed.netloc, prefix


def resolve_execution_target(
    *, project: str = "", project_id: str = "", context: str = "",
    region: str = "", output_uris: Sequence[str] = (),
    credentials: SubmitCredentialContext | None = None,
    provenance: Mapping[str, str] | None = None,
    output_kinds: Mapping[str, str] | None = None,
) -> ExecutionTarget:
    from npa.clients.config import default_project_name, resolve_environment

    alias = project or default_project_name()
    saved = resolve_environment(alias)
    if saved is None or not saved.project_id or not saved.tenant_id or not saved.region:
        raise ExecutionPreflightError("scope", "selected project requires saved project, tenant and region identity")
    if project_id and project_id != saved.project_id:
        raise ExecutionPreflightError("scope", "explicit project identity disagrees with the selected project")
    if region and region != saved.region:
        raise ExecutionPreflightError("region", "explicit region disagrees with the selected project")
    uris = tuple(dict.fromkeys(str(uri) for uri in output_uris if uri))
    for uri in uris:
        _destination(uri, (output_kinds or {}).get(uri, ""))
    selected = credentials if credentials is not None else resolve_submit_credentials(project=alias)
    if uris:
        endpoint = urlparse(selected.endpoint_url)
        if (
            endpoint.scheme != "https" or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment or endpoint.path not in {"", "/"}
            or endpoint.netloc != f"storage.{saved.region}.nebius.cloud"
        ):
            raise ExecutionPreflightError("endpoint", "storage endpoint must match the selected Nebius project region")
        if not selected.access_key_id or not selected.secret_access_key:
            raise ExecutionPreflightError("credentials", "execution storage credential pair is missing")
    return ExecutionTarget(
        alias, saved.project_id, saved.tenant_id, saved.region, context, uris,
        selected,
        {**dict(selected.provenance), "project": "cli" if project else "project.default",
         "scope": "project.config", "context": "submit.infra" if context else "not-required",
         "outputs": "workload", **dict(provenance or {})}, dict(output_kinds or {}),
    )


def verify_execution_scope(target: ExecutionTarget, *, verify_cluster: bool = True) -> dict[str, str]:
    """Read-only ownership checks. A denied/unknown owner never permits S3 writes."""
    from npa.clients.nebius import get_bucket_by_name, get_project_identity

    try:
        remote = get_project_identity(target.project_id, tenant_id=target.tenant_id)
    except Exception as exc:
        raise ExecutionPreflightError("scope", "provider project identity is unavailable or mismatched", status="unknown") from exc
    if remote is None:
        raise ExecutionPreflightError("scope", "selected project does not exist")
    if remote.project_id != target.project_id or remote.tenant_id != target.tenant_id or remote.region != target.region:
        raise ExecutionPreflightError("scope", "provider project, tenant or region disagrees with the execution target")
    if target.context:
        from npa.cluster.state import load_cluster_state

        try:
            local = load_cluster_state(target.context)
        except Exception as exc:
            raise ExecutionPreflightError("cluster_owner", "saved cluster identity is unreadable", status="unknown") from exc
        if local is not None and local.project_id != target.project_id:
            raise ExecutionPreflightError("cluster_owner", "saved context belongs to a different project")
    for bucket in dict.fromkeys(_destination(uri)[0] for uri in target.output_uris):
        try:
            bucket_record = get_bucket_by_name(target.project_id, bucket)
        except Exception as exc:
            raise ExecutionPreflightError("storage_owner", "exact output bucket ownership is unavailable or mismatched", status="unknown") from exc
        metadata = bucket_record.get("metadata", {}) if isinstance(bucket_record, Mapping) else {}
        parent = metadata.get("parent_id") or metadata.get("parentId")
        if not bucket_record or parent != target.project_id or metadata.get("name") != bucket:
            raise ExecutionPreflightError("storage_owner", "output bucket is not proven to belong to the selected project")
    if target.context and verify_cluster:
        from npa.cluster.identity import resolve_verified_cluster_identity

        try:
            cluster = resolve_verified_cluster_identity(project=target.project, context=target.context)
        except Exception as exc:
            raise ExecutionPreflightError("cluster_owner", "exact Kubernetes context and provider ownership could not be verified", status="unknown") from exc
        if cluster.cluster_absent or cluster.project_id != target.project_id or cluster.context != target.context:
            raise ExecutionPreflightError("cluster_owner", "Kubernetes context does not identify the selected live project cluster")
    return {"scope": "pass", "storage_owner": "pass" if target.output_uris else "not-required",
            "cluster_owner": "pass" if target.context and verify_cluster else "not-checked"}


def verify_execution_target(
    target: ExecutionTarget, *, gpu_check: Callable[[], Any] | None = None,
    verify_cluster: bool = True,
) -> dict[str, Any]:
    """Verify scope/GPU before exact-prefix PUT/GET using the executing keys."""
    from npa.clients.storage_validation import StorageCapabilityProfile, probe_storage_write

    checks = verify_execution_scope(target, verify_cluster=verify_cluster)
    if gpu_check is not None:
        try:
            gpu_check()
        except ExecutionPreflightError:
            raise
        except Exception as exc:
            raise ExecutionPreflightError("gpu", "requested product/shape/capacity is unsupported or could not be verified", status="unknown") from exc
        checks["gpu"] = "pass"
    else:
        checks["gpu"] = "not-checked"
    destinations = tuple(dict.fromkeys(_destination(uri, target.output_kinds.get(uri, "")) for uri in target.output_uris))
    retained = 0
    for bucket, prefix in destinations:
        probe = probe_storage_write(
            bucket=bucket, prefix=prefix, endpoint_url=target.credentials.endpoint_url,
            access_key_id=target.credentials.access_key_id,
            secret_access_key=target.credentials.secret_access_key,
            region=target.region, profile=StorageCapabilityProfile.STANDARD,
        )
        if not probe.ok:
            # Summary can include a retained private URI. Keep only the typed
            # phase/failure category in ordinary diagnostics.
            kind = probe.error.kind.value if probe.error else "unknown"
            raise ExecutionPreflightError("storage_access", f"exact output prefix {probe.phase} failed ({kind})")
        retained += int(probe.retained_object)
    checks["storage_write_read"] = "pass" if destinations else "not-required"
    return {"schema_version": "npa.execution-preflight.v1", "presence": "pass",
            "access": "pass", "execution_readiness": "pass" if verify_cluster else "pending-cluster-verification", "checks": checks,
            "destination_count": len(destinations), "retained_probe_count": retained,
            "provenance": dict(target.provenance)}


def verify_serverless_gpu(*, project_id: str, gpu_type: str, gpu_count: int, preset: str) -> None:
    """Query current project-regional offerings; a catalog is not allocation proof."""
    from npa.clients.nebius import _run_json

    try:
        payload = _run_json(["compute", "platform", "list", "--parent-id", project_id, "--all"])
    except Exception as exc:
        raise ExecutionPreflightError("gpu_product", "provider platform catalog is unavailable", status="unknown") from exc
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        raise ExecutionPreflightError("gpu_product", "provider platform catalog is incomplete", status="unknown")
    matches = [item for item in items if isinstance(item, Mapping) and item.get("metadata", {}).get("name") == gpu_type]
    if not matches:
        raise ExecutionPreflightError("gpu_product", "requested platform is unavailable in the selected project region")
    if len(matches) != 1:
        raise ExecutionPreflightError("gpu_product", "provider product identity is ambiguous", status="unknown")
    # Supported shape must be proven by the current provider response, never a
    # static list or a substituted product. Missing evidence is not unsupported.
    platform = matches[0]
    metadata = platform.get("metadata", {})
    if not metadata.get("parent_id") or not metadata.get("id"):
        raise ExecutionPreflightError("gpu_product", "provider platform identity is incomplete", status="unknown")
    if metadata["parent_id"] != project_id:
        # PlatformService.List and GetByName are scoped by the request's parent.
        # A regional catalog product can itself belong to a provider catalog
        # project. Corroborate its exact identity and offered shape through the
        # selected workload project; do not confuse product ownership with
        # workload ownership or accept an unrelated catalog response.
        try:
            scoped = _run_json(["compute", "platform", "get-by-name", "--parent-id", project_id, "--name", gpu_type])
        except Exception as exc:
            raise ExecutionPreflightError("gpu_product", "project-scoped product lookup is unavailable", status="unknown") from exc
        if not isinstance(scoped, Mapping) or any(
            scoped.get("metadata", {}).get(key) != metadata.get(key) for key in ("id", "name", "parent_id")
        ) or scoped.get("spec", {}).get("presets") != platform.get("spec", {}).get("presets"):
            raise ExecutionPreflightError("gpu_product", "project-scoped product lookup disagrees with the catalog", status="unknown")
    presets = matches[0].get("spec", {}).get("presets")
    if not isinstance(presets, list):
        raise ExecutionPreflightError("gpu_product", "provider preset evidence is incomplete", status="unknown")
    shapes = [item for item in presets if isinstance(item, Mapping) and item.get("name") == preset]
    if not shapes:
        raise ExecutionPreflightError("gpu_product", "requested preset is unavailable for the selected platform")
    if shapes[0].get("resources", {}).get("gpu_count") != gpu_count:
        raise ExecutionPreflightError("gpu_product", "requested GPU count disagrees with the selected preset")


def verify_serverless_execution(
    *, project: str, project_id: str, gpu_type: str, gpu_count: int, preset: str,
    output_uri: str, extra_env: Mapping[str, str],
) -> dict[str, Any]:
    """Shared CLI/SDK gate using exactly the environment passed to the worker."""
    access = str(extra_env.get("AWS_ACCESS_KEY_ID") or "")
    secret = str(extra_env.get("AWS_SECRET_ACCESS_KEY") or "")
    endpoint = storage_endpoint_from_environment(extra_env)
    credentials = SubmitCredentialContext(
        access_key_id=access, secret_access_key=secret, endpoint_url=endpoint,
        provenance={"credentials": "worker.environment", "endpoint": "worker.environment"},
    )
    target = resolve_execution_target(project=project, project_id=project_id,
                                      output_uris=[output_uri], output_kinds={output_uri: "directory"}, credentials=credentials)
    verify_worker_environment(target, [{"envs": extra_env}])
    return verify_execution_target(target, gpu_check=lambda: verify_serverless_gpu(
        project_id=project_id, gpu_type=gpu_type, gpu_count=gpu_count, preset=preset,
    ))


def workflow_output_destinations(spec: Any, *, run_id: str, assume_decision: str = "") -> dict[str, str]:
    """Read destinations from the same resolved plan and ledger config as submit."""
    from npa.orchestration.npa_workflow.runtime import plan_preview
    from npa.orchestration.npa_workflow.interpreter import _make_context

    plan = plan_preview(spec, run_id=run_id, assume_decision=assume_decision)
    destinations = {str(output["uri"]): str(output.get("kind") or "") for step in plan.steps for output in step.outputs
                    if str(output.get("uri") or "").startswith("s3://")}
    config = _make_context(spec, run_id=run_id).config
    bucket = str(config.get("bucket") or "")
    if bucket:
        prefix = str(config.get("prefix") or run_id).strip("/")
        destinations[f"s3://{bucket}/{prefix}/"] = "directory"
    return destinations


def workflow_output_uris(spec: Any, *, run_id: str, assume_decision: str = "") -> tuple[str, ...]:
    """Compatibility accessor; launch code retains roles with the destination map."""
    return tuple(workflow_output_destinations(spec, run_id=run_id, assume_decision=assume_decision))


def verify_worker_environment(target: ExecutionTarget, documents: Sequence[Mapping[str, Any]]) -> None:
    """Reject runtime/pod overrides that would change the preflight principal.

    Secret references in custom pod configuration cannot be equated to the
    checked executing credentials without reading them. Refuse that uncertainty
    rather than silently relying on a different secret at execution time.
    """
    expected = {
        "AWS_ACCESS_KEY_ID": target.credentials.access_key_id,
        "AWS_SECRET_ACCESS_KEY": target.credentials.secret_access_key,
        # Static Nebius S3 keys are the checked principal. A pod-level token
        # reference must not change signing after those checks passed.
        "AWS_SESSION_TOKEN": "",
        **dict.fromkeys(STORAGE_ENDPOINT_ENV_NAMES, target.credentials.endpoint_url),
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            envs = value.get("envs")
            if isinstance(envs, Mapping):
                for key, selected in expected.items():
                    if key in envs and envs[key] != selected:
                        raise ExecutionPreflightError("worker_environment", f"rendered {key} disagrees with the verified execution target")
            env = value.get("env")
            if isinstance(env, list):
                for entry in env:
                    if isinstance(entry, Mapping) and entry.get("name") in expected:
                        if "valueFrom" in entry or entry.get("value") != expected[entry["name"]]:
                            raise ExecutionPreflightError("worker_environment", "pod storage credentials or endpoint differ from the verified execution target")
            if value.get("envFrom"):
                raise ExecutionPreflightError("worker_environment", "custom pod envFrom may override the execution principal; declare verified storage values explicitly", status="unknown")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for document in documents:
        walk(document)


def skypilot_task_documents(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Exclude SkyPilot's multi-document execution header from task mutation."""
    tasks = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise ExecutionPreflightError("worker_environment", "task document must be a mapping")
        if index == 0 and len(documents) > 1 and "execution" in document and not any(
            key in document for key in ("run", "setup", "resources", "envs")
        ):
            continue
        tasks.append(document)
    return tasks


def skypilot_workflow_environment(documents: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Collect one credential/endpoint source from actual raw task environments."""
    selected: dict[str, str] = {}
    protected = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", *STORAGE_ENDPOINT_ENV_NAMES}
    for document in skypilot_task_documents(documents):
        env = document.get("envs") or {}
        if not isinstance(env, Mapping):
            raise ExecutionPreflightError("worker_environment", "task envs must be a mapping")
        for name in protected:
            value = env.get(name)
            if value is None or value == "":
                continue
            if not isinstance(value, str) or "${" in value or "{{" in value:
                raise ExecutionPreflightError("worker_environment", "task storage environment must be fully resolved")
            if name in selected and selected[name] != value:
                raise ExecutionPreflightError("worker_environment", "tasks declare different execution credentials or endpoints")
            selected[name] = value
    return selected


def skypilot_output_destinations(documents: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Use declared destination contracts, never infer writes from shell text.

    Raw tasks may declare NPA_EXECUTION_OUTPUTS as JSON [{uri, kind}], or use
    supported directory-output environment contracts. Empty JSON [] explicitly
    declares that a task has no durable outputs (for example an input-only job).
    """
    destinations: dict[str, str] = {}
    for document in skypilot_task_documents(documents):
        env = document.get("envs") or {}
        declared = env.get("NPA_EXECUTION_OUTPUTS")
        if declared is not None:
            try:
                outputs = json.loads(str(declared))
            except (ValueError, TypeError) as exc:
                raise ExecutionPreflightError("storage_target", "NPA_EXECUTION_OUTPUTS must be a JSON array of uri/kind declarations") from exc
            if not isinstance(outputs, list):
                raise ExecutionPreflightError("storage_target", "NPA_EXECUTION_OUTPUTS must be an array")
            for output in outputs:
                if not isinstance(output, Mapping) or output.get("kind") not in {"file", "directory"}:
                    raise ExecutionPreflightError("storage_target", "each output requires an explicit file or directory kind")
                uri = str(output.get("uri") or "")
                _destination(uri, str(output["kind"]))
                destinations[uri] = str(output["kind"])
            continue
        found = False
        for name in ("NPA_OUTPUT_PATH", "NPA_OUTPUT_URI", "S3_OUTPUT_PATH", "NPA_WORKFLOW_RUN_PREFIX_URI"):
            uri = str(env.get(name) or "")
            if uri.startswith("s3://"):
                _destination(uri, "directory")
                destinations[uri] = "directory"
                found = True
        bucket = str(env.get("NPA_S3_BUCKET") or env.get("S3_BUCKET") or "")
        prefix = str(env.get("NPA_S3_PREFIX") or env.get("SONIC_OUTPUT_PREFIX") or "")
        if bucket and prefix:
            uri = f"s3://{bucket}/{prefix.strip('/')}"
            _destination(uri, "directory")
            destinations[uri] = "directory"
            found = True
        # A raw shell can hide writes in arbitrary code, so the only supported
        # storage-bearing contract is an explicit declaration or known output
        # environment. A list/head credential probe cannot fill that gap.
        if not found and ("s3://" in json.dumps(document) or bucket or prefix):
            raise ExecutionPreflightError("storage_target", "raw storage task requires NPA_EXECUTION_OUTPUTS with explicit output roles")
    return destinations


def preflight_skypilot_submission(
    documents: Sequence[dict[str, Any]], *, project: str = "", infra: str = "",
    extra_env: Mapping[str, str] | None = None,
    target: ExecutionTarget | None = None,
    global_config: Mapping[str, Any] | None = None,
    sky_bin: str = "",
    cwd: str | None = None,
) -> tuple[ExecutionTarget, dict[str, Any], dict[str, str]]:
    """Shared raw/rendered SkyPilot CLI+SDK gate before controller or job create.

    Mutates only the in-memory documents to pin resolved env values. Callers
    persist those documents owner-only before passing them to SkyPilot.
    """
    process_env = dict(os.environ)
    process_env.update(extra_env or {})
    if process_env.get("SKYPILOT_CONFIG"):
        raise ExecutionPreflightError("worker_environment", "internal SkyPilot config override prevents verification of the effective task environment", status="unknown")
    if cwd is not None or process_env.get("SKYPILOT_PROJECT_CONFIG"):
        from pathlib import Path
        import yaml

        project_config = Path(process_env.get("SKYPILOT_PROJECT_CONFIG") or str(Path(cwd or ".") / ".sky.yaml")).expanduser()
        try:
            project_values = yaml.safe_load(project_config.read_text()) if project_config.exists() else None
        except Exception as exc:
            raise ExecutionPreflightError("worker_environment", "implicit SkyPilot project configuration is unreadable", status="unknown") from exc
        if project_values:
            raise ExecutionPreflightError("worker_environment", "implicit SkyPilot project overrides require an explicit --config-path and removal of the implicit override", status="unknown")
    documents = skypilot_task_documents(documents)
    workflow_env = skypilot_workflow_environment(documents)
    if any(not isinstance(document.get("resources") or {}, Mapping) for document in documents):
        raise ExecutionPreflightError("gpu", "alternative resource targets are ambiguous; select one effective resource mapping")
    selected = target.credentials if target is not None else resolve_submit_credentials(
        project=project, environ=process_env, workflow_env=workflow_env,
    )
    task_resources = [(document, document.get("resources") or {}) for document in documents]
    controller = ((global_config or {}).get("jobs") or {}).get("controller") or {}
    if controller.get("resources"):
        if not isinstance(controller["resources"], Mapping):
            raise ExecutionPreflightError("scope", "controller resources must identify one execution target")
        task_resources.append(({"resources": controller["resources"]}, controller["resources"]))
    default_cloud = "nebius" if infra.split("/", 1)[0] == "nebius" else "kubernetes"
    native_documents = []
    kubernetes_documents = []
    for document, resources in task_resources:
        resource_infra = str(resources.get("infra") or "").split("/")
        cloud = str(resources.get("cloud") or resource_infra[0] or default_cloud).lower()
        if resource_infra[0] and resource_infra[0].lower().replace("k8s", "kubernetes") != cloud.replace("k8s", "kubernetes"):
            raise ExecutionPreflightError("scope", "resource cloud and infrastructure disagree")
        if cloud == "nebius":
            document.setdefault("resources", {})["cloud"] = "nebius"
            native_documents.append(document)
        elif cloud in {"kubernetes", "k8s"}:
            if len(resource_infra) > 1:
                if resources.get("region") not in (None, "", resource_infra[1]):
                    raise ExecutionPreflightError("cluster_owner", "resource context and infrastructure disagree")
                resources["region"] = resource_infra[1]
            kubernetes_documents.append(document)
        else:
            raise ExecutionPreflightError("scope", "task cloud is outside the selected Nebius execution contract")
    if infra and ((default_cloud == "nebius" and any(document in documents for document in kubernetes_documents)) or
                  (infra.startswith(("k8s/", "kubernetes/")) and any(document in documents for document in native_documents))):
        raise ExecutionPreflightError("scope", "task cloud disagrees with the explicit infrastructure target")
    contexts = {str((document.get("resources") or {}).get("region") or "") for document in kubernetes_documents}
    contexts.discard("")
    explicit_context = infra.split("/", 1)[1] if infra.startswith(("k8s/", "kubernetes/")) else ""
    if explicit_context and contexts and contexts != {explicit_context}:
        raise ExecutionPreflightError("cluster_owner", "task context disagrees with explicit submission context")
    if len(contexts) > 1:
        raise ExecutionPreflightError("cluster_owner", "tasks select multiple Kubernetes contexts")
    context = explicit_context or next(iter(contexts), "")
    if kubernetes_documents and not context:
        raise ExecutionPreflightError("cluster_owner", "raw/runtime submit requires one explicit Kubernetes context; ambient context is not execution evidence")
    for document in kubernetes_documents:
        resources = document.setdefault("resources", {})
        resources["cloud"] = "kubernetes"
        resources["region"] = context
        resources.pop("infra", None)
    if context and isinstance(global_config, dict):
        kube_config = global_config.setdefault("kubernetes", {})
        if not isinstance(kube_config, dict):
            raise ExecutionPreflightError("cluster_owner", "Kubernetes runtime configuration must be a mapping")
        kube_config["allowed_contexts"] = [context]
    for document in documents:
        envs = document.setdefault("envs", {})
        bucket = process_env.get("NPA_S3_BUCKET")
        prefix = process_env.get("NPA_S3_PREFIX")
        if bucket:
            envs["NPA_S3_BUCKET"] = bucket
            if "S3_BUCKET" in envs:
                envs["S3_BUCKET"] = bucket
        if prefix:
            envs["NPA_S3_PREFIX"] = prefix
            if "SONIC_OUTPUT_PREFIX" in envs and str(envs["SONIC_OUTPUT_PREFIX"]).strip("/") != prefix.strip("/"):
                envs["SONIC_OUTPUT_PREFIX"] = prefix
    destinations = skypilot_output_destinations(documents)
    from npa.orchestration.skypilot.storage_preflight import (
        nebius_mount_destinations, verify_nebius_mount_principal,
    )

    mount_destinations = nebius_mount_destinations(documents)
    destinations.update(mount_destinations)
    declared_destinations = {
        uri for document in documents
        if "NPA_EXECUTION_OUTPUTS" in (document.get("envs") or {})
        for uri in skypilot_output_destinations([document])
    }
    declared_destinations.update(mount_destinations)
    for uri in destinations:
        # Explicit artifact declarations identify actual destinations. A
        # workflow's default/ledger prefix is not a security boundary for its
        # separately declared outputs; every destination still requires exact
        # project ownership and its own access probe.
        if uri in declared_destinations:
            continue
        bucket, prefix = _destination(uri, destinations[uri])
        if process_env.get("NPA_S3_BUCKET") and bucket != process_env["NPA_S3_BUCKET"]:
            raise ExecutionPreflightError("storage_target", "declared output disagrees with the explicit bucket")
        selected_prefix = str(process_env.get("NPA_S3_PREFIX") or "").strip("/")
        if selected_prefix and prefix != selected_prefix and not prefix.startswith(selected_prefix + "/"):
            raise ExecutionPreflightError("storage_target", "declared output disagrees with the explicit task prefix")
    if target is None:
        target = resolve_execution_target(project=project, context=context,
            output_uris=list(destinations), output_kinds=destinations, credentials=selected,
            provenance={"outputs": "workflow.env"})
    else:
        if target.context != context:
            raise ExecutionPreflightError("cluster_owner", "submitted context differs from the verified target")
        # A preverified workflow target includes ledger/source destinations as
        # well as task outputs. Fresh task-specific outputs must also be checked.
        target = replace_execution_outputs(target, destinations)
    if native_documents and infra.startswith("nebius/"):
        placement = infra.split("/")
        if len(placement) > 3 or placement[1] != target.region:
            raise ExecutionPreflightError("region", "explicit native infrastructure disagrees with the selected project region")
        if len(placement) == 3:
            for document in native_documents:
                resources = document["resources"]
                if resources.get("zone") not in (None, "", placement[2]):
                    raise ExecutionPreflightError("region", "native task placement disagrees with the explicit infrastructure")
                resources["zone"] = placement[2]
    injected = {
        "AWS_ACCESS_KEY_ID": selected.access_key_id,
        "AWS_SECRET_ACCESS_KEY": selected.secret_access_key,
        **dict.fromkeys(STORAGE_ENDPOINT_ENV_NAMES, selected.endpoint_url),
    }
    for document in documents:
        env = document.setdefault("envs", {})
        for name, value in injected.items():
            if value:
                env[name] = value
    verify_worker_environment(target, [*documents, dict(global_config or {})])
    if process_env.get("AWS_SESSION_TOKEN") or workflow_env.get("AWS_SESSION_TOKEN"):
        raise ExecutionPreflightError("credentials", "session-token overrides are unsupported by the executing principal contract")
    verify_nebius_mount_principal([*documents, dict(global_config or {})], target=target, sky_bin=sky_bin,
                                  environment={**process_env, **injected}, cwd=cwd)

    def gpu_check() -> None:
        from npa.orchestration.skypilot.k8s_gpu_catalog import (
            discover_kubernetes_gpu_inventory, preflight_kubernetes_gpu_gang,
        )
        if native_documents:
            from npa.orchestration.skypilot.native_preflight import verify_native_nebius_submission

            verify_native_nebius_submission(
                documents=[document for document in native_documents if document in documents], target=target,
                global_config=global_config if global_config is not None else {},
                sky_bin=sky_bin, extra_env=process_env,
                cwd=cwd,
            )
        global_kube = (global_config or {}).get("kubernetes") or {}
        allowed = global_kube.get("allowed_nodes") or ()
        if not isinstance(allowed, (tuple, list)):
            raise ExecutionPreflightError("gpu", "global allowed_nodes shape is unknown", status="unknown")
        for document in kubernetes_documents:
            resources = document.get("resources") or {}
            if not isinstance(resources, Mapping):
                raise ExecutionPreflightError("gpu", "alternative resource targets are ambiguous; select one effective resource mapping")
            gpu = resources.get("accelerators")
            if not gpu:
                continue
            if isinstance(gpu, Mapping):
                if len(gpu) != 1:
                    raise ExecutionPreflightError("gpu", "one accelerator product is required per task")
                gpu = ":".join(str(value) for value in next(iter(gpu.items())))
            kube = resources.get("kubernetes") or {}
            pod_spec = (kube.get("pod_config") or {}).get("spec") or {}
            global_pod = (global_kube.get("pod_config") or {}).get("spec") or {}
            if any(global_pod.get(key) for key in ("nodeSelector", "affinity", "nodeName", "tolerations", "topologySpreadConstraints")) or any(
                isinstance(container, Mapping) and container.get("resources") for container in global_pod.get("containers") or []
            ):
                raise ExecutionPreflightError("gpu", "global pod placement requires explicit task pod configuration", status="unknown")
            preflight_kubernetes_gpu_gang(discover_kubernetes_gpu_inventory(context=context),
                accelerator=str(gpu), node_count=int(document.get("num_nodes") or 1),
                cpus=resources.get("cpus", 0), memory=resources.get("memory", 0),
                allowed_nodes=allowed, pod_spec=pod_spec)

    report = verify_execution_target(target, gpu_check=gpu_check)
    return target, report, {name: value for name, value in injected.items() if value}


def replace_execution_outputs(target: ExecutionTarget, destinations: Mapping[str, str]) -> ExecutionTarget:
    from dataclasses import replace

    return replace(target, output_uris=tuple(dict.fromkeys([*target.output_uris, *destinations])),
                   output_kinds={**target.output_kinds, **destinations})
