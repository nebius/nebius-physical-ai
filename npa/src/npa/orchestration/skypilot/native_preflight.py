"""Read-only native Nebius validation in the *executing* SkyPilot runtime.

SkyPilot 0.12.2's get_project_by_region otherwise chooses the first project in
its credential tenant. Pin the selected NPA project in the private generated
config, verify it through Sky's own SDK, and retain the optimizer's exact shape.
No NPA-provider credentials, cloud creates, or workload storage are used here.

The subprocess entrypoint deliberately has no NPA imports: managed SkyPilot has
its own dependency environment. All IPC files are private and transient; raw
SDK output is never included in a diagnostic.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import yaml

if TYPE_CHECKING:
    from npa.execution_preflight import ExecutionTarget


class _NativeFailure(Exception):
    def __init__(self, check: str, reason: str, status: str = "fail") -> None:
        self.check, self.reason, self.status = check, reason, status
        super().__init__(reason)


_REASONS = {
    "configuration": "native SkyPilot configuration must have one resolved target",
    "project": "native SkyPilot project disagrees with the selected project",
    "tenant": "native SkyPilot credential tenant disagrees with the selected tenant",
    "region": "native SkyPilot region disagrees with the selected project",
    "scope": "executing SkyPilot credentials returned inconsistent project ownership",
    "identity": "executing SkyPilot provider identity evidence is incomplete",
    "runtime": "managed SkyPilot runtime cannot verify native submission",
    "provider": "executing SkyPilot provider lookup is unavailable",
    "optimizer": "managed SkyPilot optimizer could not resolve the requested resources",
    "product": "optimizer selected a platform or preset unavailable in the project region",
    "shape": "optimizer GPU shape disagrees with the project-scoped provider preset",
    "catalog": "project-scoped platform or preset evidence is incomplete",
    "override": "task configuration overrides cannot change native provider identity",
}


def _check_config(config: Mapping[str, Any], project: str, tenant: str, region: str) -> None:
    native = config.get("nebius") or {}
    if not isinstance(native, dict):
        raise _NativeFailure("native_scope", "configuration")
    region_configs = native.get("region_configs") or {}
    if not isinstance(region_configs, dict):
        raise _NativeFailure("native_scope", "configuration")
    regional = region_configs.get(region) or {}
    if not isinstance(regional, dict):
        raise _NativeFailure("native_scope", "configuration")
    for part in (native, regional):
        for key, wanted, reason in (("project_id", project, "project"), ("tenant_id", tenant, "tenant")):
            if part.get(key) not in (None, "", wanted):
                raise _NativeFailure("native_scope", reason)


def _pin_resources(resources: Any, region: str) -> dict[str, Any]:
    if not isinstance(resources, dict):
        raise _NativeFailure("native_scope", "configuration")
    pinned = deepcopy(resources)
    overrides = pinned.get("_cluster_config_overrides") or {}
    if not isinstance(overrides, dict) or "nebius" in overrides:
        raise _NativeFailure("native_scope", "override")
    infra = str(pinned.pop("infra", "") or "").split("/")
    cloud = str(pinned.get("cloud") or (infra[0] if infra[0] else "nebius")).lower()
    if cloud != "nebius" or (infra[0] and infra[0].lower() != "nebius"):
        raise _NativeFailure("native_scope", "configuration")
    if pinned.get("region") not in (None, "", region) or (len(infra) > 1 and infra[1] != region):
        raise _NativeFailure("native_scope", "region")
    if len(infra) > 2:
        if pinned.get("zone") not in (None, "", infra[2]):
            raise _NativeFailure("native_scope", "configuration")
        pinned["zone"] = infra[2]
    pinned.update(cloud="nebius", region=region)
    # Each permitted alternative must share the exact provider destination.
    for key in ("any_of", "ordered"):
        if key in pinned:
            if not isinstance(pinned[key], list) or not pinned[key]:
                raise _NativeFailure("native_scope", "configuration")
            pinned[key] = [_pin_resources(item, region) for item in pinned[key]]
    return pinned


def _native(resources: Any) -> bool:
    if not isinstance(resources, dict):
        return False
    cloud = str(resources.get("cloud") or str(resources.get("infra") or "").split("/")[0]).lower()
    return cloud == "nebius"


def _managed_probe(request: dict[str, Any]) -> dict[str, Any]:
    """Pinned upstream APIs, executed by the managed environment, never NPA's."""
    try:
        import sky
        from sky import skypilot_config
        from sky.adaptors import nebius
        from sky.optimizer import Optimizer
        from sky.provision.nebius import utils as provider_utils
    except Exception:
        raise _NativeFailure("native_scope", "runtime", "unknown") from None
    if sky.__version__ != "0.12.2":
        raise _NativeFailure("native_scope", "runtime", "unknown")
    project, tenant, region = request["project"], request["tenant"], request["region"]
    # Uses normal Sky config loading (including workspace, project-file and
    # internal-file precedence), exactly as the executing client. Never replace
    # a mismatching credential tenant with the NPA CLI's credential identity.
    try:
        effective_project = skypilot_config.get_effective_region_config(
            cloud="nebius", region=region, keys=("project_id",), default_value=None,
        )
        actual_tenant = nebius.get_tenant_id()
    except Exception:
        raise _NativeFailure("native_scope", "identity", "unknown") from None
    if effective_project != project:
        raise _NativeFailure("native_scope", "project")
    if not actual_tenant:
        raise _NativeFailure("native_scope", "identity", "unknown")
    if actual_tenant != tenant:
        raise _NativeFailure("native_scope", "tenant")
    try:
        if provider_utils.get_project_by_region(region) != project:
            raise _NativeFailure("native_scope", "project")
        sdk = nebius.sdk()
        observed = nebius.sync_call(nebius.iam().ProjectServiceClient(sdk).get(
            nebius.iam().GetProjectRequest(id=project), timeout=nebius.READ_TIMEOUT,
        ))
    except _NativeFailure:
        raise
    except Exception:
        raise _NativeFailure("native_scope", "provider", "unknown") from None
    if not all((observed.metadata.id, observed.metadata.parent_id, observed.status.region)):
        raise _NativeFailure("native_scope", "identity", "unknown")
    if (observed.metadata.id, observed.metadata.parent_id, observed.status.region) != (project, tenant, region):
        raise _NativeFailure("native_scope", "scope")

    chosen = []
    for shape in request["shapes"]:
        try:
            # Keep all resource constraints, but never materialize a Task's
            # eager file_mounts/Storage objects or its executable payload.
            with sky.Dag() as dag:
                task = sky.Task.from_yaml_config(deepcopy(shape))
            Optimizer.optimize(dag, quiet=True)
            best = task.best_resources
        except Exception:
            raise _NativeFailure("gpu_product", "optimizer", "unknown") from None
        if best is None or str(best.cloud).lower() != "nebius" or best.region != region:
            raise _NativeFailure("gpu_product", "optimizer", "unknown")
        parts = str(best.instance_type or "").split("_")
        if len(parts) != 2 or not all(parts):
            raise _NativeFailure("gpu_product", "catalog", "unknown")
        platform, preset = parts
        try:
            product = nebius.sync_call(nebius.compute().PlatformServiceClient(sdk).get_by_name(
                nebius.nebius_common().GetByNameRequest(parent_id=project, name=platform),
                timeout=nebius.READ_TIMEOUT,
            ))
        except Exception:
            raise _NativeFailure("gpu_product", "provider", "unknown") from None
        if not product.metadata.id or not product.metadata.parent_id or not product.metadata.name:
            raise _NativeFailure("gpu_product", "catalog", "unknown")
        # Catalog products can be owned by a regional provider catalog project;
        # this GetByName request itself proves availability to the exact target.
        if product.metadata.name != platform:
            raise _NativeFailure("gpu_product", "product")
        matching = [item for item in product.spec.presets if item.name == preset]
        if len(matching) != 1:
            raise _NativeFailure("gpu_product", "product")
        count = sum((best.accelerators or {}).values())
        if matching[0].resources.gpu_count != count:
            raise _NativeFailure("gpu_product", "shape")
        selected = best.to_yaml_config()
        selected.pop("infra", None)
        if best.zone is not None:
            selected["zone"] = best.zone
        selected.update(cloud="nebius", region=region, instance_type=best.instance_type)
        chosen.append(selected)
    return {"status": "pass", "resources": chosen}


def verify_native_nebius_submission(
    *, documents: Sequence[dict[str, Any]], target: ExecutionTarget,
    global_config: dict[str, Any], sky_bin: str,
    extra_env: Mapping[str, str], cwd: str | os.PathLike[str] | None = None,
) -> None:
    """Verify and pin native tasks/controller in caller-owned runtime copies.

    Call after provider scope verification, before storage probes or creation.
    ``extra_env`` and ``cwd`` must be the actual managed Sky launch environment.
    Mutations are committed only when every native shape passes; caller persists
    the generated config and rendered task documents with owner-only access.
    """
    from npa.execution_preflight import ExecutionPreflightError

    try:
        config = deepcopy(global_config)
        _check_config(config, target.project_id, target.tenant_id, target.region)
        native = config.setdefault("nebius", {})
        native.setdefault("region_configs", {}).setdefault(target.region, {})["project_id"] = target.project_id
        shapes: list[dict[str, Any]] = []
        native_docs = []
        for document in documents:
            if not _native(document.get("resources")):
                continue
            shape = {"resources": _pin_resources(document["resources"], target.region)}
            if "num_nodes" in document:
                shape["num_nodes"] = document["num_nodes"]
            # Sky Task.config can override provider identity in resource config.
            # Native identity is exclusively the verified generated config.
            overrides = document.get("config") or document["resources"].get("_cluster_config_overrides") or {}
            if not isinstance(overrides, dict) or "nebius" in overrides:
                raise _NativeFailure("native_scope", "override")
            if overrides:
                shape["resources"].pop("_cluster_config_overrides", None)
                shape["config"] = deepcopy(overrides)
            shapes.append(shape)
            native_docs.append(document)
        controller = config.get("jobs", {}).get("controller", {}).get("resources")
        has_native_controller = _native(controller)
        if has_native_controller:
            shapes.append({"resources": _pin_resources(controller, target.region)})
        if not shapes:
            raise _NativeFailure("native_scope", "configuration")

        # Same sibling interpreter convention as ensure_skypilot_version; do
        # not resolve bin/python's symlink out of its managed venv.
        interpreter = Path(sky_bin).expanduser().absolute().parent / "python"
        with tempfile.TemporaryDirectory(prefix="npa-native-preflight-") as directory:
            root = Path(directory)
            root.chmod(0o700)
            request_path, result_path, config_path = (root / name for name in ("request.json", "result.json", "config.yaml"))
            request = {"project": target.project_id, "tenant": target.tenant_id,
                       "region": target.region, "shapes": shapes}
            for path, contents in ((request_path, json.dumps(request)), (config_path, yaml.safe_dump(config))):
                with open(path, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as handle:
                    handle.write(contents)
            env = {**os.environ, **extra_env, "SKYPILOT_GLOBAL_CONFIG": str(config_path)}
            try:
                result = subprocess.run(
                    [str(interpreter), str(Path(__file__).absolute()), str(request_path), str(result_path)],
                    env=env, cwd=cwd, capture_output=True, text=True, check=False,
                )
                payload = json.loads(result_path.read_text()) if result.returncode == 0 else {}
            except (OSError, ValueError, subprocess.SubprocessError):
                raise _NativeFailure("native_scope", "runtime", "unknown") from None
        if payload.get("status") != "pass":
            check = payload.get("check") if payload.get("check") in {"native_scope", "gpu_product"} else "native_scope"
            reason = payload.get("reason") if payload.get("reason") in _REASONS else "runtime"
            status = payload.get("status") if payload.get("status") in {"fail", "unknown"} else "unknown"
            raise _NativeFailure(check, reason, status)
        selections = payload.get("resources")
        if not isinstance(selections, list) or len(selections) != len(shapes) or any(
            not isinstance(item, dict) or item.get("cloud") != "nebius"
            or item.get("region") != target.region or not item.get("instance_type") for item in selections
        ):
            raise _NativeFailure("gpu_product", "catalog", "unknown")
        for document, selected in zip(native_docs, selections):
            if document.get("config"):
                selected.pop("_cluster_config_overrides", None)
            document["resources"] = selected
        if has_native_controller:
            config["jobs"]["controller"]["resources"] = selections[-1]
        global_config.clear()
        global_config.update(config)
    except _NativeFailure as exc:
        raise ExecutionPreflightError(exc.check, _REASONS[exc.reason], status=exc.status) from None


def _main() -> None:
    try:
        payload = _managed_probe(json.loads(Path(sys.argv[1]).read_text()))
    except _NativeFailure as exc:
        payload = {"check": exc.check, "reason": exc.reason, "status": exc.status}
    except Exception:
        payload = {"check": "native_scope", "reason": "runtime", "status": "unknown"}
    with open(sys.argv[2], "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    _main()
