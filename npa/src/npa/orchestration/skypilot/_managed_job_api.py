"""Transport one pinned native managed-job request/result in the Sky interpreter.

This private channel is an observation, not a provider attestation. The caller
must independently verify the owned API and controller before and after it.
SkyPilot is imported only by the isolated executable entry point, never by NPA.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping


SKY_VERSION = "0.12.2"
SKY_SOURCE_COMMIT = "158acd60038fb714e70b099038454bb23ce9ade7"
MAX_OBSERVATION_BYTES = 65536
_SOURCE_HASHES = {
    "jobs/client/sdk.py": "9a03c5e3020023557d170bb389bf8beb66f66b11c4cbc4297082cb19949e3ffe",
    "client/sdk.py": "2cdc6f8e28b00c27d74c69fec4c59b3e6d95f7738d167bd2d6dbba751eb5ecec",
    "jobs/server/server.py": "c39b85c04af342210b0327270cc06ea4f85a85ade9c0d3b19cc510b386137053",
    "jobs/server/core.py": "566908a4389cb83730727176ac4d73b8238c9d109ff36314d64b8d53dc3f1984",
    "server/common.py": "2f0fd98fbcad7c2440685e92ad2aa22bd17a273bfbed6d8ca2e37c6261e47d37",
    "server/server.py": "9451e7d1997b9fbcb0d22e60dbe525de65ba147e43c82f401713201607ede7f6",
    "server/requests/requests.py": "243f091123e637754cf33ab1d3494e777b7eac77ae139eef7be72afb74ca59c7",
    "server/requests/executor.py": "9d38c07f899ec3280fe1ddf62bcdffd3bd5cdd2671f827297448b874d411a3e3",
    "server/requests/payloads.py": "e5587675a77a72e9003be7e490a3074d61d76348f1b1543c423fcf2da487162d",
    "server/rest.py": "28b5de4346fa90576758fe5ba989ddc265eaf4a1ae3ff6142d7715d96e67138e",
    "server/requests/serializers/encoders.py": "9c06efca236572c5c8dd29c666a68ab8e0de63236cfc9573adb2d7de1e13d9bd",
    "server/requests/serializers/decoders.py": "701c5a952879f5bc58f7703683d7093d009ab886dc3dc6b7b212111da70d5584",
    "server/requests/serializers/return_value_serializers.py": "c830758b1cc4017b03b34eed435e52154dc072c91b06115b75fc0aa0c88e6135",
    "jobs/state.py": "03400bf4ff8626b2e22e183a001f6fb6f9f3efa603701293e4d095640e403569",
    "backends/cloud_vm_ray_backend.py": "8160bbb323c1cd64ad365c382bfd674c0e80c27b9f73db0a55b0910de917cce7",
    "utils/dag_utils.py": "1b7dabdb3539d8153e584d5a010f8ff1cb9d6ca932fb98c74d7d31b0000eea6e",
    "server/requests/storage.py": "910131ccf55347fb90d150c5da44c75fd7f12c7dff86f0d3eceb45a30feecf58",
    "skylet/job_lib.py": "937f8e4518aa70c8c93c9825cb8f7272ba1ce453c34cda982931114678a63fdf",
    "skylet/services.py": "c181b243b58a3ace56646d527e2a8d8577443956b7f54444b27d01957db7a02e",
    "server/constants.py": "1e88b0b837500c9bcfb886422db53b5c7049af02aeb4eaf5230b3a58678290f5",
    "utils/controller_utils.py": (
        "e00b1e4ac2" "49a32763b9ae3c5a4e970f40098a7b671100818e90ced6f1fa186c"
    ),
    "utils/common.py": "b2604e6629d52d36c7837b5968b4e55c642839b2490aa230e8865091a4f7df46",
    "templates/jobs-controller-provision.yaml.j2": "d04fd84bce03559968227beab07235707d025f8481d24fa015afdfe7b17c7249",
}
_REQUEST_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


class NativeResultUnavailable(ValueError):
    """The native success chain is incomplete; retain context and never retry."""


@dataclass(frozen=True, repr=False)
class NativeLaunchResult:
    """Private invocation observations, distinct from a reconciled queue record.

    Args:
        attempt: This local attempt, not an upstream idempotency key.
        request_id: Full ID returned by this invocation of jobs.launch.
        job_id: Controller-local ID from its successful native result.
        task_ids: Complete expected task IDs from the loaded DAG.
        context: Independently verified API/controller binding digest.
    Returns:
        None.
    Raises:
        None; use the strict observation decoder at the transport boundary.
    """

    attempt: str
    request_id: str
    job_id: str
    task_ids: tuple[int, ...]
    context: str


def _require(condition: bool) -> None:
    if not condition:
        raise NativeResultUnavailable(
            "native launch identity unavailable; preserve private recovery context"
        )


def _request_id(value: Any) -> str:
    _require(isinstance(value, str) and _REQUEST_ID.fullmatch(value) is not None)
    return str(value)


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def decode_observation(
    data: bytes,
    *,
    attempt: str,
    context: str,
    task_count: int,
) -> NativeLaunchResult:
    """Decode exactly one request followed by its complete successful result.

    Args:
        data: Bounded private IPC bytes, never CLI output.
        attempt: Independently retained attempt identity.
        context: Independently rechecked owned API/controller digest.
        task_count: Expected complete rendered DAG population.
    Returns:
        Validated observations; not permission to delete a shared controller.
    Raises:
        NativeResultUnavailable: Partial, foreign, malformed or excessive IPC.
    """
    _require(0 < len(data) <= MAX_OBSERVATION_BYTES and data.endswith(b"\n"))
    try:
        rows = [
            json.loads(line, object_pairs_hook=_unique_mapping)
            for line in data.splitlines()
        ]
        return _decode_rows(rows, attempt, context, task_count)
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        pass
    raise NativeResultUnavailable("native result incomplete; recovery context retained")


def _decode_rows(rows, attempt, context, task_count) -> NativeLaunchResult:
    _require(len(rows) == 2 and all(type(row) is dict for row in rows))
    request, result = rows
    common = {"attempt", "context", "request_id", "event"}
    _require(set(request) == common)
    _require(set(result) == common | {"job_id", "task_ids"})
    _require(request["event"] == "request" and result["event"] == "result")
    _require(bool(attempt) and re.fullmatch(r"[a-f0-9]{64}", context) is not None)
    for row in rows:
        _require(row["attempt"] == attempt and row["context"] == context)
    request_id = _request_id(request["request_id"])
    _require(result["request_id"] == request_id)
    _require(type(result["job_id"]) is int and result["job_id"] > 0)
    _require(type(task_count) is int and task_count > 0)
    tasks = result["task_ids"]
    _require(type(tasks) is list and all(type(value) is int for value in tasks))
    _require(tasks == list(range(task_count)))
    return NativeLaunchResult(
        attempt, request_id, str(result["job_id"]), tuple(tasks), context
    )


def _append_observation(descriptor: int, record: Mapping[str, Any]) -> None:
    metadata = os.fstat(descriptor)
    _require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid())
    _require(metadata.st_mode & 0o077 == 0 and metadata.st_size < MAX_OBSERVATION_BYTES)
    data = (json.dumps(record, sort_keys=True) + "\n").encode()
    _require(metadata.st_size + len(data) <= MAX_OBSERVATION_BYTES)
    while data:
        written = os.write(descriptor, data)
        _require(written > 0)
        data = data[written:]
    os.fsync(descriptor)


def decode_controller_observation(data: bytes, *, attempt: str, context: str) -> tuple[str, str, str]:
    """Decode a controller provision request; it conveys no managed-job ownership."""
    _require(0 < len(data) <= MAX_OBSERVATION_BYTES and data.endswith(b"\n"))
    try:
        rows = [json.loads(line, object_pairs_hook=_unique_mapping) for line in data.splitlines()]
        _require(len(rows) == 2 and all(type(row) is dict for row in rows))
        request, result = rows
        fields = {"event", "attempt", "context", "request_id", "controller"}
        _require(set(request) == fields and set(result) == fields | {"incarnation", "controller_cloud_name"})
        _require(request["event"] == "controller_request" and result["event"] == "controller_result")
        _require(re.fullmatch(r"[a-f0-9]{64}", context) is not None)
        _require(_request_id(request["request_id"]) == result["request_id"])
        for key, value in (("attempt", attempt), ("context", context), ("controller", request["controller"])):
            _require(request[key] == result[key] == value)
        _require(re.fullmatch(r"sky-jobs-controller-[a-z0-9-]+", request["controller"]) is not None)
        _require(re.fullmatch(r"[a-f0-9]{64}", result["incarnation"]) is not None)
        _controller_cloud_name(result["controller_cloud_name"])
        return request["controller"], result["controller_cloud_name"], result["incarnation"]
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        raise NativeResultUnavailable("controller ensure result incomplete; context retained") from None


def _controller_cloud_name(value):
    _require(isinstance(value, str) and len(value) <= 63)
    _require(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", value) is not None)
    return value


def _controller_provision_task(dag):
    """Use the pinned upstream controller-only template and resource selector."""
    from sky import Task, skypilot_config
    from sky.jobs import constants
    from sky.skylet import constants as skylet_constants
    from sky.utils import common, common_utils, controller_utils

    controller = controller_utils.Controllers.JOBS_CONTROLLER
    name = controller.value.cluster_name
    resources = controller_utils.get_controller_resources(
        controller=controller,
        task_resources=[resource for task in dag.tasks for resource in task.resources],
    )
    variables = {
        "dag_name": "ensure_controller_up",
        "job_controller_indicator_file": constants.JOB_CONTROLLER_INDICATOR_FILE,
        **controller_utils.controller_only_vars_to_fill(controller),
    }
    path = Path(constants.JOBS_CONTROLLER_YAML_PREFIX).expanduser() / "npa-ensure-controller.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with skypilot_config.local_active_workspace_ctx(skylet_constants.SKYPILOT_DEFAULT_WORKSPACE):
        with common.with_server_user():
            common_utils.fill_template(constants.JOBS_CONTROLLER_PROVISION_TEMPLATE, variables, output_path=str(path))
            task = Task.from_yaml(str(path))
            task.set_resources(resources)
    _require(task.run is None)
    return name, task


def _ensure_controller_native(payload, *, sky, load_dag, prepare_controller,
                              verify_absent, verify_context, verify_incarnation, observe):
    _require(sky.__version__ == SKY_VERSION and sky.__commit__ == SKY_SOURCE_COMMIT)
    _require(verify_context() == payload["context"])
    dag = load_dag(payload["yaml"])
    name, task = prepare_controller(dag)
    _require(not payload["controller"] or name == payload["controller"])
    _require(verify_absent(name))
    _require(verify_context() == payload["context"])
    request_id = _request_id(sky.launch(
        task, cluster_name=name, retry_until_up=True, fast=True,
        _disable_controller_check=True, _need_confirmation=False,
    ))
    request = {"event": "controller_request", "attempt": payload["attempt"],
               "context": payload["context"], "request_id": request_id, "controller": name}
    observe(request)
    _require(verify_context() == payload["context"])
    result = sky.get(request_id)
    _require(type(result) is tuple and len(result) == 2 and result[0] is None)
    _require(getattr(result[1], "cluster_name", None) == name)
    _require(verify_context() == payload["context"])
    cloud_name = _controller_cloud_name(getattr(result[1], "cluster_name_on_cloud", None))
    incarnation = verify_incarnation(name, cloud_name)
    _require(re.fullmatch(r"[a-f0-9]{64}", incarnation) is not None)
    observe({**request, "event": "controller_result", "incarnation": incarnation,
             "controller_cloud_name": cloud_name})


def _launch_native(payload, *, sky, load_dag, observe, verify_context) -> None:
    _require(sky.__version__ == SKY_VERSION and sky.__commit__ == SKY_SOURCE_COMMIT)
    _require(verify_context() == payload["context"])
    dag = load_dag(payload["yaml"], secrets_overrides=payload["secrets"])
    _require(len(dag.tasks) == payload["task_count"] and bool(dag.tasks))
    if payload.get("robotwin_image_sha256"):
        _bind_robotwin_image(dag, payload["secrets"], payload["robotwin_image_sha256"])
    _require(verify_context() == payload["context"])
    request_id = _request_id(
        sky.jobs.launch(dag, name=payload["name"], _need_confirmation=False)
    )
    request = {
        "event": "request",
        "attempt": payload["attempt"],
        "context": payload["context"],
        "request_id": request_id,
    }
    observe(request)
    _require(verify_context() == payload["context"])
    native_result = sky.get(request_id)
    _require(verify_context() == payload["context"])
    job_id = _successful_job_id(native_result, payload["controller"])
    observe(
        {
            **request,
            "event": "result",
            "job_id": job_id,
            "task_ids": list(range(len(dag.tasks))),
        }
    )


def _bind_robotwin_image(dag, secrets, expected_sha256: str) -> None:
    """Resolve the one authorized image in memory; SkyPilot does not expand it.

    The validated inner submit supplies a hash of its immutable image. Other
    secret interpolation and persisted template contents remain unchanged.
    """
    name = "NPA_INTERNAL_BYOF_ROBOTWIN_IMAGE"
    values = [value for key, value in secrets if key == name]
    _require(len(values) == 1 and isinstance(values[0], str))
    image = values[0]
    _require(re.fullmatch(r"[^@\s]+/npa-robotwin@sha256:[0-9a-f]{64}", image) is not None)
    _require(hashlib.sha256(image.encode()).hexdigest() == expected_sha256)
    _require(len(dag.tasks) == 1)
    task = dag.tasks[0]
    _require(task.name == "byof-solution-smoke-robotwin-rtxpro")
    _require(len(task.resources) == 1)
    resources = next(iter(task.resources))
    image_ids = resources.image_id
    _require(isinstance(image_ids, dict) and len(image_ids) == 1)
    _require(next(iter(image_ids.values())) == f"docker:${{{name}}}")
    resolved = {region: "docker:" + image for region in image_ids}
    task.set_resources(resources.copy(image_id=resolved))
    _require(next(iter(task.resources)).image_id == resolved)


def _successful_job_id(result: Any, controller: str) -> int:
    _require(type(result) is tuple and len(result) == 2)
    job_ids, handle = result
    _require(type(job_ids) is list and len(job_ids) == 1)
    _require(type(job_ids[0]) is int and job_ids[0] > 0)
    _require(bool(controller) and getattr(handle, "cluster_name", None) == controller)
    # A handle identifies where the request ran, never ownership of that server.
    return job_ids[0]


def observation_digest(value: Mapping[str, Any]) -> str:
    """Hash private context observations without returning their values.

    Args:
        value: Independently obtained context, never caller declarations.
    Returns:
        SHA-256 used to compare observations, not attest their provenance.
    Raises:
        TypeError: Context is not JSON-serializable.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_bridge(
    payload: dict[str, Any],
    *,
    descriptor: int,
    sky: Any,
    load_dag: Callable[..., Any],
    verify_context: Callable[[], str],
) -> None:
    """Execute one native request using injected pinned-runtime interfaces.

    Args:
        payload: Private rendered DAG, expected task count and attempt binding.
        descriptor: Parent-owned observation file descriptor.
        sky: Pinned native SDK, supplied only in its isolated interpreter.
        load_dag: Pinned full-DAG/JobGroup loader.
        verify_context: Rechecks the same verified owned API/controller context.
    Returns:
        None; fsynced request/result records remain in the private descriptor.
    Raises:
        NativeResultUnavailable: Missing or changed identity/result.
        Exception: Native failure; no retries or name-based adoption occur here.
    """
    with contextlib.redirect_stdout(sys.stderr):
        _launch_native(
            payload,
            sky=sky,
            load_dag=load_dag,
            observe=lambda row: _append_observation(descriptor, row),
            verify_context=verify_context,
        )


def _main() -> int:
    # This file is executed by the already selected isolated Sky interpreter.
    # No bootstrap, default API selection or SDK installation occurs here.
    raw = sys.stdin.buffer.read(MAX_OBSERVATION_BYTES + 1)
    _require(len(raw) <= MAX_OBSERVATION_BYTES)
    payload = json.loads(raw, object_pairs_hook=_unique_mapping)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from npa.orchestration.skypilot.workflow import (
        _native_api_context_digest,
        _native_context_digest,
        _native_controller_namespace_empty,
    )

    def verify_context():
        if payload.get("mode") == "ensure_controller":
            return _native_api_context_digest(
                isolated_dir=Path(payload["isolated_dir"]),
                sky_executable=payload["sky_executable"], environment=os.environ,
            )
        return _native_context_digest(
            isolated_dir=Path(payload["isolated_dir"]),
            controller=payload["controller"],
            controller_cloud_name=payload.get("controller_cloud_name", ""),
            context=payload["kube_context"],
            sky_executable=payload["sky_executable"],
            environment=os.environ,
        )

    _require(verify_context() == payload["context"])
    specification = importlib.util.find_spec("sky")
    _require(specification is not None and bool(specification.origin))
    root = Path(specification.origin).parent
    _verify_native_sources(root)
    import sky
    from sky.utils.dag_utils import load_dag_from_yaml_str

    _require(Path(sky.__file__).parent == root)
    if payload.get("mode") == "ensure_controller":
        def absent(_name):
            return _native_controller_namespace_empty(
                context=payload["kube_context"], environment=os.environ,
            )

        def incarnation(name, cloud_name):
            return _native_context_digest(
                isolated_dir=Path(payload["isolated_dir"]), controller=name,
                controller_cloud_name=cloud_name, context=payload["kube_context"],
                sky_executable=payload["sky_executable"], environment=os.environ,
            )

        with contextlib.redirect_stdout(sys.stderr):
            _ensure_controller_native(
                payload, sky=sky, load_dag=load_dag_from_yaml_str,
                prepare_controller=_controller_provision_task, verify_absent=absent,
                verify_context=verify_context, verify_incarnation=incarnation,
                observe=lambda row: _append_observation(payload["descriptor"], row),
            )
        _verify_native_sources(root)
        return 0
    payload["secrets"] = [(name, os.environ[name]) for name in payload["secrets"]]
    run_bridge(
        payload,
        descriptor=payload["descriptor"],
        sky=sky,
        load_dag=load_dag_from_yaml_str,
        verify_context=verify_context,
    )
    _verify_native_sources(Path(sky.__file__).parent)
    return 0


def _verify_native_sources(root: Path) -> None:
    # These are the inspected request/result chain files at the pinned source
    # commit, not a claim of wheel, dependency or remote-controller equivalence.
    for relative, expected in _SOURCE_HASHES.items():
        path = root / relative
        _require(path.resolve() == path.absolute())
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected)


if __name__ == "__main__":
    try:
        status = _main()
    except BaseException:
        # Native exceptions can carry secrets, URLs and arbitrary log output.
        # Request observations, if any, remain in the parent's private file.
        status = 2
    raise SystemExit(status)
