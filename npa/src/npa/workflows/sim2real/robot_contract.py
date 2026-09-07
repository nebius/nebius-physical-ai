"""Immutable Stage-2 robot embodiment materialization for compositional Sim2Real."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from npa.genesis import robot_assets
from npa.workflows.sim2real import isaac_byo_robot_task, onboarding_derive


ROBOT_CONTRACT_SCHEMA = "npa.sim2real.robot_contract.v1"
RESOLVED_ROBOT_SCHEMA = "npa.sim2real.resolved_robot_spec.v1"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class RobotContractError(RuntimeError):
    """Raised when Stage 2 cannot produce one trustworthy embodiment contract."""


def _canonical_digest(payload: dict[str, Any], *, omit: tuple[str, ...] = ()) -> str:
    normalized = {key: value for key, value in payload.items() if key not in omit}
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        sha256 = _file_digest(path)
        size = path.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256.encode())
        digest.update(b"\0")
        files.append({"path": relative, "sha256": sha256, "size_bytes": size})
    if not files:
        raise RobotContractError("robot asset bundle is empty")
    return digest.hexdigest(), files


def _s3_relative(uri: str, root_uri: str) -> str:
    source = urlparse(uri)
    root = urlparse(root_uri)
    if source.scheme != "s3" or root.scheme != "s3" or source.netloc != root.netloc:
        raise RobotContractError(
            "robot_uri and asset_root_uri must be s3:// URIs in the same bucket"
        )
    source_raw = source.path.lstrip("/")
    root_raw = root.path.lstrip("/").rstrip("/")
    prefix = root_raw + "/"
    relative = source_raw[len(prefix) :] if source_raw.startswith(prefix) else ""
    relative_path = PurePosixPath(relative)
    root_path = PurePosixPath(root_raw)
    unsafe = (
        not root_raw
        or root_path.is_absolute()
        or root_raw != root_path.as_posix()
        or any(part in {"", ".", ".."} for part in root_path.parts)
        or not relative
        or relative_path.is_absolute()
        or relative != relative_path.as_posix()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or "\\" in source_raw
        or "\\" in root_raw
        or "\x00" in source_raw
        or "\x00" in root_raw
    )
    if unsafe:
        raise RobotContractError(
            f"robot_uri {uri!r} is not contained by asset_root_uri {root_uri!r}"
        )
    return relative_path.as_posix()


def _mesh_dependencies(urdf_path: Path, bundle_root: Path) -> list[str]:
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError as exc:
        raise RobotContractError(
            f"invalid URDF XML in {urdf_path.name}: line {exc.position[0]}, "
            f"column {exc.position[1]}: {exc.msg}"
        ) from exc
    root = tree.getroot()
    if root.tag != "robot":
        raise RobotContractError("URDF root element must be <robot>")
    links = [str(node.get("name") or "").strip() for node in root.findall("link")]
    joints = [str(node.get("name") or "").strip() for node in root.findall("joint")]
    if not links or not joints or "" in links or "" in joints:
        raise RobotContractError("URDF requires non-empty named links and joints")
    if len(set(links)) != len(links) or len(set(joints)) != len(joints):
        raise RobotContractError("URDF link and joint names must be unique")

    resolved_root = bundle_root.resolve()

    def safe_candidate(relative: PurePosixPath) -> Path | None:
        raw = relative.as_posix()
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in raw
            or "\x00" in raw
        ):
            raise RobotContractError(
                f"URDF dependency path {raw!r} is unsafe; mesh references must "
                "remain below RobotSpec.asset_root_uri"
            )
        candidate = bundle_root.joinpath(*relative.parts)
        if not candidate.is_file():
            return None
        cursor = bundle_root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise RobotContractError(
                    f"URDF dependency path {raw!r} is unsafe; symbolic links are "
                    "not accepted in robot asset bundles"
                )
        resolved = candidate.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise RobotContractError(
                f"URDF dependency path {raw!r} is unsafe; mesh references must "
                "remain below RobotSpec.asset_root_uri"
            ) from exc
        return resolved

    urdf_parent = urdf_path.resolve().parent.relative_to(resolved_root)
    dependencies: list[str] = []
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.get("filename") or "").strip()
        if not filename:
            raise RobotContractError("URDF mesh element has no filename")
        if filename.startswith("package://"):
            relative = filename.removeprefix("package://")
            candidates = [PurePosixPath(relative)]
            parts = PurePosixPath(relative).parts
            if len(parts) > 1:
                candidates.append(PurePosixPath(*parts[1:]))
        elif "://" in filename:
            raise RobotContractError(
                f"URDF mesh URI {filename!r} is unsupported; stage dependencies "
                "under asset_root_uri and use package:// or relative references"
            )
        else:
            candidates = [PurePosixPath(urdf_parent.as_posix()) / filename]
        resolved = next(
            (path for candidate in candidates if (path := safe_candidate(candidate))),
            None,
        )
        if resolved is None:
            raise RobotContractError(
                f"URDF dependency {filename!r} is missing. Upload the complete "
                "package (including meshes) below RobotSpec.asset_root_uri."
            )
        dependencies.append(resolved.relative_to(resolved_root).as_posix())
    return sorted(set(dependencies))


def _validate_urdf_structure(
    urdf_path: Path, spec: robot_assets.RobotSpec
) -> dict[str, Any]:
    root = ET.parse(urdf_path).getroot()
    links = {str(node.get("name") or "").strip() for node in root.findall("link")}
    joint_nodes = {
        str(node.get("name") or "").strip(): node for node in root.findall("joint")
    }
    missing_links = sorted({spec.base_link, spec.ee_link, *spec.finger_links} - links)
    if missing_links:
        raise RobotContractError(
            "RobotSpec names links absent from the URDF: " + ", ".join(missing_links)
        )
    ordered = list(spec.joint_names)
    if len(ordered) != spec.dof_count:
        raise RobotContractError(
            f"RobotSpec.joint_names must contain exactly dof_count={spec.dof_count} "
            f"ordered arm-then-gripper joints; got {len(ordered)}"
        )
    missing_joints = [name for name in ordered if name not in joint_nodes]
    if missing_joints:
        raise RobotContractError(
            "RobotSpec names joints absent from the URDF: " + ", ".join(missing_joints)
        )
    fixed_joints = [
        name for name in ordered if str(joint_nodes[name].get("type") or "") == "fixed"
    ]
    if fixed_joints:
        raise RobotContractError(
            "RobotSpec ordered control joints cannot be fixed: "
            + ", ".join(fixed_joints)
        )
    gripper = list(spec.gripper_joint_names)
    expected_gripper = ordered[spec.n_arm_joints :]
    if gripper != expected_gripper:
        raise RobotContractError(
            "RobotSpec.gripper_joint_names must exactly equal the ordered trailing "
            f"gripper joints {expected_gripper!r}; got {gripper!r}"
        )
    return {
        "robot_name": str(root.get("name") or spec.name),
        "link_count": len(links),
        "joint_count": len(joint_nodes),
        "controlled_joint_count": len(ordered),
    }


def _task_config(doc: dict[str, Any], spec: robot_assets.RobotSpec) -> dict[str, Any]:
    supplied = doc.get("task") or doc.get("task_config") or {}
    if supplied and not isinstance(supplied, dict):
        raise RobotContractError("RobotSpec.task/task_config must be a JSON object")
    try:
        json.dumps(supplied, allow_nan=False)
    except ValueError as exc:
        raise RobotContractError(
            "RobotSpec.task/task_config numeric values must be finite"
        ) from exc
    reach = onboarding_derive.PRESET_REACH_M.get(
        str(doc.get("preset") or spec.name).lower(),
        onboarding_derive.FRANKA_REACH_M,
    )
    object_range, goal_range, _ = onboarding_derive.derive_placement(reach)
    base = {
        "skill": "lift",
        "action_scale": onboarding_derive.STOCK_ACTION_SCALE,
        "workspace_reach_m": reach,
        "object_init_range": {key: list(value) for key, value in object_range.items()},
        "goal_range": {key: list(value) for key, value in goal_range.items()},
        "goal_pos": [],
        "minimal_height_m": onboarding_derive.STOCK_MINIMAL_HEIGHT_M,
        "success_distance_m": onboarding_derive.STOCK_SUCCESS_DISTANCE_M,
        "gripper_open": spec.gripper_open,
        "gripper_close": spec.gripper_close,
        "init_joint_pos": list(spec.home_qpos),
        "source": {"contract": "robot_spec"},
    }
    base.update(dict(supplied))
    if str(base.get("skill") or "").lower() != "lift":
        raise RobotContractError(
            f"unsupported RobotSpec task {base.get('skill')!r}; canonical Sim2Real "
            "currently supports the gripper-actuated lift contract"
        )
    for key, expected in (
        ("gripper_open", spec.gripper_open),
        ("gripper_close", spec.gripper_close),
        ("init_joint_pos", list(spec.home_qpos)),
    ):
        if base.get(key) != expected:
            raise RobotContractError(
                f"RobotSpec.task.{key} conflicts with the embodiment value; "
                "declare it once in the RobotSpec morphology/control fields"
            )
    validated = isaac_byo_robot_task.task_config_overrides(base)
    required = {
        "action_scale",
        "object_init_range",
        "goal_range",
        "minimal_height_m",
        "success_distance_m",
        "gripper_open",
        "gripper_close",
    }
    missing = sorted(required - validated.keys())
    if missing:
        raise RobotContractError(
            "RobotSpec task workspace/reset/reward configuration is invalid: "
            + ", ".join(missing)
        )
    return base


def materialize_robot_contract(
    *,
    robot_spec_uri: str,
    root_uri: str,
    work_dir: Path,
    client: Any,
) -> dict[str, Any]:
    """Fetch, validate, normalize, and publish one content-addressed RobotSpec."""

    if not robot_spec_uri.startswith("s3://"):
        raise RobotContractError(
            "robot_spec_uri must be an accessible s3:// object URI for the "
            "standard workflow; stage public HTTP sources into operator storage first"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / "robot-spec-input.json"
    try:
        client.download_file(robot_spec_uri, str(spec_path))
    except Exception as exc:  # noqa: BLE001 - convert provider errors to operator action
        raise RobotContractError(
            f"robot_spec_uri is inaccessible: {robot_spec_uri!r}: {exc}"
        ) from exc
    try:
        doc = json.loads(spec_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RobotContractError(f"RobotSpec is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise RobotContractError("RobotSpec must be a JSON object")
    if doc.get("schema") not in {None, "", robot_assets.ROBOT_SPEC_SCHEMA}:
        raise RobotContractError(
            f"unsupported RobotSpec schema {doc.get('schema')!r}; expected "
            f"{robot_assets.ROBOT_SPEC_SCHEMA!r}"
        )
    try:
        spec = robot_assets.parse_robot_spec(doc)
    except robot_assets.RobotSpecError as exc:
        raise RobotContractError(f"invalid RobotSpec: {exc}") from exc
    if spec.is_stock_franka():
        raise RobotContractError(
            "a non-empty robot_spec_uri must describe an explicit articulated BYO "
            "robot; omit robot_spec_uri to use the byte-for-byte stock Franka path"
        )
    if spec.n_gripper_joints != 2 or len(spec.finger_links) != 2:
        raise RobotContractError(
            "unsupported gripper contract: canonical Isaac Lift currently requires "
            "a two-joint, two-finger parallel-jaw gripper with explicit ordered "
            "gripper_joint_names and finger_links"
        )

    bundle = work_dir / "source"
    asset_root_uri = str(doc.get("asset_root_uri") or "").strip()
    if asset_root_uri:
        if not asset_root_uri.startswith("s3://"):
            raise RobotContractError("RobotSpec.asset_root_uri must be an s3:// prefix")
        client.download_directory(asset_root_uri.rstrip("/") + "/", str(bundle))
        source_relative = _s3_relative(spec.robot_uri, asset_root_uri)
        source_path = bundle / source_relative
    else:
        source_path = bundle / Path(urlparse(spec.robot_uri).path).name
        try:
            client.download_file(spec.robot_uri, str(source_path))
        except Exception as exc:  # noqa: BLE001
            raise RobotContractError(
                f"RobotSpec.robot_uri is inaccessible: {spec.robot_uri!r}: {exc}"
            ) from exc
        source_relative = source_path.relative_to(bundle).as_posix()
    try:
        source_path.resolve().relative_to(bundle.resolve())
    except ValueError as exc:
        raise RobotContractError(
            "RobotSpec.robot_uri resolved outside RobotSpec.asset_root_uri"
        ) from exc
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise RobotContractError(
            f"RobotSpec.robot_uri did not resolve to a non-empty file: {spec.robot_uri!r}"
        )

    dependencies: list[str] = []
    urdf: dict[str, Any] = {}
    if spec.robot_source == robot_assets.ROBOT_SOURCE_BYO_URDF:
        dependencies = _mesh_dependencies(source_path, bundle)
        urdf = _validate_urdf_structure(source_path, spec)
    elif spec.robot_source != robot_assets.ROBOT_SOURCE_BYO_USD:
        raise RobotContractError(
            f"Isaac canonical workflow supports byo_urdf and byo_usd robot sources; "
            f"got {spec.robot_source!r}"
        )

    source_tree_sha256, files = _tree_digest(bundle)
    source_sha256 = _file_digest(source_path)
    source_root_uri = (
        f"{root_uri.rstrip('/')}/stage_02_assets/robot/by-sha256/"
        f"{source_tree_sha256}/source/"
    )
    client.upload_directory(str(bundle), source_root_uri)
    immutable_source_uri = source_root_uri + source_relative

    normalized_spec = asdict(spec)
    normalized_spec.update(
        {
            "schema": robot_assets.ROBOT_SPEC_SCHEMA,
            "robot_uri": immutable_source_uri,
            "asset_root_uri": source_root_uri,
            "source_sha256": source_sha256,
            "source_tree_sha256": source_tree_sha256,
            "source_relative_path": source_relative,
            "source_format": "urdf" if spec.robot_source == "byo_urdf" else "usd",
            "source_files": files,
            "urdf_dependencies": dependencies,
        }
    )
    task_config = _task_config(doc, spec)
    compatibility = isaac_byo_robot_task.task_robot_compatibility(
        {**normalized_spec, "usd_path": immutable_source_uri},
        task_kind=str(task_config["skill"]),
    )
    if not compatibility.get("task_robot_compatible"):
        raise RobotContractError(str(compatibility.get("reason") or compatibility))
    expected_action_dim = spec.n_arm_joints + (1 if spec.has_gripper else 0)
    expected_observation_dim = 2 * spec.dof_count + 10 + expected_action_dim
    embodiment = {
        "name": spec.name,
        "robot_source": spec.robot_source,
        "base_link": spec.base_link,
        "ee_link": spec.ee_link,
        "arm_joint_names": list(spec.joint_names[: spec.n_arm_joints]),
        "gripper_joint_names": list(spec.gripper_joint_names),
        "finger_links": list(spec.finger_links),
        "home_qpos": list(spec.home_qpos),
        "actuators": {
            "kp": list(spec.kp),
            "kv": list(spec.kv),
            "force_lower": list(spec.force_lower),
            "force_upper": list(spec.force_upper),
        },
        "gripper_commands": {
            "open": spec.gripper_open,
            "close": spec.gripper_close,
        },
        "dimensions": {
            "dof": spec.dof_count,
            "action": expected_action_dim,
            "observation": expected_observation_dim,
            "observation_formula": "2*dof+object_pose7+goal3+last_action",
        },
        "task_config": task_config,
        "compatibility": compatibility,
        "source": {
            "source_uri": immutable_source_uri,
            "asset_root_uri": source_root_uri,
            "source_sha256": source_sha256,
            "source_tree_sha256": source_tree_sha256,
            "license": dict(doc.get("license") or {}),
            "provenance": dict(doc.get("provenance") or {}),
        },
        "urdf": urdf,
    }
    embodiment_digest = _canonical_digest(embodiment)
    normalized_spec.update(
        {
            "embodiment_digest": embodiment_digest,
            "expected_action_dim": expected_action_dim,
            "expected_observation_dim": expected_observation_dim,
            "task_config": task_config,
        }
    )
    resolved_usd_uri = (
        immutable_source_uri
        if spec.robot_source == robot_assets.ROBOT_SOURCE_BYO_USD
        else f"{root_uri.rstrip('/')}/stage_02_assets/robot/resolved/{embodiment_digest}/robot.usd"
    )
    normalized_spec["resolved_usd_uri"] = resolved_usd_uri
    normalized_spec["resolved_manifest_uri"] = (
        f"{root_uri.rstrip('/')}/stage_02_assets/robot/resolved/"
        f"{embodiment_digest}/manifest.json"
    )
    contract = {
        "schema": ROBOT_CONTRACT_SCHEMA,
        "status": (
            "resolved_usd"
            if spec.robot_source == "byo_usd"
            else "immutable_urdf_validated"
        ),
        "embodiment_digest": embodiment_digest,
        "robot_spec": normalized_spec,
        "embodiment": embodiment,
        "isaac_asset": {
            "source_format": normalized_spec["source_format"],
            "source_uri": immutable_source_uri,
            "source_root_uri": source_root_uri,
            "source_relative_path": source_relative,
            "resolved_usd_uri": resolved_usd_uri,
            "conversion": (
                "isaaclab.sim.converters.UrdfConverter"
                if spec.robot_source == "byo_urdf"
                else "none"
            ),
        },
        "requested_inputs": {
            "robot_spec_uri": robot_spec_uri,
            "robot_uri": spec.robot_uri,
            "asset_root_uri": asset_root_uri,
        },
    }
    contract["contract_digest"] = _canonical_digest(contract)
    if not _HEX64.fullmatch(contract["contract_digest"]):  # defensive schema pin
        raise AssertionError("invalid robot contract digest")
    contract_uri = (
        f"{root_uri.rstrip('/')}/stage_02_assets/robot/contracts/by-sha256/"
        f"{contract['contract_digest']}/robot-contract.json"
    )
    contract["content_addressed_uri"] = contract_uri
    contract_path = work_dir / "robot-contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    client.upload_file(str(contract_path), contract_uri)
    return contract


def assert_robot_contract(payload: dict[str, Any]) -> str:
    """Validate a consumed robot contract and return its embodiment digest."""

    if payload.get("schema") != ROBOT_CONTRACT_SCHEMA:
        raise RobotContractError(
            f"expected {ROBOT_CONTRACT_SCHEMA}, got {payload.get('schema')!r}"
        )
    expected = str(payload.get("contract_digest") or "")
    actual = _canonical_digest(
        payload, omit=("contract_digest", "content_addressed_uri")
    )
    if not _HEX64.fullmatch(expected) or expected != actual:
        raise RobotContractError(
            f"robot contract digest mismatch: expected={expected or 'missing'} actual={actual}"
        )
    embodiment = dict(payload.get("embodiment") or {})
    embodiment_digest = str(payload.get("embodiment_digest") or "")
    if embodiment_digest != _canonical_digest(embodiment):
        raise RobotContractError("robot embodiment digest mismatch")
    return embodiment_digest


def stock_robot_contract() -> dict[str, Any]:
    """Return the stock marker without enabling any custom Isaac environment."""

    payload: dict[str, Any] = {
        "schema": ROBOT_CONTRACT_SCHEMA,
        "stage": 2,
        "name": "Stock robot preset (franka)",
        "status": "stock_franka",
        "robot_preset": "franka",
        "robot_spec_uri": "",
        "robot_source": "stock_franka",
        "robot_spec": {
            "schema": robot_assets.ROBOT_SPEC_SCHEMA,
            "preset": "franka",
            "robot_source": "stock_franka",
            "name": "franka_panda",
            "ee_link": "hand",
            "n_arm_joints": 7,
            "n_gripper_joints": 2,
            "isaac_robot_hint": "franka",
            "robot_uri": "",
            "status": "stock_preset",
        },
        "next_action": "CONTINUE",
    }
    payload["contract_digest"] = _canonical_digest(payload)
    return payload


def isaac_environment(
    payload: dict[str, Any], *, contract_uri: str, stage: int
) -> dict[str, Any]:
    """Return custom-only Isaac env values; the stock marker returns no changes."""

    if payload.get("schema") != ROBOT_CONTRACT_SCHEMA:
        raise RobotContractError(
            f"expected {ROBOT_CONTRACT_SCHEMA}, got {payload.get('schema')!r}"
        )
    if payload.get("status") == "stock_franka":
        return {}
    embodiment_digest = assert_robot_contract(payload)
    embodiment = dict(payload["embodiment"])
    dimensions = dict(embodiment["dimensions"])
    return {
        "NPA_BYO_ROBOT_TASK": "1",
        "NPA_SIM2REAL_ROBOT_SPEC_URI": contract_uri,
        "NPA_SIM2REAL_EMBODIMENT_DIGEST": embodiment_digest,
        "NPA_SIM2REAL_EXPECTED_ACTION_DIM": dimensions["action"],
        "NPA_SIM2REAL_EXPECTED_OBSERVATION_DIM": dimensions["observation"],
        "NPA_SIM2REAL_ROBOT_ASSET_OPERATION": "prepare" if stage == 7 else "fetch",
        "NPA_BYO_TASK_CONFIG_JSON": json.dumps(
            embodiment["task_config"], sort_keys=True, separators=(",", ":")
        ),
    }


def assert_embodiment_evidence(
    contract: dict[str, Any], *, payload: dict[str, Any], stage: str
) -> dict[str, Any]:
    """Require custom stages to prove exact embodiment and policy dimensions."""

    if (
        contract.get("schema") != ROBOT_CONTRACT_SCHEMA
        or contract.get("status") == "stock_franka"
    ):
        return {}
    expected = dict(contract["embodiment"])
    dimensions = dict(expected["dimensions"])
    evidence = dict(payload.get("embodiment") or {})
    checks = {
        "embodiment_digest": contract["embodiment_digest"],
        "expected_action_dim": int(dimensions["action"]),
        "expected_observation_dim": int(dimensions["observation"]),
        "runtime_dimension_validation": "passed",
    }
    mismatches = {
        key: {"expected": value, "actual": evidence.get(key)}
        for key, value in checks.items()
        if evidence.get(key) != value
    }
    if mismatches:
        raise RobotContractError(f"{stage} embodiment parity mismatch: {mismatches}")
    return evidence
