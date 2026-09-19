"""Validate the payload-free Gymnasium-Robotics runtime-fetch contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any
import urllib.parse


MANIFEST_SCHEMA = "npa.gymnasium-robotics.public-runtime-fetch-manifest.v1"
RUNTIME_LOCK_SCHEMA = "npa.gymnasium-robotics.runtime-fetch-lock.v2"
CORRESPONDING_LOCK_SCHEMA = "npa.gymnasium-robotics.baked-corresponding-source-lock.v2"
RIGHTS_BOUNDARY = (
    "Runtime fetch changes delivery only; it does not grant or resolve use, "
    "derivative-work, output, or hosted-service rights."
)
EXPECTED_DELIVERY = "operator-owned-runtime-cache"
EXPECTED_CUSTOMER_GATE = "operator-owned-official-access-after-notice-and-acceptance"
EXPECTED_ACCEPTANCE_RECORD = "customer-run-external"
EXPECTED_ORIGINS = frozenset(
    {"https://codeload.github.com", "https://files.pythonhosted.org", "https://github.com"}
)
EXPECTED_EXCLUDED = frozenset(
    {
        "gymnasium-robotics-source",
        "shadow-hand-assets",
        "mujoco-and-python-wheels",
        "operator-runtime-cache",
        "customer-credentials",
        "customer-data",
        "vendor-runtimes",
        "checkpoints",
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RuntimeFetchContractError(RuntimeError):
    """A fail-closed runtime-fetch packaging contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeFetchContractError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            identity = (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size)
            expected = (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            if (
                identity != expected
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_uid not in {0, os.geteuid()}
                or stat.S_IMODE(opened.st_mode) & 0o022
            ):
                raise RuntimeFetchContractError(f"{label} is not a trusted stable regular file")
            raw = stream.read()
            after_descriptor = os.fstat(stream.fileno())
        after = path.lstat()
        final_identity = (
            after_descriptor.st_dev,
            after_descriptor.st_ino,
            after_descriptor.st_mode,
            after_descriptor.st_size,
        )
        path_identity = (after.st_dev, after.st_ino, after.st_mode, after.st_size)
        _require(
            identity == final_identity == path_identity and len(raw) == opened.st_size,
            f"{label} changed while being read",
        )
        value = json.loads(raw)
    except RuntimeFetchContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeFetchContractError(f"{label} is unavailable or malformed") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value, raw


def _sha256(raw: bytes, label: str) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    _require(SHA256.fullmatch(digest) is not None, f"{label} digest is malformed")
    return digest


def _origin(url: Any, label: str) -> str:
    _require(isinstance(url, str), f"{label} must be a URL")
    parsed = urllib.parse.urlsplit(url)
    _require(
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.port in {None, 443},
        f"{label} is not an immutable credential-free HTTPS URL",
    )
    host = parsed.hostname.lower()
    return f"https://{host}"


def _validate_runtime_lock(lock: dict[str, Any], raw: bytes, manifest: dict[str, Any]) -> None:
    _require(lock.get("schema") == RUNTIME_LOCK_SCHEMA, "runtime source lock schema changed")
    _require(lock.get("status") == "complete", "runtime source lock is incomplete")
    _require(lock.get("rights_boundary") == RIGHTS_BOUNDARY, "runtime rights boundary changed")
    delivery = lock.get("delivery")
    _require(
        delivery
        == {
            "source": "operator-owned-runtime-cache",
            "baked_runtime": "neutral-bootstrap-only",
            "weights": "none",
            "data_assets": "runtime-cache-only",
            "runtime_cache": "operator-owned-and-external",
            "outputs": "operator-owned-run-artifacts",
        },
        "runtime delivery boundary changed",
    )
    runtime = manifest["runtime_fetch"]
    _require(runtime["source_lock_sha256"] == _sha256(raw, "runtime source lock"), "runtime source lock digest does not match")
    artifacts = lock.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "runtime artifact inventory is empty")
    roles: list[str] = []
    for index, artifact in enumerate(artifacts):
        _require(isinstance(artifact, dict), f"runtime artifact {index} is malformed")
        _require(artifact.get("delivery_status") == "runtime-only", f"runtime artifact {index} is not runtime-only")
        role = artifact.get("role")
        _require(role in {"solution-source", "python-wheel"}, f"runtime artifact {index} has an unsupported role")
        roles.append(role)
        for field in ("url", "final_url"):
            origin = _origin(artifact.get(field), f"runtime artifact {index} {field}")
            _require(origin in EXPECTED_ORIGINS, f"runtime artifact {index} origin is not approved")
        _require(
            isinstance(artifact.get("sha256"), str) and SHA256.fullmatch(artifact["sha256"]) is not None,
            f"runtime artifact {index} has no exact digest",
        )
    _require(roles.count("solution-source") == 1, "runtime source closure must contain one source artifact")
    _require(roles.count("python-wheel") > 0, "runtime Python closure is empty")


def _validate_corresponding_lock(lock: dict[str, Any], raw: bytes, manifest: dict[str, Any]) -> None:
    _require(lock.get("schema") == CORRESPONDING_LOCK_SCHEMA, "corresponding-source lock schema changed")
    _require(lock.get("status") == "complete", "corresponding-source lock is incomplete")
    _require(
        lock.get("public_corresponding_source_delivery") == "runtime-fetch-operator-owned",
        "corresponding-source delivery is not runtime-fetch-only",
    )
    _require(
        set(lock.get("runtime_fetched_material_excluded", []))
        == EXPECTED_EXCLUDED - {"customer-credentials", "customer-data", "vendor-runtimes", "checkpoints"},
        "excluded runtime material changed",
    )
    _require(
        manifest["runtime_fetch"]["corresponding_source_lock_sha256"] == _sha256(raw, "corresponding-source lock"),
        "corresponding-source lock digest does not match",
    )


def validate_runtime_fetch_contract(
    manifest_path: Path,
    source_lock_path: Path,
    corresponding_lock_path: Path,
) -> dict[str, Any]:
    """Validate a neutral image's runtime-only source and payload boundaries."""

    manifest, _manifest_raw = _read_json(manifest_path, "runtime-fetch manifest")
    _require(
        set(manifest)
        == {"schema", "status", "tool", "image", "runtime_fetch", "excluded_from_public_image", "reason", "rights_boundary"},
        "runtime-fetch manifest fields are incomplete or unsupported",
    )
    _require(manifest["schema"] == MANIFEST_SCHEMA, "runtime-fetch manifest schema changed")
    _require(manifest["status"] == "complete" and manifest["tool"] == "gymnasium-robotics", "runtime-fetch manifest is not complete for Gymnasium-Robotics")
    _require(manifest["rights_boundary"] == RIGHTS_BOUNDARY, "runtime-fetch rights boundary changed")
    image = manifest["image"]
    _require(
        image
        == {
            "redistribution": "public",
            "payload_policy": "neutral-bootstrap-only",
            "restricted_payloads_baked": False,
        },
        "public image payload policy is not neutral",
    )
    runtime = manifest["runtime_fetch"]
    _require(
        set(runtime)
        == {
            "source_lock",
            "source_lock_sha256",
            "corresponding_source_lock",
            "corresponding_source_lock_sha256",
            "delivery",
            "official_origins",
            "customer_gate",
            "acceptance_record",
            "credential_storage",
        },
        "runtime-fetch manifest fields are incomplete or unsupported",
    )
    _require(runtime["delivery"] == EXPECTED_DELIVERY, "runtime delivery is not operator-owned")
    _require(set(runtime["official_origins"]) == EXPECTED_ORIGINS, "official runtime origins changed")
    _require(runtime["customer_gate"] == EXPECTED_CUSTOMER_GATE, "customer notice/acceptance gate is missing")
    _require(runtime["acceptance_record"] == EXPECTED_ACCEPTANCE_RECORD, "customer acceptance must remain external")
    _require(runtime["credential_storage"] == "never-in-image-or-repository", "credential storage boundary changed")
    _require(set(manifest["excluded_from_public_image"]) == EXPECTED_EXCLUDED, "public image exclusion set changed")
    _require(isinstance(manifest["reason"], str) and manifest["reason"], "runtime-fetch rationale is missing")
    source_lock, source_raw = _read_json(source_lock_path, "runtime source lock")
    corresponding_lock, corresponding_raw = _read_json(corresponding_lock_path, "corresponding-source lock")
    _validate_runtime_lock(source_lock, source_raw, manifest)
    _validate_corresponding_lock(corresponding_lock, corresponding_raw, manifest)
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "passed",
        "payload_free": True,
        "runtime_artifact_count": len(source_lock["artifacts"]),
        "customer_gate": runtime["customer_gate"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--corresponding-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_runtime_fetch_contract(args.manifest, args.source_lock, args.corresponding_lock)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
