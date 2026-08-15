"""Fail-closed dependency-closure validation for the split Wan runtime image.

The public image deliberately omits Torch and NVIDIA CUDA distributions.  A
plain ``pip check`` in that image would therefore report intentional failures,
while comparing ``pip freeze`` with the baked constraints cannot detect a
missing transitive dependency.  This validator combines the metadata from the
actually installed baked distributions with hash-bound metadata for the exact
runtime wheels and checks every applicable ``Requires-Dist`` edge.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import urllib.request
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

REPORT_SCHEMA = "npa.workbench.wan2_2.dependency_closure.v1"
PYPI_SIMPLE_INDEX = "https://pypi.org/simple"
MAX_CORE_METADATA_BYTES = 8 * 1024 * 1024

# These are the only distributions that may be absent from the public image and
# supplied solely by the operator-accepted runtime overlay.  Shared pure-Python
# dependencies may be repeated in the runtime lock, but must also exist in the
# baked environment so ordinary OSS imports remain usable before provisioning.
RUNTIME_ONLY_DISTRIBUTIONS = frozenset(
    {
        "torch",
        "torchvision",
        "triton",
        "cuda-bindings",
        "cuda-pathfinder",
        "cuda-toolkit",
        "nvidia-cublas",
        "nvidia-cuda-cupti",
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-runtime",
        "nvidia-cudnn-cu13",
        "nvidia-cufft",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusolver",
        "nvidia-cusparse",
        "nvidia-cusparselt-cu13",
        "nvidia-nccl-cu13",
        "nvidia-nvjitlink",
        "nvidia-nvshmem-cu13",
        "nvidia-nvtx",
    }
)


class DependencyClosureError(RuntimeError):
    """Raised when the baked/runtime dependency union is not closed."""


@dataclass(frozen=True)
class PinnedRequirement:
    name: str
    version: str
    sha256: str


@dataclass(frozen=True)
class DistributionMetadata:
    name: str
    version: str
    requires_dist: tuple[str, ...] = ()
    metadata_sha256: str = ""
    wheel_filename: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_requirement_lines(path: Path) -> list[str]:
    logical_lines: list[str] = []
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        current += line.removesuffix("\\").strip() + " "
        if not line.endswith("\\"):
            logical_lines.append(current.strip())
            current = ""
    if current:
        raise DependencyClosureError(
            f"unterminated line continuation in {path.name}"
        )
    return logical_lines


def parse_runtime_requirements(path: Path) -> dict[str, PinnedRequirement]:
    """Parse the exact one-wheel-per-distribution runtime lock."""

    pins: dict[str, PinnedRequirement] = {}
    for logical in _logical_requirement_lines(path):
        if logical.startswith("--"):
            if logical != f"--index-url {PYPI_SIMPLE_INDEX}":
                raise DependencyClosureError(
                    f"unsupported runtime requirement option: {logical}"
                )
            continue
        requirement_text, separator, hash_text = logical.partition(" --hash=sha256:")
        if not separator or not hash_text or " --hash=" in hash_text:
            raise DependencyClosureError(
                f"runtime requirement must carry exactly one SHA-256: {logical}"
            )
        if len(hash_text) != 64 or any(char not in "0123456789abcdef" for char in hash_text):
            raise DependencyClosureError(
                f"runtime requirement has an invalid SHA-256: {logical}"
            )
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as exc:
            raise DependencyClosureError(
                f"invalid runtime requirement {requirement_text!r}: {exc}"
            ) from exc
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.extras
            or requirement.marker is not None
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
        ):
            raise DependencyClosureError(
                f"runtime requirement is not one unconditional exact pin: {logical}"
            )
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise DependencyClosureError(f"duplicate runtime requirement: {name}")
        pins[name] = PinnedRequirement(
            name=name,
            version=specifiers[0].version,
            sha256=hash_text,
        )
    if not pins:
        raise DependencyClosureError("runtime requirements contain no distributions")
    return pins


def installed_distribution_metadata(
    site_packages: Path,
) -> dict[str, DistributionMetadata]:
    """Read metadata from the image environment, without importing packages."""

    records: dict[str, DistributionMetadata] = {}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        raw_name = distribution.metadata.get("Name") or distribution.name
        name = canonicalize_name(raw_name)
        if name in records:
            raise DependencyClosureError(
                f"baked site-packages contains duplicate distribution metadata: {name}"
            )
        records[name] = DistributionMetadata(
            name=name,
            version=distribution.version,
            requires_dist=tuple(distribution.metadata.get_all("Requires-Dist") or ()),
        )
    if not records:
        raise DependencyClosureError(
            f"baked site-packages contains no distribution metadata: {site_packages}"
        )
    return records


def _read_limited(response: Any, *, source: str) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared is not None and int(declared) > MAX_CORE_METADATA_BYTES:
        raise DependencyClosureError(f"core metadata is too large: {source}")
    payload = response.read(MAX_CORE_METADATA_BYTES + 1)
    if len(payload) > MAX_CORE_METADATA_BYTES:
        raise DependencyClosureError(f"core metadata is too large: {source}")
    return payload


def fetch_runtime_distribution_metadata(
    pins: Mapping[str, PinnedRequirement],
) -> dict[str, DistributionMetadata]:
    """Fetch PEP 658 metadata for the exact hash-selected PyPI wheels.

    Only the small, independently hash-bound ``.metadata`` sidecars are read;
    the Torch/NVIDIA wheel payloads are never downloaded into the image build.
    """

    compatible_tags = set(sys_tags())
    records: dict[str, DistributionMetadata] = {}
    for name, pin in pins.items():
        request = urllib.request.Request(  # noqa: S310 - fixed HTTPS origin
            f"{PYPI_SIMPLE_INDEX}/{name}/",
            headers={"Accept": "application/vnd.pypi.simple.v1+json"},
        )
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310
                simple = json.load(response)
        except (OSError, ValueError) as exc:
            raise DependencyClosureError(
                f"cannot read PyPI simple metadata for {name}: {exc}"
            ) from exc
        matches = [
            item
            for item in simple.get("files", [])
            if isinstance(item, dict)
            and item.get("hashes", {}).get("sha256") == pin.sha256
        ]
        if len(matches) != 1:
            raise DependencyClosureError(
                f"{name}=={pin.version} SHA-256 selects {len(matches)} PyPI files"
            )
        selected = matches[0]
        filename = str(selected.get("filename") or "")
        try:
            wheel_name, wheel_version, _build, wheel_tags = parse_wheel_filename(
                filename
            )
        except (InvalidVersion, ValueError) as exc:
            raise DependencyClosureError(
                f"hash-selected runtime artifact is not a valid wheel: {filename}"
            ) from exc
        if (
            canonicalize_name(wheel_name) != name
            or wheel_version != Version(pin.version)
            or not compatible_tags.intersection(wheel_tags)
        ):
            raise DependencyClosureError(
                f"hash-selected runtime wheel is not compatible: {filename}"
            )
        if selected.get("yanked") not in (False, None):
            raise DependencyClosureError(f"hash-selected runtime wheel is yanked: {filename}")
        requires_python = selected.get("requires-python")
        if requires_python:
            try:
                python_specifier = SpecifierSet(str(requires_python))
            except InvalidSpecifier as exc:
                raise DependencyClosureError(
                    f"runtime wheel has invalid Requires-Python: {filename}"
                ) from exc
            current_python = Version(
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            if current_python not in python_specifier:
                raise DependencyClosureError(
                    f"runtime wheel rejects Python {current_python}: {filename}"
                )
        metadata_identity = selected.get("core-metadata") or selected.get(
            "data-dist-info-metadata"
        )
        if not isinstance(metadata_identity, dict):
            raise DependencyClosureError(
                f"runtime wheel lacks hash-bound core metadata: {filename}"
            )
        metadata_sha256 = str(metadata_identity.get("sha256") or "")
        if len(metadata_sha256) != 64:
            raise DependencyClosureError(
                f"runtime wheel core metadata lacks SHA-256: {filename}"
            )
        wheel_url = str(selected.get("url") or "")
        parsed_url = urlparse(wheel_url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "files.pythonhosted.org":
            raise DependencyClosureError(
                f"runtime wheel URL is outside files.pythonhosted.org: {filename}"
            )
        metadata_url = wheel_url + ".metadata"
        try:
            with urllib.request.urlopen(metadata_url) as response:  # noqa: S310
                raw_metadata = _read_limited(response, source=metadata_url)
        except OSError as exc:
            raise DependencyClosureError(
                f"cannot read hash-bound core metadata for {filename}: {exc}"
            ) from exc
        if hashlib.sha256(raw_metadata).hexdigest() != metadata_sha256:
            raise DependencyClosureError(
                f"core metadata SHA-256 mismatch for {filename}"
            )
        message = BytesParser(policy=compat32).parsebytes(raw_metadata)
        metadata_name = canonicalize_name(str(message.get("Name") or ""))
        metadata_version = str(message.get("Version") or "")
        if metadata_name != name or metadata_version != pin.version:
            raise DependencyClosureError(
                f"core metadata identity mismatch for {filename}: "
                f"{metadata_name}=={metadata_version}"
            )
        records[name] = DistributionMetadata(
            name=name,
            version=metadata_version,
            requires_dist=tuple(message.get_all("Requires-Dist") or ()),
            metadata_sha256=metadata_sha256,
            wheel_filename=filename,
        )
    return records


def validate_dependency_union(
    baked: Mapping[str, DistributionMetadata],
    runtime: Mapping[str, DistributionMetadata],
    *,
    runtime_only_allowlist: frozenset[str] = RUNTIME_ONLY_DISTRIBUTIONS,
) -> dict[str, Any]:
    """Validate runtime-only policy and every active transitive requirement."""

    normalized_baked = {canonicalize_name(name): item for name, item in baked.items()}
    normalized_runtime = {
        canonicalize_name(name): item for name, item in runtime.items()
    }
    runtime_only = frozenset(normalized_runtime).difference(normalized_baked)
    if runtime_only != runtime_only_allowlist:
        unexpected = sorted(runtime_only.difference(runtime_only_allowlist))
        absent = sorted(runtime_only_allowlist.difference(runtime_only))
        raise DependencyClosureError(
            "runtime-only distribution set drifted; "
            f"unexpected={unexpected}, absent={absent}"
        )

    # Runtime site-packages precedes the baked .pth path, so an exact runtime
    # pin is the effective version when a pure-Python dependency is duplicated.
    effective = {**normalized_baked, **normalized_runtime}
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    checked_edges = 0
    declared_edges = 0
    errors: list[str] = []
    for source, distributions in (
        ("baked", normalized_baked),
        ("runtime", normalized_runtime),
    ):
        for parent_name, parent in distributions.items():
            for raw_requirement in parent.requires_dist:
                declared_edges += 1
                try:
                    requirement = Requirement(raw_requirement)
                except InvalidRequirement as exc:
                    errors.append(
                        f"{source} {parent_name}=={parent.version} has invalid "
                        f"Requires-Dist {raw_requirement!r}: {exc}"
                    )
                    continue
                if requirement.marker and not requirement.marker.evaluate(
                    marker_environment
                ):
                    continue
                checked_edges += 1
                dependency_name = canonicalize_name(requirement.name)
                dependency = effective.get(dependency_name)
                if dependency is None:
                    errors.append(
                        f"{source} {parent_name}=={parent.version} requires missing "
                        f"{requirement}"
                    )
                    continue
                try:
                    compatible = (
                        not requirement.specifier
                        or Version(dependency.version) in requirement.specifier
                    )
                except InvalidVersion as exc:
                    errors.append(
                        f"{dependency_name} has invalid installed version "
                        f"{dependency.version!r}: {exc}"
                    )
                    continue
                if not compatible:
                    errors.append(
                        f"{source} {parent_name}=={parent.version} requires "
                        f"{requirement}; effective {dependency_name}=={dependency.version}"
                    )
    if errors:
        raise DependencyClosureError(
            "dependency union is not closed:\n- " + "\n- ".join(sorted(errors))
        )
    return {
        "status": "validated",
        "baked_distribution_count": len(normalized_baked),
        "runtime_distribution_count": len(normalized_runtime),
        "effective_distribution_count": len(effective),
        "declared_dependency_edges": declared_edges,
        "applicable_dependency_edges_checked": checked_edges,
        "runtime_only_distributions": sorted(runtime_only),
    }


def build_closure_report(
    *,
    baked_site: Path,
    baked_inventory: Path,
    runtime_requirements: Path,
) -> dict[str, Any]:
    pins = parse_runtime_requirements(runtime_requirements)
    baked = installed_distribution_metadata(baked_site)
    runtime = fetch_runtime_distribution_metadata(pins)
    report = validate_dependency_union(baked, runtime)
    report.update(
        {
            "schema": REPORT_SCHEMA,
            "baked_inventory_sha256": _sha256_file(baked_inventory),
            "runtime_requirements_sha256": _sha256_file(runtime_requirements),
            "runtime_wheels": {
                name: {
                    "version": runtime[name].version,
                    "wheel_sha256": pin.sha256,
                    "metadata_sha256": runtime[name].metadata_sha256,
                    "wheel_filename": runtime[name].wheel_filename,
                }
                for name, pin in sorted(pins.items())
            },
        }
    )
    return report


def verify_closure_report(
    report_path: Path,
    *,
    baked_inventory: Path,
    runtime_requirements: Path,
) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyClosureError(f"cannot read dependency closure report: {exc}") from exc
    if not isinstance(report, dict):
        raise DependencyClosureError("dependency closure report is not an object")
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "validated":
        raise DependencyClosureError("dependency closure report is not validated")
    if report.get("baked_inventory_sha256") != _sha256_file(baked_inventory):
        raise DependencyClosureError("baked inventory changed after closure validation")
    if report.get("runtime_requirements_sha256") != _sha256_file(runtime_requirements):
        raise DependencyClosureError(
            "runtime requirements changed after closure validation"
        )
    if frozenset(report.get("runtime_only_distributions") or ()) != RUNTIME_ONLY_DISTRIBUTIONS:
        raise DependencyClosureError("dependency closure report runtime-only set drifted")
    runtime_wheels = report.get("runtime_wheels")
    if not isinstance(runtime_wheels, dict):
        raise DependencyClosureError("dependency closure report lacks runtime wheels")
    pins = parse_runtime_requirements(runtime_requirements)
    if set(runtime_wheels) != set(pins):
        raise DependencyClosureError("dependency closure report runtime wheel set drifted")
    for name, pin in pins.items():
        wheel = runtime_wheels.get(name)
        if not isinstance(wheel, dict) or (
            wheel.get("version") != pin.version
            or wheel.get("wheel_sha256") != pin.sha256
        ):
            raise DependencyClosureError(
                f"dependency closure report runtime wheel drifted: {name}"
            )
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--baked-site", type=Path, required=True)
    validate.add_argument("--baked-inventory", type=Path, required=True)
    validate.add_argument("--runtime-requirements", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    verify = subparsers.add_parser("verify-report")
    verify.add_argument("--baked-inventory", type=Path, required=True)
    verify.add_argument("--runtime-requirements", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "validate":
            report = build_closure_report(
                baked_site=args.baked_site,
                baked_inventory=args.baked_inventory,
                runtime_requirements=args.runtime_requirements,
            )
            _write_report(args.report, report)
        else:
            report = verify_closure_report(
                args.report,
                baked_inventory=args.baked_inventory,
                runtime_requirements=args.runtime_requirements,
            )
    except DependencyClosureError as exc:
        print(f"Wan dependency closure validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
