"""Replace Python 3.10's bundled installer with verified repaired wheel bytes."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile


PIP_VERSION = "26.2.1+npa.1"
SETUPTOOLS_VERSION = "84.0.0"
SETUPTOOLS_SHA256 = "f4695c21257f0d9b537ec2692c941d02ee143b7cc1276941349a546573b2ef73"


def _wheel(path: Path, name: str, version: str, expected_sha256: str) -> bytes:
    if path.is_symlink() or path.name != f"{name}-{version}-py3-none-any.whl":
        raise ValueError("Bootstrap wheel identity is invalid")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Bootstrap wheel bytes do not match their verified identity")
    with zipfile.ZipFile(path) as wheel:
        member = f"{name}-{version}.dist-info/METADATA"
        metadata = BytesParser().parsebytes(wheel.read(member))
    if metadata["Name"] != name or metadata["Version"] != version:
        raise ValueError("Bootstrap wheel metadata contradicts its identity")
    return payload


def _pip_bytes(wheel: Path, receipt: Path, inputs: Path) -> bytes:
    recorded = json.loads(receipt.read_text())
    approved = json.loads(inputs.read_text())
    if recorded["wheel"] != wheel.name or recorded["provenance"]["inputs"] != approved:
        raise ValueError("Installer receipt does not bind the reviewed source inputs")
    payload = _wheel(wheel, "pip", PIP_VERSION, recorded["sha256"])
    with zipfile.ZipFile(wheel) as archive:
        provenance = json.loads(archive.read("pip/NPA_VENDOR_REPAIR.json"))
    if provenance != recorded["provenance"] or recorded["bytes"] != len(payload):
        raise ValueError("Installer source repair provenance does not match")
    return payload


def _versions(source: str) -> dict[str, str]:
    found = re.findall(
        r'^_(PIP|SETUPTOOLS)_VERSION = "([^"\n]+)"$', source, re.MULTILINE
    )
    if len(found) != 2 or {name for name, _ in found} != {"PIP", "SETUPTOOLS"}:
        raise ValueError("Unsupported Python 3.10 ensurepip version declarations")
    return {name.lower(): version for name, version in found}


def _replace_bundle(root: Path, source: str, payloads: dict[str, bytes]) -> str:
    versions = _versions(source)
    old_files = {
        f"{name}-{version}-py3-none-any.whl" for name, version in versions.items()
    }
    bundled = root / "_bundled"
    if root.is_symlink() or bundled.is_symlink() or (root / "__init__.py").is_symlink():
        raise ValueError("Bootstrap directories and source must not be symlinks")
    if {path.name for path in bundled.glob("*.whl")} != old_files:
        raise ValueError("Unexpected ensurepip wheel population")
    for name in payloads:
        if (bundled / name).is_symlink():
            raise ValueError("Bootstrap wheel destination must not be a symlink")
    for name, payload in payloads.items():
        (bundled / name).write_bytes(payload)
    for name in old_files - payloads.keys():
        (bundled / name).unlink()
    for name, version in {"pip": PIP_VERSION, "setuptools": SETUPTOOLS_VERSION}.items():
        source = source.replace(
            f'_{name.upper()}_VERSION = "{versions[name]}"',
            f'_{name.upper()}_VERSION = "{version}"',
        )
    return source


def refresh_bootstrap(
    root: Path, pip_wheel: Path, pip_receipt: Path, setuptools_wheel: Path, inputs: Path
) -> dict[str, str]:
    """Install source-identified wheel bytes into the standard-library bootstrap.

    Args:
        root: Python 3.10's ensurepip package directory.
        pip_wheel: Repaired wheel produced by the reviewed secure-pip builder.
        pip_receipt: Its matching build receipt.
        setuptools_wheel: Exact upstream Setuptools wheel.
        inputs: Reviewed secure-pip input manifest.
    Returns:
        Before/after source hashes and installed wheel hashes.
    Raises:
        ValueError: Wheel identity, provenance, or bootstrap layout is invalid.
        OSError: A required input cannot be read or replaced.
    """
    pip_payload = _pip_bytes(pip_wheel, pip_receipt, inputs)
    setuptools_payload = _wheel(
        setuptools_wheel, "setuptools", SETUPTOOLS_VERSION, SETUPTOOLS_SHA256
    )
    path = root / "__init__.py"
    original = path.read_text()
    payloads = {pip_wheel.name: pip_payload, setuptools_wheel.name: setuptools_payload}
    updated = _replace_bundle(root, original, payloads)
    path.write_text(updated)
    return {
        "source_before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "source_after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "pip_wheel_sha256": hashlib.sha256(pip_payload).hexdigest(),
        "setuptools_wheel_sha256": hashlib.sha256(setuptools_payload).hexdigest(),
    }


def main() -> None:
    """Refresh the current image's bootstrap from verified build-stage inputs.

    Args:
        None. Paths are supplied through command-line options.
    Returns:
        None.
    Raises:
        RuntimeError: The interpreter is outside the tested Python 3.10 contract.
        ValueError: Supplied artifacts or the bootstrap layout are invalid.
    """
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("This bootstrap repair requires Python 3.10")
    import ensurepip

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pip-wheel", "pip-receipt", "setuptools-wheel", "inputs", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = refresh_bootstrap(
        Path(ensurepip.__file__).parent,
        args.pip_wheel,
        args.pip_receipt,
        args.setuptools_wheel,
        args.inputs,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
