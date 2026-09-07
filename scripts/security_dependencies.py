"""Inventory declared dependencies and compare real Trivy vulnerability reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _run(arguments: list[str], directory: Path) -> None:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_", "TRIVY_"))
    }
    with (directory / "commands.log").open("a") as log:
        result = subprocess.run(arguments, cwd=directory, env=environment,
                                stdout=log, stderr=log, check=False)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed; inspect private commands.log")


def _exact_pins(text: str) -> list[str]:
    pins = []
    for line in text.replace("\\\n", "").splitlines():
        declaration = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        declaration = re.split(r"\s+--(?:hash|config-settings)(?:=|\s)", declaration, maxsplit=1)[0]
        if not declaration or declaration.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(declaration)
        except InvalidRequirement:
            continue
        for constraint in requirement.specifier:
            if constraint.operator == "==" and "*" not in constraint.version:
                pins.append(f"{requirement.name}=={constraint.version}")
    return pins


def _write_pins(destination: Path, pins: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(sorted(set(pins))) + "\n")


def _validate_npm_manifests(root: Path) -> None:
    for manifest in root.rglob("package.json"):
        project = json.loads(manifest.read_text())
        sections = ("dependencies", "devDependencies", "optionalDependencies")
        if not any(project.get(section) for section in sections):
            continue
        lock_path = manifest.with_name("package-lock.json")
        if not lock_path.is_file():
            raise ValueError("Every npm dependency manifest requires a package lock")
        lock = json.loads(lock_path.read_text())
        declared = lock.get("packages", {}).get("")
        if not isinstance(declared, dict):
            raise TypeError("npm package lock is missing its root dependency declarations")
        if any(project.get(section, {}) != declared.get(section, {}) for section in sections):
            raise ValueError("npm dependency declarations and package lock are out of sync")
        _require_direct_packages(project, lock["packages"], sections)


def _require_direct_packages(project: dict, packages: dict, sections: tuple[str, ...]) -> None:
    for section in sections:
        for name, declaration in project.get(section, {}).items():
            version = packages.get(f"node_modules/{name}", {}).get("version")
            if not version:
                raise ValueError("npm package lock is missing a resolved direct dependency")
            exact_version = re.fullmatch(r"\d+\.\d+\.\d+(?:-[\w.-]+)?(?:\+[\w.-]+)?", declaration)
            if exact_version and version != declaration:
                raise ValueError("npm resolved direct dependency contradicts its exact version pin")


def _resolve_project(project: dict, output: Path, cache: Path) -> Path:
    declarations = project["dependencies"] + project.get(
        "optional-dependencies", {}).get("dev", [])
    for declaration in declarations:
        if Requirement(declaration).url:
            raise ValueError("Core/development dependencies must use package index requirements")
    content = "\n".join(sorted(set(declarations))) + "\n"
    digest = hashlib.sha256(content.encode()).hexdigest()
    resolved = cache / f"{digest}.txt"
    if resolved.exists():
        return resolved
    source = output / "project.in"
    source.write_text(content)
    _run(["uv", "pip", "compile", str(source), "--no-config", "--no-sources",
          "--no-build", "--python-version", "3.12", "--default-index",
          "https://pypi.org/simple", "--output-file", str(resolved)], output)
    return resolved


def _inventory(root: Path, output: Path, cache: Path) -> dict[str, str]:
    _validate_npm_manifests(root)
    inventory = {}
    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(root).as_posix()
        if source.name == "package-lock.json":
            destination = output / "inputs" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        elif "requirements" in source.name and source.suffix in {".txt", ".lock", ".in"}:
            destination = output / "inputs" / relative / "requirements.txt"
            _write_pins(destination, _exact_pins(source.read_text()))
        elif source.name == "pyproject.toml":
            project = tomllib.loads(source.read_text()).get("project", {})
            declarations = list(project.get("dependencies", []))
            for extra in project.get("optional-dependencies", {}).values():
                declarations.extend(extra)
            destination = output / "inputs" / relative / "requirements.txt"
            _write_pins(destination, _exact_pins("\n".join(declarations)))
        else:
            continue
        inventory[destination.relative_to(output / "inputs").as_posix()] = relative
    project_file = root / "npa/pyproject.toml"
    if project_file.exists():
        project = tomllib.loads(project_file.read_text())["project"]
        resolved = _resolve_project(project, output, cache)
        destination = output / "inputs/npa-resolved/requirements.txt"
        _write_pins(destination, _exact_pins(resolved.read_text()))
        inventory["npa-resolved/requirements.txt"] = "npa/pyproject.toml (resolved core + dev)"
    return inventory


def _findings(report: dict, inventory: dict[str, str]) -> list[dict]:
    if report.get("SchemaVersion") != 2 or not isinstance(report.get("Results"), list):
        raise ValueError("Trivy did not return a complete version 2 report")
    findings = []
    for target in report["Results"]:
        path = inventory[target["Target"]]
        for vulnerability in target.get("Vulnerabilities", []):
            package = vulnerability["PkgName"]
            version = vulnerability["InstalledVersion"]
            rule = vulnerability["VulnerabilityID"]
            findings.append({"scanner": "trivy", "path": path, "rule": rule,
                             "identity": f"{package}=={version}:{rule}", "line": 1,
                             "message": f"{package}=={version}; fixed: "
                             f"{vulnerability.get('FixedVersion') or 'not available'}"})
    return findings


def _expected_packages(source: Path) -> set[tuple[str, str]]:
    if source.name == "requirements.txt":
        return {(canonicalize_name(pin.split("==")[0]), pin.split("==")[1])
                for pin in _exact_pins(source.read_text())}
    lock = json.loads(source.read_text())
    if lock.get("lockfileVersion") not in (2, 3) or not isinstance(lock.get("packages"), dict):
        raise ValueError("npm dependencies require a complete version 2 or 3 package lock")
    packages = set()
    for path, package in lock["packages"].items():
        if not path or package.get("link"):
            continue
        name = package.get("name") or path.rsplit("node_modules/", 1)[-1]
        packages.add((canonicalize_name(name), package["version"]))
    return packages


def _validate_coverage(report: dict, inventory: dict[str, str], output: Path) -> None:
    observed = {}
    for target in report["Results"]:
        observed[target["Target"]] = {
            (canonicalize_name(package["Name"]), package["Version"])
            for package in target.get("Packages", [])}
    for target, manifest in inventory.items():
        expected = _expected_packages(output / "inputs" / target)
        if expected - observed.get(target, set()):
            raise ValueError(f"Trivy omitted declared packages from {manifest}")


def _scanner_versions(output: Path) -> None:
    for scanner, expected in (("trivy", "Version: 0.74.0"), ("uv", "uv 0.12.5")):
        result = subprocess.run([scanner, "--version"], check=True,
                                capture_output=True, text=True)
        if not result.stdout.startswith(expected):
            raise RuntimeError(f"Install the pinned {scanner} version")
        (output / f"{scanner}-version.txt").write_text(result.stdout)


def scan_dependencies(root: Path, output: Path, cache: Path) -> list[dict]:
    """Scan exact declarations and the resolved core/development dependency closure.

    Args:
        root: Materialized source snapshot, never executed or installed.
        output: Private report directory.
        cache: Shared resolution and Trivy database directory for both revisions.
    Returns:
        Vulnerability occurrences keyed by manifest, package version and advisory.
    Raises:
        RuntimeError: Resolution or scanning fails.
        ValueError: A dependency or report cannot be safely interpreted.
    """
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    _scanner_versions(output)
    inventory = _inventory(root, output, cache)
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    if not inventory:
        return []
    (output / "empty.yaml").write_text("{}\n")
    (output / "empty.ignore").write_text("")
    arguments = ["trivy", "fs", "--config", str(output / "empty.yaml"),
                 "--ignorefile", str(output / "empty.ignore"), "--scanners", "vuln",
                 "--pkg-types", "library", "--include-dev-deps", "--ignore-unfixed=false",
                 "--severity", "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL", "--disable-telemetry",
                 "--cache-dir", str(cache / "trivy"), "--format", "json", "--output",
                 str(output / "trivy.json"), "--exit-code", "0"]
    if (cache / "trivy/db/trivy.db").exists():
        arguments.append("--skip-db-update")
    _run(arguments + [str(output / "inputs")], output)
    report = json.loads((output / "trivy.json").read_text())
    findings = _findings(report, inventory)
    _validate_coverage(report, inventory, output)
    return findings
