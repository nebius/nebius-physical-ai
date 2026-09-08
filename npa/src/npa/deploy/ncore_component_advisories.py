"""Execute pinned Grype CPE and official OSV commit evaluations for NCore components."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
import subprocess
import ssl
import tarfile

from npa._public_https import download_public_https
from npa.deploy import ncore_component_sources as sources
from npa.deploy.ncore_component_inventory import _canonical, _require, _sha

_GRYPE_VERSION = "0.118.0"
_GRYPE_ARCHIVE_SHA256 = "1d444c5e7360471815f7158f71935fcecc68a3c417d85c7344f770854300bba2"
_GRYPE_BINARY_SHA256 = "91705979c6ccb736b87e3250831f5e1a35f13767fd2032ffa85c55b1e6f58f90"
_GRYPE_URL = "https://github.com/anchore/grype/releases/download/v0.118.0/grype_0.118.0_linux_amd64.tar.gz"
_OSV_URL = "https://api.osv.dev/v1/query"
_GRYPE_DEFAULT_IGNORES_SHA256 = "de91806bbc574052d009c46ef8fcbe768232385cd2639ae9853cb6f73692d1d5"


def _write(path, value):
    path.write_bytes(json.dumps(value, indent=2).encode() + b"\n")
    return _sha(path.read_bytes())


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _environment(directory):
    # No ambient scanner filters, auth, proxies, config, plugins, or user cache.
    return {"PATH": "/usr/bin:/bin", "HOME": str(directory),
            "XDG_CACHE_HOME": str(directory / "cache")}


def _run(command, directory, stem):
    completed = subprocess.run(command, cwd=directory, env=_environment(directory),
                               capture_output=True, check=False)
    (directory / (stem + ".stdout")).write_bytes(completed.stdout)
    (directory / (stem + ".stderr")).write_bytes(completed.stderr)
    _write(directory / (stem + ".command.json"), {
        "argv": command, "exit": completed.returncode,
        "stdout_sha256": _sha(completed.stdout), "stderr_sha256": _sha(completed.stderr),
    })
    _require(completed.returncode == 0, "scanner command failed: " + stem)
    return completed.stdout


def _grype_binary(directory, executable):
    destination = directory / "grype"
    if executable is not None:
        raw = Path(executable).read_bytes()
    else:
        archive = _download_scanner(_GRYPE_URL)
        _require(_sha(archive) == _GRYPE_ARCHIVE_SHA256, "Grype download hash differs")
        (directory / "grype.tar.gz").write_bytes(archive)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            members = [item for item in source if item.name == "grype"]
            _require(len(members) == 1 and members[0].isfile(), "Grype archive member differs")
            raw = source.extractfile(members[0]).read()
    _require(_sha(raw) == _GRYPE_BINARY_SHA256, "Grype executable hash differs")
    destination.write_bytes(raw)
    destination.chmod(0o700)
    return destination


def _download_scanner(url):
    output = io.BytesIO()
    download_public_https(url, output, allowed_hosts=frozenset({"github.com"}),
                          redirect_hosts=frozenset({"release-assets.githubusercontent.com"}))
    return output.getvalue()


def _prepare_grype(directory, executable):
    binary = _grype_binary(directory, executable)
    config = directory / "grype.json"
    _write(config, {"check-for-app-update": False, "db": {
        "cache-dir": str(directory / "db"), "auto-update": False,
        "validate-by-hash-on-start": True, "validate-age": True,
    }})
    _run([str(binary), "db", "update", "--config", str(config)], directory, "grype-db-update")
    return binary, config


def _grype_configuration(config):
    for field in ("only-fixed", "only-notfixed", "ignore-wontfix", "exclude", "vex-documents", "vex-add"):
        _require(not config.get(field), "Grype filtering is not allowed")
    _require(config.get("match", {}).get("stock", {}).get("using-cpes") is True,
             "Grype CPE matcher required")
    db = config.get("db", {})
    _require(db.get("validate-by-hash-on-start") is True and db.get("validate-age") is True,
             "Grype database validation required")
    # Grype ships four kernel-header rules. They cannot apply to a direct CPE
    # target. Refuse any other rule rather than accepting ambient exclusions.
    rules = config.get("ignore", [])
    _require(rules == [] or _sha(_canonical(rules)) == _GRYPE_DEFAULT_IGNORES_SHA256,
             "unexpected Grype ignore rule")


def grype_findings(payload: dict, cpe: str) -> list[dict]:
    """Validate the evaluated CPE and preserve all fixed/unfixed findings.

    Args:
        payload: JSON emitted by this module's pinned Grype invocation.
        cpe: Exact CPE derived from the authenticated component population.
    Returns:
        Findings, including policy blockers; an empty list is only this CPE's scope.
    Raises:
        ValueError: Wrong scanner/target, invalid database, filtering or suppression.
        KeyError, TypeError: Malformed report fields.
    """
    _require(payload.get("source") == {"type": "cpe", "target": cpe}, "Grype target differs")
    descriptor = payload.get("descriptor", {})
    _require(descriptor.get("name") == "grype" and descriptor.get("version") == _GRYPE_VERSION,
             "pinned Grype version required")
    _grype_configuration(descriptor.get("configuration", {}))
    db = descriptor.get("db", {})
    _require(db.get("status", {}).get("valid") is True and db["status"].get("built")
             and db.get("providers", {}).get("nvd", {}).get("captured"), "Grype database incomplete")
    _require(not payload.get("ignoredMatches"), "suppressed Grype matches are not accepted")
    _require(isinstance(payload.get("matches"), list), "Grype match population missing")
    return [_grype_finding(row, cpe) for row in payload["matches"]]


def _grype_finding(row, cpe):
    artifact = row["artifact"]
    _require(cpe in artifact["cpes"] and artifact["version"] == cpe.split(":")[5],
             "Grype finding component differs")
    vulnerability = row["vulnerability"]
    severity = vulnerability["severity"].upper()
    _require(severity in {"UNKNOWN", "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "CRITICAL"},
             "unknown Grype severity")
    fix = vulnerability["fix"]
    _require(isinstance(fix["versions"], list), "Grype fix population missing")
    return {"id": vulnerability["id"], "severity": severity,
            "fixed_versions": fix["versions"], "fix_state": fix["state"],
            "blocking": severity == "CRITICAL" and bool(fix["versions"] or fix["state"] == "fixed")}


def _grype_evaluation(component, directory, binary, config):
    cpe = component["query"]["cpe"]
    command = [str(binary), cpe, "--config", str(config), "--output", "json"]
    raw = _run(command, directory, component["name"] + ".grype")
    payload = json.loads(raw)
    findings = grype_findings(payload, cpe)
    database = Path(payload["descriptor"]["db"]["status"]["path"])
    _require(database == directory / "db/6/vulnerability.db", "unexpected Grype database path")
    return {"method": "grype-cpe", "query": component["query"], "findings": findings,
            "report_sha256": _sha(raw), "database_sha256": _file_hash(database),
            "database_status": payload["descriptor"]["db"]["status"],
            "scanner_sha256": _file_hash(binary)}


def _osv_page(query):
    connection = http.client.HTTPSConnection("api.osv.dev", port=443,
                                             context=ssl.create_default_context())
    try:
        connection.request("POST", "/v1/query", body=_canonical(query),
                           headers={"Content-Type": "application/json"})
        with connection.getresponse() as response:
            _require(response.status == 200, "OSV response status differs")
            return response.read()
    except (http.client.HTTPException, OSError):
        raise ValueError("NCore component scan: OSV query unavailable") from None
    finally:
        connection.close()


def _osv_evaluation(component, directory):
    query = dict(component["query"])
    pages, findings, seen = [], [], set()
    while True:
        stem = component["name"] + ".osv-" + str(len(pages))
        _write(directory / (stem + ".request.json"), query)
        raw = _osv_page(query)
        (directory / (stem + ".response.json")).write_bytes(raw)
        payload = json.loads(raw)
        _require(isinstance(payload, dict) and set(payload) <= {"vulns", "next_page_token"},
                 "invalid OSV response")
        _require(isinstance(payload.get("vulns", []), list), "invalid OSV findings")
        findings.extend(payload.get("vulns", []))
        pages.append({"query": dict(query), "response_sha256": _sha(raw), "http_status": 200})
        token = payload.get("next_page_token")
        if token is None:
            break
        _require(isinstance(token, str) and token and token not in seen, "invalid OSV pagination")
        seen.add(token)
        query["page_token"] = token
    # These exact source commits lack a package-version/severity policy mapping.
    # Any returned advisory requires review; never turn missing severity into zero.
    return {"method": "osv-commit", "endpoint": _OSV_URL, "query": component["query"], "pages": pages,
            "findings": [{"id": row["id"], "blocking": True} for row in findings]}


def _attach_source_proof(component, result, proofs, directory):
    proof = proofs.get(component["name"])
    if proof is None:
        return
    result["source_proof_sha256"] = _write(
        directory / (component["name"] + ".source-proof.json"), proof)
    result["source_mapping"] = component["source_mapping"]
    result["parent_advisory_scope"] = proof["parent_advisory_scope"]
    if "upstream_review" in proof:
        result["upstream_review"] = proof["upstream_review"]


def evaluate_components(components: list[dict], directory: Path,
                        *, grype_executable: Path | None = None) -> list[dict]:
    """Evaluate every component with a verified mapping, retaining unresolved rows.

    Args:
        components: Byte-bound inventory, including explicitly unmapped components.
        directory: Existing private evidence directory.
        grype_executable: Optional host executable; the same exact binary hash is required.
    Returns:
        One result per component, with real evaluations or explicit coverage failures.
    Raises:
        ValueError: Scanner identity, process, target, schema or database failure.
        OSError, KeyError, TypeError: Missing or malformed inputs.
    """
    proofs = sources.verify_bundled_sources(components, directory)
    binary, config = _prepare_grype(directory, grype_executable)
    results = []
    for component in components:
        query = component["query"]
        if query is None:
            result = {"method": "unmapped", "query": None, "findings": [],
                      "reason": "No verified standalone advisory identity for this vendored snapshot"}
        elif "cpe" in query:
            result = _grype_evaluation(component, directory, binary, config)
        else:
            result = _osv_evaluation(component, directory)
        _attach_source_proof(component, result, proofs, directory)
        result.update({"component": component["name"], "version": component["version"],
                       "files_sha256": component["files_sha256"]})
        results.append(result)
        _write(directory / (component["name"] + ".evaluation.json"), result)
    _require(_file_hash(binary) == _GRYPE_BINARY_SHA256, "Grype changed during evaluation")
    return results
