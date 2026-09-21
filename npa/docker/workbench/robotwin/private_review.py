"""Transport an independent review of one failed RoboTwin byte-scan phase."""

from __future__ import annotations

import io
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
from urllib.parse import unquote, urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import adjudicate as A, prepare as P  # noqa: E402

W = A.W
REQUEST_ENV = "ROBOTWIN_PRIVATE_REVIEW_REQUEST_URL"
RESPONSE_ENV = "ROBOTWIN_PRIVATE_REVIEW_RESPONSE_URL"
REQUEST_SCHEMA = "npa.robotwin.private-byte-review-request.v1"
RESPONSE_SCHEMA = "npa.robotwin.private-byte-review-response.v1"
RESPONSE_NAME = re.compile(
    r"(?:response|manifest|review)\.json|proofs/[0-9a-f]{64}\.json|"
    r"evidence/[0-9a-f]{64}\.(?:json|txt|bin)"
)


def _url(value):
    parsed = urlsplit(value)
    W.require(parsed.scheme == "https" and bool(parsed.hostname)
              and parsed.username is None and parsed.password is None
              and not parsed.fragment, "robotwin_review_url")
    return value


def _object_identity(value):
    parsed = urlsplit(value)
    return parsed.hostname, parsed.port or 443, unquote(parsed.path)


def _scan_inputs(phase):
    fixed = {name: P.binding(phase / name) for name in (
        "image.tar", "scan/report.json", "scan/records.jsonl",
        "graph/verification.json", "authorization/authorization.json")}
    authorization = W.bound_json(fixed["authorization/authorization.json"])
    report = W.bound_json(fixed["scan/report.json"])
    ledger = A.bound_bytes(fixed["scan/records.jsonl"])
    population = A.population(report, [A.decode(line) for line in ledger.splitlines()])
    W.require(report["valid"] is False and population, "robotwin_review_not_failed")
    W.require(authorization["archive"] == fixed["image.tar"]
              and authorization["verification_report"] == fixed["graph/verification.json"]
              and authorization.get("confidentiality") is not None
              and "literal_inventory" not in authorization, "robotwin_review_inputs")
    verification = W.bound_json(authorization["verification_report"])
    W.require(verification["schema_version"] == "npa.robotwin.image-verification.v1",
              "robotwin_review_graph_schema")
    context = A.context(authorization, report, fixed["scan/report.json"]["sha256"],
                        fixed["scan/records.jsonl"]["sha256"],
                        fixed["authorization/authorization.json"]["sha256"],
                        A.image_revision(authorization, verification, report),
                        A.committed_sources(authorization))
    W.require(report["authorization_sha256"] == context["authorization_sha256"]
              and report["archive_sha256"] == context["archive_sha256"]
              and A.policy_receipt(authorization) == report["confidentiality_policy"],
              "robotwin_review_scan_binding")
    snapshots = W.input_snapshots(authorization)
    expected = [{"role": role, "sha256": spec["sha256"], "stat": list(before)}
                for role, spec, _secret, _path, before in snapshots]
    W.require(report["input_snapshot_receipts"] == expected, "robotwin_review_snapshots")
    return fixed, authorization, context, snapshots


def _request_members(fixed, authorization):
    members = {name: (binding, True) for name, binding in fixed.items()}
    members["authorization/confidentiality.json"] = (authorization["confidentiality"], True)
    members["tools/dependency-receipt.json"] = (authorization["tools_receipt"], True)
    tools = W.bound_json(authorization["tools_receipt"])
    members["tools/ready.json"] = (tools["ready"], True)
    members["tools/config.toml"] = (authorization["config"], False)
    if "literal_engine" in authorization:
        native = P.binding(Path(os.environ["ROBOTWIN_BYTE_GATE_ROOT"]) / "native/dependencies.json")
        receipt = W.bound_json(native)
        W.require({key: receipt[key] for key in ("kind", *W.AHO_PINS)}
                  == authorization["literal_engine"], "robotwin_review_native_receipt")
        members["native/dependencies.json"] = (native, True)
    # Only the pinned scanner policy and dependency receipts are exported here.
    # Executables, arbitrary environment and registry/storage credentials are not.
    return members


def _tar_bytes(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size, member.mode = len(data), 0o600
    archive.addfile(member, io.BytesIO(data))


def _tar_bound(archive, name, binding, secret):
    with W.bound_open(binding, secret=secret) as (_path, descriptor, before):
        member = tarfile.TarInfo(name)
        member.size, member.mode = before.st_size, 0o600
        with os.fdopen(os.dup(descriptor), "rb") as source:
            archive.addfile(member, source)
        W.require(W.descriptor_digest(descriptor) == binding["sha256"]
                  and W.stat_fingerprint(os.fstat(descriptor)) == W.stat_fingerprint(before),
                  "robotwin_review_export_changed")


def _put_request(client, url, phase, request, members):
    raw = W.canonical(request)
    P.save_bytes(phase, "review-request.json", raw)
    with tempfile.TemporaryFile(dir=phase) as bundle:
        with tarfile.open(fileobj=bundle, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            _tar_bytes(archive, "request.json", raw)
            for name, (binding, secret) in members.items():
                _tar_bound(archive, name, binding, secret)
        size = bundle.tell()
        bundle.seek(0)
        with client.stream("PUT", url, content=iter(lambda: bundle.read(W.CHUNK), b""),
                           headers={"Content-Type": "application/x-tar",
                                    "Content-Length": str(size)}) as response:
            W.require(200 <= response.status_code < 300, "robotwin_review_upload_status")
    return W.sha(raw)


def _response_files(stream):
    stream.seek(0)
    result, end = {}, 0
    with tarfile.open(fileobj=stream, mode="r:") as archive:
        for member in archive:
            W.require(member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
                      and not member.pax_headers and not member.linkname
                      and member.offset == end and member.offset_data == member.offset + 512
                      and RESPONSE_NAME.fullmatch(member.name) is not None
                      and member.name not in result, "robotwin_review_response_member")
            raw = archive.extractfile(member).read()
            W.require(len(raw) == member.size, "robotwin_review_response_truncated")
            result[member.name] = raw
            end = member.offset_data + ((member.size + 511) // 512) * 512
    stream.seek(end)
    trailer = stream.read()
    W.require(len(trailer) >= 1024 and not any(trailer), "robotwin_review_response_trailer")
    return result


def _get_response(client, url, phase):
    with tempfile.TemporaryFile(dir=phase) as stream:
        with client.stream("GET", url) as response:
            W.require(response.status_code == 200, "robotwin_review_download_status")
            for data in response.iter_bytes():
                stream.write(data)
        return _response_files(stream)


def _review_pins(files, phase_name, request_sha):
    envelope = A.decode(files["response.json"])
    common = {"schema_version", "status", "phase"}
    W.require(envelope.get("schema_version") == RESPONSE_SCHEMA
              and envelope.get("phase") == phase_name, "robotwin_review_response_binding")
    if envelope.get("status") == "pending":
        A.fields(envelope, common, "robotwin_review_pending_fields")
        W.require(set(files) == {"response.json"}, "robotwin_review_pending_payload")
        return None
    A.fields(envelope, common | {"request_sha256", "manifest_sha256", "review_sha256"},
             "robotwin_review_response_fields")
    W.require(envelope["status"] == "reviewed" and envelope["request_sha256"] == request_sha,
              "robotwin_review_response_binding")
    for name in ("manifest", "review"):
        W.require(W.sha(files[name + ".json"]) == A.digest(envelope[name + "_sha256"]),
                  "robotwin_review_response_hash")
    return envelope


def _store_review(directory, files):
    for name, raw in files.items():
        path = directory / name
        if path.parent != directory and not path.parent.exists():
            path.parent.mkdir(mode=0o700)
        P.save_bytes(path.parent, path.name, raw)


def _review_paths(directory, files):
    available = {str(directory / name): name for name in files}
    consumed = {"response.json", "manifest.json", "review.json"}

    def referenced(binding, prefix):
        name = available.get(binding["path"])
        W.require(name is not None and name.startswith(prefix), "robotwin_review_proof_scope")
        W.require(W.sha(files[name]) == binding["sha256"], "robotwin_review_proof_hash")
        consumed.add(name)
        return files[name]

    manifest = A.decode(files["manifest.json"])
    for disposition in manifest["dispositions"]:
        proof = A.decode(referenced(disposition["proof"], "proofs/"))
        for role in ("provenance_evidence", "semantic_evidence"):
            for binding in proof[role]:
                referenced(binding, "evidence/")
    W.require(consumed == set(files), "robotwin_review_unreferenced_files")


def _adjudicate(phase, review_directory, pins):
    argv = ["--analysis-root", os.environ["ROBOTWIN_BYTE_GATE_ROOT"], "--trusted-root", str(ROOT)]
    paths = {"authorization": phase / "authorization/authorization.json",
             "report": phase / "scan/report.json", "records": phase / "scan/records.jsonl",
             "manifest": review_directory / "manifest.json", "review": review_directory / "review.json",
             "output-dir": phase / "independent-acceptance"}
    for key, path in paths.items():
        argv.extend(("--" + key, str(path)))
    for name in ("manifest", "review"):
        argv.extend(("--" + name + "-sha256", pins[name + "_sha256"]))
    W.require(A.main(argv) == 0, "robotwin_review_not_accepted")


def review_phase(phase, phase_name, request_url, response_url):
    """Request and verify an externally approved review without changing raw evidence.

    Args:
        phase: Original private scan directory, retained throughout review.
        phase_name: The exact pre-publication or post-pull phase.
        request_url: HTTPS PUT capability for one operator-owned request object.
        response_url: HTTPS GET capability for a separate reviewer-written object.
    Returns:
        None only after the existing adjudicator accepts every occurrence.
    Raises:
        W.ScanError, OSError, httpx.HTTPError: Any binding or transport failure.
    """
    W.require(phase_name in {"pre", "post"}
              and phase == Path(os.environ["ROBOTWIN_BYTE_GATE_ROOT"]).absolute() / phase_name,
              "robotwin_review_phase")
    request_url, response_url = _url(request_url), _url(response_url)
    W.require(_object_identity(request_url) != _object_identity(response_url),
              "robotwin_review_capabilities")
    fixed, authorization, context, snapshots = _scan_inputs(phase)
    review_directory, descriptor = W.create_output(phase / "review-input")
    os.close(descriptor)
    members = _request_members(fixed, authorization)
    request = {"schema_version": REQUEST_SCHEMA, "phase": phase_name, "context": context,
               "review_directory": str(review_directory),
               "members": {name: binding["sha256"] for name, (binding, _secret) in members.items()}}
    with httpx.Client(verify=True, follow_redirects=False, trust_env=False) as client:
        request_sha = _put_request(client, request_url, phase, request, members)
        while True:
            files = _get_response(client, response_url, phase)
            pins = _review_pins(files, phase_name, request_sha)
            if pins is not None:
                break
            W.recheck_snapshots(snapshots)
            time.sleep(5)
    W.recheck_snapshots(snapshots)
    _review_paths(review_directory, files)
    _store_review(review_directory, files)
    _adjudicate(phase, review_directory, pins)


def main():
    """Run one private review; never expose transport, policy or exception values.

    Args:
        None; phase arguments and the two environment capabilities are required.
    Returns:
        Zero only for a separately accepted exact review, otherwise one.
    Raises:
        None for ordinary input, evidence or transport failures.
    """
    os.umask(0o077)
    try:
        with W.authorized_roots(Path(os.environ["ROBOTWIN_BYTE_GATE_ROOT"]), ROOT):
            review_phase(Path(sys.argv[1]).absolute(), sys.argv[2],
                         os.environ[REQUEST_ENV], os.environ[RESPONSE_ENV])
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
