"""Private phase transport uses actual graphs and the existing strict adjudicator."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import httpx
import pytest
import yaml

from test_image_byte_adjudication import committed_source_oracle  # noqa: F401
from test_image_byte_scan import CHECKOUT, FakeDetector, file, js, tar_data, write
from test_image_byte_scan_oci import MARKER, authorize, oci_fixture

IMAGE = CHECKOUT / "npa/docker/workbench/robotwin"
SPEC = importlib.util.spec_from_file_location("robotwin_private_review", IMAGE / "private_review.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
A, W = R.A, R.W
REQUEST_URL = "https://example.invalid/request?signature=synthetic-request"
RESPONSE_URL = "https://example.invalid/response?signature=synthetic-response"


def phase_fixture(tmp_path):
    phase = tmp_path / "pre"
    phase.mkdir(mode=0o700)
    for name in ("graph", "authorization", "scan"):
        (phase / name).mkdir(mode=0o700)
    files, verification, _ = oci_fixture(marker="ancestor", nested=False, revision="a" * 40)
    runtime = json.loads(files["blobs/sha256/" + verification["image_manifest_digest"][7:]])
    files["manifest.json"] = js([{
        "Config": "blobs/sha256/" + verification["image_config_digest"][7:],
        "RepoTags": ["example.invalid/robotwin:synthetic"],
        "Layers": ["blobs/sha256/" + d["digest"][7:] for d in runtime["layers"]],
    }])
    verification["schema_version"] = "npa.robotwin.image-verification.v1"
    auth = authorize(phase, files, verification)
    auth.pop("literal_inventory")
    auth["confidentiality"] = write(phase / "authorization/confidentiality.json", js({
        "customer_pattern": MARKER, "infra_pattern": None,
    }))
    graph = W.bound_json(auth["verification_report"])
    auth["verification_report"] = write(phase / "graph/verification.json", js(graph))
    write(phase / "authorization/authorization.json", js(auth))
    report = W._scan(auth, phase / "scan", detector_type=FakeDetector)
    assert report["complete"] and not report["valid"]
    write(phase / "scan/report.json", js(report))
    return phase


def response_packet(request, rows, report, *, mutation="none"):
    """Independent synthetic review producer, separate from the transport consumer."""
    context, directory = request["context"], Path(request["review_directory"])
    evidence = b"synthetic independently reviewed immutable source and semantics"
    evidence_name = "evidence/" + W.sha(evidence) + ".txt"
    binding = {"path": str(directory / evidence_name), "sha256": W.sha(evidence)}
    files, dispositions, decisions = {evidence_name: evidence}, [], []
    for key, occurrence in A.population(report, rows).items():
        proof = {"schema_version": A.PROOF_SCHEMA, "context": context, "occurrence_id": key,
                 "record_sha256": occurrence["record_sha256"], "record_bytes": occurrence["record_bytes"],
                 "semantic_role": "non-operational-source-example", "operational_credential": False,
                 "provenance_evidence": [binding], "semantic_evidence": [binding]}
        if mutation == "active_credential":
            proof["operational_credential"] = True
        if mutation == "unsupported_role":
            proof["semantic_role"] = "blanket-public-file"
        raw, name = js(proof), "proofs/" + key + ".json"
        files[name] = raw
        dispositions.append({"occurrence_id": key, "proof": {"path": str(directory / name), "sha256": W.sha(raw)}})
        decisions.append({"occurrence_id": key, "proof_sha256": W.sha(raw), "decision": "accept"})
    if mutation == "unreviewed":
        decisions.pop()
    manifest = {"schema_version": A.MANIFEST_SCHEMA, "context": context, "dispositions": dispositions}
    files["manifest.json"] = js(manifest)
    review = {"schema_version": A.REVIEW_SCHEMA, "context": context, "decision": "accept",
              "manifest_sha256": W.sha(files["manifest.json"]), "reviewed_occurrences": decisions}
    files["review.json"] = js(review)
    response = {"schema_version": R.RESPONSE_SCHEMA, "status": "reviewed", "phase": request["phase"],
                "request_sha256": W.sha(W.canonical(request)), "manifest_sha256": W.sha(files["manifest.json"]),
                "review_sha256": W.sha(files["review.json"])}
    if mutation == "stale_request":
        response["request_sha256"] = "0" * 64
    if mutation == "wrong_phase":
        response["phase"] = "post"
    files["response.json"] = js(response)
    return tar_data([file(name, body) for name, body in files.items()])


@pytest.mark.usefixtures("committed_source_oracle")
@pytest.mark.parametrize("mutation", ["none", "unreviewed", "active_credential", "unsupported_role",
                                     "stale_request", "wrong_phase", "changed_original"])
def test_real_robotwin_graph_review_requires_external_complete_decision(tmp_path, monkeypatch, mutation):
    tmp_path.chmod(0o700)
    monkeypatch.setenv("ROBOTWIN_BYTE_GATE_ROOT", str(tmp_path))
    monkeypatch.setenv("SYNTHETIC_AMBIENT_CREDENTIAL", "never-export-this-credential")
    with W.authorized_roots(tmp_path, CHECKOUT):
        phase = phase_fixture(tmp_path)
        before = {p: (p.read_bytes(), W.stat_fingerprint(p.stat())) for p in (
            phase / "image.tar", phase / "scan/report.json", phase / "scan/records.jsonl")}
        request, calls = {}, []
        client_type = httpx.Client

        def receive(call):
            calls.append(call.method)
            assert "authorization" not in call.headers and "cookie" not in call.headers
            if call.method == "PUT":
                assert str(call.url) == REQUEST_URL
                body = call.read()
                assert b"never-export-this-credential" not in body
                assert REQUEST_URL.encode() not in body and RESPONSE_URL.encode() not in body
                with tarfile.open(fileobj=io.BytesIO(body)) as archive:
                    request.update(json.load(archive.extractfile("request.json")))
                    assert set(archive.getnames()) == {"request.json", *request["members"]}
                    for name, digest in request["members"].items():
                        assert W.sha(archive.extractfile(name).read()) == digest
                return httpx.Response(200)
            assert call.method == "GET" and str(call.url) == RESPONSE_URL
            if calls == ["PUT", "GET"]:
                return httpx.Response(200, content=tar_data([file("response.json", js({
                    "schema_version": R.RESPONSE_SCHEMA, "status": "pending", "phase": "pre"}))]))
            report = json.loads(before[phase / "scan/report.json"][0])
            rows = [json.loads(line) for line in before[phase / "scan/records.jsonl"][0].splitlines()]
            packet = response_packet(request, rows, report, mutation=mutation)
            if mutation == "changed_original":
                (phase / "image.tar").write_bytes(b"changed original artifact")
            return httpx.Response(200, content=packet)

        def client(**kwargs):
            assert kwargs == {"verify": True, "follow_redirects": False, "trust_env": False}
            return client_type(**kwargs, transport=httpx.MockTransport(receive))

        monkeypatch.setattr(R.httpx, "Client", client)
        monkeypatch.setattr(R.time, "sleep", lambda _seconds: None)
        if mutation == "none":
            R.review_phase(phase, "pre", REQUEST_URL, RESPONSE_URL)
            accepted = json.loads((phase / "independent-acceptance/adjudication.json").read_bytes())
            assert accepted["accepted"] and accepted["raw_scan_valid"] is False
            assert accepted["accepted_occurrences"] == json.loads(before[phase / "scan/report.json"][0])["findings"]
        else:
            with pytest.raises(W.ScanError):
                R.review_phase(phase, "pre", REQUEST_URL, RESPONSE_URL)
            assert not (phase / "independent-acceptance").exists()
        assert calls == ["PUT", "GET", "GET"]
        for path, (raw, stat) in before.items():
            if mutation != "changed_original" or path.name != "image.tar":
                assert path.read_bytes() == raw and W.stat_fingerprint(path.stat()) == stat


@pytest.mark.parametrize("name,kind", [("../outside", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                                      ("manifest.json", tarfile.SYMTYPE), ("review.json", tarfile.LNKTYPE),
                                      ("proofs/", tarfile.DIRTYPE), ("unknown.json", tarfile.REGTYPE)])
def test_response_members_refuse_unsafe_names_and_types(name, kind):
    with pytest.raises(W.ScanError):
        R._response_files(io.BytesIO(tar_data([file(name, b"private value", kind=kind)])))


@pytest.mark.parametrize("mutation", ["duplicate", "trailer", "truncated", "pax", "gnu_extension"])
def test_response_archive_refuses_ambiguous_or_incomplete_bytes(mutation):
    entries = [file("response.json", b"{}")] * (2 if mutation == "duplicate" else 1)
    if mutation == "pax":
        entries = [file("response.json", b"{}", pax={"comment": "private value"})]
    raw = tar_data(entries)
    if mutation == "gnu_extension":
        extension = tarfile.TarInfo("././@LongLink")
        extension.type, extension.size = tarfile.GNUTYPE_LONGNAME, len(b"response.json\0")
        raw = extension.tobuf(format=tarfile.GNU_FORMAT) + b"response.json\0".ljust(512, b"\0")
        raw += tar_data([file("placeholder", b"{}")])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            assert archive.getnames() == ["response.json"]
    if mutation == "trailer":
        raw += b"private trailing bytes"
    if mutation == "truncated":
        raw = raw[:513]
    with pytest.raises((W.ScanError, tarfile.TarError)):
        R._response_files(io.BytesIO(raw))


@pytest.mark.parametrize("status", [302, 403, 404, 500])
def test_response_http_errors_never_follow_or_retry(tmp_path, status):
    calls = []
    def receive(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": REQUEST_URL})
    with httpx.Client(transport=httpx.MockTransport(receive), follow_redirects=False) as client:
        with pytest.raises(W.ScanError):
            R._get_response(client, RESPONSE_URL, tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("url", ["http://example.invalid/x", "https://user:pass@example.invalid/x",
                                  "https://example.invalid/x#fragment", "invalid"])
def test_private_review_invalid_urls_are_refused(url):
    with pytest.raises(W.ScanError):
        R._url(url)


def test_request_and_response_capabilities_cannot_address_same_object(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTWIN_BYTE_GATE_ROOT", str(tmp_path))
    with pytest.raises(W.ScanError, match="robotwin_review_capabilities"):
        R.review_phase(tmp_path / "pre", "pre", REQUEST_URL,
                       "https://example.invalid:443/reque%73t?signature=different-get-capability")


@pytest.mark.parametrize("changed", [False, True])
def test_request_native_receipt_must_match_authorized_engine(tmp_path, monkeypatch, changed):
    tmp_path.chmod(0o700)
    monkeypatch.setenv("ROBOTWIN_BYTE_GATE_ROOT", str(tmp_path))
    with W.authorized_roots(tmp_path, CHECKOUT):
        phase = phase_fixture(tmp_path)
        authorization = W.bound_json(R.P.binding(phase / "authorization/authorization.json"))
        engine = {"kind": "aho-corasick-v1", **{name: {"sha256": digest, "path": "synthetic"}
                  for name, digest in W.AHO_PINS.items()}}
        authorization["literal_engine"] = engine
        (tmp_path / "native").mkdir(mode=0o700)
        receipt = {**engine, "schema_version": "synthetic-receipt"}
        if changed:
            receipt["kind"] = "changed-engine"
        write(tmp_path / "native/dependencies.json", js(receipt))
        if changed:
            with pytest.raises(W.ScanError, match="robotwin_review_native_receipt"):
                R._request_members({}, authorization)
        else:
            members = R._request_members({}, authorization)
            assert W.bound_json(members["native/dependencies.json"][0]) == receipt


@pytest.mark.parametrize("mutation", ["outside", "extra", "changed"])
def test_review_references_stay_in_fixed_directory_and_bind_all_files(tmp_path, mutation):
    raw = b"independently reviewed evidence"
    evidence_name = "evidence/" + W.sha(raw) + ".bin"
    binding = {"path": str(tmp_path / evidence_name), "sha256": W.sha(raw)}
    if mutation == "outside":
        binding["path"] = str(tmp_path.parent / evidence_name)
    proof = js({"provenance_evidence": [binding], "semantic_evidence": [binding]})
    proof_name = "proofs/" + W.sha(proof) + ".json"
    files = {"response.json": b"{}", "review.json": b"{}", evidence_name: raw, proof_name: proof,
             "manifest.json": js({"dispositions": [{"proof": {
                 "path": str(tmp_path / proof_name), "sha256": W.sha(proof)}}]})}
    if mutation == "extra":
        files["evidence/" + "a" * 64 + ".bin"] = b"unreviewed extra evidence"
    if mutation == "changed":
        files[evidence_name] += b" changed"
    with pytest.raises(W.ScanError):
        R._review_paths(tmp_path, files)


def test_private_review_main_does_not_disclose_hostile_exceptions(tmp_path, monkeypatch, capsys):
    tmp_path.chmod(0o700)
    monkeypatch.setenv("ROBOTWIN_BYTE_GATE_ROOT", str(tmp_path))
    monkeypatch.setenv(R.REQUEST_ENV, REQUEST_URL)
    monkeypatch.setenv(R.RESPONSE_ENV, RESPONSE_URL)
    monkeypatch.setattr(sys, "argv", ["private_review.py", str(tmp_path / "pre"), "pre"])
    def fail(*args):
        raise ValueError(REQUEST_URL + " synthetic private policy and response")
    monkeypatch.setattr(R, "review_phase", fail)
    assert R.main() == 1 and capsys.readouterr() == ("", "")


def test_workflow_private_review_capabilities_are_phase_and_tool_scoped():
    workflow = yaml.safe_load((CHECKOUT / ".github/workflows/publish-public-images.yml").read_text())
    selected = [s for s in workflow["jobs"]["build-development"]["steps"] if R.REQUEST_ENV in s.get("env", {})]
    assert len(selected) == 2
    for step, phase in zip(selected, ("PRE", "POST"), strict=True):
        for direction in ("REQUEST", "RESPONSE"):
            assert step["env"][f"ROBOTWIN_PRIVATE_REVIEW_{direction}_URL"] == (
                "${{ matrix.tool == 'robotwin' && secrets.ROBOTWIN_PRIVATE_REVIEW_"
                + phase + "_" + direction + "_URL || '' }}")


@pytest.mark.parametrize("scan_status,configured,review_status", [(0, True, 0), (1, False, 0),
                                                                (1, True, 0), (1, True, 71), (23, True, 0)])
def test_shell_accepts_only_separate_review_and_keeps_original_failure(tmp_path, scan_status, configured, review_status):
    interpreter = tmp_path / "npa/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f"#!{sys.executable}\n" + '''
import os, pathlib, sys
script = pathlib.Path(sys.argv[1]).name
if script == "scan_image_bytes.py":
    raise SystemExit(int(os.environ["SCAN_STATUS"]))
if script == "private_review.py":
    pathlib.Path(os.environ["REVIEW_CALLED"]).touch()
    print(os.environ["ROBOTWIN_PRIVATE_REVIEW_REQUEST_URL"])
    print("synthetic-private-policy", file=sys.stderr)
    raise SystemExit(int(os.environ["REVIEW_STATUS"]))
''')
    interpreter.chmod(0o755)
    docker = interpreter.parent / "docker"
    docker.write_text("#!/bin/sh\nprintf '%s\\n' sha256:synthetic\n")
    docker.chmod(0o755)
    analysis, saved, called = tmp_path / "analysis", tmp_path / "saved.tar", tmp_path / "called"
    analysis.mkdir(mode=0o700)
    saved.write_bytes(b"synthetic saved image")
    env = {**os.environ, "PATH": str(interpreter.parent) + os.pathsep + os.defpath,
           "ROBOTWIN_BYTE_GATE_ROOT": str(analysis), "ROBOTWIN_PUBLIC_NATIVE_POLICY_SHA256": "a" * 64,
           "SCAN_STATUS": str(scan_status), "REVIEW_STATUS": str(review_status), "REVIEW_CALLED": str(called)}
    for name in (R.REQUEST_ENV, R.RESPONSE_ENV, "ROBOTWIN_PRIVATE_FAILURE_UPLOAD_URL"):
        env.pop(name, None)
    if configured:
        env.update({R.REQUEST_ENV: REQUEST_URL, R.RESPONSE_ENV: RESPONSE_URL})
    result = subprocess.run(["bash", str(IMAGE / "byte_gate.sh"), str(saved), "synthetic-image", "pre"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == (0 if scan_status == 1 and configured and review_status == 0 else scan_status)
    assert called.exists() is (scan_status == 1 and configured)
    assert REQUEST_URL not in result.stdout + result.stderr and "synthetic-private-policy" not in result.stdout + result.stderr
    assert result.stderr == ""
