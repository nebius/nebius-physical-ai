"""Require externally signed, independently reviewed dispositions for exact scans."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_byte_scan import adjudicate as A, core as W, prepare as P
from image_byte_scan import review_signature_build as S

SCHEMA = "npa.image-byte-signed-review.v1"
ACCEPTANCE_SCHEMA = "npa.image-byte-signed-acceptance.v1"
DOMAIN = b"npa.image-byte-independent-review.Ed25519.v1\x00"
IDENTITY_KEYS = ("repository", "workflow_ref", "workflow_sha", "run_id", "run_attempt", "job")


def _read(path):
    binding = P.binding(path)
    with W.bound_open(binding) as (_, fd, before):
        W.require(before.st_nlink == 1, "private_review_input_hardlink")
        data = W.descriptor_bytes(fd)
        after = os.fstat(fd)
        W.require(after.st_nlink == 1 and W.stat_fingerprint(before) == W.stat_fingerprint(after)
                  and W.sha(data) == binding["sha256"], "private_review_input_changed")
    return A.decode(data), binding


def _single_file(binding):
    with W.bound_open(binding) as (_, fd, before):
        W.require(before.st_nlink == os.fstat(fd).st_nlink == 1, "private_review_input_hardlink")


def _identity(source_sha, phase, environment):
    W.require(re.fullmatch(r"[0-9a-f]{40}", source_sha) and phase in {"pre", "post"},
              "private_review_source_or_phase")
    values = {key: environment.get("GITHUB_" + key.upper()) for key in IDENTITY_KEYS}
    W.require(all(isinstance(value, str) and value for value in values.values()),
              "private_review_workflow_identity_missing")
    W.require(values["repository"] == "nebius/nebius-physical-ai"
              and values["workflow_ref"].startswith(values["repository"] +
                  "/.github/workflows/publish-public-images.yml@refs/heads/")
              and values["workflow_sha"] == source_sha
              and values["job"] == "build-development", "private_review_workflow_identity")
    W.require(all(re.fullmatch(r"[1-9][0-9]*", values[key])
                  for key in ("run_id", "run_attempt")), "private_review_run_identity")
    return {**values, "source_sha": source_sha, "tool": "curobo", "phase": phase}


def _context(phase_root):
    authorization, auth_binding = _read(phase_root / "authorization/authorization.json")
    raw, report_binding = _read(phase_root / "scan/report.json")
    records = P.binding(phase_root / "scan/records.jsonl")
    rows = [A.decode(line) for line in A.bound_bytes(records).splitlines()]
    A.population(raw, rows)
    verification = W.bound_json(authorization["verification_report"])
    image_source = A.image_revision(authorization, verification, raw)
    scanner_source = A.committed_sources(authorization)
    context = A.context(authorization, raw, report_binding["sha256"], records["sha256"],
                        auth_binding["sha256"], image_source, scanner_source)
    return context, raw


def _run_verifier(binary, packet):
    process = None
    try:
        with W.bound_open(binary) as (_, source_fd, _):
            with W.sealed_execution_input(source_fd, binary["sha256"], executable=True) as fd:
                W._SPAWNING = True
                try:
                    process = subprocess.Popen([f"/proc/self/fd/{fd}"], pass_fds=(fd,),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        env={"PATH": os.defpath}, start_new_session=True)
                finally:
                    W._SPAWNING = False
                W.require(not W._CANCEL_REQUESTED, "private_review_cancelled")
                output, error = process.communicate(packet)
        W.require(process.returncode == 0 and output == error == b"", "private_review_signature")
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.communicate()


def _signed_bundle(phase_root, key_pin, expected_identity, verifier_receipt):
    A.digest(key_pin)
    inbox = phase_root / "review-inbox"
    signed, binding = _read(inbox / "signed-envelope.json")
    A.fields(signed, {"envelope", "public_key_hex", "signature_hex"}, "private_review_bundle_schema")
    for name, length in (("public_key_hex", 64), ("signature_hex", 128)):
        W.require(isinstance(signed[name], str) and
                  re.fullmatch(r"[0-9a-f]{" + str(length) + "}", signed[name]),
                  "private_review_signature_encoding")
    key, signature = bytes.fromhex(signed["public_key_hex"]), bytes.fromhex(signed["signature_hex"])
    W.require(W.sha(key) == key_pin, "private_review_key_not_authorized")
    envelope = signed["envelope"]
    A.fields(envelope, {"schema_version", "algorithm", "identity", "context", "manifest_sha256",
                       "review_sha256", "public_key_sha256", "verifier_sha256", "scan_command_sha256"},
             "private_review_envelope_schema")
    W.require(envelope["schema_version"] == SCHEMA and envelope["algorithm"] == "Ed25519"
              and envelope["identity"] == expected_identity
              and envelope["public_key_sha256"] == key_pin, "private_review_envelope_identity")
    for name in ("manifest_sha256", "review_sha256", "verifier_sha256", "scan_command_sha256"):
        A.digest(envelope[name])
    binary = S.verified_binary(verifier_receipt)
    W.require(envelope["verifier_sha256"] == binary["sha256"], "private_review_verifier_changed")
    _run_verifier(binary, key + signature + DOMAIN + W.canonical(envelope))
    W.bound_file(binding)
    return envelope, binding


def _adjudicate(phase_root, envelope):
    args = SimpleNamespace(authorization=phase_root / "authorization/authorization.json",
        report=phase_root / "scan/report.json", records=phase_root / "scan/records.jsonl",
        manifest=phase_root / "review-inbox/manifest.json", review=phase_root / "review-inbox/review.json",
        manifest_sha256=envelope["manifest_sha256"], review_sha256=envelope["review_sha256"])
    _review_inputs(args.manifest, args.review)
    result = A.verify(args)
    W.require(result["context"] == envelope["context"], "private_review_adjudication_context")
    return result


def _review_inputs(manifest_path, review_path):
    manifest, _ = _read(manifest_path)
    _read(review_path)
    checked = set()
    for row in manifest["dispositions"]:
        proof, binding = _read(Path(row["proof"]["path"]))
        W.require(binding == row["proof"], "private_review_proof_changed")
        for name in ("provenance_evidence", "semantic_evidence"):
            for evidence in proof[name]:
                key = (evidence["path"], evidence["sha256"])
                if key in checked:
                    continue
                with W.bound_open(evidence) as (_, _, info):
                    W.require(info.st_nlink == 1, "private_review_evidence_hardlink")
                checked.add(key)


def _wait_for_bundle(phase_root):
    inbox, fd = W.create_output(phase_root / "review-inbox")
    try:
        while not (inbox / "signed-envelope.json").exists():
            W.output_identity(inbox, fd)
            W.require(not W._CANCEL_REQUESTED, "private_review_cancelled")
            time.sleep(1)
        W.output_identity(inbox, fd)
    finally:
        os.close(fd)


def _request(phase_root, identity, context, raw_exit, key_pin, verifier_receipt):
    binary = S.verified_binary(verifier_receipt)
    actual_exit, command_binding = _command_exit(phase_root)
    W.require(actual_exit == raw_exit, "private_review_command_exit")
    request = {"schema_version": "npa.image-byte-private-review-request.v1",
        "identity": identity, "context": context, "raw_scan_exit_code": raw_exit,
        "public_key_sha256": key_pin, "verifier_sha256": binary["sha256"],
        "scan_command_sha256": command_binding["sha256"]}
    W.write_private_json(phase_root, "private-review-request.json", request)


def _accept(phase_root, key_pin, identity, verifier_receipt, raw_exit):
    context, raw = _context(phase_root)
    W.require(context["image_source_sha"] == context["scanner_source_sha"] == identity["source_sha"],
              "private_review_source_binding")
    _request(phase_root, identity, context, raw_exit, key_pin, verifier_receipt)
    _wait_for_bundle(phase_root)
    envelope, signed_binding = _signed_bundle(phase_root, key_pin, identity, verifier_receipt)
    W.require(envelope["context"] == context, "private_review_scan_context")
    actual_exit, command_binding = _command_exit(phase_root)
    W.require(actual_exit == raw_exit and command_binding["sha256"] == envelope["scan_command_sha256"],
              "private_review_signed_command_exit")
    adjudication = _adjudicate(phase_root, envelope)
    W.bound_file(signed_binding)
    receipt = {"schema_version": ACCEPTANCE_SCHEMA, "accepted": True,
        "mode": "signed-private-occurrence-review", "identity": identity,
        "context": context, "raw_scan_exit_code": raw_exit,
        "raw_scan_valid": raw["valid"], "raw_findings": raw["findings"],
        "native_findings": raw["helper_summary"]["findings"],
        "accepted_occurrences": adjudication["accepted_occurrences"],
        "unresolved_occurrences": adjudication["unresolved_occurrences"],
        "public_key_sha256": key_pin, "signed_envelope_sha256": signed_binding["sha256"],
        "scan_command_sha256": envelope["scan_command_sha256"],
        "manifest_sha256": envelope["manifest_sha256"], "review_sha256": envelope["review_sha256"],
        "verifier_sha256": envelope["verifier_sha256"]}
    _write_acceptance(phase_root, adjudication, receipt, signed_binding)


def _write_acceptance(phase_root, adjudication, receipt, signed_binding):
    directory, fd = W.create_output(phase_root / "accepted-review")
    published = False
    try:
        A.write_result(directory, fd, adjudication)
        identity = W.write_private_json(directory, "signed-acceptance.json", receipt)
        published = True
        W.verify_private_json(directory, fd, "signed-acceptance.json", receipt, identity)
        W.bound_file(signed_binding)
    except BaseException:
        if published:
            os.unlink("signed-acceptance.json", dir_fd=fd)
            os.fsync(fd)
        raise
    finally:
        os.close(fd)


def verified_acceptance(phase_root, key_pin, identity, verifier_receipt):
    """Reverify signed prior acceptance without claiming to rerun its byte scan.

    Args:
        phase_root: Original private pre or post scan directory.
        key_pin: Public-key digest authorized by authenticated workflow dispatch.
        identity: Current trusted workflow, run, source and phase identity.
        verifier_receipt: Exact separate verifier build receipt.
    Returns:
        The bound acceptance receipt and its file digest.
    Raises:
        ScanError: Any signature, context, input or acceptance binding differs.
        OSError: Required private evidence cannot be read.
    """
    envelope, signed_binding = _signed_bundle(phase_root, key_pin, identity, verifier_receipt)
    receipt, binding = _read(phase_root / "accepted-review/signed-acceptance.json")
    result, _ = _read(phase_root / "accepted-review/adjudication.json")
    _check_acceptance(receipt, result, envelope, signed_binding, key_pin, identity)
    count, raw = _reviewed_population(phase_root, envelope)
    W.require(count == receipt["accepted_occurrences"], "private_review_derived_population")
    W.require(receipt["raw_scan_valid"] is raw["valid"]
              and type(receipt["native_findings"]) is int
              and receipt["native_findings"] == raw["helper_summary"]["findings"],
              "private_review_derived_counts")
    for name, relative in (("report_sha256", "scan/report.json"),
                           ("records_sha256", "scan/records.jsonl"),
                           ("authorization_file_sha256", "authorization/authorization.json")):
        _single_file({"path": str(phase_root / relative), "sha256": envelope["context"][name]})
    _single_file(binding)
    _final_inputs(phase_root, envelope, receipt, signed_binding)
    return receipt, binding["sha256"]


def _final_inputs(phase_root, envelope, receipt, signed_binding):
    W.bound_file(signed_binding)
    for name, relative in (("manifest_sha256", "review-inbox/manifest.json"),
                           ("review_sha256", "review-inbox/review.json")):
        _, current = _read(phase_root / relative)
        W.require(current["sha256"] == envelope[name], "private_review_signed_proof_changed")
    actual_exit, command_binding = _command_exit(phase_root)
    W.require(actual_exit == receipt["raw_scan_exit_code"]
              and command_binding["sha256"] == receipt["scan_command_sha256"] == envelope["scan_command_sha256"],
              "private_review_signed_command_exit")


def _command_exit(phase_root):
    command, binding = _read(phase_root / "raw-scan-exit.json")
    A.fields(command, {"exit_code", "completed", "interrupted"}, "private_review_command_schema")
    W.require(type(command["exit_code"]) is int and command["exit_code"] in {0, 1}
              and command["completed"] is True and command["interrupted"] is False,
              "private_review_command_not_completed")
    return command["exit_code"], binding


def _reviewed_population(phase_root, envelope):
    context = envelope["context"]
    raw = A.pinned_json(phase_root / "scan/report.json", context["report_sha256"])
    records = A.bound_bytes({"path": str(phase_root / "scan/records.jsonl"),
                            "sha256": context["records_sha256"]})
    population = A.population(raw, [A.decode(line) for line in records.splitlines()])
    manifest = A.pinned_json(phase_root / "review-inbox/manifest.json", envelope["manifest_sha256"])
    review = A.pinned_json(phase_root / "review-inbox/review.json", envelope["review_sha256"])
    consumed = {}

    def load(binding, json_body=True):
        key = (binding["path"], binding["sha256"], json_body)
        if key not in consumed:
            with W.bound_open(binding) as (_, _, info):
                W.require(info.st_nlink == 1 and info.st_size > 0, "private_review_evidence_type")
            consumed[key] = A.pinned_json(Path(binding["path"]), binding["sha256"]) if json_body else None
        return consumed[key]

    count = A.dispositions(manifest, review, population, context, envelope["manifest_sha256"], load)
    for path, digest, _ in consumed:
        with W.bound_open({"path": path, "sha256": digest}) as (_, _, info):
            W.require(info.st_nlink == 1, "private_review_evidence_hardlink")
    _prior_inputs(phase_root, context, raw)
    return count, raw


def _prior_inputs(phase_root, context, raw):
    authorization = A.pinned_json(phase_root / "authorization/authorization.json",
                                   context["authorization_file_sha256"])
    snapshots = W.input_snapshots(authorization)
    expected = [{"role": role, "sha256": spec["sha256"], "stat": list(before)}
                for role, spec, _secret, _path, before in snapshots]
    W.require(raw.get("input_snapshot_receipts") == expected, "private_review_prior_inputs_changed")
    W.require(A.committed_sources(authorization) == context["scanner_source_sha"]
              and W.sha(W.canonical(authorization["sources"])) == context["scanner_sources_sha256"]
              and A.policy_receipt(authorization) == raw["confidentiality_policy"]
              and W.sha(W.canonical(raw["confidentiality_policy"])) == context["confidentiality_policy_sha256"],
              "private_review_prior_source_or_policy")
    W.recheck_snapshots(snapshots)


def _check_acceptance(receipt, result, envelope, signed_binding, key_pin, identity):
    W.require(receipt.get("schema_version") == ACCEPTANCE_SCHEMA and receipt.get("accepted") is True
              and receipt.get("mode") == "signed-private-occurrence-review",
              "private_review_acceptance_schema")
    W.require(receipt.get("identity") == identity and receipt.get("context") == envelope["context"]
              and receipt.get("signed_envelope_sha256") == signed_binding["sha256"]
              and receipt.get("public_key_sha256") == key_pin, "private_review_acceptance_binding")
    W.require(result.get("schema_version") == A.SCHEMA and result.get("accepted") is True
              and result.get("context") == envelope["context"], "private_review_prior_adjudication")
    for name in ("manifest_sha256", "review_sha256"):
        W.require(receipt.get(name) == result.get(name) == envelope[name], "private_review_proof_binding")
    for name in ("verifier_sha256", "scan_command_sha256"):
        W.require(receipt.get(name) == envelope[name], "private_review_signature_binding")
    for name in ("accepted_occurrences", "unresolved_occurrences", "raw_scan_valid"):
        W.require(type(receipt.get(name)) is type(result.get(name))
                  and receipt.get(name) == result.get(name), "private_review_prior_counts")
    W.require(receipt.get("raw_findings") == result.get("raw_scan_findings")
              == receipt.get("accepted_occurrences") and receipt.get("unresolved_occurrences") == 0
              and type(receipt.get("raw_scan_exit_code")) is int
              and receipt["raw_scan_exit_code"] in {0, 1}, "private_review_prior_verdict")
    for name in ("raw_findings", "accepted_occurrences", "unresolved_occurrences"):
        A.integer(receipt[name])


def _arguments(argv):
    parser = W.SanitizedArgumentParser(description=__doc__)
    for name in ("analysis-root", "trusted-root", "authorization", "output-dir", "public-native-policy"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--public-native-policy-sha256", required=True)
    parser.add_argument("--review-public-key-sha256", default="")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    return parser.parse_args(argv)


def _run(args):
    scan_args = []
    for name in ("analysis_root", "trusted_root", "authorization", "output_dir",
                 "public_native_policy", "public_native_policy_sha256"):
        scan_args.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    if not args.review_public_key_sha256:
        return W.main(scan_args)
    A.digest(args.review_public_key_sha256)
    identity = _identity(args.source_sha, args.phase, os.environ)
    phase_root = args.analysis_root / args.phase
    W.require(args.authorization == phase_root / "authorization/authorization.json"
              and args.output_dir == phase_root / "scan", "private_review_phase_paths")
    raw_exit = _scan_command(args.trusted_root, phase_root, scan_args)
    with W.authorized_roots(args.analysis_root, args.trusted_root):
        _accept(phase_root, args.review_public_key_sha256, identity,
                args.analysis_root / "review-signature/dependency-receipt.json", raw_exit)
    return 0


def _scan_command(trusted_root, phase_root, scan_args):
    process = None
    try:
        W._SPAWNING = True
        try:
            process = subprocess.Popen([sys.executable, str(trusted_root / "npa/scripts/scan_image_bytes.py"),
                                        *scan_args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env={"PATH": os.defpath}, start_new_session=True)
        finally:
            W._SPAWNING = False
        W.require(not W._CANCEL_REQUESTED, "private_review_scan_cancelled")
        output, error = process.communicate()
        W.require(not W._CANCEL_REQUESTED, "private_review_scan_cancelled")
        with W.authorized_roots(phase_root.parent, trusted_root):
            P.save_bytes(phase_root, "raw-scan.stdout.log", output)
            P.save_bytes(phase_root, "raw-scan.stderr.log", error)
            W.write_private_json(phase_root, "raw-scan-exit.json", {"exit_code": process.returncode,
                                 "completed": True, "interrupted": False})
        W.require(process.returncode in {0, 1}, "private_review_scan_command_failed")
        return process.returncode
    finally:
        if process is not None and process.poll() is None:
            S.B._stop_owned(process)


def main(argv=None):
    """Run unchanged scanners, then require explicit signed private review if selected.

    Args:
        argv: Explicit arguments, or the process argument vector.
    Returns:
        Zero only for the selected complete and accepted gate.
    Raises:
        None.
    """
    os.umask(0o077)
    try:
        with W.cancellation_scope():
            code = _run(_arguments(argv))
        if code == 0:
            print("image byte publication gate passed")
        return code
    except (*W.INPUT_ERRORS, S.B.BuildError, subprocess.SubprocessError):
        print("image byte publication gate failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
