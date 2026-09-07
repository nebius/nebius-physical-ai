#!/usr/bin/env python3
"""Prepare pinned scanning dependencies or owner-only image authorization."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import core as W
else:
    from . import core as W

WHEEL_URL = "https://files.pythonhosted.org/packages/a1/f2/d13807476195e4ec5999a78f22db592a64da54229c9183438f3165105779/pyahocorasick-2.3.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"


def binding(path, *, secret=True):
    path, fd, info = W.open_private_fd(path, secret=secret)
    try:
        result = {"path": str(path), "sha256": W.descriptor_digest(fd)}
        W.require(W.stat_fingerprint(info) == W.stat_fingerprint(os.fstat(fd)), "preparation_input_changed")
        return result
    finally:
        os.close(fd)


def save_bytes(directory, name, payload):
    parent = W.directory_fd(directory)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(parent)
    return binding(Path(directory) / name)


def dependencies(directory):
    W.require(sys.version_info[:2] == (3, 12) and sys.platform == "linux" and os.uname().machine == "x86_64", "native_platform_requires_cpython312_linux_x86_64")
    with urllib.request.urlopen(WHEEL_URL) as response:  # noqa: S310 - one pinned official HTTPS artifact.
        wheel = response.read()
    W.require(W.sha(wheel) == W.AHO_PINS["wheel"], "native_wheel_download_digest")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        native = archive.read(W.AHO_MEMBER)
    W.require(W.sha(native) == W.AHO_PINS["extension"], "native_extension_digest")
    roots = W._ROOTS.get()
    receipt = {"schema_version": "npa.image-byte-scan-dependencies.v1", "kind": "aho-corasick-v1",
               "source": binding(roots[1] / "npa/scripts/image_byte_scan/aho_matcher.py", secret=False),
               "wheel": save_bytes(directory, "pyahocorasick.whl", wheel),
               "extension": save_bytes(directory, W.AHO_MEMBER, native)}
    engine = W.AuthorizedAho({key: value for key, value in receipt.items() if key != "schema_version"})
    try:
        receipt["verification"] = engine.receipt()
    finally:
        engine.close()
    W.write_private_json(directory, "dependencies.json", receipt)
    return receipt


def native_engine(path):
    receipt = W.bound_json(binding(path))
    W.require(receipt.get("schema_version") == "npa.image-byte-scan-dependencies.v1", "native_receipt_schema")
    engine_binding = {key: receipt[key] for key in ("kind", *W.AHO_PINS)}
    engine = W.AuthorizedAho(engine_binding)
    try:
        W.require(receipt.get("verification") == engine.receipt(), "native_receipt_verification")
    finally:
        engine.close()
    return engine_binding


def tools_bindings(path):
    return W.verified_tools(binding(path))


def authorize(args, directory):
    helper, config = tools_bindings(args.tools_receipt)
    engine = native_engine(args.native_receipt)
    archive, verification = binding(args.archive), binding(args.verification_report)
    report = W.bound_json(verification)
    W.require(report.get("valid") is True and report.get("schema_version") == "npa.curobo.image-verification.v1", "accepted_graph_report_required")
    W.require(report.get("docker_save_sha256") == archive["sha256"] and report.get("expected_image_id") == args.expected_image_id, "prepared_image_binding")
    W.require(W.DIGEST.fullmatch(args.expected_image_id) is not None, "expected_image_digest")
    authorization = {"schema_version": "npa.image-byte-scan-authorization.v1", "accepted_verification": True,
                     "archive": archive, "verification_report": verification, "expected_image_id": args.expected_image_id,
                     "helper": helper, "config": config, "sources": W.source_bindings(), "literal_engine": engine,
                     "tools_receipt": binding(args.tools_receipt)}
    if args.policy_mode == "ci-regex":
        policy = {"customer_pattern": os.environ.get("CUSTOMER_DENYLIST"), "infra_pattern": os.environ.get("INFRA_DENYLIST")}
        W.C.compile_policy(policy["customer_pattern"], policy["infra_pattern"])
        authorization["confidentiality"] = save_bytes(directory, "confidentiality.json", W.canonical(policy))
    if args.literal_inventory is not None:
        literal_binding = binding(args.literal_inventory)
        literal_binding["matching_policy"] = args.literal_matching_policy
        inventory = W.bound_json(literal_binding)
        values = inventory.get("literals")
        W.require(isinstance(values, list) and all(type(value) is str and value for value in values), "literal_inventory_schema")
        W.require(args.policy_mode != "exact-literals" or bool(values), "nonempty_confidentiality_policy_required")
        authorization["literal_inventory"] = literal_binding
    W.require(args.policy_mode != "exact-literals" or "literal_inventory" in authorization, "literal_inventory_required")
    # Even an empty real handshake proves the configured binary and policy before
    # authorizing image analysis. Synthetic test code substitutes only this seam.
    detector = W.Detector(authorization, directory / "helper-preparation-stderr.jsonl")
    try:
        detector.finish()
    finally:
        if not detector.joined:
            detector.abort()
    W.input_snapshots(authorization)
    W.write_private_json(directory, "authorization.json", authorization)
    return authorization


def _main(argv=None):
    os.umask(0o077)
    directory = fd = None
    try:
        parser = W.SanitizedArgumentParser(description=__doc__)
        parser.add_argument("action", choices=("dependencies", "authorize", "check-policy"))
        parser.add_argument("--analysis-root", type=Path)
        parser.add_argument("--trusted-root", type=Path)
        parser.add_argument("--output-dir", type=Path)
        parser.add_argument("--tools-receipt", type=Path)
        parser.add_argument("--native-receipt", type=Path)
        parser.add_argument("--archive", type=Path)
        parser.add_argument("--verification-report", type=Path)
        parser.add_argument("--expected-image-id")
        parser.add_argument("--policy-mode", choices=("ci-regex", "exact-literals"), default="ci-regex")
        parser.add_argument("--literal-inventory", type=Path)
        parser.add_argument("--literal-matching-policy", choices=("exact-substring-v1", W.POLICY), default="exact-substring-v1")
        args = parser.parse_args(argv)
        if args.action == "check-policy":
            W.require(args.policy_mode == "ci-regex", "policy_preflight_mode")
            policy = W.C.compile_policy(os.environ.get("CUSTOMER_DENYLIST"), os.environ.get("INFRA_DENYLIST"))
            print(json.dumps(policy.receipt(), sort_keys=True))
            return 0
        W.require(all(value is not None for value in (args.analysis_root, args.trusted_root, args.output_dir)), "preparation_roots_required")
        with W.authorized_roots(args.analysis_root, args.trusted_root):
            if args.action == "authorize":
                W.require(all(getattr(args, name) is not None for name in ("tools_receipt", "native_receipt", "archive", "verification_report", "expected_image_id")), "authorization_arguments_required")
                # Required secret configuration is rejected before image hashing,
                # native execution, output creation or detector launch.
                if args.policy_mode == "ci-regex":
                    W.C.compile_policy(os.environ.get("CUSTOMER_DENYLIST"), os.environ.get("INFRA_DENYLIST"))
            directory, fd = W.create_output(args.output_dir)
            if args.action == "dependencies":
                dependencies(directory)
            else:
                authorize(args, directory)
        print("image byte preparation completed")
        return 0
    except W.INPUT_ERRORS:
        print("image byte preparation failed")
        return 1
    finally:
        if fd is not None:
            os.close(fd)


def main(argv=None):
    try:
        with W.cancellation_scope():
            return _main(argv)
    except W.INPUT_ERRORS:
        print("image byte preparation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
