"""Conserve original scan authorization while authenticating transported inputs.

The retention digest must come from the completed trusted hosted scan receipt.
It is an authorization input, never a digest inferred from untrusted uploads.
Original authorization bytes and inode receipts remain unchanged. New locations
have independent live snapshots; they never masquerade as the original files.
Policy values are deliberately excluded and must be supplied anew by CI secrets.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import adjudicate as A, core as W, prepare as P
else:
    from . import adjudicate as A, core as W, prepare as P

SCHEMA = "npa.image-byte-retained-inputs.v1"
_CURRENT = W.ContextVar("image_byte_retention_receipt", default=None)


def _snapshot_rows(snapshots):
    return [
        {"role": role, "sha256": spec["sha256"], "stat": list(before)}
        for role, spec, _secret, _path, before in snapshots
    ]


def _policy_mode(authorization):
    W.require(
        authorization.get("confidentiality") is not None
        and "literal_inventory" not in authorization,
        "retention_requires_ci_regex_policy",
    )


def _copy_input(directory, spec, *, secret=True):
    name = spec["sha256"]
    path = directory / name
    if path.exists():
        W.bound_file({"path": str(path), "sha256": name})
        return name
    with W.bound_open(spec, secret=secret) as (_path, descriptor, _info):
        parent = W.directory_fd(directory)
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            with os.fdopen(os.open(name, flags, 0o600, dir_fd=parent), "wb") as output:
                offset = 0
                while data := os.pread(descriptor, W.CHUNK, offset):
                    offset += len(data)
                    output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.fsync(parent)
        finally:
            os.close(parent)
    W.bound_file({"path": str(path), "sha256": name})
    return name


def _retained_roles(directory, snapshots, sources):
    result = []
    for role, spec, secret, _path, before in snapshots:
        delivery = "secret" if role == "confidentiality" else "retained"
        if any(spec == source for source in sources.values()):
            delivery = "checkout"
        row = {
            "role": role,
            "binding": spec,
            "stat": list(before),
            "delivery": delivery,
        }
        if delivery == "retained":
            row["file"] = _copy_input(directory, spec, secret=secret)
        result.append(row)
    return result


def _complete_original_report(authorization, snapshots, report_binding, ledger_binding):
    report = W.bound_json(report_binding)
    rows = [
        A.decode(line)
        for line in W.bound_bytes(ledger_binding, secret=True).splitlines()
    ]
    A.population(report, rows)
    W.require(
        report["input_snapshot_receipts"] == _snapshot_rows(snapshots),
        "retention_stale_snapshot",
    )
    W.require(
        report["authorization_sha256"] == W.sha(W.canonical(authorization)),
        "retention_authorization_changed",
    )
    W.require(
        A.policy_receipt(authorization) == report["confidentiality_policy"],
        "retention_policy_changed",
    )
    return report


def _native_identity(binding, authorization):
    native = W.bound_json(binding)
    W.require(
        native.get("schema_version") == "npa.image-byte-native-checks.v1"
        and native.get("passed") is True
        and native.get("synthetic_only") is True,
        "retention_native_checks",
    )
    W.require(
        native.get("source_bindings") == authorization["sources"]
        and native.get("helper_sha256") == authorization["helper"]["sha256"],
        "retention_native_identity",
    )


def _capture_inputs(args, directory):
    auth_binding = P.binding(args.authorization)
    authorization = W.bound_json(auth_binding)
    _policy_mode(authorization)
    snapshots = W.input_snapshots(authorization)
    report_binding, ledger_binding = P.binding(args.report), P.binding(args.records)
    report = _complete_original_report(
        authorization, snapshots, report_binding, ledger_binding
    )
    sources = A.committed_sources(authorization)
    native_binding = P.binding(args.native_checks)
    _native_identity(native_binding, authorization)
    inputs = _retained_roles(directory, snapshots, authorization["sources"])
    with W.bound_open(authorization["archive"]) as (_path, _fd, info):
        archive_stat = list(W.stat_fingerprint(info))
    artifacts = {
        "authorization": auth_binding,
        "report": report_binding,
        "records": ledger_binding,
        "native_checks": native_binding,
    }
    files = {role: _copy_input(directory, spec) for role, spec in artifacts.items()}
    W.recheck_snapshots(snapshots)
    return authorization, report, sources, inputs, artifacts, files, archive_stat


def capture(args):
    """Retain exact non-secret inputs before the original hosted workspace closes.

    Args:
        args: Original authorization/report/ledger and trusted run identities.
    Returns:
        The private retention receipt, whose digest must be retained externally.
    Raises:
        ScanError: Original inputs, policy, source or completed population differ.
    """
    W.require(re.fullmatch(r"[0-9]+-[0-9]+", args.run_id), "retention_run_identity")
    W.require(
        re.fullmatch(r"[0-9a-f]{40}", args.interface_revision),
        "retention_interface_revision",
    )
    directory, descriptor = W.create_output(args.output_dir)
    try:
        auth, report, source, inputs, artifacts, files, archive_stat = _capture_inputs(
            args, directory
        )
        receipt = {
            "schema_version": SCHEMA,
            "run_id": args.run_id,
            "interface_revision": args.interface_revision,
            "scanner_revision": source,
            "archive": auth["archive"],
            "archive_stat": archive_stat,
            "artifacts": artifacts,
            "files": files,
            "inputs": inputs,
            "policy_receipt": report["confidentiality_policy"],
            "original_snapshot_receipts": report["input_snapshot_receipts"],
            "policy_values_retained": False,
        }
        W.write_private_json(directory, "retention.json", receipt)
        os.fsync(descriptor)
        return receipt
    finally:
        os.close(descriptor)


def _retained_file(directory, name, digest):
    A.digest(name)
    W.require(name == digest, "transport_retained_filename")
    result = {"path": str(directory / name), "sha256": digest}
    W.bound_file(result)
    return result


def _source_target(spec, sources):
    matches = [name for name, value in sources.items() if value == spec]
    W.require(len(matches) == 1, "transport_source_identity")
    name = matches[0]
    W.require(name in W.source_bindings(), "transport_source_population")
    actual = W.source_bindings()[name]
    W.require(actual["sha256"] == spec["sha256"], "transport_source_digest")
    return actual


def _input_locations(receipt, authorization, directory, archive, policy):
    locations = {}
    expected = receipt["original_snapshot_receipts"]
    actual = [
        {"role": row["role"], "sha256": row["binding"]["sha256"], "stat": row["stat"]}
        for row in receipt["inputs"]
    ]
    W.require(actual == expected, "transport_original_snapshots")
    for row in receipt["inputs"]:
        spec = row["binding"]
        delivery = row["delivery"]
        if delivery == "checkout":
            target = _source_target(spec, authorization["sources"])
        elif delivery == "secret":
            W.require(row["role"] == "confidentiality", "transport_secret_role")
            target = {"path": str(policy), "sha256": spec["sha256"]}
        else:
            W.require(delivery == "retained", "transport_delivery")
            target = _retained_file(directory, row["file"], spec["sha256"])
        _add_location(locations, spec, target)
    _add_location(
        locations,
        authorization["archive"],
        {"path": str(archive), "sha256": authorization["archive"]["sha256"]},
    )
    return locations


def _add_location(locations, original, target):
    W.require(original["sha256"] == target["sha256"], "transport_digest")
    key = original["path"]
    W.require(
        key not in locations or locations[key] == target,
        "transport_conflicting_location",
    )
    locations[key] = target


def _evidence_locations(locations, directory):
    if directory is None:
        return
    held = W.directory_fd(directory)
    try:
        for name in os.listdir(held):
            W.require(
                name in {"manifest.json", "review.json"} or W.SHA.fullmatch(name),
                "transport_evidence_name",
            )
            binding = P.binding(directory / name)
            _add_location(
                locations,
                {"path": "review/" + name, "sha256": binding["sha256"]},
                binding,
            )
        W.output_identity(directory, held)
    finally:
        os.close(held)


def _load_receipt(path, expected):
    receipt = A.pinned_json(path, expected)
    A.fields(
        receipt,
        {
            "schema_version",
            "run_id",
            "interface_revision",
            "scanner_revision",
            "archive",
            "archive_stat",
            "artifacts",
            "files",
            "inputs",
            "policy_receipt",
            "original_snapshot_receipts",
            "policy_values_retained",
        },
        "transport_receipt_fields",
    )
    W.require(
        receipt["schema_version"] == SCHEMA
        and receipt["policy_values_retained"] is False,
        "transport_receipt_schema",
    )
    directory = Path(path).parent
    bindings = {}
    for role in ("authorization", "report", "records", "native_checks"):
        bindings[role] = _retained_file(
            directory, receipt["files"][role], receipt["artifacts"][role]["sha256"]
        )
    return receipt, bindings


def _relocation_plan(args):
    receipt, bindings = _load_receipt(args.retention, args.retention_sha256)
    auth = W.bound_json(bindings["authorization"])
    _policy_mode(auth)
    W.require(receipt["archive"] == auth["archive"], "transport_archive_identity")
    for role in ("authorization", "report", "records"):
        binding = bindings[role]
        W.require(
            P.binding(getattr(args, role)) == binding, "transport_artifact_location"
        )
    locations = _input_locations(
        receipt, auth, args.retention.parent, args.archive, args.policy
    )
    _evidence_locations(locations, getattr(args, "evidence_root", None))
    _native_identity(bindings["native_checks"], auth)
    return receipt, bindings, auth, locations


@contextmanager
def relocated(args):
    """Hold independently authenticated new input snapshots during adjudication.

    Args:
        args: Explicit retention digest, new archive/policy paths and evidence args.
    Yields:
        The authenticated retention seal; original authorization stays unchanged.
    Raises:
        ScanError: Transport, source, policy, snapshot or artifact bindings differ.
    """
    receipt, bindings, auth, locations = _relocation_plan(args)
    token = W._RETAINED_INPUTS.set(locations)
    state_token = _CURRENT.set({"receipt": receipt, "sha256": args.retention_sha256})
    try:
        W.bound_file(auth["archive"])
        snapshots = W.input_snapshots(auth)
        W.require(
            A.committed_sources(auth) == receipt["scanner_revision"],
            "transport_scanner_revision",
        )
        W.require(
            A.policy_receipt(auth) == receipt["policy_receipt"],
            "transport_policy_changed",
        )
        yield receipt
        W.recheck_snapshots(snapshots)
        A.pinned_json(args.retention, args.retention_sha256)
        for binding in bindings.values():
            W.bound_file(binding)
    finally:
        _CURRENT.reset(state_token)
        W._RETAINED_INPUTS.reset(token)


def snapshot_receipts(snapshots):
    """Return authenticated original snapshots without changing current identities.

    Args:
        snapshots: Independently verified live input identities.
    Returns:
        Original retained snapshots, or current snapshots for a local adjudication.
    Raises:
        ScanError: Transport changed the complete role and hash population.
    """
    state = _CURRENT.get()
    current = _snapshot_rows(snapshots)
    if state is None:
        return current
    original = state["receipt"]["original_snapshot_receipts"]
    W.require(
        [(row["role"], row["sha256"]) for row in original]
        == [(row["role"], row["sha256"]) for row in current],
        "transport_snapshot_population",
    )
    return original


def transport_context():
    """Return the separately authorized retention digest for review binding.

    Args:
        None.
    Returns:
        Empty local context, or the authenticated retained-input digest.
    Raises:
        None.
    """
    state = _CURRENT.get()
    return {} if state is None else {"retention_sha256": state["sha256"]}


def main(argv=None):
    """Capture private scan inputs through the trusted hosted caller.

    Args:
        argv: Explicit command arguments, or process arguments.
    Returns:
        Zero on retained evidence, one on a sanitized failure.
    Raises:
        None.
    """
    parser = W.SanitizedArgumentParser(description=__doc__)
    for name in (
        "analysis-root",
        "trusted-root",
        "authorization",
        "report",
        "records",
        "output-dir",
        "native-checks",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "interface-revision"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        with W.authorized_roots(args.analysis_root, args.trusted_root):
            capture(args)
        print("private scan inputs retained")
        return 0
    except W.INPUT_ERRORS:
        print("private scan input retention failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
