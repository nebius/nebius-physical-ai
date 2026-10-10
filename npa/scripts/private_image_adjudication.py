#!/usr/bin/env python3
"""Adjudicate a reviewed retained scan on hosted CI without exporting policy values."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import private_image_qualification as Q


def _fetch(ssh, selector, role, target, *, size=None, digest=None):
    command = Q._remote_command(ssh, "review", selector, role)
    with Q._private_output(target) as output, tempfile.TemporaryFile() as errors:
        with Q._child(
            command, stdout=subprocess.PIPE, stderr=errors, env=Q._clean_environment()
        ) as process:
            try:
                if role == "request":
                    data = process.stdout.read(Q.MANIFEST_BYTES + 1)
                    Q._require(process.wait() == 0, "ssh_transfer_failed")
                    Q._adjudication_request(data, selector)
                    output.write(data)
                else:
                    Q._copy_exact(process.stdout, output, size, digest)
                Q._require(process.wait() == 0, "ssh_transfer_failed")
            except BaseException:
                process.terminate()
                raise


def _member_name(name, review):
    if review:
        return name in {"manifest.json", "review.json"} or Q.HEX.fullmatch(name)
    return name in Q.RECEIPT_FILES or re.fullmatch(r"retained/[0-9a-f]{64}", name)


def _unpack(source, destination, *, review=False):
    descriptor = Q._directory(destination, create=True)
    os.close(descriptor)
    seen = set()
    with (
        Q._private_input(source) as (stream, _info),
        tarfile.open(fileobj=stream, mode="r:") as archive,
    ):
        for member in archive:
            Q._require(
                member.name not in seen and _member_name(member.name, review),
                "receipt_identity",
            )
            Q._require(
                member.type == tarfile.REGTYPE and not member.pax_headers,
                "receipt_identity",
            )
            seen.add(member.name)
            target = destination / member.name
            parent = Q._directory(target.parent, create=True)
            os.close(parent)
            with Q._private_output(target) as output:
                content = archive.extractfile(member)
                shutil.copyfileobj(content, output, Q.CHUNK)
                Q._require(output.tell() == member.size, "receipt_identity")
    return seen


def _read(path):
    with Q._private_input(path) as (stream, _info):
        return json.load(stream, object_pairs_hook=Q._unique_object)


def _request_inputs(ssh, root, selector):
    _fetch(ssh, selector, "request", root / "request.json")
    request = _read(root / "request.json")
    Q._fetch(ssh, request["image_manifest_sha256"], "manifest", root / "manifest.json")
    manifest = _read(root / "manifest.json")
    required = (
        manifest["archive_bytes"]
        + request["workspace_bytes"]
        + 2 * (request["scan_receipt_bytes"] + request["review_tar_bytes"])
    )
    Q._require(shutil.disk_usage(root).free >= required, "insufficient_capacity")
    for role, prefix in (("receipt", "scan_receipt"), ("review", "review_tar")):
        _fetch(
            ssh,
            selector,
            role,
            root / (role + ".tar"),
            size=request[prefix + "_bytes"],
            digest=request[prefix + "_sha256"],
        )
    _unpack(root / "receipt.tar", root / "original")
    _unpack(root / "review.tar", root / "review", review=True)
    return request, manifest


def _retention_binding(root, request, manifest):
    path = root / "original/retained/retention.json"
    with Q._private_input(path) as (stream, info):
        Q._require(
            Q._descriptor_digest(stream.fileno(), info.st_size)
            == request["retention_sha256"],
            "receipt_identity",
        )
    receipt = _read(path)
    Q._require(receipt["run_id"] == request["scan_run_id"], "run_identity")
    Q._require(
        receipt["scanner_revision"] == Q.SCANNER_REVISION, "scanner_source_revision"
    )
    Q._require(
        receipt["archive"]["sha256"] == manifest["archive_sha256"], "archive_digest"
    )
    original_manifest = (root / "original/manifest.json").read_bytes()
    Q._manifest(original_manifest, request["image_manifest_sha256"])
    return receipt


def _adjudicate(scanner, root, request, receipt):
    policy = {
        "customer_pattern": os.environ.get("CUSTOMER_DENYLIST"),
        "infra_pattern": os.environ.get("INFRA_DENYLIST"),
    }
    Q._write(
        root / "current-policy.json",
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode(),
    )
    retained = root / "original/retained"
    command = Q._scanner_command(
        scanner,
        "image_byte_scan/adjudicate.py",
        root,
        manifest=root / "review/manifest.json",
        review=root / "review/review.json",
        manifest_sha256=request["disposition_manifest_sha256"],
        review_sha256=request["independent_review_sha256"],
        authorization=retained / receipt["files"]["authorization"],
        report=retained / receipt["files"]["report"],
        records=retained / receipt["files"]["records"],
        retention=retained / "retention.json",
        retention_sha256=request["retention_sha256"],
        archive=root / "image.tar",
        policy=root / "current-policy.json",
        evidence_root=root / "review",
        output_dir=root / "adjudication",
    )
    return Q._execute(command)


def _steps(scanner, root, ssh, selector):
    request, manifest = _request_inputs(ssh, root, selector)
    receipt = _retention_binding(root, request, manifest)
    Q._check_policy(scanner)
    Q._require(
        shutil.disk_usage(root).free
        >= manifest["archive_bytes"] + request["workspace_bytes"],
        "insufficient_capacity_after_preparation",
    )
    Q._fetch(
        ssh,
        request["image_manifest_sha256"],
        "archive",
        root / "image.tar",
        size=manifest["archive_bytes"],
        digest=manifest["archive_sha256"],
    )
    result = _adjudicate(scanner, root, request, receipt)
    Q._require(result == 0, "adjudication_failed")
    accepted = _read(root / "adjudication/adjudication.json")
    Q._require(
        accepted["accepted"] is True and accepted["unresolved_occurrences"] == 0,
        "adjudication_failed",
    )
    return {
        "status": "passed",
        "accepted": True,
        "request_sha256": selector,
        "raw_scan_valid": accepted["raw_scan_valid"],
        "raw_scan_findings": accepted["raw_scan_findings"],
        "accepted_occurrences": accepted["accepted_occurrences"],
        "native_qualified": False,
    }


def _receipt_bundle(root):
    target = root / "adjudication-result.tar"
    names = (
        "request.json",
        "summary.json",
        "adjudication/adjudication.json",
        "phases.jsonl",
    )
    with (
        Q._private_output(target) as output,
        tarfile.open(fileobj=output, mode="w|") as archive,
    ):
        for name in names:
            if not (root / name).exists():
                continue
            with Q._private_input(root / name) as (stream, info):
                member = tarfile.TarInfo(name)
                member.size, member.mode = info.st_size, 0o600
                archive.addfile(member, stream)
    return target


def _retain(root, ssh, request, run):
    target = _receipt_bundle(root)
    with Q._private_input(target) as (stream, info):
        digest = Q._descriptor_digest(stream.fileno(), info.st_size)
        command = Q._remote_command(
            ssh,
            "store",
            request["image_manifest_sha256"],
            run,
            str(info.st_size),
            digest,
        )
        with (
            tempfile.TemporaryFile() as errors,
            Q._child(
                command,
                stdin=stream,
                stdout=subprocess.PIPE,
                stderr=errors,
                env=Q._clean_environment(),
            ) as process,
        ):
            stdout, _ = process.communicate()
            Q._require(
                process.returncode == 0
                and json.loads(stdout) == {"sha256": digest, "bytes": info.st_size},
                "private_receipt_not_retained",
            )
    return {"receipt_sha256": digest, "receipt_bytes": info.st_size}


def _execute_adjudication(args, root, ssh):
    summary = {
        "status": "failed",
        "accepted": False,
        "request_sha256": args.request_sha256,
    }
    with Q._observe_run(root) as observer:
        try:
            summary = _steps(args.scanner_root, root, ssh, args.request_sha256)
            Q._check_cancelled()
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            Q._QualificationError,
        ) as error:
            summary.update(Q._failure(error, "adjudication"), accepted=False)
        Q._cancelled_summary(summary, observer)
        observer.retaining = True
        Q._write(root / "summary.json", Q._json_bytes(summary))
        if (root / "request.json").exists():
            try:
                summary.update(
                    _retain(root, ssh, _read(root / "request.json"), args.run_id)
                )
            except (OSError, ValueError, Q._QualificationError) as error:
                summary.update(Q._failure(error, "receipt-retention"), accepted=False)
        Q._cancelled_summary(summary, observer)
        if summary["status"] != "passed":
            summary["accepted"] = False
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def _run(args):
    Q._require(Q.HEX.fullmatch(args.request_sha256), "manifest_selector")
    Q._require(re.fullmatch(r"[0-9]+-[0-9]+", args.run_id), "run_identity")
    Q._source_binding(args.scanner_root)
    with (
        tempfile.TemporaryDirectory(
            prefix="private-adjudication-", dir=args.work_parent
        ) as directory,
        tempfile.TemporaryDirectory(
            prefix="private-adjudication-credentials-", dir=args.work_parent
        ) as credentials,
    ):
        return _execute_adjudication(
            args, Path(directory), Q._ssh_arguments(Path(credentials))
        )


def main(argv=None):
    """Run the default-branch hosted retained-adjudication interface.

    Args:
        argv: Explicit request selector, run identity and trusted checkout paths.
    Returns:
        Zero only after complete acceptance and private receipt retention.
    Raises:
        None.
    """
    os.umask(0o077)
    parser = Q._Parser(description=__doc__)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scanner-root", type=Path, required=True)
    parser.add_argument("--work-parent", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, KeyError, TypeError, Q._QualificationError):
        print('{"status":"failed","accepted":false}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
