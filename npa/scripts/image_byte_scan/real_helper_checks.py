#!/usr/bin/env python3
"""Mandatory native regressions; missing binaries or native modules fail."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import core as W, prepare as P, synthetic as F
else:
    from . import core as W, prepare as P, synthetic as F


def cancellation_check(authorization, directory):
    """Interrupt a synthetic scan using the actual helper and selected matcher."""
    auth = directory / "cancellation-authorization.json"
    F.write(auth, F.js(authorization))
    marker = directory / "cancellation-helper.json"
    wrapper = directory / "cancellation-wrapper.py"
    output = directory / "cancellation-output"
    analysis, trusted = W._ROOTS.get()
    script = f"""import json,os,signal,sys
from pathlib import Path
sys.path.insert(0,{str(trusted / "npa/scripts")!r})
from image_byte_scan import core as W
original=W.Ledger
class PauseAfterNativeImport(original):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  pid=self.detector.process.pid
  Path({str(marker)!r}).write_text(json.dumps({{'pid':pid,'session':os.getsid(pid),'start':Path(f'/proc/{{pid}}/stat').read_text().split()[21]}}))
  signal.pause()
W.Ledger=PauseAfterNativeImport
raise SystemExit(W.main())
"""
    F.write(wrapper, script.encode())
    sibling = worker = None
    identity = None
    try:
        W._SPAWNING = True
        try:
            sibling = subprocess.Popen(
                [sys.executable, "-c", "import signal; signal.pause()"],
                start_new_session=True,
                env={"PATH": os.defpath},
            )
        finally:
            W._SPAWNING = False
        W.require(not W._CANCEL_REQUESTED, "scan_cancelled")
        W._SPAWNING = True
        try:
            worker = subprocess.Popen(
                [
                    sys.executable,
                    str(wrapper),
                    "--analysis-root",
                    str(analysis),
                    "--trusted-root",
                    str(trusted),
                    "--authorization",
                    str(auth),
                    "--output-dir",
                    str(output),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env={"PATH": os.defpath},
            )
        finally:
            W._SPAWNING = False
        W.require(not W._CANCEL_REQUESTED, "scan_cancelled")
        deadline = (
            time.monotonic() + 10
        )  # Synthetic synchronization, never an image workload limit.
        while (
            not marker.exists()
            and worker.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        W.require(marker.exists(), "native_cancellation_startup")
        identity = json.loads(marker.read_bytes())
        W.require(identity["pid"] == identity["session"], "native_helper_session")
        worker.send_signal(signal.SIGTERM)
        stdout, stderr = worker.communicate(timeout=10)
        W.require(
            worker.returncode == 1
            and stdout == b"complete image byte scan failed\n"
            and stderr == b"",
            "native_cancellation_exit",
        )
        report = W.bound_json(P.binding(output / "report.json"))
        W.require(
            not report["complete"]
            and not report["valid"]
            and report["helper_joined"]
            and report["failure_code"] == "scan_cancelled",
            "native_cancellation_receipt",
        )
        W.require(
            sibling.poll() is None and not Path(f"/proc/{identity['pid']}").exists(),
            "native_cancellation_child_join",
        )
        return {
            "passed": True,
            "helper_joined": True,
            "unrelated_sibling_preserved": True,
        }
    finally:
        try:
            if worker is not None:
                if worker.poll() is None:
                    worker.terminate()
                worker.communicate()
        finally:
            if sibling is not None:
                sibling.terminate()
                sibling.wait()
        if identity is not None:
            target = Path(f"/proc/{identity['pid']}/stat")
            if target.exists() and target.read_text().split()[21] == identity["start"]:
                os.kill(identity["pid"], signal.SIGTERM)
                raise W.ScanError("native_cancellation_helper_not_joined")


# The caller's queue must be deeper than the helper's so the helper stops reading
# first; one helper worker holds four records, so this depth is unreachable for it
# on any host, including a single-CPU one.
DUPLEX_HELPER_WORKERS = 1
DUPLEX_CALLER_DEPTH = 64
DUPLEX_LOUD_SECRETS = 12000


def duplex_fixture(helper, config, directory, case):
    """Build an archive whose first record's result exceeds a pipe buffer.

    Args:
        helper: The verified helper binding.
        config: The verified configuration binding.
        directory: Directory holding the shared tools receipt copy.
        case: Directory receiving this case's fixture files.

    Returns:
        The path of the written authorization.

    Raises:
        None.
    """
    token = b"gh" + b"p_" + b"aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hI6"
    entries = [
        F.file("opt/loud", (b'token = "' + token + b'"\n') * DUPLEX_LOUD_SECRETS)
    ]
    entries += [
        F.file(f"opt/quiet-{index}", b"plain public control\n" * 512)
        for index in range(23)
    ]
    authorization = F.fixture(case, entries=entries)
    authorization.update(
        helper=helper, config=config, tools_receipt=P.binding(directory / "tools.json")
    )
    auth = case / "duplex-authorization.json"
    F.write(auth, F.js(authorization))
    return auth


def duplex_wrapper(case, trusted):
    """Write a scan entry point with a deep caller queue and a one-worker helper.

    Constraining the helper's own affinity rather than the whole scan reproduces
    what a cgroup CPU quota does to a Go child: the caller's queue is sized from
    its own CPUs while the helper runs fewer workers. Both settings are explicit
    here, so the asymmetry does not depend on how many CPUs the host has.

    Args:
        case: Directory receiving the wrapper script.
        trusted: Checkout supplying the scanner package.

    Returns:
        The path of the written wrapper script.

    Raises:
        None.
    """
    wrapper = case / "duplex-wrapper.py"
    script = f"""import os,sys
sys.path.insert(0,{str(trusted / "npa/scripts")!r})
from image_byte_scan import core as W
W.PIPELINE_RECORDS={DUPLEX_CALLER_DEPTH}
cpus=sorted(os.sched_getaffinity(0))[:{DUPLEX_HELPER_WORKERS}]
spawn=W.subprocess.Popen
def constrained(*args,**kwargs):
 kwargs['preexec_fn']=lambda: os.sched_setaffinity(0,cpus)
 return spawn(*args,**kwargs)
W.subprocess.Popen=constrained
raise SystemExit(W.main())
"""
    F.write(wrapper, script.encode())
    return wrapper


def scan_argv(wrapper, auth, output):
    """Build the command line for one wrapped scan.

    Args:
        wrapper: The scan entry point to run.
        auth: Authorization path for the case.
        output: Directory the scan writes into.

    Returns:
        The argument vector for :class:`subprocess.Popen`.

    Raises:
        None.
    """
    analysis, trusted = W._ROOTS.get()
    return [
        sys.executable,
        str(wrapper),
        "--analysis-root",
        str(analysis),
        "--trusted-root",
        str(trusted),
        "--authorization",
        str(auth),
        "--output-dir",
        str(output),
    ]


def run_duplex_scan(wrapper, auth, output):
    """Run the wrapped scan to completion, failing closed if it stalls.

    Args:
        wrapper: The scan entry point written by :func:`duplex_wrapper`.
        auth: Authorization path for the case.
        output: Directory the scan writes into.

    Returns:
        None.

    Raises:
        ScanError: If the scan does not finish, which is the defect this case
            exists to catch; no deadline here turns stalling into success.
    """
    worker = None
    try:
        W._SPAWNING = True
        try:
            worker = subprocess.Popen(
                scan_argv(wrapper, auth, output),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env={"PATH": os.defpath},
            )
        finally:
            W._SPAWNING = False
        W.require(not W._CANCEL_REQUESTED, "scan_cancelled")
        try:
            worker.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            raise W.ScanError("native_duplex_stalled")
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.communicate()


def duplex_receipt(report):
    """Require that the saturated-pipe scan completed with nothing lost.

    Args:
        report: The scan report the wrapped run wrote.

    Returns:
        None.

    Raises:
        ScanError: If the scan did not complete or lost records or findings.
    """
    W.require(
        report["complete"] and report["helper_joined"], "native_duplex_completion"
    )
    W.require(
        report["helper_summary"]["findings"] == DUPLEX_LOUD_SECRETS
        and report["helper_summary"]["files"] == report["records"],
        "native_duplex_receipt",
    )


def duplex_check(helper, config, directory):
    """Scan while both pipe directions are saturated and the queues are unequal.

    The first record carries twelve thousand synthetic secrets, so its result is
    far larger than a pipe buffer and the helper is still writing output while the
    scan is still writing records. The helper runs fewer workers than the caller's
    queue depth, so the helper stops reading first. A caller that only drains
    output before each write stops here with both pipes full and never finishes.

    Args:
        helper: The verified helper binding.
        config: The verified configuration binding.
        directory: Directory receiving the case's fixtures and output.

    Returns:
        A mapping recording the completed receipt for this case.

    Raises:
        ScanError: If the scan stalls, fails, or loses records or findings.
    """
    case = directory / "duplex"
    case.mkdir(mode=0o700)
    _analysis, trusted = W._ROOTS.get()
    auth = duplex_fixture(helper, config, directory, case)
    output = case / "output"
    run_duplex_scan(duplex_wrapper(case, trusted), auth, output)
    report = W.bound_json(P.binding(output / "report.json"))
    duplex_receipt(report)
    return {
        "passed": True,
        "records": report["records"],
        "scanned_bytes": report["scanned_bytes"],
        "helper_findings": report["helper_summary"]["findings"],
        "helper_workers": DUPLEX_HELPER_WORKERS,
        "caller_depth": DUPLEX_CALLER_DEPTH,
    }


def checks(args, directory):
    helper, config = P.tools_bindings(args.tools_receipt)
    engine_binding = P.native_engine(args.native_receipt)
    snapshots = []
    for role in W.AHO_PINS:
        with W.bound_open(engine_binding[role], secret=False) as (path, _fd, info):
            snapshots.append(
                (
                    "native_" + role,
                    engine_binding[role],
                    False,
                    path,
                    W.stat_fingerprint(info),
                )
            )
    native = W.AuthorizedAho(engine_binding)
    differential_cases = 0
    try:
        rng = random.Random(2701)
        for policy in ("exact-substring-v1", W.POLICY):
            for _ in range(100):
                values = ["abc", "bc", "ééééé", "aaaa", "aa", "\0", "a", "abc"]
                data = (
                    bytes(rng.randrange(256) for _ in range(100))
                    + b"abc aaaa "
                    + "ééééé".encode()
                    + b"\0" * 3
                )
                reference = W.LiteralMatcher(values, policy)
                optimized = native.module.LiteralMatcher(
                    native.module.compile_literals(values, policy)
                )
                for offset in range(0, len(data), 7):
                    chunk = data[offset : offset + 7]
                    W.require(
                        reference.feed(chunk) == optimized.feed(chunk),
                        "native_literal_differential",
                    )
                W.require(
                    reference.feed(b"", final=True) == optimized.feed(b"", final=True),
                    "native_literal_differential_final",
                )
                differential_cases += 1
        native_receipt = native.receipt()
    finally:
        native.close()
    image_results = []
    token = b"glpat" + b"-" + b"J9aL7mN2pQ8rS4tU6vW0"
    raw_body = (
        b"\x7fELF\0"
        + b"\0" * (W.CHUNK - 9)
        + token
        + b"\0synthetic-literal\nfirst\nlast\xff"
    )
    entries = [
        F.file("opt/binary", raw_body),
        F.file("opt/empty"),
        F.file("opt/path-only.p12"),
    ]
    raw = F.tar_data(entries)
    compressed, _ = F.optional_gzip(
        raw,
        flags=30,
        name=b"advisory-" + token,
        comment=b"synthetic-literal",
        extra=b"ID" + (len(token)).to_bytes(2, "little") + token,
    )
    for optimized in (False, True):
        case = directory / ("native" if optimized else "reference")
        case.mkdir(mode=0o700)
        authorization = F.fixture(
            case,
            entries=entries,
            raw=raw,
            compressed=compressed,
            repeat=2,
            literals=["synthetic-literal"],
        )
        authorization.update(
            helper=helper, config=config, tools_receipt=P.binding(args.tools_receipt)
        )
        authorization["confidentiality"] = F.write(
            case / "policy.json",
            F.js({"customer_pattern": "first[\\s\\S]*last|\\x00{600}"}),
        )
        if optimized:
            authorization["literal_engine"] = engine_binding
        output = case / "output"
        output.mkdir(mode=0o700)
        report = W._scan(authorization, output)
        W.write_private_json(output, "report.json", report)
        W.require(
            report.get("complete") is True
            and report.get("valid") is False
            and report.get("helper_joined") is True,
            "native_canary_outcome",
        )
        W.require(
            report["helper_summary"]["findings"] >= 4 and report["regular_files"] == 6,
            "native_canary_population",
        )
        rows = [
            json.loads(line)
            for line in (output / "records.jsonl").read_text().splitlines()
        ]
        by_kind = {}
        for row in rows:
            if row.get("type") == "record":
                by_kind.setdefault(row["kind"], []).extend(
                    item["rule_id"] for item in row["findings"]
                )
        W.require(
            "gitlab-pat" in by_kind["layer_regular_content"]
            and "gitlab-pat" in by_kind["raw_gzip_header"],
            "native_binary_and_metadata_canary",
        )
        W.require(
            "customer-denylist" in by_kind["verified_zero_content"],
            "native_zero_regex_canary",
        )
        W.require(
            any(row.get("rule_id") == "pkcs12-file" for row in rows),
            "native_path_rule_canary",
        )
        W.require(
            token.decode() not in json.dumps(report) + json.dumps(rows),
            "native_receipt_disclosure",
        )
        cancellation = cancellation_check(authorization, case)
        image_results.append(
            {
                "cancellation": cancellation,
                "engine": "native" if optimized else "reference",
                "records": report["records"],
                "findings": report["findings"],
                "helper_findings": report["helper_summary"]["findings"],
                "regular_files": report["regular_files"],
                "complete": True,
                "helper_joined": True,
            }
        )
    W.require(
        image_results[0]["findings"] == image_results[1]["findings"],
        "native_full_scan_differential",
    )
    F.write(directory / "tools.json", args.tools_receipt.read_bytes())
    duplex = duplex_check(helper, config, directory)
    W.recheck_snapshots(snapshots)
    result = {
        "schema_version": "npa.image-byte-native-checks.v1",
        "passed": True,
        "synthetic_only": True,
        "literal_differential_cases": differential_cases,
        "native": native_receipt,
        "archive_checks": image_results,
        "duplex_check": duplex,
        "helper_sha256": helper["sha256"],
        "source_bindings": W.source_bindings(),
    }
    W.write_private_json(directory, "native-checks.json", result)
    return result


def _main(argv=None):
    os.umask(0o077)
    output_fd = None
    try:
        parser = W.SanitizedArgumentParser(description=__doc__)
        parser.add_argument("--analysis-root", type=Path, required=True)
        parser.add_argument("--trusted-root", type=Path, required=True)
        parser.add_argument("--tools-receipt", type=Path, required=True)
        parser.add_argument("--native-receipt", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args(argv)
        with W.authorized_roots(args.analysis_root, args.trusted_root):
            directory, output_fd = W.create_output(args.output_dir)
            try:
                result = checks(args, directory)
            except (*W.INPUT_ERRORS, subprocess.SubprocessError) as error:
                result = {
                    "schema_version": "npa.image-byte-native-checks.v1",
                    "passed": False,
                    "synthetic_only": True,
                    "failure_code": str(error)
                    if isinstance(error, W.ScanError)
                    else "native_check_execution_failed",
                }
                W.write_private_json(directory, "native-checks-failure.json", result)
        print(
            "native image byte checks " + ("passed" if result["passed"] else "failed")
        )
        return 0 if result["passed"] else 1
    except W.INPUT_ERRORS:
        print("native image byte checks failed")
        return 1
    finally:
        if output_fd is not None:
            os.close(output_fd)


def main(argv=None):
    try:
        with W.cancellation_scope():
            return _main(argv)
    except W.INPUT_ERRORS:
        print("native image byte checks failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
