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
sys.path.insert(0,{str(trusted / 'npa/scripts')!r})
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
            sibling = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"], start_new_session=True, env={"PATH": os.defpath})
        finally:
            W._SPAWNING = False
        W.require(not W._CANCEL_REQUESTED, "scan_cancelled")
        W._SPAWNING = True
        try:
            worker = subprocess.Popen([sys.executable, str(wrapper), "--analysis-root", str(analysis), "--trusted-root", str(trusted),
                                       "--authorization", str(auth), "--output-dir", str(output)], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, start_new_session=True, env={"PATH": os.defpath})
        finally:
            W._SPAWNING = False
        W.require(not W._CANCEL_REQUESTED, "scan_cancelled")
        deadline = time.monotonic() + 10  # Synthetic synchronization, never an image workload limit.
        while not marker.exists() and worker.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        W.require(marker.exists(), "native_cancellation_startup")
        identity = json.loads(marker.read_bytes())
        W.require(identity["pid"] == identity["session"], "native_helper_session")
        worker.send_signal(signal.SIGTERM)
        stdout, stderr = worker.communicate(timeout=10)
        W.require(worker.returncode == 1 and stdout == b"complete image byte scan failed\n" and stderr == b"", "native_cancellation_exit")
        report = W.bound_json(P.binding(output / "report.json"))
        W.require(not report["complete"] and not report["valid"] and report["helper_joined"] and report["failure_code"] == "scan_cancelled", "native_cancellation_receipt")
        W.require(sibling.poll() is None and not Path(f'/proc/{identity["pid"]}').exists(), "native_cancellation_child_join")
        return {"passed": True, "helper_joined": True, "unrelated_sibling_preserved": True}
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
            target = Path(f'/proc/{identity["pid"]}/stat')
            if target.exists() and target.read_text().split()[21] == identity["start"]:
                os.kill(identity["pid"], signal.SIGTERM)
                raise W.ScanError("native_cancellation_helper_not_joined")


def checks(args, directory):
    helper, config = P.tools_bindings(args.tools_receipt)
    engine_binding = P.native_engine(args.native_receipt)
    snapshots = []
    for role in W.AHO_PINS:
        with W.bound_open(engine_binding[role], secret=False) as (path, _fd, info):
            snapshots.append(("native_" + role, engine_binding[role], False, path, W.stat_fingerprint(info)))
    native = W.AuthorizedAho(engine_binding)
    differential_cases = 0
    try:
        rng = random.Random(2701)
        for policy in ("exact-substring-v1", W.POLICY):
            for _ in range(100):
                values = ["abc", "bc", "ééééé", "aaaa", "aa", "\0", "a", "abc"]
                data = bytes(rng.randrange(256) for _ in range(100)) + b"abc aaaa " + "ééééé".encode() + b"\0" * 3
                reference = W.LiteralMatcher(values, policy)
                optimized = native.module.LiteralMatcher(native.module.compile_literals(values, policy))
                for offset in range(0, len(data), 7):
                    chunk = data[offset:offset + 7]
                    W.require(reference.feed(chunk) == optimized.feed(chunk), "native_literal_differential")
                W.require(reference.feed(b"", final=True) == optimized.feed(b"", final=True), "native_literal_differential_final")
                differential_cases += 1
        native_receipt = native.receipt()
    finally:
        native.close()
    image_results = []
    token = b"glpat" + b"-" + b"J9aL7mN2pQ8rS4tU6vW0"
    raw_body = b"\x7fELF\0" + b"\0" * (W.CHUNK - 9) + token + b"\0synthetic-literal\nfirst\nlast\xff"
    entries = [F.file("opt/binary", raw_body), F.file("opt/empty"), F.file("opt/path-only.p12")]
    raw = F.tar_data(entries)
    compressed, _ = F.optional_gzip(raw, flags=30, name=b"advisory-" + token, comment=b"synthetic-literal", extra=b"ID" + (len(token)).to_bytes(2, "little") + token)
    for optimized in (False, True):
        case = directory / ("native" if optimized else "reference")
        case.mkdir(mode=0o700)
        authorization = F.fixture(case, entries=entries, raw=raw, compressed=compressed, repeat=2, literals=["synthetic-literal"])
        authorization.update(helper=helper, config=config, tools_receipt=P.binding(args.tools_receipt))
        authorization["confidentiality"] = F.write(case / "policy.json", F.js({"customer_pattern": "first[\\s\\S]*last|\\x00{600}"}))
        if optimized:
            authorization["literal_engine"] = engine_binding
        output = case / "output"
        output.mkdir(mode=0o700)
        report = W._scan(authorization, output)
        W.write_private_json(output, "report.json", report)
        W.require(report.get("complete") is True and report.get("valid") is False and report.get("helper_joined") is True, "native_canary_outcome")
        W.require(report["helper_summary"]["findings"] >= 4 and report["regular_files"] == 6, "native_canary_population")
        rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
        by_kind = {}
        for row in rows:
            if row.get("type") == "record":
                by_kind.setdefault(row["kind"], []).extend(item["rule_id"] for item in row["findings"])
        W.require("gitlab-pat" in by_kind["layer_regular_content"] and "gitlab-pat" in by_kind["raw_gzip_header"], "native_binary_and_metadata_canary")
        W.require("customer-denylist" in by_kind["verified_zero_content"], "native_zero_regex_canary")
        W.require(any(row.get("rule_id") == "pkcs12-file" for row in rows), "native_path_rule_canary")
        W.require(token.decode() not in json.dumps(report) + json.dumps(rows), "native_receipt_disclosure")
        cancellation = cancellation_check(authorization, case)
        image_results.append({"cancellation": cancellation, "engine": "native" if optimized else "reference", "records": report["records"],
                              "findings": report["findings"], "helper_findings": report["helper_summary"]["findings"],
                              "regular_files": report["regular_files"], "complete": True, "helper_joined": True})
    W.require(image_results[0]["findings"] == image_results[1]["findings"], "native_full_scan_differential")
    W.recheck_snapshots(snapshots)
    result = {"schema_version": "npa.image-byte-native-checks.v1", "passed": True, "synthetic_only": True,
              "literal_differential_cases": differential_cases, "native": native_receipt, "archive_checks": image_results,
              "helper_sha256": helper["sha256"], "source_bindings": W.source_bindings()}
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
                result = {"schema_version": "npa.image-byte-native-checks.v1", "passed": False, "synthetic_only": True,
                          "failure_code": str(error) if isinstance(error, W.ScanError) else "native_check_execution_failed"}
                W.write_private_json(directory, "native-checks-failure.json", result)
        print("native image byte checks " + ("passed" if result["passed"] else "failed"))
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
