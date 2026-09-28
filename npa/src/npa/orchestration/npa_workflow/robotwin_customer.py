"""Explicit customer-terminal decisions for one RoboTwin run and manifest.

The customer runs the decision command on their authenticated operator host.
This local consent path does not assert an external identity-provider signature
and does not authorize a manager to make a customer decision. Hosted callers
can continue to inject their existing authenticated authorization boundary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)


DECISION_ENV = "NPA_BYOF_ROBOTWIN_CUSTOMER_DECISION"
SCHEMA = "npa.robotwin.customer-terminal-decision.v1"
ISSUER = "customer-authenticated-operator-terminal"
ACTIVITY = (
    "noncommercial-containerization-and-technical-workload-validation-and-evaluation"
)


class CustomerDecisionError(ValueError):
    """A fixed refusal safe to include in redacted diagnostics."""


def _fail(category: str) -> None:
    raise CustomerDecisionError(category)


def _timestamp(value: str) -> datetime:
    try:
        result = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError):
        _fail("timestamp-invalid")
    return result


def owner_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        _fail("owner-directory-required")
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        _fail("owner-directory-required")


def notice(lock_path: Path) -> dict[str, Any]:
    raw = lock_path.read_bytes()
    lock = json.loads(raw)
    return {
        "solution": "robotwin",
        "runtime_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "terms": lock["access"]["terms_without_vendor_gate"],
        "intended_activity": ACTIVITY,
        "customer_action": "An authorized customer representative must review the named terms and explicitly accept or decline on their authenticated operator terminal.",
        "accept": "Run customer-decision with --decision accept, the exact run and customer scope, an expiry, and an owner-only receipt destination; type accept at the prompt.",
        "decline": "Run customer-decision with --decision decline, or take no action; no governed fetch/install/cache/provisioning will occur.",
        "resume": f"Supply the resulting private receipt path through {DECISION_ENV} when resuming the same run. Receipts are consumed once at submission; a failed consumed attempt requires a new customer decision for that run and manifest.",
        "manager_acceptance": False,
    }


def record_decision(
    *,
    lock_path: Path,
    run_id: str,
    customer_scope_id: str,
    expires_at: str,
    decision: str,
    receipt: Path,
) -> int:
    document = notice(lock_path)
    print(json.dumps(document, indent=2, sort_keys=True))
    if decision == "decline":
        print("RoboTwin customer decision: declined; no runtime side effects.")
        return 78
    if decision != "accept" or not sys.stdin.isatty():
        _fail("customer-terminal-required")
    if not run_id or not customer_scope_id:
        _fail("run-binding-required")
    expiry = _timestamp(expires_at)
    now = datetime.now(timezone.utc)
    if expiry <= now:
        _fail("stale")
    owner_directory(receipt.parent)
    if (
        input(
            "Type accept only if authorized to bind this customer to these exact terms for this run: "
        ).strip()
        != "accept"
    ):
        _fail("declined")
    payload = {
        "schema_version": SCHEMA,
        "issuer": ISSUER,
        "customer_scope_id": customer_scope_id,
        "run_id": run_id,
        "runtime_manifest_sha256": document["runtime_manifest_sha256"],
        "issued_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires_at,
        "decision": "accepted",
        "intended_activity": ACTIVITY,
        "terms": document["terms"],
        "assertion_id": "customer-terminal-" + secrets.token_hex(16),
        "nonce": secrets.token_hex(32),
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    print(
        "Customer decision recorded privately. Resume the same run with the receipt environment variable shown above."
    )
    return 0


def read_receipt(path: Path) -> dict[str, Any]:
    owner_directory(path.parent)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
                or metadata.st_size > 16384
            ):
                _fail("receipt-not-owner-only")
            raw = stream.read()
        value = json.loads(raw)
    except (OSError, ValueError, TypeError):
        _fail("receipt-unreadable")
    expected = {
        "schema_version",
        "issuer",
        "customer_scope_id",
        "run_id",
        "runtime_manifest_sha256",
        "issued_at",
        "expires_at",
        "decision",
        "intended_activity",
        "terms",
        "assertion_id",
        "nonce",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["schema_version"] != SCHEMA
        or value["issuer"] != ISSUER
    ):
        _fail("receipt-invalid")
    return value


def validate_receipt(
    value: Mapping[str, Any], request: Any, *, now: datetime | None = None
) -> None:
    if value["decision"] != "accepted":
        _fail("declined")
    for key in (
        "customer_scope_id",
        "run_id",
        "runtime_manifest_sha256",
        "intended_activity",
    ):
        if value[key] != getattr(request, key):
            _fail("binding-mismatch")
    if value["terms"] != [
        dict(id=identifier, url=url) for identifier, url in request.terms
    ]:
        _fail("terms-mismatch")
    current = datetime.now(timezone.utc) if now is None else now
    issued, expiry = _timestamp(value["issued_at"]), _timestamp(value["expires_at"])
    if issued > current or expiry <= current or issued >= expiry:
        _fail("stale")
    if (
        not isinstance(value["assertion_id"], str)
        or not value["assertion_id"].startswith("customer-terminal-")
        or not isinstance(value["nonce"], str)
        or len(value["nonce"]) != 64
    ):
        _fail("receipt-invalid")


@dataclass(frozen=True)
class CustomerTerminalAssertion:
    issuer: str
    customer_scope_id: str
    run_id: str
    runtime_manifest_sha256: str
    issued_at: str
    expires_at: str
    decision: str
    intended_activity: str
    terms: list[dict[str, str]]
    assertion_id: str
    nonce: str


class CustomerTerminalBoundary:
    """Consume the customer's explicit private terminal decision once."""

    trusted_issuer = ISSUER

    def __init__(self, receipt: Path):
        self.receipt = receipt

    def consume_once(self, request: Any) -> CustomerTerminalAssertion:
        value = read_receipt(self.receipt)
        validate_receipt(value, request)
        marker = self.receipt.parent / (
            ".robotwin-consumed-"
            + hashlib.sha256(value["assertion_id"].encode()).hexdigest()
        )
        try:
            fd = os.open(
                marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
        except FileExistsError:
            _fail("replayed")
        with os.fdopen(fd, "wb") as target:
            target.write(b"consumed\n")
            target.flush()
            os.fsync(target.fileno())
        fields = {key: item for key, item in value.items() if key != "schema_version"}
        return CustomerTerminalAssertion(**fields)


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            _fail("upstream-redirect-invalid")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def probe_runtime_inputs(lock_path: Path, *, expected_sha256: str) -> dict[str, int]:
    """Probe exact anonymous input bytes after customer consent, before GPU work.

    Only the first payload byte is read; complete sizes/hashes are verified by
    the runtime downloader. No payload, credential, URL or receipt is persisted.
    The caller must already have validated the customer's decision.
    """
    raw = lock_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail("probe-runtime-manifest-mismatch")
    lock = json.loads(raw)
    urls = [item["url"] for item in lock["runtime_artifacts"]]
    urls.extend(
        f"https://huggingface.co/datasets/{item['repository']}/resolve/{item['revision']}/{item['name']}"
        for item in lock["assets"]
    )
    urls.extend(
        item["license_url"]
        .replace("https://github.com/", "https://raw.githubusercontent.com/")
        .replace("/blob/", "/")
        for item in lock["sources"]
    )
    if not urls:
        _fail("probe-runtime-manifest-empty")

    def probe(url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            _fail("upstream-url-invalid")
        try:
            opener = build_opener(ProxyHandler({}), HTTPSHandler(), _HTTPSRedirects())
            with opener.open(Request(url, headers={"Range": "bytes=0-0"})) as response:
                if not response.read(1):
                    _fail("upstream-payload-empty")
        except Exception:
            _fail("upstream-payload-unavailable")

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(probe, urls))
    return {"payloads_probed": len(urls)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("notice", "customer-decision"))
    parser.add_argument(
        "--lock",
        type=Path,
        default=(
            Path(__file__).with_name("runtime-lock.json")
            if Path(__file__).with_name("runtime-lock.json").exists()
            else Path(__file__).resolve().parents[4]
            / "docker/workbench/robotwin/runtime-lock.json"
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--customer-scope-id")
    parser.add_argument("--expires-at")
    parser.add_argument("--decision", choices=("accept", "decline"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "notice":
            print(json.dumps(notice(args.lock), indent=2, sort_keys=True))
            return 0
        if not all(
            (
                args.run_id,
                args.customer_scope_id,
                args.expires_at,
                args.decision,
                args.receipt,
            )
        ):
            parser.error(
                "customer-decision requires run, customer scope, expiry, decision, and receipt"
            )
        return record_decision(
            lock_path=args.lock,
            run_id=args.run_id,
            customer_scope_id=args.customer_scope_id,
            expires_at=args.expires_at,
            decision=args.decision,
            receipt=args.receipt,
        )
    except CustomerDecisionError as failure:
        print("ROBOTWIN_CUSTOMER_REFUSED:" + str(failure), file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
