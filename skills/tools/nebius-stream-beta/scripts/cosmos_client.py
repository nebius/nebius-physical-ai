#!/usr/bin/env python3
"""Portable verified-TLS Cosmos client. Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import getpass
from email.utils import parsedate_to_datetime
from datetime import datetime
import math
import random
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import stat
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlencode
from urllib.request import Request, HTTPSHandler, HTTPRedirectHandler, ProxyHandler, build_opener
import uuid


class ClientError(Exception):
    pass


def read_private(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_uid != os.geteuid():
                raise ClientError("credential/state file must be private, regular and owned by this user")
            value = stream.read(1_048_577)
            if len(value) > 1_048_576:
                raise ClientError("private file is too large")
            return value
    except OSError:
        raise ClientError("cannot read private file") from None


def read_json(path):
    try:
        result = json.loads(read_private(path))
    except (ValueError, UnicodeError):
        raise ClientError("invalid private JSON file") from None
    if not isinstance(result, dict):
        raise ClientError("JSON input must be an object")
    return result


def private_directory(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_uid != os.geteuid():
        raise ClientError("output directory must be private and owned by this user")


def save_private(path, value):
    path = Path(path)
    private_directory(path.parent)
    if path.exists() or path.is_symlink():
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_uid != os.geteuid():
                raise ClientError("existing output must be private, regular and owned by this user")
        finally:
            os.close(descriptor)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".cosmos-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value if isinstance(value, bytes) else json.dumps(value, indent=2).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def state_lock(path):
    path = Path(path)
    private_directory(path.parent)
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_uid != os.geteuid():
            raise ClientError("state lock is not private")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ClientError("API redirect refused; verify the supplied HTTPS origin")


class Client:
    def __init__(self, origin, key_file, ca_bundle=None):
        parsed = urlsplit(origin)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ClientError("WMH_BASE_URL must be one HTTPS API origin")
        self.origin = origin.rstrip("/")
        self.key = read_private(key_file).decode("utf-8").strip()
        if not self.key or any(char.isspace() for char in self.key):
            raise ClientError("API key file must contain one key")
        context = ssl.create_default_context(cafile=ca_bundle)
        self.opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), NoRedirect())

    def open(self, method, route, payload=None, identity=None, *, query=None):
        if not route.startswith("/v1/") or "?" in route or "#" in route or ".." in route or "\\" in route:
            raise ClientError("invalid API route")
        suffix = ""
        if query:
            if method != "GET" or any(not isinstance(key, str) or not re.fullmatch(r"[a-z_]+", key)
                                      or not isinstance(value, (str, int, float, bool))
                                      or isinstance(value, float) and not math.isfinite(value)
                                      for key, value in query.items()):
                raise ClientError("invalid API query")
            suffix = "?" + urlencode({key: str(value).lower() if isinstance(value, bool) else value
                                       for key, value in query.items()})
        headers = {"Authorization": "Bearer " + self.key, "Accept": "application/json"}
        if identity:
            headers["Idempotency-Key"] = identity
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        while True:
            try:
                return self.opener.open(Request(self.origin + route + suffix, data=body, headers=headers, method=method), timeout=60)
            except HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After", "1") if error.headers else "1"
                error.close()
                if method == "GET" and status == 429:
                    try:
                        delay = float(retry_after)
                    except (ValueError, TypeError):
                        try:
                            delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
                        except (ValueError, TypeError, OverflowError):
                            delay = 1.0
                    if not math.isfinite(delay):
                        delay = 1.0
                    time.sleep(max(0.0, delay) + random.uniform(0.0, 0.25))
                    continue
                # No response body, URL, provider error or submitted field is echoed.
                raise ClientError(f"API returned HTTP {status}; inspect grants, capability, request or retained job state") from None
            except (URLError, OSError, ValueError):
                raise ClientError("HTTPS request failed; retain the same state for recovery") from None

    def request(self, method, route, payload=None, identity=None, *, query=None):
        with self.open(method, route, payload, identity, query=query) as response:
            try:
                value = json.load(response)
            except (ValueError, UnicodeError):
                raise ClientError("API returned invalid JSON") from None
        if not isinstance(value, dict):
            raise ClientError("API returned an unexpected JSON shape")
        return value

    def contract(self):
        return {"auth": self.request("GET", "/v1/auth/me"), "models": self.request("GET", "/v1/models"), "openapi": self.request("GET", "/v1/openapi.json")}


def storage_secrets(path):
    value = read_json(path)
    required = {"aws_access_key_id", "aws_secret_access_key"}
    if not required.issubset(value) or set(value) - required - {"aws_session_token"} or not all(isinstance(v, str) and v for v in value.values()):
        raise ClientError("S3 credential file must contain only access key, secret key and optional session token")
    return value


def verified_output(connection):
    if connection.get("purpose") not in {"output", "input_output"} or connection.get("provider") != "nebius" or connection.get("output_eligible") is not True or connection.get("verification", {}).get("status") != "verified":
        raise ClientError("output connection has not passed server Nebius verification")


def qualify(contract, payload, route):
    if route not in {"/v1/videos", "/v1/images", "/v1/augmentations"} or "post" not in contract.get("openapi", {}).get("paths", {}).get(route, {}):
        raise ClientError("requested generation route is unsupported")
    model = next((item for item in contract["models"].get("data", []) if item.get("id") == payload.get("model")), None)
    if model is None or model.get("availability") in {"unavailable", "disabled"}:
        raise ClientError("explicit requested model is not available to this key")
    profile = payload.get("generation", {}).get("profile")
    rows = model.get("capability_matrix", [])
    if route == "/v1/augmentations":
        row = next((row for row in rows if row.get("task") == "full_source_v2v_augmentation"), {})
    else:
        row = next((row for row in rows if profile and profile in row.get("profiles", [])), {})
    if not row.get("endpoint_exposed") or not row.get("runtime_implemented") or row.get("profile_evidence", {}).get(profile, row.get("evidence_status")) != "qualified":
        raise ClientError("requested task/profile has no exposed qualified runtime capability")


SUBMISSIONS = {
    "/v1/videos": "job_id", "/v1/images": "job_id", "/v1/augmentations": "job_id",
    "/v1/batches": "batch_id", "/v1/augmentation-batches": "batch_id",
    "/v1/source-inventories": "inventory_id",
}


def authenticated_owner(client):
    owner = client.request("GET", "/v1/auth/me").get("owner_id")
    if not isinstance(owner, str) or not owner:
        raise ClientError("API did not return an authenticated owner")
    return owner


def qualify_submission(client, contract, payload, route):
    if route not in SUBMISSIONS or "post" not in contract.get("openapi", {}).get("paths", {}).get(route, {}):
        raise ClientError("requested submission route is unsupported")
    if route == "/v1/source-inventories":
        return
    requests = [item["request"] for item in payload["items"]] if route == "/v1/batches" else [payload]
    for value in requests:
        generation_route = "/v1/videos" if route == "/v1/batches" else "/v1/augmentations" if route == "/v1/augmentation-batches" else route
        qualify(contract, value, generation_route)
        connection = client.request("GET", "/v1/storage/connections/" + identifier(value["output"]["connection_id"]))["connection"]
        verified_output(connection)
    if route == "/v1/augmentation-batches":
        client.request("POST", route + "/preflight", payload)


def submit(client, payload, state_path, route):
    contract = client.contract()
    owner = contract["auth"].get("owner_id")
    if not isinstance(owner, str) or not owner:
        raise ClientError("API did not return an authenticated owner")
    qualify_submission(client, contract, payload, route)
    resource_field = SUBMISSIONS[route]
    canonical = json.dumps({"origin": client.origin, "owner": owner, "route": route, "request": payload}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    with state_lock(state_path):
        if Path(state_path).exists():
            state = read_json(state_path)
            if state.get("request_sha256") != digest:
                raise ClientError("state belongs to a different request, origin or owner")
        else:
            state = {"request_sha256": digest, "idempotency_key": "cosmos-" + uuid.uuid4().hex, "route": route, "origin": client.origin, "owner_id": owner}
            save_private(state_path, state)
        if not state.get(resource_field):
            result = client.request("POST", route, payload, state["idempotency_key"])
            state[resource_field] = identifier(result[resource_field])
            save_private(state_path, state)
        return state


def owned_state(client, state_path):
    state = read_json(state_path)
    if state.get("origin") != client.origin or state.get("owner_id") != authenticated_owner(client):
        raise ClientError("state belongs to a different API origin or owner")
    if state.get("route") not in SUBMISSIONS:
        raise ClientError("state has an unsupported submission route")
    field = SUBMISSIONS[state["route"]]
    if not state.get(field):
        raise ClientError("submission outcome is ambiguous; run submit with the same request and state")
    identifier(state[field])
    return state


def status_route(state):
    return ("/v1/jobs" if SUBMISSIONS[state["route"]] == "job_id" else state["route"]) + "/" + state[SUBMISSIONS[state["route"]]]


def inventory_resume(client, state_path, output):
    state = owned_state(client, state_path)
    if state["route"] != "/v1/source-inventories":
        raise ClientError("inventory operation requires an inventory state")
    route = status_route(state)
    record = client.request("GET", route)
    save_private(output, record)
    while record.get("state") == "discovering":
        # Each POST advances one durable discovery page, never a GPU job.
        record = client.request("POST", route + "/resume")
        save_private(output, record)
    if record.get("state") != "complete":
        raise ClientError("inventory discovery needs attention; inspect its private output")
    return record


def inventory_objects(client, state_path, output_dir, limit=100):
    state = owned_state(client, state_path)
    if state["route"] != "/v1/source-inventories":
        raise ClientError("inventory operation requires an inventory state")
    if client.request("GET", status_route(state)).get("state") != "complete":
        raise ClientError("finish inventory discovery before exporting its objects")
    cursor = -1
    pages = 0
    while True:
        record = client.request("GET", status_route(state) + "/objects", query={"cursor": cursor, "limit": limit})
        save_private(Path(output_dir) / ("objects-first.json" if cursor == -1 else f"objects-after-{cursor:012d}.json"), record)
        pages += 1
        next_cursor = record.get("next_cursor")
        if next_cursor is None:
            return {"pages": pages}
        if type(next_cursor) is not int or next_cursor <= cursor:
            raise ClientError("API returned a non-advancing inventory cursor")
        cursor = next_cursor


def usage_export(client, state_path, output_dir, filters, limit=100, after=0):
    """Persist each fixed-watermark page before advancing the recovery cursor."""
    owner = authenticated_owner(client)
    output_dir = Path(output_dir).absolute()
    identity = {"origin": client.origin, "owner_id": owner, "filters": filters,
                "output_dir": str(output_dir), "limit": limit, "after": after}
    with state_lock(state_path):
        if Path(state_path).exists():
            state = read_json(state_path)
            if state.get("identity") != identity:
                raise ClientError("usage export state belongs to different options, origin or owner")
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ClientError("new usage export requires an empty output directory")
            state = {"identity": identity, "cursor": after, "watermark": None, "complete": False, "pages": 0, "events": 0}
            save_private(state_path, state)
        while not state["complete"]:
            cursor = state["cursor"]
            query = {**filters, "owner_id": owner, "cursor": cursor, "limit": limit}
            if state["watermark"] is not None:
                query["watermark"] = state["watermark"]
            page = client.request("GET", "/v1/usage/events", query=query)
            watermark = page.get("watermark")
            events = page.get("events")
            if type(watermark) is not int or watermark < cursor or not isinstance(events, list) or len(events) > limit:
                raise ClientError("API returned invalid usage pagination")
            if state["watermark"] is not None and watermark != state["watermark"]:
                raise ClientError("API changed the usage export watermark")
            previous = cursor
            for event in events:
                sequence = event.get("sequence")
                if type(sequence) is not int or not previous < sequence <= watermark or event.get("owner_id") != owner:
                    raise ClientError("API returned invalid or cross-owner usage events")
                previous = sequence
            next_cursor = page.get("next_cursor")
            if next_cursor is not None and (not events or type(next_cursor) is not int or next_cursor != previous):
                raise ClientError("API returned a non-advancing usage cursor")
            resume_cursor = next_cursor if next_cursor is not None else watermark
            if page.get("resume_cursor") != resume_cursor:
                raise ClientError("API returned an inconsistent usage resume cursor")
            if state["watermark"] is None:
                state["watermark"] = watermark
                save_private(state_path, state)
            save_private(output_dir / f"events-after-{cursor:012d}.json", page)
            state.update(cursor=resume_cursor, complete=next_cursor is None,
                         pages=state["pages"] + 1, events=state["events"] + len(events))
            save_private(state_path, state)
        return {key: state[key] for key in ("watermark", "cursor", "complete", "pages", "events")}


def utc_epoch(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError()
        return parsed.timestamp()
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("use an ISO-8601 timestamp with an explicit timezone") from None


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) or ".." in value:
        raise ClientError("invalid API identifier")
    return value


def resume(client, state_path, output_dir, interval=3):
    state = owned_state(client, state_path)
    if state["route"] == "/v1/source-inventories":
        raise ClientError("use inventory-resume for source discovery")
    output_dir = Path(output_dir)
    while True:
        record = client.request("GET", status_route(state))
        is_batch = "batch_id" in state
        save_private(output_dir / ("batch.json" if is_batch else "job.json"), record)
        if is_batch:
            if record.get("status") == "completed":
                results = []
                failed = False
                for child in record.get("jobs", []):
                    job_id = identifier(child["job_id"])
                    if child.get("status") == "succeeded":
                        results.append(download_job(client, job_id, output_dir / job_id))
                    else:
                        failed = True
                if failed or not results:
                    raise ClientError("batch has unsuccessful children; inspect private batch.json and retained successful artifacts")
                return {"batch_id": state["batch_id"], "status": "completed", "verified_jobs": len(results)}
            active = {"queued", "running", "processing"}
        else:
            if record.get("status") == "succeeded":
                return download_job(client, state["job_id"], output_dir)
            active = {"queued", "staging", "running", "publishing", "publication_queued", "cancel_requested", "submitted", "dispatching"}
        if record.get("status") not in active:
            raise ClientError("job or batch requires attention; inspect its retained private status file")
        time.sleep(interval)


def download_job(client, job_id, output_dir):
    record = client.request("GET", "/v1/jobs/" + job_id + "/artifacts")
    output_dir = Path(output_dir)
    private_directory(output_dir)
    items = record.get("manifest", {}).get("artifacts", [])
    if not items:
        raise ClientError("completed job has no published artifacts")
    for item in items:
        name = identifier(item["name"])
        expected = item.get("sha256", "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ClientError("artifact has no valid SHA-256")
        fd, temporary = tempfile.mkstemp(dir=output_dir, prefix=".download-")
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as stream, client.open("GET", "/v1/jobs/" + job_id + "/artifacts/" + name) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if digest.hexdigest() != expected.lower():
                raise ClientError("artifact SHA-256 mismatch; output rejected")
            os.replace(temporary, output_dir / name)
        finally:
            Path(temporary).unlink(missing_ok=True)
    save_private(output_dir / "manifest.json", record)
    return {"job_id": job_id, "status": "succeeded", "verified_artifacts": len(items)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-secrets")
    commands.add_parser("init-storage-secret")
    discover = commands.add_parser("discover")
    discover.add_argument("--output", required=True)
    register = commands.add_parser("register")
    register.add_argument("--config", required=True)
    connection = commands.add_parser("connection")
    connection.add_argument("--id", required=True)
    connection.add_argument("--output", required=True)
    rotate = commands.add_parser("rotate")
    rotate.add_argument("--connection", required=True)
    revoke = commands.add_parser("revoke")
    revoke.add_argument("--connection", required=True)
    generation = commands.add_parser("submit")
    generation.add_argument("--request", required=True)
    generation.add_argument("--state", required=True)
    generation.add_argument("--route", choices=list(SUBMISSIONS), default="/v1/videos")
    poll = commands.add_parser("resume")
    poll.add_argument("--state", required=True)
    poll.add_argument("--output-dir", required=True)
    status = commands.add_parser("status")
    status.add_argument("--state", required=True)
    status.add_argument("--output", required=True)
    inventory = commands.add_parser("inventory-resume")
    inventory.add_argument("--state", required=True)
    inventory.add_argument("--output", required=True)
    objects = commands.add_parser("inventory-objects")
    objects.add_argument("--state", required=True)
    objects.add_argument("--output-dir", required=True)
    objects.add_argument("--page-size", type=int, default=100)
    preflight = commands.add_parser("batch-preflight")
    preflight.add_argument("--request", required=True)
    preflight.add_argument("--output", required=True)
    for name in ("usage-summary", "usage-export"):
        usage = commands.add_parser(name)
        usage.add_argument("--include-synthetic", action="store_true")
        usage.add_argument("--since", type=utc_epoch)
        usage.add_argument("--until", type=utc_epoch)
        if name == "usage-summary":
            usage.add_argument("--bucket", choices=["hour", "day"], default="day")
            usage.add_argument("--output", required=True)
        else:
            usage.add_argument("--state", required=True)
            usage.add_argument("--output-dir", required=True)
            usage.add_argument("--page-size", type=int, default=100)
            usage.add_argument("--after", type=int, default=0)
    args = parser.parse_args()
    if hasattr(args, "page_size") and not 1 <= args.page_size <= 1000:
        raise ClientError("page size must be between 1 and 1000")
    if hasattr(args, "after") and args.after < 0:
        raise ClientError("usage starting cursor must not be negative")
    key_file = os.environ.get("WMH_API_KEY_FILE")
    s3_file = os.environ.get("WMH_S3_CREDENTIAL_FILE")
    if args.command in {"init-secrets", "init-storage-secret"}:
        with_api_key = args.command == "init-secrets"
        if not s3_file or Path(s3_file).exists() or Path(s3_file).is_symlink():
            raise ClientError("set an unused WMH_S3_CREDENTIAL_FILE path")
        if with_api_key and (not key_file or Path(key_file).exists() or Path(key_file).is_symlink() or Path(key_file).absolute() == Path(s3_file).absolute()):
            raise ClientError("set a distinct unused WMH_API_KEY_FILE path")
        if not sys.stdin.isatty():
            raise ClientError("secret initialization requires a terminal with hidden input")
        key = getpass.getpass("Inference API key (hidden): ") if with_api_key else None
        access = getpass.getpass("S3 access key (hidden): ")
        secret = getpass.getpass("S3 secret key (hidden): ")
        token = getpass.getpass("S3 session token (optional, hidden): ")
        if with_api_key and not key or not access or not secret:
            raise ClientError("required credential was empty")
        if with_api_key:
            save_private(key_file, key.encode())
        save_private(s3_file, {"aws_access_key_id": access, "aws_secret_access_key": secret, **({"aws_session_token": token} if token else {})})
        print("Private credential files created.")
        return
    if not key_file or not os.environ.get("WMH_BASE_URL"):
        raise ClientError("set WMH_BASE_URL and WMH_API_KEY_FILE")
    client = Client(os.environ["WMH_BASE_URL"], key_file, os.environ.get("WMH_CA_BUNDLE"))
    if args.command == "discover":
        save_private(args.output, client.contract())
        print("Authenticated contract saved privately.")
    elif args.command == "connection":
        save_private(args.output, client.request("GET", "/v1/storage/connections/" + identifier(args.id)))
        print("Owner-scoped connection metadata saved privately.")
    elif args.command in {"register", "rotate"}:
        if not s3_file:
            raise ClientError("set WMH_S3_CREDENTIAL_FILE")
        payload = read_json(args.config) if args.command == "register" else {}
        if "credentials" in payload or "credential_ref" in payload:
            raise ClientError("storage config must not contain secrets or credential references")
        payload["credentials"] = storage_secrets(s3_file)
        route = "/v1/storage/credentials" + ("/" + identifier(args.connection) + "/rotate" if args.command == "rotate" else "")
        result = client.request("POST", route, payload)
        if result.get("connection", {}).get("purpose") in {"output", "input_output"}:
            verified_output(result["connection"])
        print("Storage credentials verified and registered.")
    elif args.command == "revoke":
        client.request("DELETE", "/v1/storage/credentials/" + identifier(args.connection))
        print("Storage credentials revoked.")
    elif args.command == "submit":
        state = submit(client, read_json(args.request), args.state, args.route)
        field = SUBMISSIONS[args.route]
        print(json.dumps({field: state[field], "status": "accepted"}))
    elif args.command == "resume":
        print(json.dumps(resume(client, args.state, args.output_dir)))
    elif args.command == "status":
        state = owned_state(client, args.state)
        save_private(args.output, client.request("GET", status_route(state)))
        print("Owner-scoped status saved privately.")
    elif args.command == "inventory-resume":
        record = inventory_resume(client, args.state, args.output)
        print(json.dumps({"state": record["state"], "counts": record["counts"]}))
    elif args.command == "inventory-objects":
        print(json.dumps(inventory_objects(client, args.state, args.output_dir, args.page_size)))
    elif args.command == "batch-preflight":
        payload = read_json(args.request)
        qualify(client.contract(), payload, "/v1/augmentations")
        save_private(args.output, client.request("POST", "/v1/augmentation-batches/preflight", payload))
        print("Batch preflight saved privately; no inference was admitted.")
    elif args.command in {"usage-summary", "usage-export"}:
        filters = {"include_synthetic": args.include_synthetic}
        for name in ("since", "until"):
            if getattr(args, name) is not None:
                filters[name] = getattr(args, name)
        if args.since is not None and args.until is not None and args.since >= args.until:
            raise ClientError("usage time interval must increase")
        if args.command == "usage-summary":
            filters.update(owner_id=authenticated_owner(client), bucket=args.bucket)
            save_private(args.output, client.request("GET", "/v1/usage/summary", query=filters))
            print("Owner-scoped usage summary saved privately.")
        else:
            print(json.dumps(usage_export(client, args.state, args.output_dir, filters, args.page_size, args.after)))


if __name__ == "__main__":
    try:
        main()
    except (ClientError, KeyError, TypeError, ValueError, OSError, UnicodeError):
        # Preserve useful curated errors but never serialize arbitrary input,
        # provider exceptions, locals, request bodies or tracebacks.
        error = sys.exc_info()[1]
        print(str(error) if isinstance(error, ClientError) else "Client configuration or response is invalid; private state retained.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Stopped locally; server work was not cancelled. Resume with the retained state.", file=sys.stderr)
        sys.exit(130)
