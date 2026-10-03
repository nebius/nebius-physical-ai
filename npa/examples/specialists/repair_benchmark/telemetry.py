"""Capture only permitted Codex usage fields through an operator-owned loopback OTLP receiver."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from urllib.parse import urlsplit

EVENTS = {
    "codex.conversation_starts",
    "codex.api_request",
    "codex.sse_event",
    "codex.websocket_request",
    "codex.websocket_event",
}
COUNTERS = {
    "input_token_count",
    "output_token_count",
    "cached_token_count",
    "cache_write_token_count",
    "reasoning_token_count",
    "total_token_count",
    "attempt",
    "attempts",
    "retry",
    "retries",
    "status",
    "status_code",
    "http.status_code",
    "http.response.status_code",
    "duration_ms",
    "duration_seconds",
    "duration",
}
IDENTIFIERS = {
    "conversation.id",
    "conversation_id",
    "thread_id",
    "response.id",
    "response_id",
    "request.id",
    "request_id",
    "turn.id",
    "turn_id",
}
RESPONSE_EVENTS = {
    "response.completed",
    "response.created",
    "response.failed",
    "response.incomplete",
    "response.done",
    "response.error",
}
TIERS = {"auto", "default", "standard", "priority", "fast", "flex", "scale"}
ENUMS = {
    "auth_mode": {"swic", "api", "apikey", "chatgpt", "unknown"},
    "model": {"gpt-6-astra"},
    "response.model": {"gpt-6-astra"},
    "request.model": {"gpt-6-astra"},
    "event.kind": RESPONSE_EVENTS,
    "kind": RESPONSE_EVENTS,
    "event_kind": RESPONSE_EVENTS,
    "service_tier": TIERS,
    "requested_service_tier": TIERS,
    "effective_service_tier": TIERS,
}


def _value(item):
    if isinstance(item, dict):
        return next(
            (
                item[key]
                for key in ("stringValue", "intValue", "doubleValue", "boolValue")
                if key in item
            ),
            None,
        )
    return None


def _attributes(items):
    return {
        item["key"]: _value(item.get("value"))
        for item in items
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }


def _allowed(fields):
    kept = {}
    for key, value in fields.items():
        if key in COUNTERS and not isinstance(value, bool):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number >= 0:
                kept[key] = int(number) if number.is_integer() else number
        elif key in IDENTIFIERS and isinstance(value, str) and 0 < len(value) <= 512:
            kept[key + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
        elif key in ENUMS and isinstance(value, str) and value in ENUMS[key]:
            kept[key] = value
        elif key == "success" and type(value) is bool:
            kept[key] = value
    return kept


def _sanitize(record, resource):
    fields = {**resource, **_attributes(record.get("attributes", []))}
    candidates = [
        record.get("eventName"),
        fields.get("event.name"),
        fields.get("event"),
        fields.get("name"),
        _value(record.get("body")),
    ]
    event = next(
        (value for value in candidates if isinstance(value, str) and value in EVENTS),
        None,
    )
    kept = _allowed(fields)
    if event is None and not any("token" in key for key in kept):
        return None
    return {"event": event, "attributes": kept, "received_epoch_ns": time.time_ns()}


def _extract(payload):
    result = []
    for resource in payload.get("resourceLogs", []):
        base = _attributes(resource.get("resource", {}).get("attributes", []))
        for scope in resource.get(
            "scopeLogs", resource.get("instrumentationLibraryLogs", [])
        ):
            for record in scope.get("logRecords", []):
                cleaned = _sanitize(record, base)
                if cleaned is not None:
                    result.append(cleaned)
    return result


def _receive(handler, records):
    size = int(handler.headers.get("Content-Length", "0"))
    if (
        handler.path != "/v1/logs"
        or not 0 < size <= 16 * 1024 * 1024
        or handler.headers.get("Content-Encoding", "identity") != "identity"
    ):
        raise ValueError("unsupported telemetry transport")
    rows = _extract(json.loads(handler.rfile.read(size)))
    with records.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return len(rows)


def _handler(records, stats):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            stats["http_requests"] += 1
            try:
                stats["retained_records"] += _receive(self, records)
                stats["accepted_batches"] += 1
                code = 200
            except (ValueError, TypeError, KeyError, AttributeError):
                stats["rejected_batches"] += 1
                code = 400
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    return Handler


def _endpoint(value):
    url = urlsplit(value)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.username
        or url.password
        or url.path != "/v1/logs"
        or url.query
        or url.fragment
    ):
        raise ValueError("a credential-free loopback OTEL endpoint is required")
    return value


def _invoke(real_codex):
    arguments = sys.argv[1:]
    if not arguments or arguments[0] != "exec" or arguments[-1] != "-":
        raise ValueError("benchmark wrapper supports only explicit exec")
    endpoint = _endpoint(os.environ["NPA_CODEX_OTEL_ENDPOINT"])
    settings = 'otel.exporter={otlp-http={endpoint="' + endpoint + '",protocol="json"}}'
    argv = [
        real_codex,
        *arguments[:-1],
        "-c",
        settings,
        "-c",
        "otel.log_user_prompt=false",
        "-",
    ]
    environment = dict(os.environ)
    for name in ("NO_PROXY", "no_proxy"):
        entries = environment.get(name, "").split(",")
        environment[name] = ",".join(
            dict.fromkeys([*entries, "localhost", "127.0.0.1", "::1"])
        )
    os.execve(real_codex, argv, environment)


def _wrapper(directory, real_codex):
    binary = directory / "bin/codex"
    binary.parent.mkdir()
    source = (
        f"#!{sys.executable}\nimport sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        f"from telemetry import _invoke\n_invoke({real_codex!r})\n"
    )
    binary.write_text(source)
    binary.chmod(0o700)


def _serve(directory, real_codex):
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    _wrapper(directory, real_codex)
    records = directory / "events.jsonl"
    records.touch(mode=0o600)
    stats = {
        "http_requests": 0,
        "accepted_batches": 0,
        "rejected_batches": 0,
        "retained_records": 0,
        "raw_payloads_persisted": False,
    }
    server = HTTPServer(("127.0.0.1", 0), _handler(records, stats))
    server.timeout = 1
    ready = {
        "endpoint": f"http://127.0.0.1:{server.server_port}/v1/logs",
        "pid": os.getpid(),
        "records": str(records),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (directory / "ready.json").write_text(json.dumps(ready) + "\n")
    stopping = False

    def stop(*_args):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        server.handle_request()
    server.server_close()
    (directory / "summary.json").write_text(json.dumps(stats) + "\n")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--codex", default=shutil.which("codex"))
    args = parser.parse_args()
    if not args.codex or not Path(args.codex).is_file():
        parser.error("a real Codex executable is required")
    os.umask(0o077)
    _serve(args.directory.resolve(), str(Path(args.codex).absolute()))


if __name__ == "__main__":
    _main()
