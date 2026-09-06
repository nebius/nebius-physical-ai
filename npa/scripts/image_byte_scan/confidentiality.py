"""Lossless image-record confidentiality policy, independent of archive I/O.

The legacy view searches each ``str.splitlines()`` line once. The additive view
uses Python ``finditer`` over the whole UTF-8/surrogateescape record. Neither
view changes operator regex flags or discards undecodable bytes. Configuration
loaders and the separately verified exact-literal matcher belong to the caller.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import hashlib
import json
import re
import sys
from typing import Iterator


_SEMANTICS = "python-search-lines-plus-finditer-record/v1"
_LINE_BREAK = re.compile(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ConfidentialityError(ValueError):
    """A fixed error code that contains no input or regex diagnostic."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _valid_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


@dataclass(frozen=True)
class LiteralPolicyBinding:
    """Identity of a caller-verified matcher and its exact literal policy.

    This module does not load the matcher or verify its implementation. The
    caller must bind these hashes to the matcher/configuration it actually uses.
    """

    policy_sha256: str
    matcher_sha256: str
    pattern_count: int

    def __post_init__(self) -> None:
        if (
            not _valid_digest(self.policy_sha256)
            or not _valid_digest(self.matcher_sha256)
            or type(self.pattern_count) is not int
            or self.pattern_count < 0
        ):
            raise ConfidentialityError("literal_policy_invalid")

    def receipt(self) -> dict[str, object]:
        return {
            "status": "configured",
            "policy_sha256": self.policy_sha256,
            "matcher_sha256": self.matcher_sha256,
            "pattern_count": self.pattern_count,
            "verification": "caller_supplied",
        }


@dataclass(frozen=True)
class LiteralMatch:
    """An external match: zero-based pattern index and half-open byte range."""

    pattern_index: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class LiteralScan:
    """A complete, caller-supplied literal scan of the exact record bytes."""

    binding: LiteralPolicyBinding
    record_sha256: str
    byte_count: int
    matches: tuple[LiteralMatch, ...]
    complete: bool


@dataclass(frozen=True)
class Finding:
    """A redacted rule/location; line numbers are one-based, bytes half-open.

    A zero-width match at EOF after a line separator belongs to the following
    line. Empty input therefore has zero split lines but position zero is line 1.
    """

    rule_id: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    views: tuple[str, ...]


@dataclass(frozen=True)
class RecordScan:
    """Completed policy evaluation; findings contain no matched content."""

    policy_sha256: str
    record_sha256: str
    byte_count: int
    line_count: int
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    expression: re.Pattern[str] = field(repr=False)


def _lines(text: str) -> Iterator[tuple[int, int, int]]:
    """Yield content start/end and separator end with splitlines semantics.

    Streaming the documented separators avoids a second whole-record list of
    copied strings. In particular CRLF is one separator and a trailing separator
    does not produce another empty line. Tests compare this to str.splitlines.
    """
    start = 0
    for match in _LINE_BREAK.finditer(text):
        yield start, match.start(), match.end()
        start = match.end()
    if start < len(text):
        yield start, len(text), len(text)


def _byte_positions(text: str, positions: set[int]) -> dict[int, int]:
    """Convert only needed character offsets, encoding each intervening span once."""
    previous = 0
    byte_offset = 0
    result = {}
    for position in sorted(positions):
        byte_offset += len(text[previous:position].encode("utf-8", "surrogateescape"))
        result[position] = byte_offset
        previous = position
    return result


@dataclass(frozen=True)
class ConfidentialityPolicy:
    """An immutable compiled policy; create it with :func:`compile_policy`.

    Receipts bind exact configuration and regex runtime, but are not signatures
    or evidence that an external literal matcher was independently verified.
    """

    policy_sha256: str
    configuration_sha256: str
    _rules: tuple[_Rule, ...] = field(repr=False)
    _literal_policy: LiteralPolicyBinding | None = field(repr=False)

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "npa.image-byte-confidentiality-policy.v1",
            "policy_sha256": self.policy_sha256,
            "configuration_sha256": self.configuration_sha256,
            "semantics": _SEMANTICS,
            "engine": "python.re",
            "runtime": {
                "implementation": sys.implementation.name,
                "version": list(sys.version_info[:3]),
            },
            "customer": "configured",
            "infra": "configured" if len(self._rules) == 2 else "not_configured",
            "exact_literals": (
                self._literal_policy.receipt()
                if self._literal_policy is not None
                else {"status": "not_configured"}
            ),
        }

    def scan_record(self, raw: bytes, *, literal_scan: LiteralScan | None = None) -> RecordScan:
        """Scan a complete regular-file/metadata record with no size or finding cap.

        No record bytes, regex text, exception text, or external filenames appear
        in the result. Callers must treat failure or process termination as an
        incomplete scan, never as a clean result.
        """
        if type(raw) is not bytes:
            raise ConfidentialityError("record_type_invalid")
        try:
            result = self._scan(raw, literal_scan)
        except (re.error, RecursionError, OverflowError, MemoryError):
            result = None
        # Raise outside the handler so even __context__ has no private diagnostic.
        if result is None:
            raise ConfidentialityError("record_scan_failed")
        return result

    def _scan(self, raw: bytes, literal_scan: LiteralScan | None) -> RecordScan:
        record_sha = hashlib.sha256(raw).hexdigest()
        external = self._literal_matches(literal_scan, record_sha, len(raw))
        text = raw.decode("utf-8", "surrogateescape")
        spans: dict[tuple[str, int, int], set[str]] = {}
        line_starts = [0]
        line_count = 0
        for start, end, next_start in _lines(text):
            line_count += 1
            if next_start > end:
                line_starts.append(next_start)
            line = text[start:end]
            for rule in self._rules:
                match = rule.expression.search(line)
                if match is not None:
                    key = (rule.rule_id, start + match.start(), start + match.end())
                    spans.setdefault(key, set()).add("line")
        for rule in self._rules:
            for match in rule.expression.finditer(text):
                key = (rule.rule_id, match.start(), match.end())
                spans.setdefault(key, set()).add("record")
        needed = set(line_starts)
        for _, start, end in spans:
            needed.update((start, end))
        offsets = _byte_positions(text, needed)
        byte_line_starts = [offsets[position] for position in line_starts]
        byte_spans = {
            (rule, offsets[start], offsets[end]): views
            for (rule, start, end), views in spans.items()
        }
        for match in external:
            key = (f"literal-{match.pattern_index}", match.start_byte, match.end_byte)
            byte_spans.setdefault(key, set()).add("external_literal")
        findings = tuple(
            Finding(
                rule_id=rule,
                start_byte=start,
                end_byte=end,
                start_line=bisect_right(byte_line_starts, start),
                end_line=bisect_right(byte_line_starts, end - 1 if end > start else end),
                views=tuple(sorted(views)),
            )
            for (rule, start, end), views in sorted(byte_spans.items())
        )
        return RecordScan(self.policy_sha256, record_sha, len(raw), line_count, findings)

    def _literal_matches(
        self, scan: LiteralScan | None, record_sha: str, byte_count: int
    ) -> tuple[LiteralMatch, ...]:
        if self._literal_policy is None:
            if scan is not None:
                raise ConfidentialityError("literal_scan_unexpected")
            return ()
        if type(scan) is not LiteralScan:
            raise ConfidentialityError("literal_scan_missing")
        if (
            scan.complete is not True
            or type(scan.binding) is not LiteralPolicyBinding
            or scan.binding != self._literal_policy
            or type(scan.record_sha256) is not str
            or scan.record_sha256 != record_sha
            or type(scan.byte_count) is not int
            or scan.byte_count != byte_count
            or type(scan.matches) is not tuple
        ):
            raise ConfidentialityError("literal_scan_binding_invalid")
        for match in scan.matches:
            if (
                type(match) is not LiteralMatch
                or type(match.pattern_index) is not int
                or not 0 <= match.pattern_index < self._literal_policy.pattern_count
                or type(match.start_byte) is not int
                or type(match.end_byte) is not int
                or not 0 <= match.start_byte < match.end_byte <= byte_count
            ):
                raise ConfidentialityError("literal_scan_match_invalid")
        return scan.matches


def compile_policy(
    customer_pattern: str | None,
    infra_pattern: str | None = None,
    *,
    literal_policy: LiteralPolicyBinding | None = None,
) -> ConfidentialityPolicy:
    """Compile explicit configuration without reading environment or local files.

    Customer configuration is required. Absent/blank optional infrastructure
    configuration is reported as not configured. Nonblank expressions that
    match an empty string are valid and their zero-width findings are retained.
    """
    if type(customer_pattern) is not str or not customer_pattern.strip():
        raise ConfidentialityError("customer_pattern_missing")
    if infra_pattern is not None and type(infra_pattern) is not str:
        raise ConfidentialityError("infra_pattern_invalid")
    if literal_policy is not None and type(literal_policy) is not LiteralPolicyBinding:
        raise ConfidentialityError("literal_policy_invalid")
    rules = []
    configurations = []
    for rule_id, pattern, flags in (
        ("customer-denylist", customer_pattern, 0),
        ("infra-denylist", infra_pattern, re.IGNORECASE),
    ):
        if pattern is None or not pattern.strip():
            configurations.append({"rule_id": rule_id, "pattern": pattern, "configured": False})
            continue
        try:
            expression = re.compile(pattern, flags=flags)
        except (re.error, ValueError, RecursionError, OverflowError, MemoryError):
            expression = None
        if expression is None:
            raise ConfidentialityError(f"{rule_id}_invalid")
        rules.append(_Rule(rule_id, expression))
        configurations.append({
            "rule_id": rule_id,
            "pattern": pattern,
            "flags": int(flags),
            "effective_flags": expression.flags,
            "configured": True,
        })
    configuration_sha = _digest({
        "regex": configurations,
        "exact_literals": literal_policy.receipt() if literal_policy is not None else None,
    })
    policy_sha = _digest({
        "configuration_sha256": configuration_sha,
        "semantics": _SEMANTICS,
        "engine": "python.re",
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:3]),
    })
    return ConfidentialityPolicy(policy_sha, configuration_sha, tuple(rules), literal_policy)
