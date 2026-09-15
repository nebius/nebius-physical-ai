"""Real Python-regex policy tests; no image, credentials or optional engine."""

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import traceback

import pytest

from npa.guardrails.confidentiality import compile_denylist, scan_text


SOURCE = Path(__file__).resolve().parents[2] / "scripts/image_byte_scan/confidentiality.py"
SPEC = importlib.util.spec_from_file_location("image_byte_confidentiality_under_test", SOURCE)
policy_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy_module
SPEC.loader.exec_module(policy_module)
compile_policy = policy_module.compile_policy
Error = policy_module.ConfidentialityError
Binding = policy_module.LiteralPolicyBinding
LiteralMatch = policy_module.LiteralMatch
LiteralScan = policy_module.LiteralScan


def spans(result):
    return {(h.rule_id, h.start_byte, h.end_byte): h.views for h in result.findings}


def oracle(raw, pattern, flags=0):
    """Independent exact legacy calls plus whole-record matches, on small inputs."""
    text = raw.decode("utf-8", "surrogateescape")
    regex = re.compile(pattern, flags)
    result = {}
    start = 0
    for content, with_end in zip(text.splitlines(), text.splitlines(keepends=True)):
        match = regex.search(content)
        if match is not None:
            positions = (start + match.start(), start + match.end())
            key = tuple(len(text[:p].encode("utf-8", "surrogateescape")) for p in positions)
            result.setdefault(key, set()).add("line")
        start += len(with_end)
    for match in regex.finditer(text):
        key = tuple(
            len(text[:p].encode("utf-8", "surrogateescape"))
            for p in (match.start(), match.end())
        )
        result.setdefault(key, set()).add("record")
    return {key: tuple(sorted(views)) for key, views in result.items()}


@pytest.mark.parametrize("value", [None, "", " ", "\n\t", b"pattern", 0, False])
def test_customer_configuration_required(value):
    with pytest.raises(Error, match="^customer_pattern_missing$"):
        compile_policy(value)


@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_optional_infra_explicitly_not_configured(value):
    policy = compile_policy("customer-fixture", value)
    assert policy.receipt()["infra"] == "not_configured"
    assert policy.receipt()["exact_literals"] == {"status": "not_configured"}
    assert policy.scan_record(b"no match").findings == ()


@pytest.mark.parametrize("infra", [False, [], {}, b"fixture"])
def test_invalid_optional_configuration_is_not_absence(infra):
    with pytest.raises(Error, match="^infra_pattern_invalid$"):
        compile_policy("fixture", infra)


@pytest.mark.parametrize("role", ["customer", "infra"])
def test_invalid_regex_diagnostic_never_exposes_pattern(role, capsys):
    private = "sensitive-pattern-fixture(?P<"
    with pytest.raises(Error) as error:
        compile_policy(private if role == "customer" else "okay", private if role == "infra" else None)
    assert str(error.value) == f"{role}-denylist_invalid"
    rendered = "".join(traceback.format_exception(error.type, error.value, error.tb))
    assert private not in rendered
    assert private not in repr(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("role", ["customer", "infra"])
@pytest.mark.parametrize("flags", ["(?a)(?u)", "(?u)(?a)"])
def test_incompatible_inline_flags_have_a_sanitized_compile_error(role, flags):
    private = "sensitive-pattern-fixture"
    pattern = f"(?# {private}){flags}"
    # This real compiler boundary raises ValueError instead of re.error.
    with pytest.raises(ValueError, match="ASCII and UNICODE flags are incompatible"):
        re.compile(pattern, re.IGNORECASE if role == "infra" else 0)
    with pytest.raises(Error, match=f"^{role}-denylist_invalid$") as error:
        compile_policy(pattern if role == "customer" else "okay", pattern if role == "infra" else None)
    assert error.value.__context__ is None and error.value.__cause__ is None
    assert private not in repr(error.value)


@pytest.mark.parametrize("separator", ["\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"])
def test_every_python_splitlines_separator_preserves_anchored_search(separator):
    text = separator + "before" + separator + "FIXTURE" + separator + separator + "after" + separator
    result = compile_policy("^FIXTURE$").scan_record(text.encode())
    expected = scan_text(text, compile_denylist("^FIXTURE$"), source="fixture")
    assert [h.start_line for h in result.findings if "line" in h.views] == [h.line_number for h in expected]
    assert result.line_count == len(text.splitlines()) == 5
    assert result.findings[0].views == ("line",)
    assert result.findings[0].start_line == result.findings[0].end_line == 3


def test_separator_iterator_matches_python_for_all_single_byte_codepoints():
    for codepoint in range(256):
        text = "A" + chr(codepoint) + "B" + chr(codepoint)
        observed = [text[start:end] for start, end, _ in policy_module._lines(text)]
        assert observed == text.splitlines()


def test_whole_record_is_additive_without_forced_multiline():
    raw = b"before\nFIXTURE\nafter\nBEGIN\0middle\nEND"
    result = compile_policy(r"^FIXTURE$|(?s:BEGIN.*END)").scan_record(raw)
    assert [(h.views, raw[h.start_byte:h.end_byte]) for h in result.findings] == [
        (("line",), b"FIXTURE"), (("record",), b"BEGIN\0middle\nEND")
    ]
    assert (result.findings[-1].start_line, result.findings[-1].end_line) == (4, 5)


def test_customer_case_and_infra_unicode_ignorecase_and_inline_flags():
    raw = "FIXTURE K café".encode()
    result = compile_policy(r"fixture", r"fixture|k|CAFÉ").scan_record(raw)
    assert {h.rule_id for h in result.findings} == {"infra-denylist"}
    assert [raw[h.start_byte:h.end_byte].decode() for h in result.findings] == ["FIXTURE", "K", "café"]
    inline = compile_policy(r"(?i)fixture", r"(?-i:fixture)").scan_record(b"FIXTURE")
    assert [h.rule_id for h in inline.findings] == ["customer-denylist"]
    assert compile_policy(r"(?a)\w+").scan_record("é".encode()).findings == ()


def test_no_latin1_conversion_or_invalid_byte_deletion():
    raw = "é🙂".encode() + b"\xffA\0B\xfe" + " MÜLLER".encode()
    result = compile_policy(r"\udcffA\x00B\udcfe|MÜLLER").scan_record(raw)
    assert [(h.start_byte, h.end_byte) for h in result.findings] == [(6, 11), (12, 19)]
    assert compile_policy("AB").scan_record(b"A\xffB").findings == ()
    assert compile_policy(r"\x85").scan_record(b"\x85").findings == ()
    assert compile_policy(r"\udc85").scan_record(b"\x85").findings[0].end_byte == 1


def test_dedup_retains_both_views_but_not_other_rules_or_occurrences():
    result = compile_policy("FIXTURE", "FIXTURE").scan_record(b"FIXTURE FIXTURE")
    assert len(result.findings) == 4
    for rule in ("customer-denylist", "infra-denylist"):
        assert spans(result)[rule, 0, 7] == ("line", "record")
        assert spans(result)[rule, 8, 15] == ("record",)


@pytest.mark.parametrize("pattern", [r"(?:)", r"^", r"$", r"\Z", r"(?=A)", r"A*", r"(?m)^", r"(?s:.*)"])
@pytest.mark.parametrize("raw", [b"", b"A", b"\n", b"A\r\n\n", "é\u2028A".encode()])
def test_zero_width_and_empty_record_semantics(pattern, raw):
    result = compile_policy(pattern).scan_record(raw)
    actual = {(h.start_byte, h.end_byte): h.views for h in result.findings}
    assert actual == oracle(raw, pattern)
    assert result.line_count == len(raw.decode("utf-8", "surrogateescape").splitlines())
    if not raw:
        assert all(h.start_line == h.end_line == 1 for h in result.findings)


def test_zero_width_at_end_after_crlf_belongs_to_following_line():
    result = compile_policy(r"\Z").scan_record(b"A\r\n")
    assert [(h.start_byte, h.start_line, h.end_line, h.views) for h in result.findings] == [
        (1, 1, 1, ("line",)), (3, 2, 2, ("record",))
    ]


def test_python_lookbehind_backreference_and_context_views():
    raw = b"owner=ABC-ABC\nABC-ABC"
    result = compile_policy(r"(?<=owner=)([A-Z]+)-\1").scan_record(raw)
    assert [(h.start_byte, h.end_byte, h.views) for h in result.findings] == [(6, 13, ("line", "record"))]
    contextual = compile_policy(r"(?<=\n)ABC|(?<!\n)^ABC").scan_record(raw)
    assert spans(contextual)["customer-denylist", 14, 17] == ("line", "record")


def test_whole_file_match_crosses_large_binary_record_without_cap():
    raw = b"BEGIN" + b"\0\xff" * 150000 + b"\n" + b"x" * 900000 + b"END"
    result = compile_policy(r"(?s)BEGIN.*END").scan_record(raw)
    assert len(result.findings) == 1
    assert result.findings[0].end_byte == result.byte_count == len(raw)
    assert result.findings[0].views == ("record",)
    assert result.record_sha256 == hashlib.sha256(raw).hexdigest()


def test_all_findings_are_retained():
    result = compile_policy("x").scan_record(b"x " * 12001)
    assert len(result.findings) == 12001
    assert result.findings[-1].start_byte == 24000


def test_receipts_bind_configuration_and_are_safe_immutable_snapshots():
    private = "sensitive-pattern-fixture"
    policy = compile_policy(private)
    result = policy.scan_record(private.encode())
    exported = json.dumps({"receipt": policy.receipt(), "scan": asdict(result)})
    assert private not in exported
    assert private not in repr(policy)
    assert "expression" not in exported and "pattern" not in exported
    assert compile_policy(private).receipt() == policy.receipt()
    assert compile_policy(private + " ").policy_sha256 != policy.policy_sha256
    assert compile_policy(private, "").policy_sha256 != policy.policy_sha256
    assert compile_policy("(?i)" + private).policy_sha256 != policy.policy_sha256
    receipt = policy.receipt()
    receipt["runtime"]["version"][0] = -1
    assert policy.receipt()["runtime"]["version"][0] == sys.version_info.major
    with pytest.raises(FrozenInstanceError):
        policy.policy_sha256 = "changed"


@pytest.mark.parametrize("raw", ["text", bytearray(b"text"), memoryview(b"text"), None])
def test_requires_exact_complete_bytes_input(raw):
    with pytest.raises(Error, match="^record_type_invalid$"):
        compile_policy("fixture").scan_record(raw)


@pytest.mark.parametrize("error_type", [re.error, RecursionError, OverflowError, MemoryError])
def test_scan_failures_are_incomplete_and_have_no_private_exception_chain(monkeypatch, error_type):
    def fail(*args):
        raise error_type("sensitive-diagnostic-fixture")
    monkeypatch.setattr(policy_module.ConfidentialityPolicy, "_scan", fail)
    with pytest.raises(Error, match="^record_scan_failed$") as caught:
        compile_policy("fixture").scan_record(b"input")
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_configuration_is_explicit_and_does_not_read_ambient_environment(monkeypatch):
    monkeypatch.setenv("CUSTOMER_DENYLIST", "ambient-fixture")
    monkeypatch.setenv("INFRA_DENYLIST", "ambient-fixture")
    policy = compile_policy("explicit-fixture")
    assert policy.scan_record(b"ambient-fixture").findings == ()
    assert len(policy.scan_record(b"explicit-fixture").findings) == 1
    with pytest.raises(Error, match="^customer_pattern_missing$"):
        compile_policy(None)


def binding():
    return Binding(hashlib.sha256(b"synthetic-policy").hexdigest(), hashlib.sha256(b"synthetic-matcher").hexdigest(), 2)


def literal_result(raw, **changes):
    result = LiteralScan(binding(), hashlib.sha256(raw).hexdigest(), len(raw), (LiteralMatch(0, 0, 1),), True)
    return replace(result, **changes)


def test_external_literals_compose_without_a_matcher_import_or_validation_claim():
    raw = "é\nfixture".encode()
    matches = (LiteralMatch(0, 0, 2), LiteralMatch(1, 0, 2), LiteralMatch(0, 0, 2), LiteralMatch(1, 3, len(raw)))
    policy = compile_policy("fixture", literal_policy=binding())
    result = policy.scan_record(raw, literal_scan=literal_result(raw, matches=matches))
    assert len(result.findings) == 4
    assert {h.rule_id for h in result.findings} == {"customer-denylist", "literal-0", "literal-1"}
    assert spans(result)["literal-1", 3, len(raw)] == ("external_literal",)
    assert result.findings[-1].start_line == 2
    assert policy.receipt()["exact_literals"]["verification"] == "caller_supplied"


def test_external_literal_bytes_may_be_inside_utf8_encoding():
    raw = "é".encode()
    result = compile_policy("absent", literal_policy=binding()).scan_record(raw, literal_scan=literal_result(raw, matches=(LiteralMatch(0, 1, 2),)))
    assert result.findings[0].start_byte == 1
    assert result.findings[0].start_line == result.findings[0].end_line == 1


def test_literal_policy_requires_result_even_if_zero_patterns():
    empty = replace(binding(), pattern_count=0)
    policy = compile_policy("absent", literal_policy=empty)
    with pytest.raises(Error, match="^literal_scan_missing$"):
        policy.scan_record(b"")
    result = policy.scan_record(b"", literal_scan=LiteralScan(empty, hashlib.sha256(b"").hexdigest(), 0, (), True))
    assert result.findings == ()
    with pytest.raises(Error, match="^literal_scan_unexpected$"):
        compile_policy("absent").scan_record(b"x", literal_scan=literal_result(b"x"))


@pytest.mark.parametrize("changes", [
    {"complete": False}, {"complete": 1}, {"byte_count": 2}, {"byte_count": True},
    {"record_sha256": "0" * 64}, {"matches": []},
    {"binding": Binding("0" * 64, "1" * 64, 2)},
    {"binding": None}, {"record_sha256": None},
])
def test_literal_result_rejects_incomplete_or_wrong_provenance(changes):
    with pytest.raises(Error, match="^literal_scan_binding_invalid$"):
        compile_policy("x", literal_policy=binding()).scan_record(b"x", literal_scan=literal_result(b"x", **changes))


@pytest.mark.parametrize("match", [
    LiteralMatch(-1, 0, 1), LiteralMatch(2, 0, 1), LiteralMatch(True, 0, 1),
    LiteralMatch(0, -1, 1), LiteralMatch(0, 0, 0), LiteralMatch(0, 0, 2),
    LiteralMatch(0, 0.0, 1), LiteralMatch(0, 0, True), "untrusted",
])
def test_external_literal_matches_must_be_valid_nonempty_byte_spans(match):
    with pytest.raises(Error, match="^literal_scan_match_invalid$"):
        compile_policy("x", literal_policy=binding()).scan_record(b"x", literal_scan=literal_result(b"x", matches=(match,)))


@pytest.mark.parametrize("changes", [
    {"policy_sha256": "private-invalid-fixture"}, {"matcher_sha256": "A" * 64},
    {"pattern_count": -1}, {"pattern_count": False}, {"pattern_count": 1.0},
])
def test_literal_binding_rejects_malformed_fields(changes):
    with pytest.raises(Error, match="^literal_policy_invalid$"):
        replace(binding(), **changes)


@pytest.mark.parametrize("seed", range(20))
def test_randomized_lossless_semantics_against_independent_oracle(seed):
    rng = random.Random(seed)
    pieces = [b"A", b"B", b" ", b"\n", b"\r\n", b"\v", b"\x00", b"\xff", b"\x85", "é".encode(), "\u2028".encode()]
    patterns = [r"A+", r"^A$", r"(?m)^B", r"\w+", r"(?=A)", r"\Z", r"(?s)A.*B", r"\udcff", r"(?<!\n)A", r"(A)\1", r"(?:)"]
    for _ in range(25):
        raw = b"".join(rng.choice(pieces) for _ in range(rng.randrange(50)))
        pattern = rng.choice(patterns)
        result = compile_policy(pattern).scan_record(raw)
        assert {(h.start_byte, h.end_byte): h.views for h in result.findings} == oracle(raw, pattern)
        text = raw.decode("utf-8", "surrogateescape")
        expected_lines = scan_text(text, compile_denylist(pattern), source="fixture")
        assert sorted(h.start_line for h in result.findings if "line" in h.views) == [h.line_number for h in expected_lines]
        assert result.line_count == len(text.splitlines())
