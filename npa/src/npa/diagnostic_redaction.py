"""Redact credential formats from diagnostics without changing line structure."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import re


SECRET_ENV_NAMES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
_HORIZONTAL_WHITESPACE = r"[^\S\r\n\v\f\x1c-\x1e\x85\u2028\u2029]"
_SECRET_ASSIGNMENT_PREFIX = re.compile(
    r"(?i)(?<![a-z0-9_-])"
    r"(?P<key>[\"']?[a-z0-9_-]+[\"']?)"
    rf"(?P<separator>{_HORIZONTAL_WHITESPACE}*[:=]{_HORIZONTAL_WHITESPACE}*)"
    rf"(?P<scheme>(?:bearer|basic){_HORIZONTAL_WHITESPACE}+)?"
)
_SECRET_KEY_MARKERS = (
    "token",
    "password",
    "passwd",
    "secret",
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "cookie",
    "private_key",
    "private-key",
    "access_key",
    "access-key",
)
_NON_SECRET_WORKFLOW_TOKEN_REFERENCES = tuple(
    (f"unknown {scope} ", f"{scope}.") for scope in ("config", "run", "state", "loop")
)
_NON_SECRET_ASSIGNMENT_CONTEXT_WIDTH = max(
    len(context) for context, _value_prefix in _NON_SECRET_WORKFLOW_TOKEN_REFERENCES
)
_BEARER_TOKEN = re.compile(
    rf"(?i)\bbearer{_HORIZONTAL_WHITESPACE}+"
    r"(?![a-z][a-z0-9+.-]*://)"
    r"[A-Za-z0-9._~+/=-]+"
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_KNOWN_TOKEN_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"hf_[A-Za-z0-9_=-]{8,}",
        r"nvapi-[A-Za-z0-9_=-]{8,}",
        r"gh[pousr]_[A-Za-z0-9_=-]{20,}",
        r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    )
)


def _quoted_secret_end(text: str, start: int) -> tuple[int, bool]:
    quote = text[start]
    line_end = text.find("\n", start + 1)
    if line_end < 0:
        line_end = len(text)
    index = start + 1
    while index < line_end:
        if text[index] == "\\":
            index += 2
            continue
        if text[index] != quote:
            index += 1
            continue
        tail = index + 1
        while tail < line_end and text[tail] in " \t":
            tail += 1
        if tail == line_end or text[tail] in ",;}]":
            return index + 1, True
        index += 1
    return line_end, False


def _unquoted_secret_end(text: str, start: int) -> int:
    end = start
    while end < len(text) and not text[end].isspace() and text[end] not in ",;}]\"'":
        end += 1
    return end


def _is_public_workflow_token(text: str, match: re.Match[str], value: str) -> bool:
    if match.group("key").strip("\"'").lower() != "token":
        return False
    prefix = text[
        max(0, match.start() - _NON_SECRET_ASSIGNMENT_CONTEXT_WIDTH) : match.start()
    ].lower()
    return any(
        prefix.endswith(context)
        and value.lower().startswith(value_prefix)
        and len(value) > len(value_prefix)
        and all(
            character.isascii() and (character.isalnum() or character in "_.-")
            for character in value
        )
        for context, value_prefix in _NON_SECRET_WORKFLOW_TOKEN_REFERENCES
    )


def _redact_secret_assignments(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _SECRET_ASSIGNMENT_PREFIX.finditer(text):
        if match.start() < cursor:
            continue
        key = match.group("key")
        normalized_key = key.strip("\"'").lower()
        if not any(marker in normalized_key for marker in _SECRET_KEY_MARKERS):
            continue
        value_start = match.end()
        if value_start >= len(text) or text[value_start] == "\n":
            continue
        quote = text[value_start] if text[value_start] in {'"', "'"} else ""
        value_end, closed = (
            _quoted_secret_end(text, value_start)
            if quote
            else (_unquoted_secret_end(text, value_start), False)
        )
        if value_end == value_start:
            continue
        value = text[value_start:value_end]
        if not quote and _is_public_workflow_token(text, match, value):
            continue
        pieces.extend((text[cursor : match.start()], key, match.group("separator")))
        pieces.append(f"{quote}<redacted>{quote if closed else ''}")
        cursor = value_end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _nonspace_ranges(text: str) -> Iterator[tuple[int, int]]:
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        end = start
        while end < len(text) and not text[end].isspace():
            end += 1
        if start == end:
            return
        yield start, end
        start = end


def _url_starts(text: str, start: int, end: int) -> list[tuple[int, int]]:
    urls: list[tuple[int, int]] = []
    search_from = start
    while True:
        delimiter = text.find("://", search_from, end)
        if delimiter < 0:
            return urls
        scheme_start = delimiter
        while scheme_start > start:
            candidate = text[scheme_start - 1]
            if not (
                candidate.isascii() and (candidate.isalnum() or candidate in "+.-")
            ):
                break
            scheme_start -= 1
        while scheme_start < delimiter and not (
            text[scheme_start].isascii() and text[scheme_start].isalpha()
        ):
            scheme_start += 1
        if scheme_start < delimiter:
            urls.append((scheme_start, delimiter))
        search_from = delimiter + 3


def _url_credential_ranges(text: str) -> list[tuple[int, int]]:
    replacements: list[tuple[int, int]] = []
    for token_start, token_end in _nonspace_ranges(text):
        urls = _url_starts(text, token_start, token_end)
        for index, (scheme_start, delimiter) in enumerate(urls):
            url_end = urls[index + 1][0] if index + 1 < len(urls) else token_end
            authority_start = delimiter + 3
            boundaries = (
                text.find(boundary, authority_start, url_end) for boundary in "/?#"
            )
            authority_end = min(
                (value for value in boundaries if value >= 0),
                default=url_end,
            )
            userinfo_end = text.find("@", authority_start, authority_end)
            if userinfo_end >= 0:
                replacements.append((authority_start, userinfo_end))
            query_start = text.find("?", authority_start, url_end)
            if 0 <= query_start < token_end - 1:
                replacements.append((query_start + 1, token_end))
    return replacements


def _replace_ranges(text: str, replacements: Sequence[tuple[int, int]]) -> str:
    if not replacements:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(replacements):
        if start < cursor:
            cursor = max(cursor, end)
            continue
        pieces.extend((text[cursor:start], "<redacted>"))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_url_credentials(text: str) -> str:
    return _replace_ranges(text, _url_credential_ranges(text))


def _redact_private_key_block(match: re.Match[str]) -> str:
    return "<redacted>" + ("\n" * match.group().count("\n"))


def redact_diagnostic_text(
    reason: object,
    *,
    secrets: Sequence[str] = (),
) -> str:
    """Redact explicit and patterned credentials without changing line structure.

    Args:
        reason: Exception, subprocess output, log, or provider diagnostic.
        secrets: Exact opaque credential values known at the call boundary.
    Returns:
        The diagnostic with credential values replaced by ``<redacted>``.
    Raises:
        None.
    """

    text = str(reason or "")
    for secret in sorted(
        (str(value) for value in secrets if value), key=len, reverse=True
    ):
        text = text.replace(secret, "<redacted>")
    text = _PRIVATE_KEY_BLOCK.sub(_redact_private_key_block, text)
    text = _redact_secret_assignments(text)
    text = _BEARER_TOKEN.sub("Bearer <redacted>", text)
    for pattern in _KNOWN_TOKEN_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return _redact_url_credentials(text)
