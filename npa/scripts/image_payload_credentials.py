"""Credential detection shared by the per-image payload scanners.

Every ``scan_image_*_payload.py`` grew its own credential rules, and they
drifted apart. Measured over six scanners and seven planted cases, the rule sets
did not order cleanly: the two that detected the private-key family missed
``.netrc``, and the only scanner that caught ``.netrc`` missed every private
key. These shared rules are used by the Alpamayo2, Cosmos3 serving and Cosmos3
Ray Serve scanners; other image scanners retain their own policies.

Paths are not sufficient on their own. A private key copied to an application
directory matches no conventional name, so content detection is here too.

Content detection has to survive three things that review demonstrated it did
not, each with a working counterexample rather than a hypothetical:

*A key anywhere in the member.* Reading a bounded prefix is not a cheaper
approximation of scanning a file, it is a documented place to hide a key. Every
byte is examined.

*A match on a chunk boundary.* Fixed-size markers are found with an overlap
between consecutive chunks.

*Whitespace longer than any window.* Private-key headers and named credential
assignments can be separated from their values by arbitrarily much whitespace.
Their matchers retain state across chunks, never the intervening bytes. Named
values retain at most 513 bytes to distinguish source syntax from literals;
oversized or ambiguous values remain findings.

Generic quoted assignments are a Dockerfile policy, not proof of a credential
in arbitrary library bytes: pip and Pydantic contain ordinary password examples.
These detectors cover the declared paths and credential shapes only. Complete
image qualification additionally requires the complete-byte/Gitleaks gate and
its evidence-based disposition; this module cannot prove absence of arbitrary
plaintext secrets.
"""

from __future__ import annotations

import ast
import base64
import re
from dataclasses import dataclass
from typing import IO

try:
    from re import _parser as _regex_parser
except ImportError:  # Python 3.10 predates the re._parser module.
    import sre_parse as _regex_parser


def normalise_member_name(name: str) -> str:
    """Return a tar member name resolved to its canonical path form.

    The rules below are substring matches, so any spelling that pushes a ``.``,
    an empty segment or a ``..`` *inside* a rule's matched span defeats it while
    naming the identical file. ``root/.docker/config.json`` is caught and
    ``root/.docker/./config.json`` was not. Stripping a leading ``./`` is not
    enough, because the leading position is the one place an alias is harmless.

    So resolve the path properly: drop empty segments and ``.``, and pop a
    directory for each ``..``.

    A leading ``..`` is kept rather than discarded. Dropping it would rewrite a
    name that escapes the archive root into an innocuous-looking one, which
    trades a bypass for a different bypass. Only ``..`` that has a directory to
    cancel is resolved.

    Note this is purely lexical, which is the right scope here: the input is an
    archive member name, nothing is resolved against a filesystem, and a symlink
    cannot be followed at scan time anyway.
    """

    segments: list[str] = []
    for segment in name.split("/"):
        if segment in ("", "."):
            # Collapses "//" and "/./". Note "" also drops a leading slash.
            continue
        if segment == ".." and segments and segments[-1] != "..":
            segments.pop()
            continue
        segments.append(segment)
    return "/".join(segments)


# Conventional locations. ``ssh_user_key`` covers both root's home and a normal
# one, which is the case a path list keyed only on ``etc/ssh`` misses.
CREDENTIAL_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ssh_host_key",
        re.compile(r"(?:^|/)etc/ssh/ssh_host_(?:rsa|dsa|ecdsa|ed25519)_key$", re.I),
    ),
    (
        "ssh_user_key",
        re.compile(r"(?:^|/)\.ssh/id_(?:rsa|dsa|ecdsa|ed25519)$", re.I),
    ),
    (
        "cloud_credential_file",
        re.compile(
            r"(?:^|\A|/)(?:\.aws/credentials|\.docker/config\.json|\.git-credentials"
            r"|\.netrc|\.npa/credentials\.yaml|\.kube/config|kubeconfig)$",
            re.I,
        ),
    ),
)

# PEM permits arbitrarily much whitespace before the body. Only its finite
# header uses a window; the whitespace phase retains state instead of bytes.
_PRIVATE_KEY_HEADER = re.compile(
    rb"-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
)
_PEM_WHITESPACE = re.compile(rb"[ \t\r\n\f\v]*")
_PEM_BODY_START = re.compile(rb"[A-Za-z0-9+/=]")

# These complete patterns have finite maximum widths and can use a window.
MARKER_CONTENT: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("aws_access_key_id", re.compile(rb"AKIA[0-9A-Z]{16}")),
)

_WHITESPACE = re.compile(rb"[ \t\r\n\f\v]*")


@dataclass(frozen=True)
class AssignmentRule:
    """A ``NAME <ws> SEP <ws> VALUE`` credential, matched across chunks.

    Rules need ``min_value`` non-space bytes. Source references are checked only
    after a complete bounded value; label and separator gaps have no size bound.
    """

    kind: str
    names: re.Pattern[bytes]
    separators: bytes
    min_value: int = 0


class _DeclaredAssignmentPattern:
    """Describe a whole-input assignment independently of chunk assembly."""

    def __init__(self, rule: AssignmentRule) -> None:
        self.rule = rule
        self.pattern = rule.names.pattern + rb"\s*[=:]\s*"
        self.header = re.compile(
            rb"(?:" + rule.names.pattern + rb")\s*([=:])\s*", rule.names.flags
        )

    def search(self, data: bytes) -> re.Match[bytes] | None:
        """Return the first assignment whose bounded value is credential data."""
        for header in self.header.finditer(data):
            value = _AssignmentValue(header.group(1))
            value.feed(data, header.end())
            if value.credential(self.rule.min_value):
                return header
        return None


ASSIGNMENT_RULES: tuple[AssignmentRule, ...] = (
    AssignmentRule(
        kind="credential_assignment",
        names=re.compile(rb"aws_secret_access_key|hf_token|ngc_api_key", re.I),
        separators=b"=:",
        min_value=8,
    ),
)

# Whole-input markers and named-value policy describe the streaming verdict.
# Source expressions require syntax validation beyond a regular expression;
# chunk assembly remains independent so boundary behavior can be compared.
# test_image_payload_credentials.py asserts the two agree; a reviewer checking
# this module should compare against these rather than read the state machine.
DECLARED_CONTENT_PATTERNS: tuple[
    tuple[str, re.Pattern[bytes] | _DeclaredAssignmentPattern], ...
] = MARKER_CONTENT + (
    (
        "private_key_content",
        re.compile(
            rb"-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
            rb"[ \t\r\n\f\v]*[A-Za-z0-9+/=]"
        ),
    ),
    (
        "credential_assignment",
        _DeclaredAssignmentPattern(ASSIGNMENT_RULES[0]),
    ),
)

# Retained under its former name: the review lane's reproducers import it.
CREDENTIAL_CONTENT = DECLARED_CONTENT_PATTERNS

CONTENT_CHUNK = 1024 * 1024
# Headroom for a future marker, not a measurement of the current ones. Review
# showed this is not load-bearing today: setting it to zero changes no verdict,
# because the carry takes the largest of the three constants below and the
# longest finite header declared here is 37 bytes.
MARKER_OVERLAP = 4096
# Enough to hold the longest NAME token whole when one straddles a boundary.
# This is the effective floor on the carry today.
_NAME_CARRY = 64


def _max_match_length(pattern: re.Pattern[bytes]) -> int:
    """Return the longest byte string ``pattern`` can match.

    Measuring the source with ``len(pattern.pattern)`` is the obvious thing and
    it is wrong in the unsafe direction: ``AKIA[0-9A-Z]{16}`` is 16 source
    characters and matches 20 bytes, so a carry sized that way would be an
    under-estimate presented as a bound. Ask the parser for the real width.

    The regex parser (``sre_parse`` on Python 3.10, ``re._parser`` on newer
    versions) is private. It is used deliberately, because the alternative is a
    hand-maintained number that silently goes stale, and there is no public
    equivalent. If neither parser is available, importing this module raises
    rather than guessing, and the carry is never quietly too small.
    """

    width = _regex_parser.parse(pattern.pattern.decode("latin-1")).getwidth()[1]
    if width >= _regex_parser.MAXREPEAT:
        raise ValueError("streaming markers must have a finite maximum width")
    return width


# An upper bound on any marker match, derived so that adding a longer marker
# widens the carry automatically instead of silently outgrowing it.
_LONGEST_MARKER = max(_max_match_length(pattern) for _, pattern in MARKER_CONTENT)
_HEADER_CARRY = _max_match_length(_PRIVATE_KEY_HEADER)
# What actually gets carried between chunks, and why: long enough for any marker
# to be found whole, any NAME token to survive a split, and whatever headroom
# MARKER_OVERLAP asks for.
CARRY = max(MARKER_OVERLAP, _NAME_CARRY, _LONGEST_MARKER, _HEADER_CARRY)


class _PrivateKeyMatcher:
    """Recognize PEM bodies after unbounded whitespace with constant state."""

    def __init__(self) -> None:
        self._awaiting_body = False

    def feed(self, data: bytes, carried: int) -> bool:
        """Continue after the prior chunk, revisiting carry only to find headers."""
        position = carried if self._awaiting_body else max(0, carried - _HEADER_CARRY)
        while position < len(data):
            if not self._awaiting_body:
                header = _PRIVATE_KEY_HEADER.search(data, position)
                if header is None:
                    return False
                position = header.end()
                self._awaiting_body = True
            position = _PEM_WHITESPACE.match(data, position).end()
            if position == len(data):
                return False
            # OpenSSH binaries contain NUL-terminated parser constants. A body
            # byte is required, but its distance from the header is unbounded.
            if _PEM_BODY_START.match(data, position):
                return True
            self._awaiting_body = False
        return False


_SEEK, _GAP_BEFORE_SEP, _GAP_AFTER_SEP, _VALUE = range(4)
_VALUE_LIMIT = 512
_SHELL_REFERENCE = re.compile(
    rb"""(?P<quote>"?)(?:\$[A-Za-z_][A-Za-z0-9_]*|"""
    rb"""\$\{[A-Za-z_][A-Za-z0-9_]*(?::-)?\})(?P=quote)\Z"""
)
_PLACEHOLDER = re.compile(rb"<[A-Za-z_][A-Za-z0-9_]*>\Z")
_COMPACT_TOKEN = re.compile(rb"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*){1,2}\Z")
_CREDENTIAL_LOOKUP_NAMES = frozenset(
    {
        "aws_secret_access_key",
        "hf_token",
        "ngc_api_key",
        "hugging_face_hub_token",
        "huggingface_hub_token",
        "hf_access_token",
    }
)
_REFERENCE_NODES = (
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Call,
    ast.keyword,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.BoolOp,
    ast.And,
    ast.Or,
)
_ANNOTATION_NAMES = frozenset(
    {
        "str",
        "bytes",
        "bool",
        "int",
        "float",
        "list",
        "dict",
        "tuple",
        "set",
        "Any",
        "Optional",
        "Union",
        "Mapping",
        "Sequence",
        "Literal",
    }
)


def _compact_token_literal(data: bytes) -> bool:
    """Keep encoded JSON headers from becoming qualified Python references."""
    if not _COMPACT_TOKEN.fullmatch(data):
        return False
    header = data.split(b".", 1)[0]
    # An incomplete final base64 quantum must not hide a recognizable header.
    # This detects a token shape; it does not authenticate or approve a JWT.
    if len(header) % 4 == 1:
        header = header[:-1]
    try:
        decoded = base64.b64decode(
            header + b"=" * (-len(header) % 4), altchars=b"-_", validate=True
        )
    except ValueError:
        return False
    return decoded.lstrip().startswith(b"{")


def _source_reference(expression: ast.expr | None) -> bool:
    """Accept source references only when their literals cannot contain a value."""
    if expression is None or isinstance(expression, (ast.Name, ast.Constant)):
        return False
    literal_bytes = 0
    for node in ast.walk(expression):
        if not isinstance(node, _REFERENCE_NODES):
            return False
        if isinstance(node, ast.Attribute) and _compact_token_literal(
            ast.unparse(node).encode()
        ):
            return False
        if (
            isinstance(node, ast.keyword)
            and node.arg
            and node.arg.casefold() in _CREDENTIAL_LOOKUP_NAMES
        ):
            # Do not consume a nested named assignment as part of an outer
            # reference: it needs its own literal/source verdict.
            return False
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, bool):
                continue
            if isinstance(node.value, str):
                if node.value.casefold() in _CREDENTIAL_LOOKUP_NAMES:
                    continue
                literal_bytes += len(node.value.encode())
            else:
                literal_bytes += len(str(node.value))
    return literal_bytes < 8


def _source_annotation(annotation: ast.expr) -> bool:
    """Recognize finite builtin typing syntax rather than arbitrary bare data."""
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id not in _ANNOTATION_NAMES | {"typing"}:
            return False
        if isinstance(node, ast.Attribute) and node.attr not in _ANNOTATION_NAMES:
            return False
        if isinstance(node, ast.Constant) and node.value is not None:
            return False
        if not isinstance(
            node,
            (
                ast.Name,
                ast.Load,
                ast.Attribute,
                ast.Subscript,
                ast.Tuple,
                ast.Constant,
                ast.BinOp,
                ast.BitOr,
            ),
        ):
            return False
    return True


def _annotated_reference(source: str) -> bool:
    """Accept known annotations with no literal credential default."""
    statement = ast.parse("value: " + source).body[0]
    if not isinstance(statement, ast.AnnAssign):
        return False
    if not _source_annotation(statement.annotation):
        return False
    default = statement.value
    if default is None:
        return True
    if isinstance(default, ast.Constant) and default.value in (None, "", False, True):
        return True
    return _source_reference(default)


def _nonliteral_value(data: bytes, separator: bytes) -> bool:
    """Keep literals, unknown expressions and ambiguous syntax blocking."""
    candidate = data.strip().rstrip(b",;)] ").strip()
    if _compact_token_literal(candidate):
        return False
    if _SHELL_REFERENCE.fullmatch(candidate) or _PLACEHOLDER.fullmatch(candidate):
        return True
    if b"#" in data:
        return False
    # External argument closers are not part of the assigned expression. Try
    # the complete candidate first so valid subscription/call closers survive.
    candidate = data.strip().rstrip(b",; ")
    for _ in range(5):
        try:
            source = candidate.decode("utf-8")
            if separator == b":":
                if _annotated_reference(source):
                    return True
                expression = None
            else:
                expression = ast.parse(source, mode="eval").body
        except (UnicodeDecodeError, SyntaxError, RecursionError):
            expression = None
        if _source_reference(expression):
            return True
        if not candidate.endswith((b")", b"]", b"}")):
            return False
        candidate = candidate[:-1].rstrip()
    return False


class _AssignmentValue:
    """Retain a bounded source candidate, never an unbounded value or gap."""

    def __init__(self, separator: bytes) -> None:
        self.separator = separator
        self.data = bytearray()
        self.quote = 0
        self.escaped = False
        self.depth = 0
        self.seen = 0
        self.run = 0

    def feed(self, data: bytes, position: int) -> tuple[int, bool]:
        """Read through a complete expression or fail closed at the size bound."""
        while position < len(data):
            byte = data[position]
            if byte in b"\r\n" and not self.quote and not self.depth:
                return position, True
            self.data.append(byte)
            self.run = 0 if byte in b" \t\r\n\f\v" else self.run + 1
            self.seen = max(self.seen, self.run)
            position += 1
            if len(self.data) > _VALUE_LIMIT:
                return position, True
            if self.quote:
                self._quoted_byte(byte)
            elif byte in b"\"'":
                self.quote = byte
            elif byte in b"([{":
                self.depth += 1
            elif byte in b")]}" and self.depth:
                self.depth -= 1
            elif byte in b",;)]}" and not self.depth:
                return position, True
        return position, False

    def _quoted_byte(self, byte: int) -> None:
        if self.escaped:
            self.escaped = False
        elif byte == ord("\\"):
            self.escaped = True
        elif byte == self.quote:
            self.quote = 0

    def credential(self, minimum: int) -> bool:
        """Unknown, literal and oversized values retain the credential verdict."""
        if len(self.data) > _VALUE_LIMIT:
            return True
        if self.seen < minimum:
            return False
        return not _nonliteral_value(bytes(self.data), self.separator)


class _AssignmentMatcher:
    """Resume unbounded label gaps and a bounded literal/source value check."""

    def __init__(self, rule: AssignmentRule) -> None:
        self._rule = rule
        self._phase = _SEEK
        self._value: _AssignmentValue | None = None

    def resume_at(self, carried: int) -> int:
        """Where in the next window this matcher should resume.

        Mid-assignment the matcher must continue exactly where the previous
        window ended. Still seeking, it has to look back over the carried bytes,
        or a NAME split across the boundary would be missed: its first half was
        an incomplete match last time and its second half alone matches nothing.
        """

        if self._phase != _SEEK:
            return carried
        return max(0, carried - _NAME_CARRY)

    def _label(self, data: bytes, position: int) -> int:
        """Consume a name or separator without retaining whitespace gaps."""
        if self._phase == _SEEK:
            found = self._rule.names.search(data, position)
            if found is None:
                return len(data)
            self._phase = _GAP_BEFORE_SEP
            return found.end()
        position = _WHITESPACE.match(data, position).end()
        if position == len(data):
            return position
        if self._phase == _GAP_AFTER_SEP:
            self._phase = _VALUE
        elif data[position] in self._rule.separators:
            self._value = _AssignmentValue(data[position : position + 1])
            self._phase = _GAP_AFTER_SEP
            position += 1
        else:
            # Revisit this position so an overlapping NAME is not skipped.
            self._phase = _SEEK
        return position

    def feed(self, data: bytes, start: int) -> bool:
        """Consume ``data`` from ``start``; return True if the rule matched."""
        position = start
        while position < len(data):
            if self._phase != _VALUE:
                position = self._label(data, position)
                continue
            assert self._value is not None
            position, complete = self._value.feed(data, position)
            if not complete:
                return False
            if self._value.credential(self._rule.min_value):
                return True
            self._phase = _SEEK
        return False

    def finish(self) -> bool:
        """Classify the final value when EOF has no following delimiter."""
        return bool(
            self._value
            and self._phase == _VALUE
            and self._value.credential(self._rule.min_value)
        )


def path_credential(name: str) -> str | None:
    """Return the credential kind for a member path, or None."""

    normalised = normalise_member_name(name)
    for kind, pattern in CREDENTIAL_PATHS:
        if pattern.search(normalised):
            return kind
    return None


def content_credential(stream: IO[bytes]) -> str | None:
    """Return the credential kind found in a member's bytes, or None.

    Reads the member in full, in fixed-size chunks. Peak memory is one chunk
    plus a small overlap regardless of member size or match length. Short reads
    are handled: only an empty read ends the scan, because a stream may
    legitimately return fewer bytes than requested.

    Args:
        stream: Binary member stream to inspect from its current position.
    Returns:
        The first detected credential kind, or None when no rule matches.
    Raises:
        OSError: If reading the member fails.
    """

    matchers = [(rule.kind, _AssignmentMatcher(rule)) for rule in ASSIGNMENT_RULES]
    private_key = _PrivateKeyMatcher()
    carry = b""
    while True:
        chunk = stream.read(CONTENT_CHUNK)
        if not chunk:
            for kind, matcher in matchers:
                if matcher.finish():
                    return kind
            return None
        window = carry + chunk
        if private_key.feed(window, len(carry)):
            return "private_key_content"
        for kind, pattern in MARKER_CONTENT:
            if pattern.search(window):
                return kind
        for kind, matcher in matchers:
            if matcher.feed(window, matcher.resume_at(len(carry))):
                return kind
        carry = window[-CARRY:]
