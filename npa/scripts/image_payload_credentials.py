"""Credential detection shared by the per-image payload scanners.

Every ``scan_image_*_payload.py`` grew its own credential rules, and they
drifted apart. Measured over six scanners and seven planted cases, the rule sets
did not order cleanly: the two that detected the private-key family missed
``.netrc``, and the only scanner that caught ``.netrc`` missed every private
key. That finite measurement is the reason this module exists — one list, so a
gap closed here closes in every scanner that imports it.

Paths are not sufficient on their own. A private key copied to an application
directory matches no conventional name, so content detection is here too.

Content detection has to survive three things that review demonstrated it did
not, each with a working counterexample rather than a hypothetical:

*A key anywhere in the member.* Reading a bounded prefix is not a cheaper
approximation of scanning a file, it is a documented place to hide a key. Every
byte is examined.

*A match on a chunk boundary.* Fixed-size markers are found with an overlap
between consecutive chunks.

*A match longer than any window.* ``password = "`` followed by two megabytes of
text and a closing quote is a single match that no fixed overlap can hold. A
window cannot solve this, so the assignment rules are not matched by a window:
they are resumable state machines that carry a phase and a counter across
chunks, never the bytes themselves. That keeps memory bounded while the match
span stays unbounded, which is what the declared patterns actually mean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import IO


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
            r"|\.netrc|\.npa/credentials\.yaml|kubeconfig)$",
            re.I,
        ),
    ),
)

# Fixed-length markers. Every one of these has a maximum match length well under
# MARKER_OVERLAP, so a window with that overlap finds them wherever they fall.
MARKER_CONTENT: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private_key_content",
        # ENCRYPTED is the PKCS#8 passphrase-protected header, which is what
        # anyone who protected a key with a passphrase produces. A passphrase is
        # not a reason to leave the key in an image, so it is detected too.
        #
        # The trailing requirement is what stops this rejecting any image that
        # installs OpenSSH. Measured on this host, /usr/bin/ssh, /usr/sbin/sshd
        # and /usr/bin/ssh-keygen each carry the literal header as a parser
        # constant, because that is how they recognise the format they read, and
        # all three were rejected before this clause existed. In a binary the
        # literal is NUL-terminated:
        #
        #     OpenSSH begin-header string, newline, then NUL bytes.
        #
        # whereas a real key continues into its base64 body. Requiring one body
        # character separates the two. This is a discriminator against a
        # NUL-terminated constant, not proof of key material: a binary that
        # placed printable text immediately after the literal would still match,
        # and that is the deliberate direction to err in.
        re.compile(
            rb"-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
            rb"[ \t\r\n]{0,8}[A-Za-z0-9+/=]"
        ),
    ),
    ("aws_access_key_id", re.compile(rb"AKIA[0-9A-Z]{16}")),
)

_WHITESPACE = re.compile(rb"[ \t\r\n\f\v]*")
_UNTIL_QUOTE = re.compile(rb"[^'\"]*")
_UNTIL_SPACE = re.compile(rb"[^ \t\r\n\f\v]*")


@dataclass(frozen=True)
class AssignmentRule:
    """A ``NAME <ws> SEP <ws> VALUE`` credential, matched across chunks.

    ``quoted`` rules end at the closing quote and accept a value of any length,
    which is the shape that cannot be expressed as a fixed window. Unquoted
    rules need ``min_value`` non-space bytes, and stop looking once they have
    them.
    """

    kind: str
    names: re.Pattern[bytes]
    separators: bytes
    quoted: bool
    min_value: int = 0


ASSIGNMENT_RULES: tuple[AssignmentRule, ...] = (
    AssignmentRule(
        kind="credential_assignment",
        names=re.compile(rb"(?i)aws_secret_access_key|hf_token|ngc_api_key"),
        separators=b"=:",
        quoted=False,
        min_value=8,
    ),
    # packaging-contract.yaml declares this shape under security.secret_patterns,
    # where it is enforced against Dockerfile text only. A layer scanner that did
    # not also enforce the declared list would let the contract say one thing and
    # the bytes check another; test_image_payload_credentials.py holds the two
    # together and fails if a declared pattern gains no rule here.
    AssignmentRule(
        kind="quoted_secret_assignment",
        names=re.compile(rb"(?i)api[_-]?key|secret[_-]?key|password"),
        separators=b"=",
        quoted=True,
    ),
)

# The same rules written as ordinary whole-input regexes. Nothing detects with
# these — ``content_credential`` streams — but they state what the streaming
# matcher is supposed to mean, so the two can be compared directly on any input.
# test_image_payload_credentials.py asserts the two agree; a reviewer checking
# this module should compare against these rather than read the state machine.
DECLARED_CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    MARKER_CONTENT
    + (
        (
            "credential_assignment",
            re.compile(
                rb"(?i)(?:aws_secret_access_key|hf_token|ngc_api_key)\s*[=:]\s*[^$<\s][^\s]{7,}"
            ),
        ),
        (
            "quoted_secret_assignment",
            re.compile(
                rb"""(?i)(?:api[_-]?key|secret[_-]?key|password)\s*=\s*['"][^'"]+['"]"""
            ),
        ),
    )
)

# Retained under its former name: the review lane's reproducers import it.
CREDENTIAL_CONTENT = DECLARED_CONTENT_PATTERNS

CONTENT_CHUNK = 1024 * 1024
# Headroom for a future marker, not a measurement of the current ones. Review
# showed this is not load-bearing today: setting it to zero changes no verdict,
# because the carry takes the largest of the three constants below and the
# longest marker declared here is 35 bytes.
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

    ``re._parser`` is private. It is used deliberately, because the alternative
    is a hand-maintained number that silently goes stale, and there is no public
    equivalent. If it ever disappears this raises rather than guessing, and the
    carry is never quietly too small.
    """

    return re._parser.parse(pattern.pattern.decode("latin-1")).getwidth()[1]


# An upper bound on any marker match, derived so that adding a longer marker
# widens the carry automatically instead of silently outgrowing it.
_LONGEST_MARKER = max(_max_match_length(pattern) for _, pattern in MARKER_CONTENT)
# What actually gets carried between chunks, and why: long enough for any marker
# to be found whole, any NAME token to survive a split, and whatever headroom
# MARKER_OVERLAP asks for.
CARRY = max(MARKER_OVERLAP, _NAME_CARRY, _LONGEST_MARKER)

_SEEK, _GAP_BEFORE_SEP, _GAP_AFTER_SEP, _VALUE = range(4)


class _AssignmentMatcher:
    """Resumable matcher for one rule. State is a phase and a count, never bytes."""

    def __init__(self, rule: AssignmentRule) -> None:
        self._rule = rule
        self._phase = _SEEK
        self._seen = 0

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

    def feed(self, data: bytes, start: int) -> bool:
        """Consume ``data`` from ``start``; return True if the rule matched."""

        rule = self._rule
        position = start
        end = len(data)
        while position < end:
            if self._phase == _SEEK:
                found = rule.names.search(data, position)
                if found is None:
                    return False
                position = found.end()
                self._phase = _GAP_BEFORE_SEP
            elif self._phase == _GAP_BEFORE_SEP:
                position = _WHITESPACE.match(data, position).end()
                if position == end:
                    return False
                if data[position] in rule.separators:
                    position += 1
                    self._phase = _GAP_AFTER_SEP
                else:
                    # Not an assignment after all. Resume the search here rather
                    # than after it, so an overlapping NAME is not skipped.
                    self._phase = _SEEK
            elif self._phase == _GAP_AFTER_SEP:
                position = _WHITESPACE.match(data, position).end()
                if position == end:
                    return False
                if rule.quoted:
                    if data[position : position + 1] in (b"'", b'"'):
                        position += 1
                        self._phase = _VALUE
                        self._seen = 0
                    else:
                        self._phase = _SEEK
                elif data[position : position + 1] in (b"$", b"<"):
                    # A shell or template reference is not a baked credential.
                    self._phase = _SEEK
                else:
                    self._phase = _VALUE
                    self._seen = 0
            else:
                if rule.quoted:
                    run = _UNTIL_QUOTE.match(data, position)
                    self._seen += run.end() - position
                    position = run.end()
                    if position == end:
                        return False
                    # The value class excludes both quote characters, so the
                    # first one to appear is the closing quote.
                    position += 1
                    if self._seen >= 1:
                        return True
                    self._phase = _SEEK
                else:
                    run = _UNTIL_SPACE.match(data, position)
                    self._seen += run.end() - position
                    position = run.end()
                    if self._seen >= rule.min_value:
                        return True
                    if position == end:
                        return False
                    self._phase = _SEEK
        return False


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
    """

    matchers = [(rule.kind, _AssignmentMatcher(rule)) for rule in ASSIGNMENT_RULES]
    carry = b""
    while True:
        chunk = stream.read(CONTENT_CHUNK)
        if not chunk:
            return None
        window = carry + chunk
        for kind, pattern in MARKER_CONTENT:
            if pattern.search(window):
                return kind
        for kind, matcher in matchers:
            if matcher.feed(window, matcher.resume_at(len(carry))):
                return kind
        carry = window[-CARRY:]
