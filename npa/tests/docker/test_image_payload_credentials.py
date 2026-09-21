"""Unit guards for the credential rules shared by the payload scanners.

The important test here is the last one. ``packaging-contract.yaml`` declares
``security.secret_patterns``, and until now those patterns were enforced only
against Dockerfile text, never against the bytes in a built layer. That let the
contract state one thing while the layer scanners checked another. The coverage
test binds the two together, so a pattern added to the contract fails here until
the shared rules detect it too.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "image_payload_credentials.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "image_payload_credentials", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
credentials = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = credentials
_SPEC.loader.exec_module(credentials)

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "workbench"
    / "packaging-contract.yaml"
)

SYNTHETIC_KEY = (
    b"-----BEGIN OPENSSH PRIVATE KEY-----\n"
    b"NOT-A-REAL-KEY-SYNTHETIC-CONTROL-FIXTURE-ONLY\n"
    b"-----END OPENSSH PRIVATE KEY-----\n"
)

# One sample per declared contract pattern. Keyed by the pattern itself so that
# adding a pattern to the contract without a sample fails the coverage test
# rather than silently going unchecked.
CONTRACT_SAMPLES: dict[str, bytes] = {
    "(?i)(api[_-]?key|secret[_-]?key|password)\\s*=\\s*['\\\"][^'\\\"]+['\\\"]": (
        b'password = "hunter2-synthetic"\n'
    ),
    "(?i)BEGIN (RSA |OPENSSH )?PRIVATE KEY": SYNTHETIC_KEY,
}


@pytest.mark.parametrize(
    "path,kind",
    [
        ("etc/ssh/ssh_host_ed25519_key", "ssh_host_key"),
        ("./etc/ssh/ssh_host_rsa_key", "ssh_host_key"),
        ("root/.ssh/id_rsa", "ssh_user_key"),
        ("home/ubuntu/.ssh/id_ed25519", "ssh_user_key"),
        ("root/.aws/credentials", "cloud_credential_file"),
        ("root/.netrc", "cloud_credential_file"),
        # Archive entries at the image root. lstrip("./") strips characters
        # rather than a prefix, so it turned these into "netrc" and "ssh/..."
        # and the rules stopped matching. Review found this one.
        (".netrc", "cloud_credential_file"),
        ("./.netrc", "cloud_credential_file"),
        ("/.netrc", "cloud_credential_file"),
        ("././.netrc", "cloud_credential_file"),
        ("./.ssh/id_rsa", "ssh_user_key"),
        ("./etc/ssh/ssh_host_rsa_key", "ssh_host_key"),
    ],
)
def test_path_credential_names_the_kind(path: str, kind: str) -> None:
    assert credentials.path_credential(path) == kind


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("./etc/ssh/sshd_config", "etc/ssh/sshd_config"),
        ("./.netrc", ".netrc"),
        ("/opt/app/main.py", "opt/app/main.py"),
        ("opt/app/main.py", "opt/app/main.py"),
        ("opt/.hidden/file", "opt/.hidden/file"),
        # Interior aliases resolve. These name the same file as the canonical
        # spelling, and before real normalisation each one slipped past a rule
        # whose matched span they interrupted.
        ("root/.docker/./config.json", "root/.docker/config.json"),
        ("root/.docker//config.json", "root/.docker/config.json"),
        ("root/.docker/./././config.json", "root/.docker/config.json"),
        ("root/.docker/sub/../config.json", "root/.docker/config.json"),
        ("root/./.docker/config.json", "root/.docker/config.json"),
        ("root/../root/.docker/config.json", "root/.docker/config.json"),
        ("root/etc/./ssh/ssh_host_rsa_key", "root/etc/ssh/ssh_host_rsa_key"),
        # A ".." with nothing to cancel is kept. Discarding it would rewrite a
        # name that escapes the archive root into an innocuous-looking one.
        ("../etc/passwd", "../etc/passwd"),
        ("../../etc/passwd", "../../etc/passwd"),
        ("../.docker/config.json", "../.docker/config.json"),
        # Dots that are part of a name are not separators and do not resolve.
        ("root/my..file.json", "root/my..file.json"),
        ("root/a..b/notes.txt", "root/a..b/notes.txt"),
    ],
)
def test_normalise_member_name_resolves_the_path(spelling: str, expected: str) -> None:
    assert credentials.normalise_member_name(spelling) == expected


@pytest.mark.parametrize(
    "alias",
    [
        "root/.docker/./config.json",
        "root/.docker//config.json",
        "root/.docker/./././config.json",
        "root/.docker/sub/../config.json",
        "root/.ssh/./id_rsa",
        "root/.ssh/x/../id_rsa",
        "root/.aws/./credentials",
        "root/etc/ssh/./ssh_host_rsa_key",
        "root/etc/./ssh/ssh_host_rsa_key",
        "root/etc/ssh//ssh_host_rsa_key",
    ],
)
def test_aliased_spellings_match_the_same_rule_as_the_canonical_path(
    alias: str,
) -> None:
    # The rules are substring matches, so an alias inserted inside a rule's
    # matched span used to defeat it while naming the identical file. Asserting
    # only "is detected" would pass if a fix over-normalised everything into one
    # bucket, so this pins the kind against the canonical spelling.
    canonical = credentials.normalise_member_name(alias)
    assert credentials.path_credential(canonical) is not None, canonical
    assert credentials.path_credential(alias) == credentials.path_credential(canonical)


@pytest.mark.parametrize(
    "benign",
    [
        "root/.docker/README.md",  # non-credential file in a credential dir
        "root/my..file.json",  # legitimate dots in a filename
        "root/a..b/notes.txt",  # legitimate dots in a directory
        "root/docker/config.json",  # no leading dot, not ~/.docker
        "opt/app/main.py",
    ],
)
def test_normalisation_does_not_turn_benign_paths_into_credentials(
    benign: str,
) -> None:
    # The other half of the alias fix. Resolving "." and ".." must not be
    # over-applied into matching things that were correctly accepted before.
    assert credentials.path_credential(benign) is None


@pytest.mark.parametrize(
    "path",
    [
        "opt/app/inference.py",
        "usr/lib/libcudart.so",
        # Near misses: a directory of that name, and a public key, are not
        # credentials. Without these the path rules could be wildly over-broad
        # and every other test here would still pass.
        "root/.ssh/id_rsa.pub",
        "etc/ssh/ssh_host_ed25519_key.pub",
    ],
)
def test_path_credential_ignores_non_credentials(path: str) -> None:
    assert credentials.path_credential(path) is None


def test_content_credential_finds_a_marker_split_across_chunks() -> None:
    # The reason the reader carries an overlap. Place the marker so it straddles
    # a chunk boundary; without the overlap this returns None.
    marker = b"-----BEGIN OPENSSH PRIVATE KEY-----\nc3ludGhldGlj"
    head = b"\x00" * (credentials.CONTENT_CHUNK - 10)
    stream = io.BytesIO(head + marker + b"\x00" * 64)
    assert credentials.content_credential(stream) == "private_key_content"


@pytest.mark.parametrize("offset", [0, 262145, credentials.CONTENT_CHUNK * 2 + 7])
def test_content_credential_reads_the_whole_member(offset: int) -> None:
    # An earlier revision read only a 256 KiB prefix, which review demonstrated
    # was a documented place to put a key: the identical marker was rejected at
    # byte 0 and accepted at 262145. Offsets here sit before the cap, just past
    # it, and past two whole chunks.
    stream = io.BytesIO(b"\x00" * offset + SYNTHETIC_KEY)
    assert credentials.content_credential(stream) == "private_key_content"


def test_content_credential_tolerates_short_reads() -> None:
    # read(n) may return fewer than n bytes without being at EOF. Only an empty
    # read ends the scan; a reader that stopped on a short one would miss
    # everything after the first partial chunk.
    class Trickle(io.RawIOBase):
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self._at = 0

        def read(self, size: int = -1) -> bytes:  # noqa: D102
            chunk = self._payload[self._at : self._at + 3]
            self._at += len(chunk)
            return chunk

    stream = Trickle(b"\x00" * 5000 + SYNTHETIC_KEY)
    assert credentials.content_credential(stream) == "private_key_content"


def test_the_carry_covers_every_fixed_marker() -> None:
    # Only the fixed-length markers are found with a window, so only they need
    # to fit inside the carry. The assignment rules deliberately do not: they
    # stream, which is why they can match spans no window could hold.
    #
    # Assert CARRY, not MARKER_OVERLAP. Review showed MARKER_OVERLAP is not
    # load-bearing today — zeroing it changes no verdict, because _NAME_CARRY is
    # the effective floor — so a test naming it would claim more than it checks.
    longest = max(
        credentials._max_match_length(pattern)
        for _, pattern in credentials.MARKER_CONTENT
    )
    assert credentials.CARRY >= longest


@pytest.mark.parametrize(
    "pattern,expected",
    [
        (rb"AKIA[0-9A-Z]{16}", 20),
        (rb"-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----", 37),
        (rb"abc", 3),
        (rb"(?:ab|cdef)", 4),
    ],
)
def test_max_match_length_measures_the_match_not_the_source(
    pattern: bytes, expected: int
) -> None:
    # The first case is why this function exists. An earlier revision used
    # len(pattern.pattern), and review caught that AKIA[0-9A-Z]{16} is 16 source
    # characters matching 20 bytes — an under-estimate presented as a bound.
    #
    # The expected values are written out rather than recomputed from the same
    # expression the implementation uses, because a test that reuses the
    # implementation cannot catch the case it exists for.
    assert credentials._max_match_length(re.compile(pattern)) == expected


@pytest.mark.parametrize(
    "header",
    [
        b"RSA PRIVATE KEY",
        b"DSA PRIVATE KEY",
        b"EC PRIVATE KEY",
        b"OPENSSH PRIVATE KEY",
        b"PRIVATE KEY",
        # PKCS#8 passphrase-protected, which is what anyone who protected a key
        # with a passphrase produces. A passphrase is not a reason to leave the
        # key in an image.
        b"ENCRYPTED PRIVATE KEY",
    ],
)
def test_every_private_key_header_is_detected(header: bytes) -> None:
    body = (
        b"-----BEGIN "
        + header
        + b"-----\nc3ludGhldGlj\n-----END "
        + header
        + b"-----\n"
    )
    assert credentials.content_credential(io.BytesIO(body)) == "private_key_content"


@pytest.mark.parametrize(
    "trailer",
    [
        # How the header appears in an OpenSSH binary: a NUL-terminated parser
        # constant, immediately followed by the END constant. Taken from the
        # real byte layout in /usr/bin/ssh, /usr/sbin/sshd and
        # /usr/bin/ssh-keygen, each of which was rejected before the rule
        # required a body character. Any image that installs OpenSSH hits this.
        b"\n\x00\x00\x00\x00-----END OPENSSH PRIVATE KEY-----\n\x00\x00",
        b"\x00",
        b"\n\x00",
        b"",
    ],
)
def test_a_nul_terminated_parser_constant_is_not_a_key(trailer: bytes) -> None:
    body = b"\x7fELF\x02\x01\x01" + b"\x00" * 64
    body += b"-----BEGIN OPENSSH PRIVATE KEY-----" + trailer
    assert credentials.content_credential(io.BytesIO(body)) is None


@pytest.mark.parametrize(
    "key",
    [
        # The two lane fixtures. Both reproducer suites plant these, so the
        # false-positive fix must not turn either into a miss.
        b"-----BEGIN OPENSSH PRIVATE KEY-----\nNOT-A-KEY-SYNTHETIC-FIXTURE\n"
        b"-----END OPENSSH PRIVATE KEY-----\n",
        SYNTHETIC_KEY,
        # Traditional encrypted PEM, whose body is preceded by RFC 1421 headers
        # rather than starting with base64 straight away.
        b"-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\n"
        b"DEK-Info: AES-128-CBC,0123456789ABCDEF\n\nMIIEow==\n"
        b"-----END RSA PRIVATE KEY-----\n",
    ],
)
def test_real_key_shapes_survive_the_parser_string_fix(key: bytes) -> None:
    assert credentials.content_credential(io.BytesIO(key)) == "private_key_content"


def test_the_longest_marker_is_measured_correctly() -> None:
    # Pins the derived constant to a number a reader can verify by counting:
    # "-----BEGIN " is 11, "ENCRYPTED " is 10, "PRIVATE KEY-----" is 16.
    assert credentials._LONGEST_MARKER == 46


def test_markers_are_found_when_split_at_a_chunk_boundary() -> None:
    # The property the carry exists for, exercised at the offsets where a marker
    # actually straddles a chunk edge rather than only in the middle of one.
    marker = b"-----BEGIN OPENSSH PRIVATE KEY-----\nc3ludGhldGlj"
    for shift in range(-3, 4):
        offset = credentials.CONTENT_CHUNK - len(marker) // 2 + shift
        stream = io.BytesIO(b"\x00" * offset + marker + b"\x00" * 32)
        assert credentials.content_credential(stream) == "private_key_content", shift


CHUNK = credentials.CONTENT_CHUNK


@pytest.mark.parametrize(
    "sample,kind",
    [
        # Review's counterexample: a quoted value two chunks long. The whole
        # declared regex matches it, and an overlap-window reader returned None.
        (b'password = "' + b"a" * (CHUNK * 2) + b'"', "quoted_secret_assignment"),
        (b'api_key   =   "' + b"b" * (CHUNK * 3) + b'"', "quoted_secret_assignment"),
        # A whitespace run longer than any window, which is the earlier
        # counterexample. It is matched now rather than defined away.
        (
            b"aws_secret_access_key" + b" " * (CHUNK + 5) + b"=abcdefgh",
            "credential_assignment",
        ),
        # The name itself straddling a chunk boundary: its first half is an
        # incomplete match and its second half alone matches nothing.
        (b"\x00" * (CHUNK - 5) + b'password = "x"', "quoted_secret_assignment"),
        (b'password = "hunter2-synthetic"', "quoted_secret_assignment"),
        (b"aws_secret_access_key=abcdefgh", "credential_assignment"),
        (b"hf_token: abcdefghij", "credential_assignment"),
    ],
)
def test_assignments_match_across_any_number_of_chunks(
    sample: bytes, kind: str
) -> None:
    assert credentials.content_credential(io.BytesIO(sample)) == kind


@pytest.mark.parametrize(
    "sample",
    [
        # Negative controls for the streaming matcher. Without these it could
        # return a kind for almost anything and every test above would pass.
        b"print('hello world')\n" * 1000,
        # No closing quote, so the declared pattern does not match and neither
        # does the reader: faithful rather than eager.
        b'password = "' + b"a" * (CHUNK * 2),
        # A shell or template reference is not a baked credential.
        b"aws_secret_access_key=$SECRET_VALUE",
        b"aws_secret_access_key=<REPLACE_ME>",
        # Too short to satisfy the declared minimum value length.
        b"hf_token=abc",
        # The name appears, but never as an assignment.
        b"see the password policy document for details",
    ],
)
def test_streaming_matcher_has_negative_controls(sample: bytes) -> None:
    assert credentials.content_credential(io.BytesIO(sample)) is None


def test_streaming_state_is_bounded_not_buffered() -> None:
    # The point of the state machine: a match spanning megabytes must not mean
    # holding megabytes. The matcher's own state is a phase and a counter.
    matcher = credentials._AssignmentMatcher(credentials.ASSIGNMENT_RULES[1])
    matcher.feed(b'password = "' + b"a" * 4096, 0)
    assert matcher.__dict__.keys() == {"_rule", "_phase", "_seen"}
    assert isinstance(matcher._seen, int)


def test_shared_rules_cover_every_declared_contract_secret_pattern() -> None:
    declared = yaml.safe_load(CONTRACT.read_text())["security"]["secret_patterns"]
    assert set(declared) == set(CONTRACT_SAMPLES), (
        "packaging-contract.yaml security.secret_patterns changed. Add a sample "
        "and, if needed, a rule in image_payload_credentials.py: a declared "
        "pattern that the layer scanners do not enforce is the gap this test exists "
        "to prevent."
    )
    for pattern, sample in CONTRACT_SAMPLES.items():
        assert re.search(pattern.encode(), sample), (
            f"sample does not match its own declared pattern {pattern}"
        )
        assert credentials.content_credential(io.BytesIO(sample)) is not None, (
            f"declared contract pattern is not enforced against layer bytes: {pattern}"
        )


SCANNERS_USING_SHARED_RULES = (
    "scan_image_alpamayo2_payload.py",
    "scan_image_cosmos3_ray_serve_payload.py",
    "scan_image_cosmos3_serving_payload.py",
)


def _single_layer_archive(path: Path, member: str, payload: bytes) -> Path:
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    layer.seek(0)
    manifest = [{"Config": "config.json", "RepoTags": [], "Layers": ["layer.tar"]}]
    with tarfile.open(path, mode="w") as archive:
        for name, body in (
            ("manifest.json", json.dumps(manifest).encode()),
            ("config.json", json.dumps({"config": {}, "history": []}).encode()),
            ("layer.tar", layer.read()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return path


@pytest.mark.parametrize("scanner", SCANNERS_USING_SHARED_RULES)
def test_scanner_runs_standalone_from_an_unrelated_directory(
    tmp_path: Path, scanner: str
) -> None:
    # These scanners are invoked as scripts by absolute path, from build.sh and
    # from publish_public.py, with the working directory set to anything. An
    # in-process importlib test would not notice a sibling import that only
    # resolves because pytest's rootdir happens to be on sys.path, so run the
    # real command line in a subprocess from an unrelated cwd.
    archive = _single_layer_archive(
        tmp_path / "image.tar", "root/.ssh/id_rsa", SYNTHETIC_KEY
    )
    completed = subprocess.run(
        [sys.executable, str(_MODULE_PATH.parent / scanner), "--tarball", str(archive)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1, (
        f"{scanner} did not reject a planted key when run standalone: "
        f"{completed.stderr[-400:]}"
    )


@pytest.mark.parametrize("scanner", SCANNERS_USING_SHARED_RULES)
def test_scanner_declares_its_sibling_dependency(scanner: str) -> None:
    # A delivery path that copies one scanner without this module would fail at
    # import. Keep the dependency explicit and co-located: the sys.path insert
    # is what makes the absolute-path invocation work, and it is the same
    # pattern scan_image_ltx_payload.py already uses to import wan.
    text = (_MODULE_PATH.parent / scanner).read_text()
    assert "image_payload_credentials" in text
    assert "sys.path.insert(0, str(Path(__file__).resolve().parent))" in text
    assert _MODULE_PATH.is_file(), "the shared module must sit beside the scanners"


EQUIVALENCE_CORPUS = (
    b'password = "' + b"a" * (CHUNK * 2) + b'"',
    b'password = "' + b"a" * (CHUNK * 2),
    b'api_key\t=\t"x"',
    b"aws_secret_access_key" + b" " * (CHUNK + 5) + b"=abcdefgh",
    b"aws_secret_access_key=abcdefgh",
    b"aws_secret_access_key=$SECRET_VALUE",
    b"aws_secret_access_key=<REPLACE_ME>",
    b"hf_token=abc",
    b"hf_token: abcdefghij",
    b"ngc_api_key   :   abcdefghij",
    b"see the password policy document for details",
    b"print('hello world')\n" * 500,
    b"\x00" * (CHUNK - 5) + b'password = "x"',
    b"-----BEGIN OPENSSH PRIVATE KEY-----\nc3ludGhldGlj",
    b"\x00" * 262145 + b"-----BEGIN RSA PRIVATE KEY-----\nc3ludGhldGlj",
    b"AKIA" + b"A" * 16,
    b"AKIA" + b"a" * 16,
    b"",
)


@pytest.mark.parametrize(
    "sample", EQUIVALENCE_CORPUS, ids=range(len(EQUIVALENCE_CORPUS))
)
def test_streaming_agrees_with_the_declared_patterns(sample: bytes) -> None:
    # The property that matters, and the one an overlap window silently broke:
    # streaming over chunks must decide exactly what the declared patterns
    # decide on the whole input. Review's counterexample was a case where the
    # declared pattern matched and the reader returned None, so assert the
    # equivalence directly instead of testing the reader against itself.
    declared = any(
        pattern.search(sample) for _, pattern in credentials.DECLARED_CONTENT_PATTERNS
    )
    streamed = credentials.content_credential(io.BytesIO(sample)) is not None
    assert streamed == declared, (
        f"declared={declared} streamed={streamed} for a {len(sample)}-byte sample"
    )


def test_the_corpus_contains_both_verdicts() -> None:
    # Guards the test above: a corpus that was all-matching or all-clean would
    # make the equivalence assertion vacuous.
    verdicts = {
        any(pattern.search(s) for _, pattern in credentials.DECLARED_CONTENT_PATTERNS)
        for s in EQUIVALENCE_CORPUS
    }
    assert verdicts == {True, False}
