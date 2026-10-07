"""Option declarations require safe defaults, decoded literals and source context."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_image_payload_assignment_source import _synthetic_compact_token
from test_image_payload_credentials import (
    SCANNERS_USING_SHARED_RULES,
    SYNTHETIC_KEY,
    _MODULE_PATH,
    _single_layer_archive,
    credentials,
)


def _parameter(option: bytes, *, annotation: bytes = b"str") -> bytes:
    return (
        b"import typer\n\ndef command(ngc_api_key: "
        + annotation
        + b" = "
        + option
        + b"):\n    pass\n"
    )


def _verdict(payload: bytes, monkeypatch: pytest.MonkeyPatch, chunk: int) -> bool:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", chunk)
    found = bool(credentials.content_credential(io.BytesIO(payload)))
    declared = any(
        pattern.search(payload) for _, pattern in credentials.CREDENTIAL_CONTENT
    )
    assert found == declared
    return found


BENIGN_OPTIONS = (
    b'typer.Option("", "--ngc-api-key", help="Optional key supplied at runtime")',
    b'typer.Option(None, "--ngc-api-key", help="Optional key supplied at runtime")',
    b'typer.Option(False, "--ngc-api-key", show_default=False)',
    b'typer.Option(True, "--ngc-api-key", hide_input=True)',
    b'typer.Option(default=None, help="Optional key supplied at runtime")',
    b'typer.Option(config.ngc_api_key, "--ngc-api-key", help="Runtime field")',
    b'typer.Option(credentials["ngc_api_key"], "--ngc-api-key", help="Runtime field")',
    b'typer.Option(os.getenv("NGC_API_KEY"), "--ngc-api-key", help="Runtime lookup")',
    b'typer.Option(os.environ.get("NGC_API_KEY", ""), "--ngc-api-key", envvar="NGC_API_KEY")',
    b'typer.Option(config.ngc_api_key or None, "--ngc-api-key", show_envvar=False)',
    b'typer.Option(\n    "",\n    "--ngc-api-key",\n    help="Optional key supplied at runtime",\n)',
)


@pytest.mark.parametrize("chunk", [1, 7, 31, 512, 1024])
@pytest.mark.parametrize("option", BENIGN_OPTIONS)
def test_known_option_defaults_do_not_treat_metadata_as_credentials(
    monkeypatch: pytest.MonkeyPatch, option: bytes, chunk: int
) -> None:
    assert not _verdict(_parameter(option), monkeypatch, chunk)


BLOCKING_OPTIONS = (
    b'typer.Option("SYNTHETICFIXTURE", "--ngc-api-key")',
    b'typer.Option(SYNTHETICFIXTURE, "--ngc-api-key")',
    b'typer.Option(0, "--ngc-api-key")',
    b'typer.Option(1, "--ngc-api-key")',
    b'typer.Option(mystery(), "--ngc-api-key")',
    b'typer.Option(os.getenv("NGC_API_KEY", "SYNTHETICFIXTURE"), "--ngc-api-key")',
    b'typer.Option(os.getenv("UNKNOWN_SELECTOR"), "--ngc-api-key")',
    b'typer.Option(config.load(), "--ngc-api-key")',
    b'typer.Option(load(ngc_api_key="SYNTHETICFIXTURE"), "--ngc-api-key")',
    b'typer.Option(load(NGC_API_KEY=config.ngc_api_key), "--ngc-api-key")',
    b'typer.Option("", "SYNTHETICFIXTURE")',
    b'typer.Option("", *flags, help="Optional key supplied at runtime")',
    b'typer.Option("", **metadata)',
    b'typer.Option("", "--ngc-api-key", unknown="SYNTHETICFIXTURE")',
    b'typer.Option("", "--ngc-api-key", help=config.help)',
    b'typer.Option("", "--ngc-api-key", callback=mystery)',
    b'typer.Option("", "--ngc-api-key", envvar="UNKNOWN_SELECTOR")',
    b'typer.Option("", "--ngc-api-key", show_default="SYNTHETICFIXTURE")',
    b'typer.Option("", default="SYNTHETICFIXTURE")',
    b'typer.Option("", "--ngc-api-key", "--ngc-api-key")',
    b'typer.Option(default=None, default="SYNTHETICFIXTURE")',
    b'typer.Option("", help="First help", help="Second help")',
    b'typer.Option("", ngc_api_key="SYNTHETICFIXTURE")',
    b'Option("", "--ngc-api-key", help="Optional key supplied at runtime")',
    b'helper.Option("", "--ngc-api-key", help="Optional key supplied at runtime")',
    b'typer.options.Option("", "--ngc-api-key", help="Optional key supplied at runtime")',
    b'typer.Option("" if ready else "SYNTHETICFIXTURE", "--ngc-api-key")',
    b'typer.Option("' + b"x" * 513 + b'", "--ngc-api-key")',
    b'typer.Option("", help="' + b"x" * 513 + b'")',
)


@pytest.mark.parametrize("chunk", [1, 7, 31, 512, 1024])
@pytest.mark.parametrize("option", BLOCKING_OPTIONS)
def test_unknown_helper_default_metadata_and_nested_assignments_stay_blocked(
    monkeypatch: pytest.MonkeyPatch, option: bytes, chunk: int
) -> None:
    assert _verdict(_parameter(option), monkeypatch, chunk)


@pytest.mark.parametrize("chunk", [1, 7, 31, 1024])
@pytest.mark.parametrize(
    "shape", ["quoted", "escaped", "metadata", "qualified", "extra", "truncated"]
)
def test_encoded_default_or_decoded_metadata_is_not_an_option_exemption(
    monkeypatch: pytest.MonkeyPatch, chunk: int, shape: str
) -> None:
    token = _synthetic_compact_token(b'{"alg":"none"}')
    escaped = b"".join(b"\\x" + format(byte, "02x").encode() for byte in token)
    options = {
        "quoted": b'typer.Option("' + token + b'", "--ngc-api-key")',
        "escaped": b'typer.Option("", help="' + escaped + b'")',
        "metadata": b'typer.Option("", help="Runtime instructions: ' + token + b'")',
        "qualified": b"typer.Option(" + token + b', "--ngc-api-key")',
        "extra": b'typer.Option("", help="' + token + b'.extra")',
        "truncated": b'typer.Option("", help="' + token.rsplit(b".", 1)[0] + b'")',
    }
    assert _verdict(_parameter(options[shape]), monkeypatch, chunk)


@pytest.mark.parametrize("chunk", [1, 7, 31, 1024])
@pytest.mark.parametrize(
    "literal",
    [
        b"AKIA" + b"A" * 16,
        b"-----BEGIN PRIVATE KEY-----\nA",
        b"hf_token=SYNTHETICFIXTURE",
        b"hf_token=config.hf_token",
    ],
)
@pytest.mark.parametrize("escaping", ["hex", "unicode", "concatenated"])
def test_decoded_literals_cannot_hide_declared_shapes_in_metadata(
    monkeypatch: pytest.MonkeyPatch, chunk: int, literal: bytes, escaping: str
) -> None:
    if escaping == "concatenated":
        encoded = (
            repr(literal[:3].decode()).encode()
            + b" "
            + repr(literal[3:].decode()).encode()
        )
    else:
        prefix, width = (b"\\x", 2) if escaping == "hex" else (b"\\u", 4)
        encoded = (
            b'"'
            + b"".join(prefix + format(byte, f"0{width}x").encode() for byte in literal)
            + b'"'
        )
    option = b'typer.Option("", "--ngc-api-key", help=' + encoded + b")"
    assert _verdict(_parameter(option), monkeypatch, chunk)


SHADOWS = (
    b"typer = replacement\n",
    b"del typer\n",
    b"typer.Option = replacement\n",
    b"del typer.Option\n",
    b"typer.__dict__['Option'] = replacement\n",
    b"typer.Option.__globals__['Option'] = replacement\n",
    b"typer.mutate()\n",
    b"setattr(typer, 'Option', replacement)\n",
    b"alias = typer\n",
    b"alias = typer.Option\n",
    b"def shadow(typer):\n    pass\n",
    b"def typer():\n    pass\n",
    b"class typer:\n    pass\n",
    b"import fake as typer\n",
    b"import typer as alias\nalias.Option = replacement\n",
    b"from typer import Option as alternate\n",
    b"from fake import typer\n",
    b"from fake import *\n",
    b"globals()['typer'] = replacement\n",
    b"vars()['typer'] = replacement\n",
    b"builtins.globals()['typer'] = replacement\n",
    b"sys.modules['typer'].Option = replacement\n",
    b"importlib.reload(typer)\n",
    b"exec('fixture')\n",
    b"try:\n    pass\nexcept Exception as typer:\n    pass\n",
    b"match value:\n    case {'fixture': typer}:\n        pass\n",
    b"match value:\n    case {**typer}:\n        pass\n",
)


@pytest.mark.parametrize("chunk", [1, 7, 31, 1024])
@pytest.mark.parametrize("shadow", SHADOWS)
@pytest.mark.parametrize("position", ["before", "after"])
def test_observable_shadow_or_mutation_never_qualifies_the_option_binding(
    monkeypatch: pytest.MonkeyPatch, chunk: int, shadow: bytes, position: str
) -> None:
    payload = _parameter(BENIGN_OPTIONS[0])
    if position == "before":
        payload = payload.replace(b"\n\ndef", b"\n" + shadow + b"\ndef")
    else:
        payload += shadow
    assert _verdict(payload, monkeypatch, chunk)


@pytest.mark.parametrize("chunk", [1, 7, 31, 1024])
@pytest.mark.parametrize(
    "payload",
    [
        b'import typer, fake as typer\ndef command(ngc_api_key: str = typer.Option("", "--ngc-api-key")):\n    pass\n',
        b'import typer, typer\ndef command(ngc_api_key: str = typer.Option("", "--ngc-api-key")):\n    pass\n',
        b'import typer as typer\ndef command(ngc_api_key: str = typer.Option("", "--ngc-api-key")):\n    pass\n',
        b'ngc_api_key: str = typer.Option("", "--ngc-api-key")\n',
        b'import typer\nngc_api_key: str = typer.Option("", "--ngc-api-key")\n',
        b'import typer\nngc_api_key = typer.Option("", "--ngc-api-key")\n',
        b'import typer\nexample = \'ngc_api_key: str = typer.Option("", "--ngc-api-key")\'\n',
        b'import typer\n# ngc_api_key: str = typer.Option("", "--ngc-api-key")\n',
        b'import typer\ndef command(ngc_api_key: str = typer.Option("", "--ngc-api-key")):\n    INVALID !\n',
        b'import typer\n\xff\ndef command(ngc_api_key: str = typer.Option("", "--ngc-api-key")):\n    pass\n',
        b'def command(ngc_api_key: str = typer.Option("", "--ngc-api-key")):\n    pass\nimport typer\n',
    ],
)
def test_fragment_quotes_comments_invalid_context_and_import_aliases_are_blocking(
    monkeypatch: pytest.MonkeyPatch, chunk: int, payload: bytes
) -> None:
    assert _verdict(payload, monkeypatch, chunk)


@pytest.mark.parametrize(
    "suffix", [b"hf_token=SYNTHETICFIXTURE", SYNTHETIC_KEY, b"AKIA" + b"A" * 16]
)
@pytest.mark.parametrize("order", ["before", "after"])
def test_later_or_earlier_credentials_win_over_a_benign_option(
    monkeypatch: pytest.MonkeyPatch, suffix: bytes, order: str
) -> None:
    payload = _parameter(BENIGN_OPTIONS[0])
    payload = suffix + b"\n" + payload if order == "before" else payload + suffix
    assert _verdict(payload, monkeypatch, 7)


def test_complete_source_context_has_a_fixed_memory_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _parameter(BENIGN_OPTIONS[0])
    monkeypatch.setattr(credentials, "_SOURCE_CONTEXT_LIMIT", len(payload))
    assert not _verdict(payload, monkeypatch, 7)
    assert _verdict(payload + b"\n", monkeypatch, 7)
    source = credentials._OptionSourceContext()
    source.feed(payload)
    source.feed(b"\n")
    source.feed(b"x" * 10000)
    assert source.oversized and source.data == b""


@pytest.mark.parametrize("chunk", [1, 7, 31, 1024])
def test_unicode_prefix_uses_byte_offsets_for_a_real_parameter(
    monkeypatch: pytest.MonkeyPatch, chunk: int
) -> None:
    payload = "# documentation: café\n".encode() + _parameter(BENIGN_OPTIONS[0])
    assert not _verdict(payload, monkeypatch, chunk)


@pytest.mark.parametrize("chunk", [1, 7, 31, 1024])
def test_only_the_exact_parameter_offset_can_qualify(
    monkeypatch: pytest.MonkeyPatch, chunk: int
) -> None:
    payload = _parameter(BENIGN_OPTIONS[0])
    payload += b'example = \'ngc_api_key: str = typer.Option("", "--ngc-api-key")\'\n'
    assert _verdict(payload, monkeypatch, chunk)


@pytest.mark.parametrize(
    "callee", [b"Option", b"helper.Option", b"typer.options.Option"]
)
def test_unknown_option_callee_with_short_metadata_does_not_use_generic_reference_rule(
    monkeypatch: pytest.MonkeyPatch, callee: bytes
) -> None:
    assert _verdict(_parameter(callee + b'("", "--x", help="hi")'), monkeypatch, 1)


@pytest.mark.parametrize("padding", [7, 1024 * 1024 + 1])
def test_large_label_gap_does_not_remove_the_source_context_bound(
    monkeypatch: pytest.MonkeyPatch, padding: int
) -> None:
    payload = _parameter(BENIGN_OPTIONS[0]).replace(
        b"ngc_api_key:", b"ngc_api_key" + b" " * padding + b":"
    )
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", 1024)
    assert _verdict(payload, monkeypatch, 1024) == (
        padding > credentials._SOURCE_CONTEXT_LIMIT
    )


def test_deferred_option_count_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "_OPTION_DECLARATION_LIMIT", 1)
    payload = _parameter(BENIGN_OPTIONS[0])
    assert not _verdict(payload, monkeypatch, 7)
    assert _verdict(payload + payload.replace(b"import typer\n", b""), monkeypatch, 7)


def test_impossible_python_prefix_does_not_retain_pem_whitespace_context() -> None:
    source = credentials._OptionSourceContext()
    for chunk in (b"-----BEGIN PRIVATE ", b"KEY-----", b"\n" * 100000):
        source.feed(chunk)
    assert source.invalid and source.data == b""
    assert not source.allows([0])


def test_impossible_prefix_is_rejected_before_a_large_context_allocation() -> None:
    source = credentials._OptionSourceContext()
    source.feed(b"-----BEGIN PRIVATE KEY-----" + b"\n" * 100000)
    assert source.invalid and source.data == b""
    assert not source.oversized


@pytest.mark.parametrize(
    "prefix",
    [
        b'"-----BEGIN PRIVATE KEY-----"\n',
        b"'''-----BEGIN PRIVATE KEY-----'''\n",
        b"# -----BEGIN PRIVATE KEY-----\n",
    ],
)
def test_quotes_and_comments_do_not_discard_valid_python_context(prefix: bytes) -> None:
    source = credentials._OptionSourceContext()
    payload = prefix + _parameter(BENIGN_OPTIONS[0])
    for byte in payload:
        source.feed(bytes([byte]))
    assert not source.invalid and not source.oversized
    assert bytes(source.data) == payload
    expected = "private_key_content" if prefix.startswith(b"#") else None
    assert credentials.content_credential(io.BytesIO(payload)) == expected


@pytest.mark.parametrize("wrapper", [b'"""', b"'''"])
def test_pem_in_python_docstrings_is_still_scanned(wrapper: bytes) -> None:
    payload = wrapper + SYNTHETIC_KEY + wrapper + b"\n" + _parameter(BENIGN_OPTIONS[0])
    assert credentials.content_credential(io.BytesIO(payload)) == "private_key_content"


@pytest.mark.parametrize("scanner", SCANNERS_USING_SHARED_RULES)
@pytest.mark.parametrize("member", ["opt/app/source.bin", "opt/vendor/client.py"])
@pytest.mark.parametrize("literal", [False, True])
def test_native_payload_scanners_use_member_bytes_without_path_exemptions(
    tmp_path: Path, scanner: str, member: str, literal: bool
) -> None:
    payload = _parameter(BLOCKING_OPTIONS[0] if literal else BENIGN_OPTIONS[0])
    image = _single_layer_archive(tmp_path / "image.tar", member, payload)
    completed = subprocess.run(
        [sys.executable, str(_MODULE_PATH.parent / scanner), "--tarball", str(image)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == int(literal), completed.stderr
    report = json.loads(completed.stdout)
    assert report["scan_complete"] is True
    assert bool(report.get("credential_hits", report.get("findings"))) == literal
