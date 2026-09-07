"""Native request and public-key identities are references, not generic secrets."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from npa.agent_backend import trajectory as emitter


REQUEST = "01234567-89ab-4cde-8fab-0123456789ab"
OTHER_REQUEST = "89abcdef-0123-4abc-8def-0123456789ab"


def public_identity(kind="ssh-ed25519", *, material=b"synthetic-public-identity-only!!!"):
    """Synthetic public wire-format data; no private key is generated or used."""
    algorithm = kind.encode()
    wire = len(algorithm).to_bytes(4, "big") + algorithm
    wire += len(material).to_bytes(4, "big") + material
    return base64.b64encode(wire).decode()


@pytest.mark.parametrize("text,identifier", [
    (f"Submitted sky.jobs.launch request: {REQUEST}", REQUEST),
    (f"sky api logs {REQUEST}", REQUEST),
    (f"sky api cancel {REQUEST}", REQUEST),
    (f"Check logs with: sky api logs {REQUEST[:8]}", REQUEST[:8]),
    (f"To cancel the request, run: sky api cancel {REQUEST[:8]}", REQUEST[:8]),
    (f"sky api logs {REQUEST.upper()}", REQUEST.upper()),
    (f"`sky api cancel {REQUEST[:8]}`", REQUEST[:8]),
    (f"\x1b[1msky api logs {REQUEST[:8]}\x1b[0m", REQUEST[:8]),
    (rf"prior\nsky api logs {REQUEST[:8]}\nnext", REQUEST[:8]),
    (rf"prior\u001b[1msky api cancel {REQUEST[:8]}\u001b[0m", REQUEST[:8]),
    (rf"sky\tapi\tlogs\t{REQUEST[:8]}", REQUEST[:8]),
    (rf"prior\nSubmitted sky.jobs.launch request: {REQUEST}\nnext", REQUEST),
])
def test_only_native_request_identifier_is_replaced(text, identifier):
    expected = text.replace(identifier, "<infra-ref>")
    assert emitter.redact(text) == expected
    assert emitter.redact(expected) == expected


@pytest.mark.parametrize("text", [
    REQUEST,
    f"Episode {REQUEST} is still open.",
    f"Submitted another.operation request: {REQUEST}",
    f"Submitted sky.jobs.launch request: {REQUEST[:8]}",
    f"sky api status {REQUEST}",
    f"my-sky api logs {REQUEST[:8]}",
    f"skylark api logs {REQUEST[:8]}",
    f"sky api logs {REQUEST[:8]}abcdef01",
    f"sky api cancel {REQUEST}-suffix",
    f"sky api logs {REQUEST[:8]}_measurement",
    "The digest is 01234567 and latency is 12.5 milliseconds.",
])
def test_other_ids_and_ordinary_prose_are_unchanged(text):
    assert emitter.redact(text) == text


@pytest.mark.parametrize("kind", [
    "ssh-ed25519", "ssh-rsa", "ssh-dss", "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521",
    "ssh-ed25519-cert-v01@openssh.com", "ssh-rsa-cert-v01@openssh.com",
    "sk-ssh-ed25519@openssh.com", "sk-ecdsa-sha2-nistp256@openssh.com",
    "sk-ssh-ed25519-cert-v01@openssh.com",
])
def test_ssh_public_identity_token_redacts_without_calling_it_a_secret(kind):
    token = public_identity(kind)
    text = f"example.invalid {kind} {token} synthetic-comment\n"
    # Keep pre-existing generic token rules for algorithm labels that overlap
    # a secret-token prefix; the newly recognized public identity uses infra-ref.
    label = ("<redacted>@openssh.com" if kind in {
        "sk-ecdsa-sha2-nistp256@openssh.com", "sk-ssh-ed25519-cert-v01@openssh.com",
    } else kind)
    assert emitter.redact(text) == f"example.invalid {label} <infra-ref> synthetic-comment\n"


@pytest.mark.parametrize("text", [
    "The ssh-ed25519 algorithm uses public keys.",
    "ssh-rsa " + base64.b64encode(b"ordinary prose without an SSH wire header").decode(),
    "ssh-ed25519 " + "abcd" * 16,
    "ssh-ed25519 " + public_identity("ssh-rsa"),
    "ssh-ed25519 " + base64.b64encode((11).to_bytes(4, "big") + b"ssh-ed25519").decode(),
    "measurement " + public_identity(),
])
def test_unrelated_base64_digests_and_algorithm_names_are_unchanged(text):
    assert emitter.redact(text) == text


def test_public_key_literal_escapes_keep_the_complete_identity_private():
    kind = "ssh-ed25519"
    token = public_identity()
    text = rf"prior\nexample.invalid {kind}\t{token}\nnext"
    assert emitter.redact(text) == text.replace(token, "<infra-ref>")
    # JSON permits escaped forward slashes inside its string values.
    wire = (11).to_bytes(4, "big") + b"ssh-ed25519" + (32).to_bytes(4, "big") + b"\xff" * 32
    slash_token = base64.b64encode(wire).decode()
    assert "/" in slash_token
    escaped = slash_token.replace("/", r"\/")
    assert emitter.redact(f"{kind} {escaped}") == f"{kind} <infra-ref>"


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_nested_json_tool_output_remains_parseable_and_redacted(depth):
    token = public_identity()
    text = f"Submitted sky.jobs.launch request: {REQUEST}\nexample.invalid ssh-ed25519 {token}\n"
    wrapped = text
    for _ in range(depth):
        wrapped = json.dumps({"output": wrapped})
    safe = emitter.redact({"observation": {"output": wrapped}})["observation"]["output"]
    for _ in range(depth):
        safe = json.loads(safe)["output"]
    assert safe == text.replace(REQUEST, "<infra-ref>").replace(token, "<infra-ref>")


@pytest.mark.parametrize("original,other", [
    (f"sky api logs {REQUEST}", f"sky api logs {OTHER_REQUEST}"),
    (f"ssh-ed25519 {public_identity()}", f"ssh-ed25519 {public_identity(material=b'synthetic-other-public-material!')}"),
])
def test_mapping_collision_and_caller_lookalikes_preserve_every_value(original, other):
    marker = emitter.redact(original)
    assert marker != original
    lookalike = marker + "-" + hashlib.sha256(original.encode()).hexdigest()[:12]
    mapping = {original: "first", other: "second", marker: "third", lookalike: "fourth",
               lookalike + "<private-ref>": "fifth"}
    safe = emitter.redact(mapping)
    assert len(safe) == len(mapping) == 5
    assert set(safe.values()) == set(mapping.values())
    assert emitter.redact(safe) == safe
    assert emitter.redact(dict(reversed(list(mapping.items())))) == safe


def test_episode_session_hashes_and_measurements_keep_existing_semantics():
    digest = "0123456789abcdef" * 4
    value = {"episode_id": REQUEST, "session_id": OTHER_REQUEST, "source_sha256": digest,
             "measurements": {"latency_ms": 12.5, "samples": 8},
             "observation": f"sky api logs {REQUEST[:8]}"}
    safe = emitter.redact(value)
    assert safe == {**value, "observation": "sky api logs <infra-ref>"}


def test_existing_uri_credential_and_address_redaction_is_retained():
    value = {"password": "synthetic-password", "nested": ["https://example.invalid/path", "203.0.113.50"]}
    assert emitter.redact(value) == {"password": "<redacted>", "nested": ["<uri-ref>", "<address-ref>"]}
