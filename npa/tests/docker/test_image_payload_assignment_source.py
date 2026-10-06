"""Source references must not hide literal credential bytes in any layer."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from test_image_payload_credentials import (
    SCANNERS_USING_SHARED_RULES,
    SYNTHETIC_KEY,
    _MODULE_PATH,
    _single_layer_archive,
    credentials,
)

# The first line is present in project_credentials.py at the failed Ray build's
# exact source e73b93fde55f3d18a84c1d1ebfa8119069ab3628. No credential value is
# baked by passing the object's field into a client constructor.
BENIGN_SOURCE = (
    b"aws_secret_access_key=self.aws_secret_access_key,",
    b"hf_token = args.hf_token",
    b'hf_token = os.getenv("HF_TOKEN")',
    b'hf_token = os.environ.get("HF_TOKEN", "")',
    b'hf_token = credentials["hf_token"]',
    b"ngc_api_key=load_credential()",
    b"hf_token = config.hf_token or None",
    b'hf_token = (os.getenv(\n    "HF_TOKEN"\n))',
    b"hf_token: Optional[str] = None,",
    b"hf_token: typing.Optional[str] = None)",
    b"hf_token: str | None = None,",
    b'hf_token: Optional[str] = "",',
    b"hf_token: Optional[str] = config.hf_token,",
    b"aws_secret_access_key=$SECRET_VALUE",
    b'aws_secret_access_key="${SECRET_VALUE}"',
    b"aws_secret_access_key=${SECRET_VALUE:-}",
    b"aws_secret_access_key=<REPLACE_ME>",
)

# Deliberately synthetic opaque values, never operator credentials. Ambiguous
# bare values, literal defaults, quoted source text and malformed syntax retain
# their finding instead of gaining a filename or expression exemption.
BLOCKING_VALUES = (
    b"aws_secret_access_key=SYNTHETIC-FIXTURE",
    b"hf_token: SYNTHETICFIXTURE",
    b'hf_token = "SYNTHETIC-FIXTURE"',
    b'hf_token = "self.hf_token"',
    b'hf_token: "Optional[str]"',
    b"hf_token: typing.UnknownCredential",
    b'hf_token: Optional[str] = "SYNTHETIC-FIXTURE",',
    b'hf_token = os.getenv("HF_TOKEN", "SYNTHETIC-FIXTURE")',
    b'hf_token = source.get("SYNTHETIC-FIXTURE")',
    b'hf_token = config.hf_token or "SYNTHETIC-FIXTURE"',
    b'hf_token = join("SYN", "THETIC", "FIXTURE")',
    b'hf_token = "SYN" "THETICFIXTURE"',
    b'hf_token = f"SYNTHETIC-{config.value}"',
    b"hf_token = call(hf_token=SYNTHETICFIXTURE)",
    b"hf_token = call(HF_TOKEN=SYNTHETICFIXTURE)",
    b"hf_token = call(hf_token=source.hf_token)",
    b"hf_token = ${SECRET_VALUE:-SYNTHETICFIXTURE}",
    b'hf_token = "${SECRET_VALUE:-SYNTHETICFIXTURE}"',
    b"hf_token = '${SECRET_VALUE}'",
    b"hf_token = $SECRET_VALUE-SYNTHETICFIXTURE",
    b"hf_token = source.value # ngc_api_key=SYNTHETICFIXTURE",
    b'hf_token = source.value + "SYNTHETICFIXTURE"',
    b"hf_token = source.value unexpected",
    b'hf_token = os.getenv("HF_TOKEN"',
    b"hf_token = source.value\x00SYNTHETICFIXTURE",
    b"hf_token = source." + b"x" * 600,
)


@pytest.mark.parametrize("chunk_size", [1, 7, 37, 512, 1024])
@pytest.mark.parametrize("payload", BENIGN_SOURCE)
def test_complete_source_references_are_not_literal_credentials(
    monkeypatch: pytest.MonkeyPatch, payload: bytes, chunk_size: int
) -> None:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", chunk_size)
    assert credentials.content_credential(io.BytesIO(payload)) is None
    assert not any(
        pattern.search(payload) for _, pattern in credentials.DECLARED_CONTENT_PATTERNS
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 37, 512, 1024])
@pytest.mark.parametrize("payload", BLOCKING_VALUES)
def test_literal_unknown_or_incomplete_values_remain_blocking(
    monkeypatch: pytest.MonkeyPatch, payload: bytes, chunk_size: int
) -> None:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", chunk_size)
    assert (
        credentials.content_credential(io.BytesIO(payload)) == "credential_assignment"
    )
    assert any(
        pattern.search(payload) for _, pattern in credentials.DECLARED_CONTENT_PATTERNS
    )


@pytest.mark.parametrize("source", BENIGN_SOURCE)
@pytest.mark.parametrize(
    "suffix", [b"hf_token=SYNTHETICFIXTURE", SYNTHETIC_KEY, b"AKIA" + b"A" * 16]
)
def test_source_reference_does_not_hide_an_adjacent_credential(
    monkeypatch: pytest.MonkeyPatch, source: bytes, suffix: bytes
) -> None:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", 13)
    assert credentials.content_credential(io.BytesIO(source + b"\n" + suffix))


@pytest.mark.parametrize("gap", [b" " * 1048585, b"\t\n" * 524293])
@pytest.mark.parametrize("literal", [False, True])
def test_unbounded_label_gaps_do_not_turn_source_or_literals_into_partial_verdicts(
    gap: bytes, literal: bool
) -> None:
    value = b'"SYNTHETICFIXTURE"' if literal else b"self.hf_token"
    payload = b"\x00" * 1048571 + b"hf_token" + gap + b"=" + gap + value
    assert bool(credentials.content_credential(io.BytesIO(payload))) == literal


@pytest.mark.parametrize("scanner", SCANNERS_USING_SHARED_RULES)
@pytest.mark.parametrize("literal", [False, True])
@pytest.mark.parametrize("member", ["opt/npa/client.py", "opt/vendor/config.bin"])
def test_native_scanners_apply_source_policy_without_path_exceptions(
    tmp_path: Path, scanner: str, literal: bool, member: str
) -> None:
    payload = b"\n".join(BENIGN_SOURCE)
    if literal:
        payload += b"\nhf_token=SYNTHETICFIXTURE"
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


def _ancestor_archive(path: Path) -> Path:
    layers = []
    for name, payload in (
        ("opt/vendor/client.py", b"hf_token=SYNTHETICFIXTURE"),
        ("opt/vendor/.wh.client.py", b""),
    ):
        layer = io.BytesIO()
        with tarfile.open(fileobj=layer, mode="w") as archive:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        layers.append(layer.getvalue())
    names = ["ancestor.tar", "deletion.tar"]
    manifest = [{"Config": "config.json", "RepoTags": [], "Layers": names}]
    entries = [
        ("manifest.json", json.dumps(manifest).encode()),
        ("config.json", b'{"config":{},"history":[]}'),
        *zip(names, layers),
    ]
    with tarfile.open(path, mode="w") as archive:
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return path


@pytest.mark.parametrize("scanner", SCANNERS_USING_SHARED_RULES)
def test_deleted_ancestor_literal_is_still_blocking(
    tmp_path: Path, scanner: str
) -> None:
    image = _ancestor_archive(tmp_path / "image.tar")
    completed = subprocess.run(
        [sys.executable, str(_MODULE_PATH.parent / scanner), "--tarball", str(image)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1, completed.stderr
    report = json.loads(completed.stdout)
    if "findings" in report:
        assert any(
            finding["kind"] == "credential_assignment"
            and finding["path"] == "opt/vendor/client.py"
            and finding["layer"] == "ancestor.tar"
            for finding in report["findings"]
        )
    else:
        assert "credential_assignment:opt/vendor/client.py" in report["credential_hits"]
