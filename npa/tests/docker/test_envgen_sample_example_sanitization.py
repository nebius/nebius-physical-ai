"""Reject unbound sample sanitization and retain numerical APIs and source provenance."""

import base64
import csv
import hashlib
import importlib.util
from io import StringIO
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/common/sanitize_envgen_grass_example.py"
)
DOCKERFILE = SCRIPT.parent.parent / "sim2real-envgen/Dockerfile"
spec = importlib.util.spec_from_file_location("envgen_sample_sanitization", SCRIPT)
sanitizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanitizer)
SOURCE = b'''def grass():
    """A CC0 image; its API and primary documentation are preserved."""
    """The following code was used to obtain the final image.
    An inert historical URL example: ?token=not-a-credential
    """
    return image_reader("grass.png")
'''


def _fixture(tmp_path, monkeypatch):
    source = tmp_path / "_fetchers.py"
    source.write_bytes(SOURCE)
    monkeypatch.setattr(
        sanitizer, "EXPECTED_SOURCE_SHA256", hashlib.sha256(SOURCE).hexdigest()
    )
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(SOURCE).digest()).decode().rstrip("=")
    )
    record = tmp_path / "RECORD"
    output = StringIO()
    csv.writer(output).writerow(
        ["skimage/data/_fetchers.py", "sha256=" + digest, str(len(SOURCE))]
    )
    record.write_text(output.getvalue())
    return source, record


def test_removes_only_inert_example_and_updates_exact_record(tmp_path, monkeypatch):
    source, record = _fixture(tmp_path, monkeypatch)
    receipt = sanitizer._sanitize_module(source, record)
    repaired = source.read_bytes()
    assert b"?token=" not in repaired
    assert b"CC0 image" in repaired
    assert b'return image_reader("grass.png")' in repaired
    rows = list(csv.reader(StringIO(record.read_text())))
    assert rows[0][2] == str(len(repaired))
    assert receipt["repaired_sha256"] == hashlib.sha256(repaired).hexdigest()


@pytest.mark.parametrize(
    "case",
    (
        "source-tamper",
        "record-tamper",
        "duplicate-record",
        "source-symlink",
        "record-symlink",
    ),
)
def test_unverified_inputs_reject_before_either_file_changes(
    tmp_path, monkeypatch, case
):
    source, record = _fixture(tmp_path, monkeypatch)
    if case == "source-tamper":
        source.write_bytes(SOURCE + b"# unreviewed\n")
    elif case == "record-tamper":
        record.write_text("skimage/data/_fetchers.py,wrong,0\n")
    elif case == "duplicate-record":
        record.write_text(record.read_text() * 2)
    else:
        target = source if case == "source-symlink" else record
        original = tmp_path / "original"
        target.rename(original)
        target.symlink_to(original)
    before = source.read_bytes(), record.read_bytes()
    with pytest.raises(RuntimeError):
        sanitizer._sanitize_module(source, record)
    assert (source.read_bytes(), record.read_bytes()) == before


@pytest.mark.parametrize(
    "case", ("missing-example", "duplicate-example", "changed-executable")
)
def test_unreviewed_semantics_are_not_silently_sanitized(case):
    raw = (
        SOURCE.replace(b"?token=", b"?example=")
        if case == "missing-example"
        else SOURCE
    )
    if case == "duplicate-example":
        raw = SOURCE.replace(
            b"    return image_reader",
            b'    """The following code was used to obtain the final image. ?token=none"""\n    return image_reader',
        )
    if case == "changed-executable":
        raw = SOURCE.replace(b'    """The following', b'    payload = """The following')
    with pytest.raises(RuntimeError):
        sanitizer._without_sample_example(raw)


def _require_pre_layer_sanitization(text):
    intermediate, final = text.split("FROM scratch AS runtime", 1)
    assert "FROM ${BASE_IMAGE} AS sanitized" in intermediate
    assert "python /usr/local/lib/npa/sanitize-envgen-grass-example.py" in intermediate
    assert "COPY --from=sanitized / /" in final
    assert "FROM " not in final
    assert "COPY --from=sanitized / /" not in intermediate
    assert "sanitize-envgen-grass-example.py" not in final
    assert "COPY --from=" not in final.replace("COPY --from=sanitized / /", "")


def test_example_is_removed_before_building_the_scratch_delivery_layer():
    _require_pre_layer_sanitization(DOCKERFILE.read_text())


@pytest.mark.parametrize(
    "case", ("parent-retained", "later-deletion", "extra-ancestor-copy")
)
def test_recipe_must_not_claim_a_later_deletion_clears_ancestor_bytes(case):
    text = DOCKERFILE.read_text()
    command = "python /usr/local/lib/npa/sanitize-envgen-grass-example.py"
    if case == "parent-retained":
        text = text.replace("FROM scratch AS runtime", "FROM sanitized AS runtime")
    elif case == "later-deletion":
        text = text.replace(command, "true") + "\nRUN " + command + "\n"
    else:
        text += "\nCOPY --from=sanitized /tmp /inherited-cache\n"
    with pytest.raises((AssertionError, ValueError)):
        _require_pre_layer_sanitization(text)
