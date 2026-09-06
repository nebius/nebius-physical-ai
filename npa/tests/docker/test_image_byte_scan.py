"""Hermetic archive and policy regression tests; no native install or network."""
from __future__ import annotations

import gzip
import hashlib
import sys
import io
import json
import os
import tarfile
from pathlib import Path
from typing import ClassVar

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
SCRIPTS = CHECKOUT / "npa/scripts"
sys.path.insert(0, str(SCRIPTS))
from image_byte_scan import core as W  # noqa: E402


@pytest.fixture(autouse=True)
def explicit_roots(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        yield


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_bytes(data)
    path.chmod(0o600)
    return {"path": str(path), "sha256": digest(data)}


def js(value):
    return json.dumps(value).encode()


def tar_data(entries, *, format=tarfile.PAX_FORMAT):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w", format=format) as archive:
        for name, data, kind, link, pax in entries:
            item = tarfile.TarInfo(name)
            item.type, item.linkname, item.pax_headers = kind, link, pax
            item.size = len(data) if kind in {tarfile.REGTYPE, tarfile.AREGTYPE} else 0
            archive.addfile(item, io.BytesIO(data) if item.isfile() else None)
    return result.getvalue()


def file(name, data=b"", *, kind=tarfile.REGTYPE, link="", pax=None):
    return name, data, kind, link, pax or {}


def fixture_tools_receipt(authorization, directory, ready=None):
    ready = {} if ready is None else ready
    authorization["helper"]["ready_sha256"] = W.sha(W.canonical(ready))
    receipt = {"schema_version": "npa.image-byte-scan-tools.v1", "source": W.current_go_sources(),
               "helper": {key: value for key, value in authorization["helper"].items() if key != "ready_sha256"},
               "config": authorization["config"], "ready": write(directory / "fixture-ready.json", js(ready))}
    authorization["tools_receipt"] = write(directory / "fixture-tools.json", js(receipt))


def fixture(tmp_path, *, entries=None, raw=None, compressed=None, literals=None, policy="exact-substring-v1", repeat=1, codec="gzip"):
    entries = entries if entries is not None else [file("opt/sample", b"actual synthetic neutral body")]
    raw = raw if raw is not None else tar_data(entries)
    data = (gzip.compress(raw, mtime=0) if codec == "gzip" else raw) if compressed is None else compressed
    config = js({"rootfs": {"type": "layers", "diff_ids": ["sha256:" + digest(raw)] * repeat}})
    config_id = "sha256:" + digest(config)
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": config_id, "size": len(config)},
                "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar" + ("+gzip" if codec == "gzip" else ""), "digest": "sha256:" + digest(data), "size": len(data)}] * repeat}
    manifest_bytes = js(manifest)
    image_id = "sha256:" + digest(manifest_bytes)
    index = js({"schemaVersion": 2, "manifests": [{"mediaType": manifest["mediaType"], "digest": image_id, "size": len(manifest_bytes)}]})
    saved = js([{"Config": "blobs/sha256/" + digest(config), "Layers": ["blobs/sha256/" + digest(data)] * repeat}])
    outer = tar_data([file("manifest.json", saved), file("index.json", index), file("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
                      file("blobs/sha256/" + digest(config), config), file("blobs/sha256/" + digest(manifest_bytes), manifest_bytes),
                      file("blobs/sha256/" + digest(data), data)])
    archive_binding = write(tmp_path / "image.tar", outer)
    regulars = [row for row in entries if row[2] in {tarfile.REGTYPE, tarfile.AREGTYPE}]
    verification = {"schema_version": "npa.curobo.image-verification.v1", "valid": True, "expected_image_id": image_id,
                    "image_config_digest": config_id, "image_manifest_digest": image_id, "docker_save_sha256": digest(outer),
                    "verified_layer_diff_ids": ["sha256:" + digest(raw)] * repeat, "layer_count": repeat,
                    "regular_files_read": len(regulars) * repeat, "content_bytes_read": sum(len(row[1]) for row in regulars) * repeat}
    inventory = write(tmp_path / "literals.json", js({"literals": literals or ["private-operator-marker"]}))
    inventory["matching_policy"] = policy
    authorization = {"schema_version": "npa.image-byte-scan-authorization.v1", "accepted_verification": True,
                     "archive": archive_binding, "verification_report": write(tmp_path / "verification.json", js(verification)),
                     "expected_image_id": image_id, "helper": {**write(tmp_path / "helper-fixture", b"Synthetic framing oracle input"), "ready_sha256": "0" * 64},
                     "config": write(tmp_path / "config-fixture", W.bound_bytes({"path": str(W._ROOTS.get()[1] / ".gitleaks.toml"),
                                      "sha256": W.sha((W._ROOTS.get()[1] / ".gitleaks.toml").read_bytes())})),
                     "literal_inventory": inventory, "sources": W.source_bindings()}
    fixture_tools_receipt(authorization, tmp_path)
    return authorization


class FakeDetector:
    """Only framing/accounting oracle; no fake secret-security success claim."""
    instances: ClassVar[list] = []

    def __init__(self, authorization, stderr_path):
        self.joined = False
        self.records = []
        self.current = None
        self.instances.append(self)

    def begin(self, length):
        self.current = bytearray()
        self.length = length

    def write(self, data):
        self.current.extend(data)

    def end(self, length, value):
        assert self.length == length == len(self.current)
        assert digest(self.current) == value
        self.records.append(bytes(self.current))
        return []

    def finish(self):
        self.joined = True
        return {"type": "summary", "files": len(self.records), "bytes": sum(map(len, self.records)), "findings": 0}

    def abort(self):
        self.joined = True


def run(tmp_path, authorization, *, real=False, detector_type=None):
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    report = W._scan(authorization, output, detector_type=detector_type or (W.Detector if real else FakeDetector))
    records_path = output / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    return report, records


def findings(records, code):
    return [row for row in records if row.get("rule_id") == code]


def test_complete_zero_length_and_large_file_are_one_record_each(tmp_path):
    data = b"neutral material\n" * (2 * W.CHUNK // 17 + 1)
    entries = [file("opt/empty"), file("opt/large", data)]
    authorization = fixture(tmp_path, entries=entries)
    report, records = run(tmp_path, authorization)
    assert report["valid"] and report["complete"] and report["helper_joined"]
    bodies = [r for r in records if r.get("kind") == "layer_regular_content"]
    assert [(r["bytes"], r["sha256"]) for r in bodies] == [(0, digest(b"")), (len(data), digest(data))]
    assert report["regular_files"] == 2 and report["regular_bytes"] == len(data)
    assert report["outer"]["decoded_bytes"] == Path(authorization["archive"]["path"]).stat().st_size
    assert report["verified_zero_bytes"] > 0


@pytest.mark.parametrize("where", ["body", "pax", "gnu", "link"])
def test_literal_metadata_and_binary_content_never_emit_input_text(tmp_path, where):
    value = "private-operator-marker"
    if where == "body":
        entries = [file("opt/object.bin", b"\x7fELF\x00" + value.encode() + b"\x00")]
    elif where == "pax":
        entries = [file("opt/metadata", b"neutral", pax={"SCHILY.xattr.user.audit": value})]
    elif where == "gnu":
        entries = [file("opt/" + "a" * 120 + "/" + value, b"neutral")]
    else:
        entries = [file("opt/link", kind=tarfile.SYMTYPE, link="/" + value)]
    raw = tar_data(entries, format=tarfile.GNU_FORMAT if where == "gnu" else tarfile.PAX_FORMAT)
    report, records = run(tmp_path, fixture(tmp_path, entries=entries, raw=raw, literals=[value]))
    assert report["complete"] and not report["valid"]
    assert findings(records, "private_literal")
    assert value not in json.dumps(report) + json.dumps(records)


@pytest.mark.parametrize("literal", ["ab", "ééééé", "abcdef"])
def test_accepted_character_count_and_binary_token_boundary_policy(literal):
    raw = literal.encode("utf-8")
    matcher = W.LiteralMatcher([literal], W.POLICY)
    data = b"prefix_" + raw + b"_suffix /" + raw + b"/\x00" + raw + b"\x00"
    found = []
    for byte in data:
        found.extend(matcher.feed(bytes([byte])))
    found.extend(matcher.feed(b"", final=True))
    expected = 2 if len(literal) < 6 else 3
    assert len(found) == expected
    assert all(data[row["byte_start"]:row["byte_end"]] == raw for row in found)


def test_strict_default_includes_short_substrings_and_boundaries():
    matcher = W.LiteralMatcher(["ab"], "exact-substring-v1")
    found = matcher.feed(b"xaby /ab", final=True)
    assert [(r["byte_start"], r["byte_end"]) for r in found] == [(1, 3), (6, 8)]


def test_literal_crosses_stream_chunk_and_is_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "CHUNK", 7)
    literal = "operator-private-marker"
    report, records = run(tmp_path, fixture(tmp_path, entries=[file("opt/blob", b"-----" + literal.encode() + b"-----")], literals=[literal]))
    hits = [row for row in findings(records, "private_literal") if "record_ordinal" in row]
    assert len(hits) == 1 and not report["valid"]
    assert hits[0]["byte_start"] == 5


@pytest.mark.parametrize("path", ["opt/private.p12", "opt/credential.PFX", "opt/" + "a" * 150 + "/credential.p12"])
def test_pkcs12_uses_actual_logical_path_even_for_empty_content(tmp_path, path):
    report, records = run(tmp_path, fixture(tmp_path, entries=[file(path)]))
    assert report["complete"] and not report["valid"]
    assert findings(records, "pkcs12-file")
    assert path not in json.dumps(records)


def test_nonzero_padding_is_accounted_scanned_and_permanently_rejected(tmp_path):
    entries = [file("opt/body", b"x")]
    raw = bytearray(tar_data(entries))
    token = b"private-operator-marker"
    raw[513:513 + len(token)] = token
    report, records = run(tmp_path, fixture(tmp_path, entries=entries, raw=bytes(raw)))
    assert report["complete"] and not report["valid"]
    assert findings(records, "nonzero_tar_padding") and findings(records, "private_literal")


def test_nonzero_trailer_is_scanned_and_rejected(tmp_path):
    entries = [file("opt/body", b"neutral")]
    raw = tar_data(entries) + b"private-operator-marker".ljust(512, b"\x00")
    report, records = run(tmp_path, fixture(tmp_path, entries=entries, raw=raw))
    assert report["complete"] and not report["valid"]
    assert findings(records, "nonzero_tar_trailer") and findings(records, "private_literal")


@pytest.mark.parametrize("change", ["fname", "comment", "extra", "concatenated", "trailing", "crc", "truncated"])
def test_gzip_hidden_metadata_members_and_corruption_fail_closed(tmp_path, change):
    raw = tar_data([file("opt/body", b"neutral")])
    compressed = bytearray(gzip.compress(raw, mtime=0))
    if change in {"fname", "comment", "extra"}:
        flag = {"fname": 8, "comment": 16, "extra": 4}[change]
        compressed[3] = flag
    elif change == "concatenated":
        compressed.extend(gzip.compress(b"hidden", mtime=0))
    elif change == "trailing":
        compressed.extend(b"hidden trailer")
    elif change == "crc":
        compressed[-8] ^= 1
    else:
        del compressed[-4:]
    report, _ = run(tmp_path, fixture(tmp_path, raw=raw, compressed=bytes(compressed)))
    assert not report["complete"] and not report["valid"] and report["helper_joined"]


@pytest.mark.parametrize("pax", [{"size": "9"}, {"GNU.sparse.map": "0,9"}, {"SCHILY.realsize": "9"}, {"hdrcharset": "BINARY"}])
def test_unsupported_pax_semantic_overrides_fail_closed(tmp_path, pax):
    entries = [file("opt/body", b"neutral", pax=pax)]
    report, _ = run(tmp_path, fixture(tmp_path, entries=entries))
    assert not report["valid"] and report["failure_code"] == "unsupported_pax_semantics"


def test_sparse_entry_type_fails_closed(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", format=tarfile.GNU_FORMAT) as archive:
        item = tarfile.TarInfo("opt/sparse")
        item.type = tarfile.GNUTYPE_SPARSE
        archive.addfile(item)
    report, _ = run(tmp_path, fixture(tmp_path, entries=[], raw=data.getvalue()))
    assert not report["valid"] and report["failure_code"] == "unsupported_tar_entry_type"


def test_checksum_failure_and_duplicate_paths_fail_closed(tmp_path):
    raw = bytearray(tar_data([file("opt/body", b"neutral")]))
    raw[0] ^= 1
    report, _ = run(tmp_path, fixture(tmp_path, raw=bytes(raw)))
    assert not report["valid"] and not report["complete"]
    assert report["helper_joined"]


def test_duplicate_inner_paths_fail_closed(tmp_path):
    entries = [file("opt/body", b"first"), file("./opt/body", b"second")]
    report, _ = run(tmp_path, fixture(tmp_path, entries=entries))
    assert not report["valid"] and report["failure_code"] == "duplicate_tar_path"


@pytest.mark.parametrize("which", ["archive", "verification_report", "literal_inventory"])
def test_bound_inputs_reject_changed_bytes_before_helper(tmp_path, which):
    authorization = fixture(tmp_path)
    path = Path(authorization[which]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    before = len(FakeDetector.instances)
    with pytest.raises(W.ScanError, match="input_binding_changed"):
        run(tmp_path, authorization)
    assert len(FakeDetector.instances) == before


def test_disabled_literal_inventory_cannot_claim_complete_qualification(tmp_path):
    authorization = fixture(tmp_path)
    authorization.pop("literal_inventory")
    with pytest.raises(W.ScanError, match="confidentiality_policy_required"):
        run(tmp_path, authorization)










@pytest.mark.parametrize("codec", ["raw", "gzip"])
def test_repeated_layer_occurrences_are_each_completely_scanned(tmp_path, codec):
    report, records = run(tmp_path, fixture(tmp_path, repeat=3, codec=codec))
    assert report["complete"] and report["valid"]
    assert len(report["layers"]) == 3 and report["regular_files"] == 3
    assert len([row for row in records if row.get("kind") == "layer_regular_content"]) == 3
    assert len([row for row in records if row.get("type") == "encoded_layer_blob"]) == 1








def test_unknown_authorization_fields_are_rejected(tmp_path):
    authorization = fixture(tmp_path)
    authorization["skip_large_files"] = True
    with pytest.raises(W.ScanError, match="authorization_fields"):
        run(tmp_path, authorization)


@pytest.mark.parametrize("policy", ["exact-substring-v1", W.POLICY])
@pytest.mark.parametrize("chunk", [1, 2, 7, 31])
def test_streaming_literals_match_independent_whole_file_regex(policy, chunk):
    import re

    values = ["aaa", "ababab", "ééééé", "operator-marker"]
    data = ("a" * 40 + " abababababab /ééééé/ ééééé_suffix /operator-marker/ ").encode()
    expected = []
    for index, value in enumerate(values):
        pattern = re.escape(value.encode())
        if policy == W.POLICY and len(value) < 6:
            pattern = rb"(?<![A-Za-z0-9_])" + pattern + rb"(?![A-Za-z0-9_])"
        expected.extend((index, match.start(), match.end()) for match in re.finditer(pattern, data))
    matcher = W.LiteralMatcher(values, policy)
    actual = []
    for start in range(0, len(data), chunk):
        actual.extend(matcher.feed(data[start:start + chunk]))
    actual.extend(matcher.feed(b"", final=True))
    assert sorted((r["literal_index"], r["byte_start"], r["byte_end"]) for r in actual) == sorted(expected)


def test_zero_ranges_scan_complete_contiguous_run_with_literal_continuity(tmp_path):
    detector = FakeDetector({}, tmp_path / "unused")
    sink = W.Ledger(tmp_path, detector, ["\x00" * 600], "exact-substring-v1")
    sink.zeros(b"\x00" * 512, {"scope": "layer", "layer_ordinal": 0, "tar_offset": 0})
    sink.zeros(b"\x00" * 512, {"scope": "layer", "layer_ordinal": 0, "tar_offset": 512})
    sink.flush_zeros()
    sink.stream.close()
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    found = findings(rows, "private_literal")
    assert len(found) == 1 and found[0]["byte_start"] == 0 and found[0]["byte_end"] == 600
    assert sink.zero_bytes == 1024


def test_zero_ranges_do_not_concatenate_across_distinct_layers(tmp_path):
    detector = FakeDetector({}, tmp_path / "unused")
    sink = W.Ledger(tmp_path, detector, ["\x00" * 600], "exact-substring-v1")
    sink.zeros(b"\x00" * 512, {"scope": "layer", "layer_ordinal": 0, "tar_offset": 0})
    sink.zeros(b"\x00" * 512, {"scope": "layer", "layer_ordinal": 1, "tar_offset": 0})
    sink.flush_zeros()
    sink.stream.close()
    assert sink.findings == 0


def test_body_and_padding_offsets_refer_to_actual_physical_ranges(tmp_path):
    report, rows = run(tmp_path, fixture(tmp_path, entries=[file("opt/file", b"x")]))
    assert report["valid"]
    body = next(row for row in rows if row.get("kind") == "layer_regular_content")
    padding = next(row for row in rows if row.get("scope") == "layer" and row.get("type") == "verified_zero_range" and row["bytes"] == 511)
    assert body["tar_offset"] == 512 and padding["tar_offset"] == 513


@pytest.mark.parametrize("role", ["config", "helper", "literal_inventory", "verification_report"])
def test_source_replacement_and_restored_mtime_cannot_pass_postscan_binding(tmp_path, role):
    authorization = fixture(tmp_path, policy=W.POLICY)
    binding = authorization[role]
    path = Path(binding["path"])

    class MutateSource(FakeDetector):
        def finish(self):
            result = super().finish()
            before = path.stat()
            data = path.read_bytes()
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            return result

    report, _ = run(tmp_path, authorization, detector_type=MutateSource)
    assert not report["valid"] and not report["complete"] and report["helper_joined"]
    assert digest(path.read_bytes()) != binding["sha256"]
    assert report["failure_code"] in {"input_changed_during_scan", "input_binding_changed"}


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_postscan_special_file_replacement_fails_before_reading_it(tmp_path, kind):
    authorization = fixture(tmp_path)
    path = Path(authorization["literal_inventory"]["path"])

    class ReplaceSource(FakeDetector):
        def finish(self):
            result = super().finish()
            path.unlink()
            if kind == "symlink":
                path.symlink_to(authorization["config"]["path"])
            else:
                os.mkfifo(path, 0o600)
            return result

    report, _ = run(tmp_path, authorization, detector_type=ReplaceSource)
    assert not report["valid"] and not report["complete"] and report["helper_joined"]
    assert report["failure_code"] in {"input_symlink", "input_ownership_or_type"}


def test_exact_json_bytes_are_rehashed_after_the_earlier_binding_check(tmp_path, monkeypatch):
    authorization = fixture(tmp_path)
    binding = authorization["literal_inventory"]
    def changed_descriptor_bytes(fd):
        return b'{"literals":["changed private inventory"]}'

    monkeypatch.setattr(W, "descriptor_bytes", changed_descriptor_bytes)
    with pytest.raises(W.ScanError, match="parsed_input_binding_changed"):
        W.bound_json(binding)



def test_literal_policy_is_compiled_once_and_record_state_is_independent(tmp_path, monkeypatch):
    calls = []
    original = W.compile_literals

    def count(values, policy):
        calls.append((len(values), policy))
        return original(values, policy)

    monkeypatch.setattr(W, "compile_literals", count)
    report, _ = run(tmp_path, fixture(tmp_path, entries=[file("opt/one", b"neutral"), file("opt/two", b"neutral")]))
    assert report["valid"] and len(calls) == 1






@pytest.mark.parametrize("kind", ["unlink", "symlink", "fifo"])
def test_archive_replaced_at_completion_still_closes_fd_and_fails(tmp_path, kind):
    authorization = fixture(tmp_path)
    archive = Path(authorization["archive"]["path"])

    class ReplaceArchive(FakeDetector):
        def finish(self):
            result = super().finish()
            archive.unlink()
            if kind == "symlink":
                archive.symlink_to(authorization["config"]["path"])
            elif kind == "fifo":
                os.mkfifo(archive, 0o600)
            return result

    descriptors_before = len(list(Path("/proc/self/fd").iterdir()))
    report, _ = run(tmp_path, authorization, detector_type=ReplaceArchive)
    assert len(list(Path("/proc/self/fd").iterdir())) == descriptors_before
    assert not report["valid"] and not report["complete"] and report["helper_joined"]
    assert report["failure_code"] == "archive_changed_during_scan"


















@pytest.mark.parametrize("engine", [None, {}, {"kind": "unreviewed"}])
def test_explicit_invalid_literal_engine_never_falls_back(tmp_path, engine):
    authorization = fixture(tmp_path)
    authorization["literal_engine"] = engine
    before = len(FakeDetector.instances)
    with pytest.raises(W.ScanError, match="literal_engine_schema"):
        run(tmp_path, authorization)
    assert len(FakeDetector.instances) == before


def optional_gzip(raw, *, flags=0, extra=b"AB\x03\x00oneXY\x03\x00two", name=b"advisory.tar", comment=b"advisory comment"):
    import struct
    import zlib

    header = bytearray(b"\x1f\x8b\x08" + bytes([flags]) + b"\0" * 4 + b"\0\xff")
    if flags & 4:
        header.extend(struct.pack("<H", len(extra)) + extra)
    if flags & 8:
        header.extend(name + b"\0")
    if flags & 16:
        header.extend(comment + b"\0")
    if flags & 2:
        header.extend(struct.pack("<H", zlib.crc32(header) & 0xFFFF))
    compressor = zlib.compressobj(wbits=-15)
    body = compressor.compress(raw) + compressor.flush()
    return bytes(header) + body + struct.pack("<II", zlib.crc32(raw), len(raw) & 0xFFFFFFFF), bytes(header)


@pytest.mark.parametrize("flags", range(32))
def test_every_defined_gzip_flag_combination_scans_entire_header(tmp_path, flags):
    entries = [file("opt/body", b"neutral body")]
    raw = tar_data(entries)
    compressed, header = optional_gzip(raw, flags=flags)
    authorization = fixture(tmp_path, entries=entries, raw=raw, compressed=compressed)
    report, records = run(tmp_path, authorization)
    assert report["valid"] and report["complete"] and report["regular_files"] == 1
    assert report["layers"][0]["diff_id"] == "sha256:" + digest(raw)
    recorded = [row for row in records if row.get("kind") == "raw_gzip_header"]
    assert len(recorded) == 1
    assert recorded[0]["bytes"] == len(header) and recorded[0]["sha256"] == digest(header)
    assert recorded[0]["compressed_offset"] == 0
    reader = io.BytesIO(compressed)
    assert W.gzip_header(reader) == header and reader.tell() == len(header)




@pytest.mark.parametrize("flag", [32, 64, 128, 224])
def test_gzip_reserved_flag_bits_rejected(flag):
    compressed, _ = optional_gzip(b"body", flags=flag)
    with pytest.raises(W.ScanError, match="reserved_gzip_header_flags"):
        W.gzip_header(io.BytesIO(compressed))


def test_optional_gzip_header_all_truncation_points_and_crc_corruption_rejected():
    compressed, header = optional_gzip(b"body", flags=31)
    for length in range(len(header)):
        with pytest.raises(W.ScanError, match="truncated_tar_range"):
            W.gzip_header(io.BytesIO(header[:length]))
    corrupted = bytearray(compressed)
    corrupted[len(header) - 1] ^= 1
    with pytest.raises(W.ScanError, match="gzip_header_crc_mismatch"):
        W.gzip_header(io.BytesIO(corrupted))


@pytest.mark.parametrize("extra", [b"A", b"AB\x05\x00x", b"AB\0\0x"])
def test_malformed_extra_subfield_lengths_rejected(extra):
    compressed, _ = optional_gzip(b"body", flags=4, extra=extra)
    with pytest.raises(W.ScanError, match="malformed_gzip_extra_subfield"):
        W.gzip_header(io.BytesIO(compressed))


def test_empty_optional_fields_and_advisory_paths_are_never_extracted(tmp_path):
    raw = tar_data([file("opt/body", b"neutral")])
    compressed, header = optional_gzip(raw, flags=31, extra=b"", name=b"", comment=b"")
    assert W.gzip_header(io.BytesIO(compressed)) == header
    compressed, _ = optional_gzip(raw, flags=10, name=b"../../advisory-must-not-be-created")
    report, _ = run(tmp_path, fixture(tmp_path, entries=[file("opt/body", b"neutral")], raw=raw, compressed=compressed))
    assert report["valid"] and not (tmp_path.parent / "advisory-must-not-be-created").exists()


def test_ci_policy_scans_complete_records_and_zero_padding(tmp_path):
    body = b"first\nprivate-regex-value\nlast\xff"
    authorization = fixture(tmp_path, entries=[file("opt/data", body)])
    authorization["confidentiality"] = write(tmp_path / "policy.json", js({"customer_pattern": "^private-regex-value$|first[\\s\\S]*last|\\x00{600}", "infra_pattern": None}))
    report, records = run(tmp_path, authorization)
    assert report["complete"] and not report["valid"]
    assert report["confidentiality_policy"]["infra"] == "not_configured"
    matched = [(row, finding) for row in records if row.get("type") == "record" for finding in row["findings"] if finding["rule_id"] == "customer-denylist"]
    assert any(row["kind"] == "layer_regular_content" and finding["start_byte"] == 6 for row, finding in matched)
    assert any(row["kind"] == "verified_zero_content" for row, _ in matched)
    assert all("private-regex-value" not in json.dumps(row) for row in records)
    receipts = [row for row in records if row["type"] == "confidentiality_record"]
    assert len(receipts) == report["records"]
    assert sum(row["bytes"] for row in receipts) == report["scanned_bytes"]


def test_regex_and_exact_literal_receipts_compose_without_dropping_or_duplicating(tmp_path):
    authorization = fixture(tmp_path, entries=[file("opt/body", b"private-operator-marker regex-only-value")])
    authorization["confidentiality"] = write(tmp_path / "policy.json", js({"customer_pattern": "regex-only-value"}))
    report, rows = run(tmp_path, authorization)
    assert report["complete"] and report["findings"] == 2
    assert len(findings(rows, "private_literal")) == 1
    assert sum(finding["rule_id"] == "customer-denylist" for row in rows if row["type"] == "record" for finding in row["findings"]) == 1
    receipt = report["confidentiality_policy"]["exact_literals"]
    assert receipt["status"] == "configured" and receipt["pattern_count"] == 1


@pytest.mark.parametrize("pattern", [None, "", " ", "["])
def test_invalid_required_regex_fails_before_archive_read_or_detector(tmp_path, monkeypatch, pattern):
    authorization = fixture(tmp_path)
    authorization["confidentiality"] = write(tmp_path / "policy.json", js({"customer_pattern": pattern}))
    original = W.open_private_fd
    def guard(value, **kwargs):
        assert str(value) != authorization["archive"]["path"]
        return original(value, **kwargs)
    monkeypatch.setattr(W, "open_private_fd", guard)
    with pytest.raises(W.C.ConfidentialityError):
        run(tmp_path, authorization)


@pytest.mark.parametrize("operation", ["bound_file", "bound_json", "open_private_fd"])
@pytest.mark.parametrize("replacement", ["symlink", "parent_symlink", "fifo"])
def test_descriptor_walk_rejects_swapped_leaf_or_parent_without_following(tmp_path, monkeypatch, operation, replacement):
    allowed = tmp_path / "allowed"
    allowed.mkdir(mode=0o700)
    folder = allowed / "folder"
    folder.mkdir(mode=0o700)
    target = folder / "input.json"
    bound = write(target, b'{}')
    outside = tmp_path / "outside.json"
    write(outside, b'{}')
    original = W.private_path
    def swap(value, **kwargs):
        result = original(value, **kwargs)
        if replacement == "parent_symlink":
            folder.rename(allowed / "old-folder")
            folder.symlink_to(allowed / "old-folder", target_is_directory=True)
        else:
            target.unlink()
            if replacement == "symlink":
                target.symlink_to(outside)
            else:
                os.mkfifo(target)
        return result
    monkeypatch.setattr(W, "private_path", swap)
    with W.authorized_roots(allowed, CHECKOUT), pytest.raises((W.ScanError, OSError)):
        getattr(W, operation)(bound if operation != "open_private_fd" else target)


def test_trusted_sources_are_readable_but_private_values_cannot_use_checkout(tmp_path):
    path = CHECKOUT / ".gitleaks.toml"
    with pytest.raises(W.ScanError, match="input_outside_authorized_roots"):
        W.open_private_fd(path)
    _, fd, _ = W.open_private_fd(path, secret=False)
    os.close(fd)


def test_source_inventory_cannot_omit_executed_policy_module(tmp_path):
    authorization = fixture(tmp_path)
    authorization["sources"].pop("npa/scripts/image_byte_scan/confidentiality.py")
    with pytest.raises(W.ScanError, match="scanner_source_binding_changed"):
        run(tmp_path, authorization)


def protocol_helper(authorization, tmp_path):
    """An executable framing fixture, not Gitleaks or a native qualification."""
    ready = {"type": "ready", "protocol": "whole-file-gitleaks.v1", "version": "8.28.0",
             "config_sha256": authorization["config"]["sha256"], "max_target_megabytes": 0,
             "ignore_inline_allow": True, "redact": 100, "removed_content_path_rules": W.REMOVED_PATH_RULES,
             "path_rules": [{"rule_id": name, "selector": "fixture", "has_content_regex": True} for name in W.REMOVED_PATH_RULES]
                           + [{"rule_id": "pkcs12-file", "selector": r"(?i)(?:^|\/)[^\/]+\.p(?:12|fx)$", "has_content_regex": False}]}
    script = f'''#!{sys.executable}
import hashlib,json,os,struct,sys
ready={ready!r}
fd=int(sys.argv[2]); data=os.pread(fd,os.fstat(fd).st_size,0)
if hashlib.sha256(data).hexdigest()!=ready["config_sha256"]: raise SystemExit(2)
print(json.dumps(ready),flush=True)
ordinal=total=0
while True:
 header=sys.stdin.buffer.read(8)
 if not header: break
 if len(header)!=8: raise SystemExit(2)
 size=struct.unpack(">Q",header)[0]; data=sys.stdin.buffer.read(size)
 if len(data)!=size: raise SystemExit(2)
 ordinal+=1;total+=size
 print(json.dumps({{"type":"result","ordinal":ordinal,"bytes":size,"sha256":hashlib.sha256(data).hexdigest(),"findings":[]}}),flush=True)
print(json.dumps({{"type":"summary","files":ordinal,"bytes":total,"findings":0}}),flush=True)
'''
    helper = tmp_path / "protocol-helper"
    authorization["helper"] = {**write(helper, script.encode()), "ready_sha256": W.sha(W.canonical(ready))}
    helper.chmod(0o700)
    fixture_tools_receipt(authorization, tmp_path, ready)
    return authorization


@pytest.mark.parametrize("alter", ["ready", "record_sha", "record_length", "record_order", "summary"])
def test_protocol_corruption_fails_and_joins_only_owned_helper(tmp_path, alter):
    authorization = protocol_helper(fixture(tmp_path), tmp_path)
    class CorruptResponse(W.Detector):
        instance = None
        def __init__(self, *args):
            type(self).instance = self
            super().__init__(*args)
        def _response(self):
            result = super()._response()
            if result["type"] == "ready" and alter == "ready":
                result["version"] = "wrong"
            if result["type"] == "result":
                if alter == "record_sha":
                    result["sha256"] = "0" * 64
                if alter == "record_length":
                    result["bytes"] += 1
                if alter == "record_order":
                    result["ordinal"] += 1
            if result["type"] == "summary" and alter == "summary":
                result["files"] += 1
            return result
    report, _ = run(tmp_path, authorization, detector_type=CorruptResponse)
    assert not report["complete"] and not report["valid"] and report["helper_joined"]
    assert CorruptResponse.instance.joined and CorruptResponse.instance.process.poll() is not None


def test_full_protocol_fixture_preserves_empty_and_all_zero_records(tmp_path):
    authorization = protocol_helper(fixture(tmp_path, entries=[file("opt/empty"), file("opt/body", b"neutral")]), tmp_path)
    report, rows = run(tmp_path, authorization, real=True)
    assert report["complete"] and report["helper_joined"]
    assert report["helper_summary"]["files"] == report["records"]
    assert any(row.get("kind") == "layer_regular_content" and row["bytes"] == 0 for row in rows)
    assert any(row.get("kind") == "verified_zero_content" for row in rows)


def test_cli_sigterm_receipt_and_helper_join_preserve_unrelated_sibling(tmp_path):
    import signal
    import subprocess
    import time
    authorization = protocol_helper(fixture(tmp_path), tmp_path)
    auth = tmp_path / "auth.json"
    write(auth, js(authorization))
    marker = tmp_path / "helper.json"
    output = tmp_path / "cancelled"
    wrapper = tmp_path / "wrapper.py"
    write(wrapper, f'''import json,os,signal,sys
from pathlib import Path
sys.path.insert(0,{str(SCRIPTS)!r})
from image_byte_scan import core as W
original=W.Ledger
class PausedLedger(original):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  pid=self.detector.process.pid
  Path({str(marker)!r}).write_text(json.dumps({{"pid":pid,"session":os.getsid(pid)}}))
  signal.pause()
W.Ledger=PausedLedger
raise SystemExit(W.main())
'''.encode())
    sibling = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"], start_new_session=True)
    child = subprocess.Popen([sys.executable, str(wrapper), "--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT),
                              "--authorization", str(auth), "--output-dir", str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.monotonic() + 10  # Synthetic process synchronization only.
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        identity = json.loads(marker.read_bytes())
        assert identity["pid"] == identity["session"]
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 1 and stdout == b"complete image byte scan failed\n" and stderr == b""
        result = json.loads((output / "report.json").read_bytes())
        assert result["failure_code"] == "scan_cancelled" and result["helper_joined"] and not result["complete"]
        assert sibling.poll() is None and not Path(f'/proc/{identity["pid"]}').exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        sibling.terminate()
        sibling.wait()


def test_confidentiality_source_same_size_replacement_fails_after_detector(tmp_path):
    authorization = fixture(tmp_path)
    policy = tmp_path / "policy.json"
    authorization["confidentiality"] = write(policy, b'{"customer_pattern":"first"}')
    class MutatingDetector(FakeDetector):
        def finish(self):
            result = super().finish()
            before = policy.stat()
            policy.write_bytes(b'{"customer_pattern":"other"}')
            os.utime(policy, ns=(before.st_atime_ns, before.st_mtime_ns))
            return result
    report, _ = run(tmp_path, authorization, detector_type=MutatingDetector)
    assert report["failure_code"] == "input_binding_changed"
    assert not report["complete"] and not report["valid"] and report["helper_joined"]


@pytest.mark.parametrize("module_name", ["ahocorasick", "aho_matcher", W.AHO_MODULE])
def test_native_engine_rejects_foreign_preloaded_module_before_reading_dependencies(monkeypatch, module_name):
    import types
    module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    binding = {"kind": "aho-corasick-v1", **{role: {"path": "must-not-be-read", "sha256": value} for role, value in W.AHO_PINS.items()}}
    with pytest.raises(W.ScanError, match="literal_engine_preloaded_module"):
        W.AuthorizedAho(binding)
    assert sys.modules[module_name] is module


def test_exact_only_empty_inventory_is_not_a_configured_confidentiality_policy(tmp_path):
    authorization = fixture(tmp_path)
    authorization["literal_inventory"] = {**write(tmp_path / "empty.json", js({"literals": []})), "matching_policy": "exact-substring-v1"}
    with pytest.raises(W.ScanError, match="nonempty_confidentiality_policy_required"):
        run(tmp_path, authorization)


def test_valid_regex_can_have_an_explicit_empty_literal_supplement(tmp_path):
    authorization = fixture(tmp_path)
    authorization["literal_inventory"] = {**write(tmp_path / "empty.json", js({"literals": []})), "matching_policy": "exact-substring-v1"}
    authorization["confidentiality"] = write(tmp_path / "policy.json", js({"customer_pattern": "synthetic-absent-name"}))
    report, _ = run(tmp_path, authorization)
    assert report["complete"] and report["valid"]
    assert report["confidentiality_policy"]["exact_literals"]["pattern_count"] == 0


def test_added_source_population_is_detected_after_scan(tmp_path, monkeypatch):
    authorization = fixture(tmp_path)
    original = W.source_bindings
    finished = False
    def added_source():
        result = original()
        if finished:
            result["synthetic-new-executable.py"] = {"path": "unread", "sha256": "0" * 64}
        return result
    class FinishingDetector(FakeDetector):
        def finish(self):
            nonlocal finished
            finished = True
            return super().finish()
    monkeypatch.setattr(W, "source_bindings", added_source)
    report, _ = run(tmp_path, authorization, detector_type=FinishingDetector)
    assert not report["complete"] and report["failure_code"] == "scanner_source_population_changed"


@pytest.mark.parametrize("change", ["source", "config", "helper"])
def test_stale_or_different_tool_receipt_cannot_be_accepted_as_current(tmp_path, change):
    authorization = fixture(tmp_path)
    receipt = W.bound_json(authorization["tools_receipt"])
    if change == "source":
        receipt["source"]["main.go"] = "0" * 64
    elif change == "config":
        receipt["config"]["sha256"] = "0" * 64
    else:
        receipt["helper"] = write(tmp_path / "different-helper", b"different synthetic helper")
    authorization["tools_receipt"] = write(tmp_path / "altered-tools.json", js(receipt))
    with pytest.raises(W.ScanError, match="tools_receipt_(source_changed|checkout_config_changed|authorization_changed)"):
        run(tmp_path, authorization)
