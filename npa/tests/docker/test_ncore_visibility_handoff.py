"""Exercise exact NCore admin handoffs using real OCI fixtures and mocked GHCR."""

from contextlib import nullcontext
import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from test_ncore_oci_publication import SHA, W, _archive, _build_receipt, _fixture, artifact, cli, registry
from test_ncore_oci_publication import private as private
from test_ncore_publication_diagnostics import FAILURE, HOSTILE
from ncore_publication import diagnostics, handoff, process


def _write(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    path.chmod(0o600)


@pytest.fixture
def publication(private):
    path, digest, _ = _archive(private)
    graph, verification = artifact.inspect(path, digest)
    build = _build_receipt(private, digest)
    build["archive_sha256"] = process.file_sha(path)
    _write(private / "build/build.json", build)
    path.rename(private / "build/image.oci.tar")
    args = SimpleNamespace(analysis_root=private, source_sha=SHA, authfile=private / "auth.json",
                           output_dir=private / "gates")
    directory = args.output_dir / "transfer"
    directory.mkdir(mode=0o700, parents=True)
    # Stand-ins for successful gate outputs are private and deliberately hostile.
    # Real gate ordering/failures are covered by the publication pipeline tests.
    for name in handoff._GATE_FILES:
        _write(args.output_dir / name, HOSTILE.encode())
    _write(args.output_dir / "graph.json", verification)
    _write(args.output_dir / "prepublication.json", {
        "status": "pass", "source_sha": SHA, "image_digest": digest,
        "archive_sha256": build["archive_sha256"], "release_acceptance": False,
    })
    _write(args.authfile, HOSTILE.encode())
    rows = [{"id": index + 1, "name": row["digest"], "metadata": {"package_type": "container",
             "container": {"tags": ["dev-" + SHA] if row["digest"] == digest else []}}}
            for index, row in enumerate(graph["receipt"]["blobs"])
            if row["mediaType"] in {registry.G.INDEX, registry.G.MANIFEST, registry.G.ATTESTATION}]
    package = {"visibility": "private", "name": "nebius-physical-ai/npa-ncore",
               "owner": {"login": "nebius"}, "package_type": "container", "version_count": len(rows)}
    return SimpleNamespace(args=args, directory=directory, build=build, graph=graph, verification=verification,
                           package=package, pages=[rows[:1], rows[1:]], commands=[])


def _registry(monkeypatch, publication, *, refreshed=None, observed=None):
    files, digest, _ = _fixture()
    raw_index = files["blobs/sha256/" + digest[7:]]

    def run(argv, output, **kwargs):
        publication.commands.append(argv)
        assert argv[:2] in (["gh", "api"], ["skopeo", "inspect"])
        assert "PATCH" not in argv and "--method" not in argv
        if output.name == "local-index.json":
            _write(output, raw_index)
        elif output.name.startswith("visibility"):
            _write(output, publication.package)
        else:
            assert "--paginate" in argv and "--slurp" in argv
            _write(output, refreshed if refreshed is not None and "refresh" in output.name else publication.pages)

    def inspect(argv, **kwargs):
        publication.commands.append(argv)
        assert argv[:3] == ["skopeo", "inspect", "--raw"]
        raw = observed if observed is not None and len(publication.commands) > 2 else raw_index
        return subprocess.CompletedProcess(argv, 0, raw, HOSTILE.encode())

    monkeypatch.setattr(registry, "run", run)
    monkeypatch.setattr(registry.subprocess, "run", inspect)


def _transfer(publication):
    return diagnostics.run_phase("registry-transfer", registry.transfer, publication.args,
                                 publication.directory, publication.build, publication.graph, publication.verification)


def _receipt(publication):
    return publication.directory / "administrator-handoff.json"


@pytest.mark.parametrize("visibility", ["private", "internal"])
def test_exact_nonpublic_inventory_yields_bound_private_failure(publication, monkeypatch, capsys, visibility):
    publication.package["visibility"] = visibility
    _registry(monkeypatch, publication)
    monkeypatch.setattr(registry, "_readback", lambda *_: pytest.fail("handoff cannot accept publication"))
    with pytest.raises(handoff.AdministratorHandoffRequired) as raised:
        _transfer(publication)
    receipt = json.loads(_receipt(publication).read_bytes())
    assert _receipt(publication).stat().st_mode & 0o777 == 0o600
    assert receipt["source_sha"] == SHA and receipt["image"] == publication.build["image"]
    assert receipt["image_digest"] == publication.build["image_digest"]
    assert receipt["platform_digest"] == publication.graph["image_manifest_digest"]
    assert receipt["config_digest"] == publication.graph["image_config_digest"]
    assert receipt["archive_sha256"] == publication.verification["archive_sha256"]
    assert receipt["graph"] == publication.graph["receipt"]["blobs"]
    assert receipt["graph_sha256"] == W.sha(W.canonical(publication.graph["receipt"]))
    assert receipt["package_inventory_sha256"] == W.sha(W.canonical(receipt["package_inventory"]))
    assert len(receipt["package_inventory"]["versions"]) == publication.package["version_count"]
    for name in handoff._GATE_FILES:
        assert receipt["evidence_sha256"][name] == process.file_sha(publication.args.output_dir / name)
    assert raised.value.receipt_sha256 == process.file_sha(_receipt(publication))
    assert receipt["release_acceptance"] is False
    assert not (publication.directory / "published.json").exists()
    assert HOSTILE not in _receipt(publication).read_text()
    output = capsys.readouterr()
    assert output.out == "" and HOSTILE not in output.err
    assert "phase=registry-visibility status=failure" in output.err
    assert output.err.endswith("phase=registry-transfer status=failure\n")


def test_public_fast_path_and_same_archive_retry_never_copy_or_patch(publication, monkeypatch):
    _registry(monkeypatch, publication)
    with pytest.raises(handoff.AdministratorHandoffRequired):
        _transfer(publication)
    original_hash = process.file_sha(publication.args.analysis_root / "build/image.oci.tar")
    publication.package = {"visibility": "public"}
    publication.directory = publication.args.output_dir / "retry-transfer"
    publication.directory.mkdir(mode=0o700)
    publication.commands.clear()
    readbacks = []
    monkeypatch.setattr(registry, "_readback", lambda *args: readbacks.append(args))
    assert _transfer(publication) == publication.build["image_digest"]
    assert len(readbacks) == 1
    assert [command for command in publication.commands if command[0] == "gh"] == [["gh", "api", registry.PACKAGE_API]]
    assert not any("copy" in command or "PATCH" in command for command in publication.commands)
    assert not _receipt(publication).exists()
    assert process.file_sha(publication.args.analysis_root / "build/image.oci.tar") == original_hash


def _bad_inventory(publication, change):
    rows = [row for page in publication.pages for row in page]
    index = next(row for row in rows if row["name"] == publication.build["image_digest"])
    platform = next(row for row in rows if row is not index)
    if change == "empty":
        publication.pages = []
    elif change == "flat":
        publication.pages = rows
    elif change == "empty-page":
        publication.pages.append([])
    elif change == "missing":
        publication.pages = [rows[:-1]]
    elif change == "missing-matching-count":
        publication.pages = [[index]]
        publication.package["version_count"] = 1
    elif change == "extra":
        rows.append({"id": 999, "name": "sha256:" + "f" * 64, "metadata": platform["metadata"]})
        publication.pages = [rows]
        publication.package["version_count"] = len(rows)
    elif change == "config-version":
        platform["name"] = publication.graph["image_config_digest"]
    elif change in {"count", "boolean-count"}:
        publication.package["version_count"] = True if change == "boolean-count" else 999
    else:
        _bad_row(index, platform, change)


def _bad_row(index, platform, change):
    if change == "duplicate-id":
        platform["id"] = index["id"]
    elif change == "duplicate-digest":
        platform["name"] = index["name"]
    elif change == "name":
        platform["name"] = HOSTILE
    elif change == "metadata":
        del platform["metadata"]["package_type"]
    elif change == "tag-on-platform":
        platform["metadata"]["container"]["tags"] = ["dev-" + SHA]
    else:
        index["metadata"]["container"]["tags"] = {
            "hostile-tag": [HOSTILE], "tags-string": "dev-" + SHA, "tags-object": {"dev-" + SHA: True},
            "missing-tag": [], "extra-tag": ["dev-" + SHA, "latest"], "invalid-tag": [None],
        }[change]


@pytest.mark.parametrize("change", [
    "empty", "flat", "empty-page", "missing", "missing-matching-count", "extra", "config-version", "count",
    "boolean-count", "duplicate-id", "duplicate-digest", "name", "metadata", "tag-on-platform", "hostile-tag",
    "tags-string", "tags-object", "missing-tag", "extra-tag", "invalid-tag",
])
def test_malformed_incomplete_extra_and_hostile_inventory_has_no_handoff(publication, monkeypatch, capsys, change):
    _bad_inventory(publication, change)
    _registry(monkeypatch, publication)
    with pytest.raises(ValueError) as raised:
        _transfer(publication)
    assert not isinstance(raised.value, handoff.AdministratorHandoffRequired)
    assert not _receipt(publication).exists()
    output = capsys.readouterr()
    assert output.out == "" and HOSTILE not in output.err


@pytest.mark.parametrize("change", ["version-id", "tag", "archive", "build", "metadata", "graph", "gate-file", "pass-file"])
def test_changed_content_or_evidence_has_no_handoff(publication, monkeypatch, change):
    refreshed = copy.deepcopy(publication.pages)
    if change == "version-id":
        refreshed[0][0]["id"] += 100
    files = {"archive": publication.args.analysis_root / "build/image.oci.tar",
             "build": publication.args.analysis_root / "build/build.json",
             "metadata": publication.args.analysis_root / "build/buildx.json",
             "graph": publication.args.output_dir / "graph.json",
             "pass-file": publication.args.output_dir / "prepublication.json"}
    if change in files:
        _write(files[change], {})
    elif change == "gate-file":
        (publication.args.output_dir / "entrypoint-bootstrap.log").unlink()
    _registry(monkeypatch, publication, refreshed=refreshed, observed=b"changed index" if change == "tag" else None)
    with pytest.raises((ValueError, OSError)) as raised:
        _transfer(publication)
    assert not isinstance(raised.value, handoff.AdministratorHandoffRequired)
    assert not _receipt(publication).exists()


@pytest.mark.parametrize("field,value", [
    ("visibility", HOSTILE), ("name", HOSTILE), ("owner", {"login": HOSTILE}),
    ("package_type", "npm"), ("version_count", None),
])
def test_unbound_package_cannot_offer_handoff(publication, monkeypatch, field, value):
    publication.package[field] = value
    _registry(monkeypatch, publication)
    with pytest.raises(ValueError) as raised:
        _transfer(publication)
    assert not isinstance(raised.value, handoff.AdministratorHandoffRequired)
    assert not _receipt(publication).exists()


@pytest.mark.parametrize("payload", [b'{"visibility":"private","visibility":"public"}', b"{broken", b"null"])
def test_malformed_json_cannot_offer_handoff(publication, monkeypatch, payload):
    _registry(monkeypatch, publication)
    monkeypatch.setattr(registry, "run", lambda argv, output, **kwargs: _write(output, payload))
    with pytest.raises((ValueError, TypeError)) as raised:
        registry._public_visibility(publication.args, publication.directory, publication.build,
                                    publication.graph, publication.verification)
    assert not isinstance(raised.value, handoff.AdministratorHandoffRequired)
    assert not _receipt(publication).exists()


@pytest.mark.parametrize("change", ["source", "dev-tag", "platform", "config", "blob", "expected-index"])
def test_handoff_requires_exact_source_and_graph_bindings(publication, monkeypatch, change):
    if change == "source":
        publication.args.source_sha = "b" * 40
    elif change == "dev-tag":
        publication.build["image"] += HOSTILE
    elif change in {"platform", "config"}:
        key = "image_manifest_digest" if change == "platform" else "image_config_digest"
        publication.verification[key] = "sha256:" + "f" * 64
    elif change == "blob":
        publication.graph["receipt"]["blobs"][0]["mediaType"] = HOSTILE
    else:
        publication.verification["expected_image_id"] = "sha256:" + "f" * 64
    _registry(monkeypatch, publication)
    with pytest.raises(ValueError) as raised:
        _transfer(publication)
    assert not isinstance(raised.value, handoff.AdministratorHandoffRequired)
    assert not _receipt(publication).exists()


def _cli(monkeypatch, publication, operation):
    monkeypatch.setattr(cli, "_inputs", lambda *_: None)
    monkeypatch.setattr(cli, "committed_npa_imports", lambda *_: nullcontext())
    monkeypatch.setattr(cli, "_check_or_publish", lambda *_: operation())
    return cli.main(["publish", "--source-sha", SHA, "--analysis-root", str(publication.args.analysis_root)])


def test_cli_handoff_is_nonzero_minimal_and_keeps_failure_phases(publication, monkeypatch, capsys):
    _registry(monkeypatch, publication)
    assert _cli(monkeypatch, publication, lambda: _transfer(publication)) == 1
    output = capsys.readouterr()
    expected = handoff.AdministratorHandoffRequired(SHA, publication.build["image_digest"],
                                                   process.file_sha(_receipt(publication)))
    assert output.out == FAILURE + handoff.public_summary(expected) + "\n"
    assert "phase=publish status=failure" in output.err
    assert "phase=publish status=pass" not in output.err
    assert HOSTILE not in output.out + output.err
    assert str(publication.args.analysis_root) not in output.out
    assert "https://" not in output.out


def test_changed_receipt_cannot_be_disclosed_as_a_valid_handoff(publication, monkeypatch, capsys):
    _registry(monkeypatch, publication)
    write = W.write_private_json

    def changed(directory, name, receipt):
        identity = write(directory, name, receipt)
        _write(directory / name, HOSTILE.encode())
        return identity

    monkeypatch.setattr(W, "write_private_json", changed)
    assert _cli(monkeypatch, publication, lambda: _transfer(publication)) == 1
    output = capsys.readouterr()
    assert output.out == FAILURE and HOSTILE not in output.err


class _HostileString(str):
    def __format__(self, specification):
        pytest.fail("hostile handoff identity was formatted")


@pytest.mark.parametrize("field", ["source_sha", "image_digest", "receipt_sha256"])
@pytest.mark.parametrize("value", [HOSTILE, None, [HOSTILE], _HostileString("a" * 40)])
def test_cli_revalidates_exception_fields_before_disclosure(publication, monkeypatch, capsys, field, value):
    error = handoff.AdministratorHandoffRequired(SHA, publication.build["image_digest"], "e" * 64)
    setattr(error, field, value)
    error.args = (HOSTILE,)

    def fail():
        raise error

    assert _cli(monkeypatch, publication, fail) == 1
    output = capsys.readouterr()
    assert output.out == FAILURE and HOSTILE not in output.err
