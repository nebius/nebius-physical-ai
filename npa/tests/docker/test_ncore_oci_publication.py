"""Exercise NCore gate ordering and exact OCI transfer without registry writes."""

import copy
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import artifact, cli, gates, process, provenance, registry  # noqa: E402

SHA = "a" * 40


def _tar(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, value in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    return output.getvalue()


def _fixture(*, layers=1, tails=(), attested=True, source=SHA):
    blobs = {}

    def blob(value, media):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        digest = "sha256:" + W.sha(raw)
        blobs["blobs/sha256/" + digest[7:]] = raw
        return {"digest": digest, "size": len(raw), "mediaType": media}

    def manifest(config, layers):
        return blob(dict(schemaVersion=2, mediaType="application/vnd.oci.image.manifest.v1+json",
                         config=config, layers=layers), "application/vnd.oci.image.manifest.v1+json")

    raw = _tar({"etc/os-release": b'ID=debian\nVERSION_ID="12"\n', "opt/source": b"actual fixture bytes"})
    runtime_layers, diff_ids = _runtime_layers(blob, raw, layers, tails)
    ncore = json.loads((ROOT / "npa/docker/workbench/ncore/source-lock.json").read_bytes())["ncore"]["revision"]
    config = blob({"architecture": "amd64", "os": "linux", "rootfs": {
        "type": "layers", "diff_ids": diff_ids}, "config": {
            "User": "1000", "Labels": {"org.opencontainers.image.revision": source,
                "npa.ncore.revision": ncore,
                "org.nebius.npa.skypilot-bootstrap-contract": "skypilot-0.12.2-v1"}}},
        "application/vnd.oci.image.config.v1+json")
    runtime = manifest(config, runtime_layers)
    runtime["platform"] = {"os": "linux", "architecture": "amd64"}
    statements = _statements(blob, runtime, source)
    att_config = blob({"architecture": "unknown", "os": "unknown",
        "rootfs": {"type": "layers", "diff_ids": [row["digest"] for row in statements]}},
        "application/vnd.oci.image.config.v1+json")
    attestation = manifest(att_config, statements)
    attestation.update(platform={"os": "unknown", "architecture": "unknown"}, annotations={
        "vnd.docker.reference.type": "attestation-manifest", "vnd.docker.reference.digest": runtime["digest"]})
    index = dict(schemaVersion=2, mediaType="application/vnd.oci.image.index.v1+json",
                 manifests=[runtime, attestation] if attested else [runtime])
    publication = blob(index, "application/vnd.oci.image.index.v1+json")
    wrapper = dict(schemaVersion=2, mediaType="application/vnd.oci.image.index.v1+json", manifests=[publication])
    files = {"oci-layout": b'{"imageLayoutVersion":"1.0.0"}', "index.json": json.dumps(wrapper).encode(), **blobs}
    return files, publication["digest"], raw


def _runtime_layers(blob, raw, count, tails):
    layer = blob(gzip.compress(raw, mtime=0), "application/vnd.oci.image.layer.v1.tar+gzip")
    runtime_layers = [layer] * count
    diff_ids = ["sha256:" + W.sha(raw)] * count
    for tail in tails:
        compressed = tail.startswith(b"\x1f\x8b")
        media = "application/vnd.oci.image.layer.v1.tar" + ("+gzip" if compressed else "")
        runtime_layers.append(blob(tail, media))
        diff_ids.append("sha256:" + W.sha(gzip.decompress(tail) if compressed else tail))
    return runtime_layers, diff_ids


def _statements(blob, runtime, source):
    statements = []
    base = json.loads((ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes())["base_image"]
    reference, digest = base.split("@")
    generator, generator_digest = provenance._sbom_generator()
    for kind, predicate in [
        (provenance.SPDX, {"packages": [{"name": "fixture-only"}]}),
        ("https://slsa.dev/provenance/v0.2", {"buildType": "https://mobyproject.org/buildkit@v1",
            "invocation": {"parameters": {"args": {"build-arg:SOURCE_SHA": source}}},
            "materials": [{"uri": "pkg:docker/" + reference.replace(":", "@") + "?platform=linux%2Famd64",
                           "digest": {"sha256": digest[7:]}},
                          {"uri": "pkg:" + generator, "digest": {"sha256": generator_digest[7:]}}]}),
    ]:
        statement = blob({"_type": "https://in-toto.io/Statement/v1", "predicateType": kind,
            "predicate": predicate, "subject": [{"digest": {"sha256": runtime["digest"][7:]}}]},
            "application/vnd.in-toto+json")
        statement["annotations"] = {"in-toto.io/predicate-type": kind}
        statements.append(statement)
    return statements


@pytest.fixture
def private(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, ROOT):
        yield tmp_path


def _archive(private, **kwargs):
    files, digest, raw = _fixture(**kwargs)
    path = private / "image.oci.tar"
    path.write_bytes(_tar(files))
    path.chmod(0o600)
    return path, digest, raw


def test_adapter_binds_real_layer_config_and_original_graph(private):
    import importlib.util

    path, digest, raw = _archive(private)
    graph, verification = artifact.inspect(path, digest)
    receipt = artifact.inspection_archives(path, digest, graph, private)
    spec = importlib.util.spec_from_file_location("publication_base_sources", ROOT / "npa/docker/workbench/ncore/base_sources.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    inventory = base.inventory(private / "inspection.tar")
    assert inventory["config_sha256"] == graph["image_config_digest"][7:]
    assert inventory["layers"][0]["diff_id"] == "sha256:" + W.sha(raw)
    assert (private / "rootfs.tar").read_bytes() == raw
    assert receipt["archive_sha256"] == verification["archive_sha256"] == process.file_sha(path)
    assert len(graph["receipt"]["blobs"]) == 8


@pytest.mark.parametrize("change", ["missing", "changed", "extra", "nonattested", "wrong-digest"])
def test_partial_changed_nonattested_graphs_cannot_reach_inspection(private, change):
    files, digest, _ = _fixture(attested=change != "nonattested")
    blob = next(name for name in files if name.startswith("blobs/"))
    if change == "missing":
        del files[blob]
    elif change == "changed":
        files[blob] += b"changed"
    elif change == "extra":
        files["unaccounted"] = b"unknown bytes"
    elif change == "wrong-digest":
        digest = "sha256:" + "e" * 64
    path = private / "image.oci.tar"
    path.write_bytes(_tar(files))
    path.chmod(0o600)
    with pytest.raises(ValueError):
        artifact.inspect(path, digest)


def test_adapter_refuses_multiple_runtime_layers(private):
    path, digest, _ = _archive(private, layers=2)
    graph, _ = artifact.inspect(path, digest)
    with pytest.raises(ValueError, match="one_scratch_layer"):
        artifact.inspection_archives(path, digest, graph, private)


def test_source_revision_and_mode_max_statement_binding(private):
    path, digest, _ = _archive(private)
    graph, _ = artifact.inspect(path, digest)
    index, config, blobs = artifact.documents(path, digest, graph)
    assert provenance.verify(index, config, blobs, graph, SHA)["packages"]
    for row in blobs.values():
        if row.get("predicateType") in provenance.SLSA:
            row["predicate"]["invocation"]["parameters"]["args"]["build-arg:SOURCE_SHA"] = "b" * 40
    with pytest.raises(ValueError, match="exact_source_sha"):
        provenance.verify(index, config, blobs, graph, SHA)


@pytest.mark.parametrize("change", ["root", "revision", "subject", "duplicate-sbom", "no-packages", "no-materials"])
def test_local_provenance_refuses_release_incompatible_evidence(private, change):
    path, digest, _ = _archive(private)
    graph, _ = artifact.inspect(path, digest)
    index, config, blobs = artifact.documents(path, digest, graph)
    statements = [row for row in blobs.values() if "predicateType" in row]
    if change == "root":
        config["config"]["User"] = "000:0"
    elif change == "revision":
        config["config"]["Labels"]["org.opencontainers.image.revision"] = "b" * 40
    elif change == "subject":
        statements[0]["subject"][0]["digest"]["sha256"] = "e" * 64
    elif change == "duplicate-sbom":
        att = blobs[index["manifests"][1]["digest"]]
        att["layers"].append(copy.deepcopy(att["layers"][0]))
    elif change == "no-packages":
        statements[0]["predicate"]["packages"] = []
    else:
        statements[1]["predicate"]["materials"] = []
    with pytest.raises(ValueError):
        provenance.verify(index, config, blobs, graph, SHA)


def _build_receipt(private, digest="sha256:" + "d" * 64):
    build = private / "build"
    build.mkdir(mode=0o700)
    metadata = build / "buildx.json"
    metadata.write_text(json.dumps({"containerimage.digest": digest}))
    metadata.chmod(0o600)
    receipt = dict(schema="npa.ncore.committed-oci-build.v1", source_sha=SHA,
                   image=gates.eligibility(SHA), image_digest=digest,
                   archive_sha256="e" * 64, context_sha256="c" * 64,
                   metadata={"path": str(metadata), "sha256": process.file_sha(metadata)})
    (build / "build.json").write_text(json.dumps(receipt))
    (build / "build.json").chmod(0o600)
    return receipt


@pytest.mark.parametrize("failure", ["source", "graph", "bytes", "delivery", "payload", "history", "trivy", "selected-base", "components", "bootstrap"])
def test_any_gate_failure_prevents_registry_or_visibility_mutation(private, monkeypatch, failure):
    path, digest, _ = _archive(private)
    receipt = _build_receipt(private, digest)
    receipt["archive_sha256"] = process.file_sha(path)
    (private / "build/build.json").write_text(json.dumps(receipt))
    path.rename(private / "build/image.oci.tar")
    args = SimpleNamespace(analysis_root=private, source_sha=SHA, action="publish",
                           output_dir=private / "gates", bootstrap_source=private)
    calls = []

    def gate(name):
        def execute(*_):
            calls.append(name)
            if name == failure:
                raise ValueError("actual gate failed")
        return execute

    monkeypatch.setattr(gates, "committed_source", lambda _: receipt["context_sha256"])
    monkeypatch.setattr(gates, "source_guards", gate("source"))
    inspect = artifact.inspect

    def graph(*args):
        gate("graph")()
        return inspect(*args)

    monkeypatch.setattr(artifact, "inspect", graph)
    monkeypatch.setattr(gates, "byte_scan", gate("bytes"))
    monkeypatch.setattr(provenance, "shipped_source", lambda *_: 1)
    for name, attribute in (("delivery", "_source_delivery"), ("payload", "_payload"),
                            ("history", "_payload_history"),
                            ("trivy", "_security"), ("selected-base", "_selected_base")):
        monkeypatch.setattr(gates, attribute, gate(name))
    monkeypatch.setattr(gates.components, "verify", gate("components"))
    monkeypatch.setattr(gates.bootstrap, "verify", gate("bootstrap"))
    monkeypatch.setattr(registry, "transfer", lambda *_: calls.append("registry mutation"))
    with pytest.raises(ValueError, match="gate failed"):
        cli._check_or_publish(args)
    ordered = ["source", "graph", "bytes", "delivery", "payload", "history", "trivy", "selected-base", "components", "bootstrap"]
    assert calls == ordered[:ordered.index(failure) + 1]
    assert not (args.output_dir / "prepublication.json").exists()


def test_publication_reexecutes_gates_instead_of_trusting_a_pass_file(private, monkeypatch):
    _build_receipt(private)
    (private / "old-pass.json").write_text('{"status":"pass"}')
    args = SimpleNamespace(analysis_root=private, source_sha=SHA, action="publish", output_dir=private / "gates")
    calls = []
    monkeypatch.setattr(gates, "verify", lambda *_: (calls.append("all gates") or {}, {}))
    monkeypatch.setattr(registry, "transfer", lambda *_: calls.append("transfer"))
    cli._check_or_publish(args)
    assert calls == ["all gates", "transfer"]


def test_build_metadata_cannot_redirect_the_gated_digest(private, monkeypatch):
    _build_receipt(private)
    (private / "build/buildx.json").write_text('{"containerimage.digest":"changed"}')
    monkeypatch.setattr(gates, "verify", lambda *_: pytest.fail("gate reached changed metadata"))
    args = SimpleNamespace(analysis_root=private, source_sha=SHA, action="publish", output_dir=private / "gates")
    with pytest.raises(ValueError):
        cli._check_or_publish(args)


@pytest.mark.parametrize("observed", [None, "same", "different"])
def test_immutable_tag_is_preserved_or_refused(private, monkeypatch, observed):
    path, digest, _ = _archive(private)
    graph, verification = artifact.inspect(path, digest)
    build = private / "build"
    build.mkdir(mode=0o700)
    path.rename(build / "image.oci.tar")
    calls = []
    files, _, _ = _fixture()
    raw_index = files["blobs/sha256/" + digest[7:]]
    monkeypatch.setattr(registry, "run", lambda argv, output, **_: output.write_bytes(raw_index))
    value = digest if observed == "same" else "sha256:" + "e" * 64 if observed else None
    monkeypatch.setattr(registry, "_observed", lambda *_: value)
    monkeypatch.setattr(registry, "_copy", lambda *_: calls.append("copy"))
    monkeypatch.setattr(registry, "_public_visibility", lambda *_: calls.append("visibility"))
    monkeypatch.setattr(registry, "_readback", lambda *_: calls.append("readback"))
    args = SimpleNamespace(analysis_root=private, authfile=private / "auth.json")
    receipt = {"image": gates.eligibility(SHA), "image_digest": digest}
    if observed == "different":
        with pytest.raises(ValueError, match="divergent"):
            registry.transfer(args, private, receipt, graph, verification)
        assert calls == []
    else:
        assert registry.transfer(args, private, receipt, graph, verification) == digest
        assert calls == (["copy"] if observed is None else []) + ["visibility", "readback"]


@pytest.mark.parametrize("error", ["unauthorized", "403 Forbidden", "429 Too Many Requests", "TLS failed"])
def test_registry_errors_are_never_tag_absence(private, monkeypatch, error):
    monkeypatch.setattr(registry.subprocess, "run", lambda *_, **__: subprocess.CompletedProcess([], 1, b"", error.encode()))
    with pytest.raises(ValueError, match="not_proven_absent"):
        registry._observed(gates.eligibility(SHA), private / "existing.json", private / "auth.json")


def test_copy_uses_all_manifests_and_preserves_original_digests(private, monkeypatch):
    path, digest, _ = _archive(private)
    _, verification = artifact.inspect(path, digest)
    commands = []

    def run(argv, output, **_):
        commands.append(argv)
        Path(argv[argv.index("--digestfile") + 1]).write_text(digest)

    monkeypatch.setattr(registry, "run", run)
    monkeypatch.setattr(registry, "_observed", lambda *_: None)
    registry._copy(SimpleNamespace(authfile=private / "auth.json"), private, gates.eligibility(SHA), digest, path, verification)
    assert commands[0][1:4] == ["copy", "--all", "--preserve-digests"]
    assert commands[0][-2] == "oci-archive:" + str(path)
    assert commands[0][-1].endswith("@" + digest)
    assert commands[1][-2] == commands[0][-1]
    assert commands[1][-1].endswith(":dev-" + SHA)
    path.write_bytes(path.read_bytes() + b"changed after gates")
    with pytest.raises(ValueError, match="gated_archive_changed"):
        registry._copy(SimpleNamespace(authfile=private / "auth.json"), private, gates.eligibility(SHA), digest, path, verification)
    assert len(commands) == 2


def test_tag_created_during_digest_upload_cannot_be_overwritten(private, monkeypatch):
    path, digest, _ = _archive(private)
    _, verification = artifact.inspect(path, digest)
    commands = []

    def run(argv, output, **_):
        commands.append(argv)
        (private / "copied-digest").write_text(digest)

    monkeypatch.setattr(registry, "run", run)
    monkeypatch.setattr(registry, "_observed", lambda *_: "sha256:" + "e" * 64)
    with pytest.raises(ValueError, match="divergent"):
        registry._copy(SimpleNamespace(authfile=private / "auth.json"), private,
                       gates.eligibility(SHA), digest, path, verification)
    assert len(commands) == 1 and commands[0][-1].endswith("@" + digest)


def test_visibility_refuses_an_unrelated_private_version(private, monkeypatch):
    commands = []

    def run(argv, output, **_):
        commands.append(argv)
        value = {"visibility": "private"} if output.name == "visibility.json" else [[{
            "name": "sha256:" + "f" * 64, "metadata": {"container": {"tags": ["unrelated"]}}}]]
        output.write_text(json.dumps(value))

    monkeypatch.setattr(registry, "run", run)
    with pytest.raises(ValueError, match="unvalidated_versions"):
        registry._public_visibility(private, gates.eligibility(SHA), "sha256:" + "d" * 64, {"receipt": {"blobs": []}})
    assert all("PATCH" not in command for command in commands)


def test_source_binding_uses_full_committed_context_and_scoped_dirty_check(monkeypatch):
    commands = []

    def output(argv, **_):
        commands.append(argv)
        if argv[1] == "rev-parse":
            return (SHA + "\n").encode()
        if argv[1] == "status":
            return b""
        return b"complete committed context tar"

    monkeypatch.setattr(process.subprocess, "check_output", output)
    monkeypatch.setattr(process, "_verify_imported_sources", lambda sha: None)
    assert process.committed_source(SHA) == W.sha(b"complete committed context tar")
    assert commands[-1] == ["git", "archive", SHA, "npa/src/npa", "npa/docker/workbench/ncore"]
    assert "--" in commands[1] and "npa/scripts/ncore_publication" in commands[1]
    assert "docs" not in commands[1]
    with pytest.raises(ValueError, match="full_source_sha_required"):
        process.committed_source(SHA[:12])


def test_build_environment_keeps_private_inputs_outside_context(monkeypatch):
    for name in ("CUSTOMER_DENYLIST", "INFRA_DENYLIST", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "GH_TOKEN", "TRIVY_SKIP_FILES"):
        monkeypatch.setenv(name, "private-fixture")
    assert "private-fixture" not in process.public_environment().values()
    assert process.CONTEXT == ("npa/src/npa", "npa/docker/workbench/ncore")


@pytest.mark.parametrize("changed", [False, True])
def test_anonymous_readback_checks_every_graph_identity_without_credentials(private, monkeypatch, changed):
    path, digest, _ = _archive(private)
    graph, _ = artifact.inspect(path, digest)
    directory = private / "readback"
    directory.mkdir(mode=0o700)
    calls = []
    files, _, _ = _fixture(source="b" * 40 if changed else SHA)

    def run(argv, output, **_):
        calls.append(argv)
        if argv[1] == "copy":
            assert "--all" in argv and "--preserve-digests" in argv
            assert "--src-no-creds" in argv and "--src-authfile" in argv
            assert json.loads((directory / "anonymous.json").read_bytes()) == {"auths": {}}
            archive = directory / "anonymous.oci.tar"
            archive.write_bytes(_tar(files))
            archive.chmod(0o600)
        else:
            assert "--no-creds" in argv
            output.write_bytes(files["blobs/sha256/" + digest[7:]])

    monkeypatch.setattr(registry, "run", run)
    monkeypatch.setattr(gates, "byte_scan", lambda *_: calls.append("native-byte-recheck"))
    args = SimpleNamespace(analysis_root=private)
    if changed:
        with pytest.raises(ValueError):
            registry._readback(args, directory, gates.eligibility(SHA), digest, graph)
        assert len(calls) == 1 and not (directory / "published.json").exists()
    else:
        registry._readback(args, directory, gates.eligibility(SHA), digest, graph)
        receipt = json.loads((directory / "published.json").read_bytes())
        assert receipt["graph"] == graph["receipt"]["blobs"]
        assert calls[-1] == "native-byte-recheck" and receipt["release_acceptance"] is False


@pytest.mark.parametrize("change", ["context", "metadata-overlap", "symlink", "auth-in-checkout"])
def test_private_inputs_reject_context_aliases_and_metadata_mixing(private, monkeypatch, change):
    monkeypatch.setattr(cli, "committed_source", lambda _: "c" * 64)
    monkeypatch.setenv("CUSTOMER_DENYLIST", "synthetic-customer-marker")
    monkeypatch.setenv("INFRA_DENYLIST", "synthetic-infra-marker")
    for name in ("annex", "metadata", "bootstrap"):
        (private / name).mkdir(mode=0o700)
    args = SimpleNamespace(analysis_root=private, source_sha=SHA, action="check",
        output_dir=private / "gates", annex=private / "annex", native_source=private / "annex",
        metadata=private / "metadata", bootstrap_source=private / "bootstrap",
        policy_mode="ci-regex", literal_inventory=None)
    if change == "context":
        args.metadata = ROOT / "npa/docker/workbench/ncore"
    elif change == "metadata-overlap":
        args.metadata = args.annex
    elif change == "symlink":
        (private / "alias").symlink_to(args.annex)
        args.annex = private / "alias"
    else:
        args.action = "publish"
        args.authfile = ROOT / ".gitleaks.toml"
    with pytest.raises((ValueError, OSError)):
        cli._inputs(args)


def test_shipped_source_is_compared_to_commit_bytes(private, monkeypatch):
    checkout = private / "source"
    packaging = checkout / "npa/docker/workbench/ncore"
    packaging.mkdir(parents=True)
    npa_source = checkout / "npa/src/npa"
    npa_source.mkdir(parents=True)
    (npa_source / "__init__.py").write_bytes(b"committed code")
    files = {"opt/npa/src/npa/__init__.py": b"committed code",
             "usr/share/doc/npa-ncore/npa-source-sha": (SHA + "\n").encode()}
    for relative, filename in (("usr/share/doc/npa-ncore/recipes/Dockerfile", "Dockerfile"),
                               ("usr/share/doc/npa-ncore/runtime-lock.json", "runtime-lock.json"),
                               ("opt/ncore/base-sources/recipes/base-source-lock.json", "base-source-lock.json")):
        (packaging / filename).write_bytes(b"committed recipe")
        files[relative] = b"committed recipe"
    monkeypatch.setattr(provenance, "ROOT", checkout)
    for name, relative in provenance.NATIVE_SOURCES.items():
        source = checkout / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"committed native input")
        files[name] = source.read_bytes()
    monkeypatch.setattr(provenance, "_committed_digest", lambda source, sha: process.file_sha(source))
    path = private / "rootfs.tar"
    path.write_bytes(_tar(files))
    assert provenance.shipped_source(path, SHA) == 9
    files["opt/npa/src/npa/__init__.py"] = b"changed code"
    path.write_bytes(_tar(files))
    with pytest.raises(ValueError, match="shipped_source_changed"):
        provenance.shipped_source(path, SHA)


def test_image_security_uses_pinned_scanner_without_hidden_package_filters(private, monkeypatch):
    commands = []
    graph = {"image_config_digest": "sha256:" + "a" * 64,
             "layers": [{"diff_id": "sha256:" + "b" * 64}]}

    def run(argv, output, **_):
        commands.append(argv)
        output.write_text(json.dumps({"SchemaVersion": 2, "ArtifactType": "container_image",
                                      "Metadata": {"ImageID": graph["image_config_digest"],
                                                   "DiffIDs": [graph["layers"][0]["diff_id"]]}}))

    monkeypatch.setattr(gates, "run", run)
    gates._security(private, graph)
    assert len(commands) == 2
    for command in commands:
        assert gates._TRIVY_CONTAINER_IMAGE in command
        assert command[command.index("--input") + 1] == "/evidence/inspection.tar"
        assert command[command.index("--scanners") + 1] == "vuln,secret,license"
    assert "--ignore-unfixed" in commands[0]
    assert "--ignore-unfixed=false" in commands[1]


def test_workflow_scopes_ncore_away_from_generic_load_push_and_attestations():
    steps = yaml.safe_load((ROOT / ".github/workflows/publish-public-images.yml").read_text())["jobs"]["build-development"]["steps"]
    by_name = {step.get("name"): step for step in steps}
    for name in ("Build immutable development image locally", "Push only after every pre-publication gate passes",
                 "Attest exact pushed digest provenance", "Attest exact pushed digest SBOM"):
        assert by_name[name]["if"] == "matrix.tool != 'ncore'"
    ncore = by_name["Gate and publish the exact NCore OCI graph"]
    assert ncore["if"] == "matrix.tool == 'ncore'"
    assert "publish_ncore_oci.py publish" in ncore["run"]
    assert "--metadata" in ncore["run"] and "--authfile" in ncore["run"]
    assert "upload-artifact" not in str(ncore)
