"""Prove component and license coverage refusals without executing image payloads."""

import copy
import http.client
import io
import json
from pathlib import Path
import tarfile

import pytest

from npa.deploy import ncore_component_advisories as advisory
from npa.deploy import ncore_component_inventory as inventory
from npa.deploy import ncore_component_scan as scan
from npa.deploy import ncore_component_sources as sources


@pytest.fixture
def grype_report():
    return {
        "source": {"type": "cpe", "target": "cpe:2.3:a:python:python:3.12.12:*:*:*:*:*:*:*"},
        "descriptor": {"name": "grype", "version": "0.118.0", "configuration": {
            "match": {"stock": {"using-cpes": True}},
            "db": {"validate-by-hash-on-start": True, "validate-age": True},
        }, "db": {"status": {"valid": True, "built": "2026-09-07T00:00:00Z"},
                   "providers": {"nvd": {"captured": "2026-09-07T00:00:00Z"}}}},
        "matches": [],
    }


def _finding(cpe, severity="Critical", versions=None):
    return {"artifact": {"cpes": [cpe], "version": "3.12.12"},
            "vulnerability": {"id": "CVE-2026-6100", "severity": severity,
                              "fix": {"versions": versions or [],
                                      "state": "fixed" if versions else "not-fixed"}}}


def test_cpe_zero_matches_has_explicit_evaluated_target(grype_report):
    cpe = grype_report["source"]["target"]
    assert advisory.grype_findings(grype_report, cpe) == []
    with pytest.raises(ValueError, match="target differs"):
        advisory.grype_findings(grype_report, cpe.replace("3.12.12", "3.12.14"))


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("matches"),
    lambda r: r.update(matches=None),
    lambda r: r["descriptor"].update(version="0.117.0"),
    lambda r: r["descriptor"]["db"]["status"].update(valid=False),
    lambda r: r["descriptor"]["db"].update(providers={}),
    lambda r: r["descriptor"]["configuration"].update(exclude=["**"]),
    lambda r: r["descriptor"]["configuration"].update({"only-fixed": True}),
    lambda r: r["descriptor"]["configuration"]["match"]["stock"].update({"using-cpes": False}),
    lambda r: r["descriptor"]["configuration"]["db"].update({"validate-age": False}),
    lambda r: r.update(ignoredMatches=[{"id": "suppressed"}]),
    lambda r: r["descriptor"]["configuration"].update(ignore=[{"package": {}}]),
])
def test_successful_exit_cannot_replace_evaluation_scope(grype_report, mutation):
    cpe = grype_report["source"]["target"]
    mutation(grype_report)
    with pytest.raises(ValueError):
        advisory.grype_findings(grype_report, cpe)


@pytest.mark.parametrize("severity,versions,blocking", [
    ("Critical", ["3.12.14"], True),
    ("Critical", [], False),
    ("High", ["3.12.14"], False),
    ("Medium", [], False),
])
def test_fixed_critical_and_unfixed_accounting(grype_report, severity, versions, blocking):
    cpe = grype_report["source"]["target"]
    grype_report["matches"] = [_finding(cpe, severity, versions)]
    result = advisory.grype_findings(grype_report, cpe)
    assert result[0]["blocking"] is blocking
    assert result[0]["fixed_versions"] == versions


@pytest.mark.parametrize("change", ["cpe", "version"])
def test_findings_cannot_belong_to_another_component(grype_report, change):
    cpe = grype_report["source"]["target"]
    finding = _finding(cpe, versions=["3.12.14"])
    if change == "cpe":
        finding["artifact"]["cpes"] = [cpe.replace("python:python", "libexpat:expat")]
    else:
        finding["artifact"]["version"] = "3.12.14"
    grype_report["matches"] = [finding]
    with pytest.raises(ValueError, match="component differs"):
        advisory.grype_findings(grype_report, cpe)


def _coverage_fixture():
    components, evaluations = [], []
    for name in ("cpython", "expat", "mpdecimal", "hacl", "karamel-runtime",
                 "blake2", "ncore", "pycolmap", "npa"):
        binding = {"version": "synthetic", "files_sha256": inventory._sha(name.encode()),
                   "query": {"commit": "a" * 40}}
        if name == "cpython":
            binding.update(version=inventory._CPYTHON_VERSION, query={
                "cpe": "cpe:2.3:a:python:python:" + inventory._CPYTHON_VERSION + ":*:*:*:*:*:*:*"})
        if name in inventory._BUNDLED_SOURCE_PROFILES:
            profile = copy.deepcopy(inventory._BUNDLED_SOURCE_PROFILES[name])
            binding.update(query=profile["query"], source_mapping=profile,
                           version="2.5.1" if name == "mpdecimal" else inventory._CPYTHON_COMMIT)
        components.append({"name": name, **binding})
        method = "grype-cpe" if "cpe" in binding["query"] else "osv-commit"
        evaluations.append({"component": name, **binding, "method": method, "findings": []})
        if "source_mapping" in binding:
            evaluations[-1].update(source_proof_sha256="d" * 64, parent_advisory_scope={
                "component": "cpython", "version": inventory._CPYTHON_VERSION,
                "query": components[0]["query"]})
        if name == "mpdecimal":
            evaluations[-1]["upstream_review"] = copy.deepcopy(sources._MPDECIMAL_REVIEW)
    return {"components": components}, evaluations, {"delivery_failures": [], "missing_license_scope": []}


def test_full_population_required_even_for_no_findings():
    population, evaluations, licenses = _coverage_fixture()
    assert scan.coverage_failures(population, evaluations, licenses) == []
    evaluations.pop(0)
    with pytest.raises(ValueError, match="population differs"):
        scan.coverage_failures(population, evaluations, licenses)


@pytest.mark.parametrize("field,value", [("version", "changed"), ("files_sha256", "b" * 64),
                                        ("query", {"commit": "c" * 40})])
def test_same_name_is_not_population_binding(field, value):
    population, evaluations, licenses = _coverage_fixture()
    evaluations[0][field] = value
    with pytest.raises(ValueError, match="binding differs"):
        scan.coverage_failures(population, evaluations, licenses)


def test_duplicate_component_does_not_inflate_coverage():
    population, evaluations, licenses = _coverage_fixture()
    evaluations.append(copy.deepcopy(evaluations[0]))
    with pytest.raises(ValueError, match="population differs"):
        scan.coverage_failures(population, evaluations, licenses)


def test_unmapped_bundled_code_and_missing_notices_refuse():
    population, evaluations, licenses = _coverage_fixture()
    population["components"][2]["query"] = None
    evaluations[2]["query"] = None
    evaluations[2]["method"] = "unmapped"
    licenses["delivery_failures"] = ["missing or changed delivered notice: hacl"]
    licenses["missing_license_scope"] = ["hacl"]
    failures = scan.coverage_failures(population, evaluations, licenses)
    assert "unmapped vulnerability evaluation: mpdecimal" in failures
    assert "missing or changed delivered notice: hacl" in failures
    assert "license scanner did not cover: hacl" in failures


def test_unmapped_component_cannot_be_relabelled_as_scanned():
    population, evaluations, licenses = _coverage_fixture()
    population["components"][2]["query"] = None
    evaluations[2]["query"] = None
    with pytest.raises(ValueError, match="method differs"):
        scan.coverage_failures(population, evaluations, licenses)


def test_empty_population_cannot_qualify():
    with pytest.raises(ValueError, match="population differs"):
        scan.coverage_failures({"components": []}, [], {
            "delivery_failures": [], "missing_license_scope": []})


def test_actual_cve_shape_is_a_separate_policy_refusal():
    population, evaluations, licenses = _coverage_fixture()
    evaluations[0]["findings"] = [{"id": "BIT-python-2026-6100", "blocking": True}]
    assert scan.coverage_failures(population, evaluations, licenses) == [
        "blocking advisory: cpython:BIT-python-2026-6100"]


def _license_report(target, name="cpython", identifiers=None):
    identifiers = identifiers or scan._LICENSE_IDS[name]
    return {"SchemaVersion": 2, "Trivy": {"Version": "0.72.0"},
            "ArtifactType": "filesystem", "ArtifactName": str(target), "Results": [
                {"Class": "license-file", "Licenses": [
                    {"Name": value, "FilePath": name + "/LICENSE"} for value in identifiers]}]}


def test_actual_license_scope_is_required_for_each_component(tmp_path):
    payload = _license_report(tmp_path)
    staged = {"cpython": {"scan_path": "cpython/LICENSE"}}
    result = scan.license_findings(payload, staged, tmp_path)
    assert "cpython" not in result["missing_license_scope"]
    assert "mpdecimal" in result["missing_license_scope"]
    payload["Results"][0]["Licenses"].pop()
    assert "cpython" in scan.license_findings(payload, staged, tmp_path)["missing_license_scope"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("Results"),
    lambda r: r.update(Results=None),
    lambda r: r.update(ArtifactName="different-directory"),
    lambda r: r["Trivy"].update(Version="0.74.0"),
    lambda r: r["Results"][0].update(Class="os-pkgs"),
    lambda r: r["Results"][0]["Licenses"][0].update(FilePath="unscanned/LICENSE"),
])
def test_license_report_from_wrong_scope_is_rejected(tmp_path, mutation):
    payload = _license_report(tmp_path)
    mutation(payload)
    with pytest.raises(ValueError):
        scan.license_findings(payload, {"cpython": {"scan_path": "cpython/LICENSE"}}, tmp_path)


def test_empty_license_analyzers_are_not_complete(tmp_path):
    payload = _license_report(tmp_path)
    payload["Results"] = []
    assert set(scan.license_findings(payload, {}, tmp_path)["missing_license_scope"]) == set(scan._LICENSE_IDS)


def test_osv_pagination_is_fully_evaluated(monkeypatch, tmp_path):
    queries = []
    pages = [{"next_page_token": "second"}, {"vulns": [{"id": "PSF-synthetic"}]}]

    def page(query):
        queries.append(dict(query))
        return json.dumps(pages.pop(0)).encode()

    monkeypatch.setattr(advisory, "_osv_page", page)
    component = {"name": "hacl", "query": {"commit": "a" * 40}}
    result = advisory._osv_evaluation(component, tmp_path)
    assert queries == [{"commit": "a" * 40}, {"commit": "a" * 40, "page_token": "second"}]
    assert result["findings"] == [{"id": "PSF-synthetic", "blocking": True}]
    assert len(result["pages"]) == 2


def test_repeated_osv_page_token_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(advisory, "_osv_page", lambda query: b'{"next_page_token":"cycle"}')
    with pytest.raises(ValueError, match="pagination"):
        advisory._osv_evaluation({"name": "hacl", "query": {"commit": "a" * 40}}, tmp_path)


def test_error_json_is_not_zero_advisories(monkeypatch, tmp_path):
    monkeypatch.setattr(advisory, "_osv_page", lambda query: b'{"error":"unavailable"}')
    with pytest.raises(ValueError, match="invalid OSV response"):
        advisory._osv_evaluation({"name": "ncore", "query": {"commit": "a" * 40}}, tmp_path)


def test_scanner_identity_is_verified_before_execution(tmp_path):
    forged = tmp_path / "forged"
    forged.write_text("not the pinned binary")
    with pytest.raises(ValueError, match="Grype executable hash"):
        advisory._grype_binary(tmp_path, forged)
    with pytest.raises(ValueError, match="Trivy executable hash"):
        scan._trivy_binary(tmp_path, forged)


def test_scanner_does_not_inherit_ambient_filters_or_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("GRYPE_ONLY_FIXED", "true")
    monkeypatch.setenv("TRIVY_SKIP_FILES", "**")
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-test-value")
    env = advisory._environment(tmp_path)
    assert set(env) == {"PATH", "HOME", "XDG_CACHE_HOME"}


def test_profile_refuses_unreviewed_embedded_binary_change():
    lock = json.loads(Path("npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    assert len(inventory._python_profile(lock)) == 665
    lock["cpython_elf_files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="embedded component profile"):
        inventory._python_profile(lock)


@pytest.mark.parametrize("field", ["python_distributions", "python_wheels"])
def test_runtime_download_lock_is_not_installed_package_evidence(field):
    lock = json.loads(Path("npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    lock[field] = [{"name": "pip", "version": "synthetic"}]
    with pytest.raises(ValueError, match="installed Python dependencies"):
        inventory._python_profile(lock)


def test_arbitrary_prefix_cannot_hide_a_new_component():
    args = ({}, {}, {}, {}, set())
    for path in ("opt/ncore/unknown.so", "usr/local/bin/other", "opt/venv/lib/new.dist-info/METADATA"):
        assert inventory._file_group(path, *args) is None


def test_unmapped_native_file_in_source_area_refuses():
    files = {"application.py": {"sha256": "a" * 64, "elf": True}}
    with pytest.raises(ValueError, match="native component"):
        inventory._classify(files, {}, {}, {}, {"application.py": "a" * 64})


def test_equal_name_changed_notice_does_not_get_staged(tmp_path):
    path = tmp_path / "rootfs.tar"
    body = b"changed license"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("LICENSE")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    population = {"components": [], "required_notices": {
        "cpython": {"path": "LICENSE", "sha256": "a" * 64}},
        "files": {"LICENSE": {"sha256": inventory._sha(body)}}}
    _, staged, failures = scan._stage_notices(path, population, tmp_path)
    assert not staged
    assert failures == ["missing or changed delivered notice: cpython"]


def test_new_distribution_cannot_be_classified_as_an_npa_file(tmp_path):
    path = tmp_path / "rootfs.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("opt/npa/injected.dist-info/METADATA")
        archive.addfile(member, io.BytesIO(b""))
    with tarfile.open(path) as archive:
        members = inventory.selected._archive_members(archive)
        with pytest.raises(ValueError, match="Python distribution"):
            inventory._files(archive, members)


@pytest.mark.parametrize("field,value", [("version", "3.12.12"), ("name", "pypy"), ("kind", "pypy")])
def test_unsupported_interpreter_identity_is_not_guessed(field, value):
    lock = json.loads(Path("npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    lock["components"][0][field] = value
    with pytest.raises(ValueError, match="component/version"):
        inventory._python_profile(lock)


@pytest.mark.parametrize("status", [301, 401, 500])
def test_osv_redirect_or_failure_never_becomes_empty_success(monkeypatch, status):
    class Response:
        def __enter__(self):
            self.status = status
            return self

        def __exit__(self, *args):
            pass

    class Connection:
        def __init__(self, host, **kwargs):
            assert host == "api.osv.dev" and kwargs["port"] == 443

        def request(self, method, path, **kwargs):
            assert method == "POST" and path == "/v1/query"
            assert set(kwargs["headers"]) == {"Content-Type"}

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", Connection)
    with pytest.raises(ValueError, match="status differs"):
        advisory._osv_page({"commit": "a" * 40})


def test_scanner_nonzero_exit_is_preserved(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(advisory.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=2, stdout=b'{"matches":[]}', stderr=b"scanner failed"))
    with pytest.raises(ValueError, match="scanner command failed"):
        advisory._run(["synthetic-scanner"], tmp_path, "probe")
    receipt = json.loads((tmp_path / "probe.command.json").read_bytes())
    assert receipt["exit"] == 2
    assert (tmp_path / "probe.stderr").read_bytes() == b"scanner failed"


@pytest.mark.parametrize("name,field", [
    ("mpdecimal", "source_mapping"), ("blake2", "source_mapping"),
    ("mpdecimal", "source_proof_sha256"), ("blake2", "source_proof_sha256"),
    ("mpdecimal", "parent_advisory_scope"), ("blake2", "parent_advisory_scope"),
    ("mpdecimal", "upstream_review"),
])
def test_empty_query_results_do_not_replace_source_or_release_review(name, field):
    population, evaluations, licenses = _coverage_fixture()
    row = next(row for row in evaluations if row["component"] == name)
    del row[field]
    with pytest.raises(ValueError):
        scan.coverage_failures(population, evaluations, licenses)


def test_python_cpe_cannot_replace_the_blake2_commit():
    population, evaluations, licenses = _coverage_fixture()
    component = next(row for row in population["components"] if row["name"] == "blake2")
    evaluation = next(row for row in evaluations if row["component"] == "blake2")
    for row in (component, evaluation):
        row["query"] = population["components"][0]["query"]
    evaluation["method"] = "grype-cpe"
    with pytest.raises(ValueError, match="source proof"):
        scan.coverage_failures(population, evaluations, licenses)


def test_parent_version_change_requires_source_review():
    population, evaluations, licenses = _coverage_fixture()
    population["components"][0]["version"] = "3.12.12"
    evaluations[0]["version"] = "3.12.12"
    with pytest.raises(ValueError, match="patch advisory scope"):
        scan.coverage_failures(population, evaluations, licenses)


def _small_source_profile():
    files = {"Modules/vendor/impl/hash.c": {
        "upstream_path": "src/hash.c", "upstream_sha256": inventory._sha(b"original\n"),
        "vendored_sha256": inventory._sha(b"original\n"), "patch_sha256": inventory._sha(b""),
    }}
    tree = {"Modules/vendor/impl/hash.c": inventory._sha(b"original\n"),
            "Modules/vendor/wrapper.c": inventory._sha(b"wrapper\n")}
    return {"vendor_prefix": "Modules/vendor/impl/", "upstream_prefix": "src/", "renames": {},
            "files_sha256": inventory._sha(inventory._canonical(files)),
            "cpython_tree_sha256": inventory._sha(inventory._canonical(tree))}


@pytest.mark.parametrize("changed", ["upstream", "vendored", "wrapper", "extra", "missing"])
def test_exact_source_proof_refuses_unreviewed_deltas(changed):
    profile = _small_source_profile()
    cpython = {"Modules/vendor/impl/hash.c": b"original\n", "Modules/vendor/wrapper.c": b"wrapper\n"}
    upstream = {"src/hash.c": b"original\n"}
    assert len(sources._file_mapping(profile, cpython, upstream)["files"]) == 1
    if changed == "upstream":
        upstream["src/hash.c"] = b"changed\n"
    elif changed == "vendored":
        cpython["Modules/vendor/impl/hash.c"] = b"changed\n"
    elif changed == "wrapper":
        cpython["Modules/vendor/wrapper.c"] = b"changed\n"
    elif changed == "extra":
        cpython["Modules/vendor/impl/injected.c"] = b"new\n"
    else:
        del cpython["Modules/vendor/impl/hash.c"]
    with pytest.raises(ValueError):
        sources._file_mapping(profile, cpython, upstream)


def _mpdecimal_sbom():
    profile = inventory._BUNDLED_SOURCE_PROFILES["mpdecimal"]
    return {"packages": [{"SPDXID": "SPDXRef-PACKAGE-mpdecimal", "name": "mpdecimal",
                          "versionInfo": "2.5.1", "downloadLocation": profile["upstream"]["url"],
                          "checksums": [{"algorithm": "SHA256",
                                         "checksumValue": profile["upstream"]["sha256"]}],
                          "externalRefs": [{"referenceCategory": "SECURITY",
                                            "referenceType": "cpe23Type",
                                            "referenceLocator": profile["query"]["cpe"]}]}]}


@pytest.mark.parametrize("field,value", [
    ("versionInfo", "4.0.0"), ("name", "libmpd"), ("checksums", []),
    ("externalRefs", []), ("downloadLocation", "https://example.org/wrong-project.tar.gz"),
])
def test_mpdecimal_query_must_belong_to_exact_psf_declared_source(field, value):
    profile = inventory._BUNDLED_SOURCE_PROFILES["mpdecimal"]
    sbom = _mpdecimal_sbom()
    sources._mpdecimal_identity({"Misc/sbom.spdx.json": json.dumps(sbom).encode()}, profile)
    sbom["packages"][0][field] = value
    with pytest.raises(ValueError, match="identity differs"):
        sources._mpdecimal_identity({"Misc/sbom.spdx.json": json.dumps(sbom).encode()}, profile)


def test_upstream_release_note_drift_is_not_an_empty_advisory_result(monkeypatch, tmp_path):
    def download(url, output, **kwargs):
        assert url == sources._MPDECIMAL_REVIEW["url"]
        assert kwargs == {"allowed_hosts": sources._SOURCE_HOSTS}
        output.write(b"new upstream security notice")

    monkeypatch.setattr(sources, "download_public_https", download)
    with pytest.raises(ValueError, match="release review changed"):
        sources._fetch(sources._MPDECIMAL_REVIEW, tmp_path, "upstream-review")
    assert (tmp_path / "upstream-review.download").read_bytes() == b"new upstream security notice"


def test_wrong_upstream_query_refuses_before_network(monkeypatch, tmp_path):
    def unexpected(*args):
        pytest.fail("unreviewed mapping reached the network")

    monkeypatch.setattr(sources, "_fetch", unexpected)
    component = {"name": "blake2", "source_mapping": inventory._BUNDLED_SOURCE_PROFILES["blake2"],
                 "query": {"commit": inventory._CPYTHON_COMMIT}}
    with pytest.raises(ValueError, match="unreviewed bundled source mapping"):
        sources._component_proof(component, {}, tmp_path)


def test_python_file_lock_cannot_disagree_with_pinned_elf_hashes():
    lock = json.loads(Path("npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    target = lock["cpython_elf_files"][0]["path"]
    next(row for row in lock["cpython_files"] if row["path"] == target)["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file identity differs"):
        inventory._python_profile(lock)


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_observed_python_elf_population_must_match_source_profile(mutation):
    lock = json.loads(Path("npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    python_files = inventory._python_profile(lock)
    files = {row["path"]: {"sha256": row["sha256"], "elf": True} for row in lock["cpython_elf_files"]}
    inventory._verify_python_elf(files, python_files)
    if mutation == "missing":
        files.pop(lock["cpython_elf_files"][0]["path"])
    elif mutation == "changed":
        files[lock["cpython_elf_files"][0]["path"]]["sha256"] = "0" * 64
    else:
        candidate = next(path for path, row in python_files.items() if "sha256" in row and path not in files)
        files[candidate] = {"sha256": python_files[candidate]["sha256"], "elf": True}
    with pytest.raises(ValueError, match="actual CPython ELF population"):
        inventory._verify_python_elf(files, python_files)


def _reviewed_lock():
    return json.loads(Path("npa/docker/workbench/ncore/base-source-lock.json").read_bytes())


def test_reviewed_python_profile_and_all_eight_notice_bytes():
    lock = _reviewed_lock()
    assert inventory._CPYTHON_COMMIT == "2abcf904b8dac8c999d2b3aac76681abb333798a"
    assert inventory._CPYTHON_ELF_INVENTORY == inventory._sha(
        inventory._canonical(lock["cpython_elf_files"]))
    assert len(lock["cpython_provenance"]["notices"]) == 8
    required = {path: digest for path, digest in inventory._NOTICES.values()}
    for row in lock["cpython_provenance"]["notices"]:
        assert required[row["path"]] == row["sha256"]
        source = Path("npa/docker/workbench/ncore/notices/cpython") / Path(row["path"]).name
        assert inventory._sha(source.read_bytes()) == row["sha256"]
        assert inventory._file_group(row["path"], {row["path"]: row}, {}, {}, {}, required) == "cpython"
    assert required["usr/local/lib/python3.12/LICENSE.txt"] == (
        "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf")
    locked = {row["path"]: row["sha256"] for row in lock["notices"] if "sha256" in row}
    for row in scan._LICENSE_TERMS.values():
        assert locked[row["path"]] == row["sha256"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(source_commit="a" * 40),
    lambda p: p.update(version="3.12.12"),
    lambda p: p["embedded_components"].update(expat="2.7.3"),
    lambda p: p["embedded_components"].update(hacl="a" * 40),
    lambda p: p["notices"].pop(),
    lambda p: p["notices"].append(p["notices"][0]),
    lambda p: p["notices"][0].update(sha256="a" * 64),
    lambda p: p["notices"][0].update(path="usr/share/doc/other/LICENSE"),
])
def test_provenance_changes_require_profile_review(mutation):
    lock = _reviewed_lock()
    mutation(lock["cpython_provenance"])
    with pytest.raises(ValueError, match="provenance differs|notice profile differs"):
        inventory._python_profile(lock)


def test_components_are_code_identities_and_queries_follow_reviewed_versions():
    lock = _reviewed_lock()
    files = inventory._python_profile(lock)
    groups = {"cpython": list(files), "ncore": ["opt/ncore/src/ncore/LICENSE"],
              "pycolmap": ["opt/ncore/src/pycolmap/LICENSE.txt"],
              "npa": ["usr/share/doc/npa-ncore/notices/NPA-LICENSE"]}
    for name in ("ncore", "pycolmap", "npa"):
        files[groups[name][0]] = {"sha256": "b" * 64}
    sources = {name: {"revision": "a" * 40} for name in ("ncore", "pycolmap")}
    components = inventory._components(groups, files, sources, "c" * 40)
    by_name = {row["name"]: row for row in components}
    assert set(by_name) == inventory._COMPONENT_NAMES
    assert by_name["cpython"]["version"] == "3.12.14"
    assert by_name["cpython"]["query"] == {"cpe": "cpe:2.3:a:python:python:3.12.14:*:*:*:*:*:*:*"}
    assert by_name["expat"]["version"] == "2.8.3"
    assert by_name["expat"]["query"] == {"cpe": "cpe:2.3:a:libexpat:expat:2.8.3:*:*:*:*:*:*:*"}
    assert by_name["blake2"]["version"] == inventory._CPYTHON_COMMIT
    assert by_name["hacl"]["query"] == {"commit": inventory._HACL_COMMIT}
    assert by_name["karamel-runtime"]["files"] == by_name["hacl"]["files"]
    assert len(by_name["karamel-runtime"]["files"]) == 4
    for name in ("mpdecimal", "blake2"):
        assert by_name[name]["query"] == inventory._BUNDLED_SOURCE_PROFILES[name]["query"]
        assert by_name[name]["source_mapping"] == inventory._BUNDLED_SOURCE_PROFILES[name]
    assert by_name["karamel-runtime"]["query"] is None


def test_license_only_rows_cannot_become_installed_components():
    population, evaluations, licenses = _coverage_fixture()
    assert scan.coverage_failures(population, evaluations, licenses) == []
    row = copy.deepcopy(population["components"][0])
    row["name"] = "hacl-fstar"
    population["components"].append(row)
    evaluations.append({"component": "hacl-fstar", "method": "osv-commit", "findings": [],
                        **{key: row[key] for key in ("version", "files_sha256", "query")}})
    with pytest.raises(ValueError, match="population differs"):
        scan.coverage_failures(population, evaluations, licenses)


def test_karamel_runtime_requires_its_own_advisory_evaluation():
    population, evaluations, licenses = _coverage_fixture()
    next(row for row in population["components"] if row["name"] == "karamel-runtime")["query"] = None
    row = next(row for row in evaluations if row["component"] == "karamel-runtime")
    row.update(query=None, method="unmapped")
    assert scan.coverage_failures(population, evaluations, licenses) == [
        "unmapped vulnerability evaluation: karamel-runtime"]


@pytest.fixture
def delivered_short_notices(monkeypatch):
    files, required = {}, {}
    for name in scan._NOTICE_TERMS:
        path, digest = inventory._NOTICES[name]
        raw = (Path("npa/docker/workbench/ncore/notices/cpython") / Path(path).name).read_bytes()
        files[path] = raw
        required[name] = {"path": path, "sha256": digest}
    terms = copy.deepcopy(scan._LICENSE_TERMS)
    for name, notice in terms.items():
        raw = ("synthetic full license terms: " + name).encode()
        files[notice["path"]] = raw
        notice["sha256"] = inventory._sha(raw)
    monkeypatch.setattr(scan, "_LICENSE_TERMS", terms)
    return files, {"components": [], "required_notices": required}


def _notice_archive(tmp_path, files, population):
    path = tmp_path / "rootfs.tar"
    with tarfile.open(path, "w") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    population["files"] = {name: {"sha256": inventory._sha(raw)} for name, raw in files.items()}
    return path


def test_scanned_bytes_include_short_notices_and_complete_terms(tmp_path, delivered_short_notices):
    files, population = delivered_short_notices
    path = _notice_archive(tmp_path, files, population)
    target, staged, failures = scan._stage_notices(path, population, tmp_path)
    assert failures == []
    assert set(staged) == set(scan._NOTICE_TERMS)
    for name, row in staged.items():
        origins = [population["required_notices"][name], scan._LICENSE_TERMS[scan._NOTICE_TERMS[name]]]
        assert row["origins"] == origins
        raw = (target / row["scan_path"]).read_bytes()
        assert raw == b"\n".join(files[origin["path"]] for origin in origins)
        assert row["sha256"] == inventory._sha(raw)


@pytest.mark.parametrize("name", ["hacl-krml", "hacl-fstar", "blake2", "blake2-python"])
@pytest.mark.parametrize("part", ["notice", "terms"])
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_each_header_and_full_terms_must_be_delivered(
        tmp_path, delivered_short_notices, name, part, mutation):
    files, population = delivered_short_notices
    origin = (population["required_notices"][name] if part == "notice"
              else scan._LICENSE_TERMS[scan._NOTICE_TERMS[name]])
    if mutation == "missing":
        files.pop(origin["path"])
    else:
        files[origin["path"]] = b"changed content"
    path = _notice_archive(tmp_path, files, population)
    _, staged, failures = scan._stage_notices(path, population, tmp_path)
    assert name not in staged
    assert "missing or changed delivered notice: " + name in failures


def test_all_notice_rows_require_actual_license_analyzer_scope(tmp_path):
    staged = {name: {"scan_path": name + "/LICENSE"} for name in scan._LICENSE_IDS}
    payload = _license_report(tmp_path)
    payload["Results"] = [_license_report(tmp_path, name)["Results"][0] for name in staged]
    assert scan.license_findings(payload, staged, tmp_path)["missing_license_scope"] == []
    for name in scan._NOTICE_TERMS:
        partial = copy.deepcopy(payload)
        for row in partial["Results"]:
            row["Licenses"] = [finding for finding in row["Licenses"]
                               if finding["FilePath"] != name + "/LICENSE"]
        assert scan.license_findings(partial, staged, tmp_path)["missing_license_scope"] == [name]


@pytest.mark.parametrize("part", ["notice", "terms"])
def test_archive_notice_bytes_are_rechecked_after_inventory(
        tmp_path, delivered_short_notices, part):
    files, population = delivered_short_notices
    path = _notice_archive(tmp_path, files, population)
    original_inventory = copy.deepcopy(population["files"])
    origin = (population["required_notices"]["hacl-fstar"] if part == "notice"
              else scan._LICENSE_TERMS["Apache-2.0"])
    files[origin["path"]] = b"changed after inventory"
    _notice_archive(tmp_path, files, population)
    population["files"] = original_inventory
    with pytest.raises(ValueError, match="notice changed while staging"):
        scan._stage_notices(path, population, tmp_path)
