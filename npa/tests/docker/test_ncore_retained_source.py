"""Prove exact retained-source closure and review binding using isolated Git objects."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import core as W  # noqa: E402
from ncore_publication import acceptance as A, process as P, retained_source as S  # noqa: E402


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)

    def git(*argv):
        return subprocess.check_output(["git", *argv], cwd=repo).decode().strip()

    git("init", "-q")
    git("config", "user.email", "synthetic@example.invalid")
    git("config", "user.name", "Synthetic fixture")
    git("config", "commit.gpgsign", "false")
    for name, content in {
        S.RECIPE + "/Dockerfile": "FROM scratch\nCOPY src/npa/image.py /image.py\n",
        S.RECIPE + "/runtime.lock": "locked input\n",
        "npa/src/npa/image.py": "IMAGE = 1\n",
        "npa/src/npa/host.py": "HOST = 1\n",
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git("add", ".")
    git("commit", "-qm", "synthetic producer")
    producer_commit = git("rev-parse", "HEAD")
    (repo / "npa/src/npa/host.py").write_text("HOST = 2\n")
    git("commit", "-qam", "synthetic consumer")
    consumer_commit = git("rev-parse", "HEAD")
    return repo, producer_commit, consumer_commit


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    repo, producer_commit, consumer_commit = _source_repository(tmp_path)
    monkeypatch.setattr(P, "ROOT", repo)
    monkeypatch.setattr(P, "SOURCE_PATHS", ("npa",))
    monkeypatch.setattr(S, "HOST_PATHS", ("npa",))
    # Import-origin enforcement is separately exercised in publication closure tests.
    # This tiny Git fixture has no importable NPA package; record the real guard call.
    imports = []
    monkeypatch.setattr(P, "_verify_imported_sources", imports.append)
    monkeypatch.setattr(S, "_build_bindings", lambda *_: None)
    monkeypatch.setattr(S, "_execution_bindings", lambda *_: None)
    producer, consumer = S.identity(producer_commit), S.identity(consumer_commit)
    value = {
        "format": S.FORMAT,
        "producer": producer,
        "consumer": consumer,
        "reviewed_host_deltas": S._deltas(producer, consumer),
    }
    evidence = tmp_path / "evidence"
    path = evidence / S.BRIDGE_PATH
    manifest = {
        "development_sha": producer_commit,
        "retained_compatibility": {
            "format": S.FORMAT,
            "producer_commit": producer_commit,
            "consumer_commit": consumer_commit,
            "consumer_tree": consumer["tree"],
            "bridge_sha256": _write(path, value),
        },
    }
    return manifest, value, evidence, repo, imports


def _verify(bridge):
    manifest, _, evidence, _, _ = bridge
    with W.authorized_roots(evidence.parent, ROOT):
        return S.verify(manifest, evidence.parent, evidence)


def _reseal(bridge):
    manifest, value, evidence, _, _ = bridge
    manifest["retained_compatibility"]["bridge_sha256"] = _write(
        evidence / S.BRIDGE_PATH, value
    )


def test_complete_bridge_preserves_distinct_contexts(bridge):
    manifest, value, _, _, imports = bridge
    assert _verify(bridge) == manifest["retained_compatibility"]
    assert imports == [value["consumer"]["commit"]]
    assert value["producer"]["context_sha256"] != value["consumer"]["context_sha256"]
    assert value["producer"]["copy_inputs"] == value["consumer"]["copy_inputs"]


@pytest.mark.parametrize("side", ["producer", "consumer"])
@pytest.mark.parametrize(
    "field",
    [
        "tree",
        "context_sha256",
        "context_files",
        "copy_inputs",
        "recipe_inputs",
        "host_files",
    ],
)
def test_bridge_rejects_omitted_or_changed_closures(bridge, side, field):
    value = bridge[1]
    if isinstance(value[side][field], list):
        value[side][field].pop()
    else:
        value[side][field] = "0" * len(value[side][field])
    _reseal(bridge)
    with pytest.raises(ValueError, match="retained_source_closure"):
        _verify(bridge)


@pytest.mark.parametrize(
    "field,value", [("mode", "100755"), ("blob", "f" * 40), ("sha256", "f" * 64)]
)
def test_copy_mode_blob_and_bytes_cannot_be_substituted(bridge, field, value):
    bridge[1]["consumer"]["copy_inputs"][0][field] = value
    _reseal(bridge)
    with pytest.raises(ValueError, match="retained_source_closure"):
        _verify(bridge)


@pytest.mark.parametrize("change", ["delta", "stale", "dirty", "head", "import"])
def test_bridge_keeps_delta_head_dirty_and_import_guards(bridge, monkeypatch, change):
    manifest, value, evidence, repo, _ = bridge
    if change == "delta":
        value["reviewed_host_deltas"] = []
        _reseal(bridge)
    elif change == "stale":
        _write(evidence / S.BRIDGE_PATH, {**value, "stale": True})
    elif change == "dirty":
        (repo / "npa/src/npa/host.py").write_text("HOST = 3\n")
    elif change == "head":
        manifest["retained_compatibility"]["consumer_commit"] = manifest[
            "development_sha"
        ]
    else:

        def wrong_import(_):
            raise ValueError("host_npa_import_outside_checkout")

        monkeypatch.setattr(P, "_verify_imported_sources", wrong_import)
    with pytest.raises(ValueError):
        _verify(bridge)


@pytest.mark.parametrize("change", ["copy", "mode", "recipe", "lock"])
def test_even_a_complete_new_git_closure_cannot_change_image_inputs(bridge, change):
    manifest, value, _, repo, _ = bridge
    name = {
        "copy": "npa/src/npa/image.py",
        "mode": "npa/src/npa/image.py",
        "recipe": S.RECIPE + "/Dockerfile",
        "lock": S.RECIPE + "/runtime.lock",
    }[change]
    path = repo / name
    if change == "mode":
        path.chmod(0o755)
    else:
        path.write_text(path.read_text() + "\n# changed input\n")
    subprocess.run(["git", "add", "--", name], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "synthetic changed image input"], cwd=repo, check=True
    )
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    )
    value["consumer"] = S.identity(commit)
    value["reviewed_host_deltas"] = S._deltas(value["producer"], value["consumer"])
    manifest["retained_compatibility"].update(
        consumer_commit=commit, consumer_tree=value["consumer"]["tree"]
    )
    _reseal(bridge)
    with pytest.raises(ValueError, match="retained_image_inputs_changed"):
        _verify(bridge)


@pytest.mark.parametrize(
    "changed", [None, "argument", "provenance", "driver", "source", "image"]
)
def test_build_arguments_and_original_process_are_not_relabelled(changed):
    build = {
        "source_sha": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
        "archive_sha256": "c" * 64,
    }
    provenance = {
        "invocation": {
            "parameters": {"args": {"build-arg:SOURCE_SHA": build["source_sha"]}}
        }
    }
    bridge = {"build_provenance": copy.deepcopy(provenance)}
    values = {
        "build": build,
        "metadata": {
            "containerimage.digest": build["image_digest"],
            "buildx.build.provenance": provenance,
        },
        "driver": {**build, "returncode": 0},
    }
    if changed == "argument":
        provenance["invocation"]["parameters"]["args"]["build-arg:SOURCE_SHA"] = (
            "d" * 40
        )
        bridge["build_provenance"] = copy.deepcopy(provenance)
    elif changed == "provenance":
        bridge["build_provenance"] = {}
    elif changed == "driver":
        values["driver"]["returncode"] = 1
    elif changed == "source":
        values["driver"]["source_sha"] = "d" * 40
    elif changed == "image":
        values["metadata"]["containerimage.digest"] = "sha256:" + "d" * 64
    if changed:
        with pytest.raises(ValueError, match="retained_"):
            S._build_provenance(values, bridge)
    else:
        S._build_provenance(values, bridge)


@pytest.mark.parametrize("changed", [None, "source", "format", "bridge", "tree"])
def test_final_verification_rechecks_current_bridge_and_distinct_identity(
    monkeypatch, tmp_path, changed
):
    contract = {
        "format": S.FORMAT,
        "producer_commit": "a" * 40,
        "consumer_commit": "b" * 40,
        "consumer_tree": "c" * 40,
        "bridge_sha256": "d" * 64,
    }
    manifest = {"retained_compatibility": contract, "acceptance_verification": {}}
    statement = {"retained_evidence_root": "evidence"}
    calls = []

    def verify(*args):
        calls.append(args)
        return contract

    monkeypatch.setattr(S, "verify", verify)
    A._retained_final_binding(manifest, statement, tmp_path)
    if changed:
        key = {
            "source": "producer_commit",
            "format": "format",
            "bridge": "compatibility_bridge_sha256",
            "tree": "consumer_tree",
        }[changed]
        manifest["acceptance_verification"][key] = "changed"
        with pytest.raises(ValueError, match="retained_final_verification_binding"):
            A._retained_final_binding(manifest, statement, tmp_path, verify_only=True)
    else:
        A._retained_final_binding(manifest, statement, tmp_path, verify_only=True)
    assert len(calls) == 2


def test_referenced_original_must_be_in_protected_inventory(tmp_path):
    evidence, gates = tmp_path / "evidence", tmp_path / "gates"
    original = evidence / "original.json"
    digest = _write(original, {"synthetic": True})
    provenance = {
        "execution": {
            "path": "original.json",
            "bytes": original.stat().st_size,
            "sha256": digest,
        }
    }
    _write(evidence / "s3-probe-provenance.json", provenance)
    manifest = {"prepublication": {}}
    with W.authorized_roots(tmp_path, ROOT):
        with pytest.raises(ValueError, match="missing_from_protected_inventory"):
            A._retained_inventory(manifest, tmp_path, evidence, gates, [])
        A._retained_inventory(
            manifest,
            tmp_path,
            evidence,
            gates,
            [{**provenance["execution"], "path": "evidence/original.json"}],
        )


def _review_records():
    contract = {
        "format": S.FORMAT,
        "producer_commit": "a" * 40,
        "consumer_commit": "b" * 40,
        "consumer_tree": "c" * 40,
        "bridge_sha256": "d" * 64,
    }
    statement = {
        "format": "npa_ncore_acceptance_statement_v2",
        "status": "pending_independent_review",
        "manifest": {"retained_compatibility": contract},
        "candidate_commit": "b" * 40,
        "retained_compatibility": copy.deepcopy(contract),
        "manifest_sha256": "a" * 64,
        "evidence_inventory": [],
        "evidence_inventory_sha256": A._sha_value([]),
    }
    review = {
        "format": "npa_ncore_acceptance_review_v2",
        "verdict": "ACCEPTED",
        "candidate_commit": "b" * 40,
        "statement_sha256": "e" * 64,
        "evidence_inventory_sha256": A._sha_value([]),
        "manifest_sha256": "a" * 64,
        "objective_evidence_reviewed": True,
        "visual_evidence_reviewed": True,
        "cleanup_reviewed": True,
        "reviewer_id": "synthetic-independent",
        "retained_compatibility_reviewed": True,
        "retained_compatibility": copy.deepcopy(contract),
    }
    return contract, statement, review


@pytest.mark.parametrize(
    "changed", [None, "consumer", "producer", "bridge", "review", "unreviewed"]
)
def test_review_binds_both_sources_and_exact_bridge(changed):
    contract, statement, review = _review_records()
    if changed in {"consumer", "producer", "bridge"}:
        key = {
            "consumer": "consumer_commit",
            "producer": "producer_commit",
            "bridge": "bridge_sha256",
        }[changed]
        review["retained_compatibility"][key] = "f" * len(contract[key])
    elif changed == "review":
        review["statement_sha256"] = "f" * 64
    elif changed == "unreviewed":
        review["retained_compatibility_reviewed"] = 1
    if changed:
        with pytest.raises(ValueError, match="review"):
            A._review_binding(statement, review, "e" * 64)
    else:
        A._review_binding(statement, review, "e" * 64)
