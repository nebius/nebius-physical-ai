"""Resume exact accepted transfers without touching a real registry."""

import copy

import pytest

from test_ncore_visibility_handoff import (
    publication as publication,
    publication_umask as publication_umask,
)
from test_ncore_oci_publication import private as private, _fixture
from ncore_publication import cli, handoff, registry, process


def _transport(monkeypatch, publication, failure):
    state = {"digest": None, "copies": 0, "visibility": 0, "readback": 0}
    _, digest, _ = _fixture()
    files, _, _ = _fixture()
    monkeypatch.setattr(
        registry,
        "run",
        lambda argv, output, **_: output.write_bytes(
            files["blobs/sha256/" + digest[7:]]
        ),
    )
    monkeypatch.setattr(registry, "_observed", lambda *_: state["digest"])

    def copy_graph(*_):
        state["copies"] += 1
        state["digest"] = digest

    def visibility(*_):
        state["visibility"] += 1
        if failure == "visibility" and state["visibility"] == 1:
            raise handoff.AdministratorHandoffRequired(
                publication.build["source_sha"], digest, "b" * 64
            )

    def readback(*_):
        state["readback"] += 1
        if failure == "readback" and state["readback"] == 1:
            raise ValueError("synthetic anonymous readback failure")

    monkeypatch.setattr(registry, "_copy", copy_graph)
    monkeypatch.setattr(registry, "_public_visibility", visibility)
    monkeypatch.setattr(registry, "_readback", readback)
    publication.args.acceptance = publication.args.analysis_root / "acceptance.json"
    process.write_json(
        publication.args.acceptance, {"synthetic_accepted_workload": True}
    )
    return state


def _transfer(publication):
    return registry.transfer(
        publication.args,
        publication.directory,
        publication.build,
        publication.graph,
        publication.verification,
    )


def _retry(publication):
    args = publication.args
    args.resume_transfer = publication.directory / "completed-transfer.json"
    args.resume_transfer_sha256 = process.file_sha(args.resume_transfer)
    publication.directory = args.output_dir / "retry"
    publication.directory.mkdir(mode=0o700)


@pytest.mark.parametrize("failure", ["visibility", "readback"])
def test_completed_transfer_resumes_visibility_and_anonymous_checks(
    publication, monkeypatch, failure
):
    state = _transport(monkeypatch, publication, failure)
    with pytest.raises(ValueError):
        _transfer(publication)
    _retry(publication)
    assert _transfer(publication) == publication.build["image_digest"]
    assert state == {
        "digest": publication.build["image_digest"],
        "copies": 1,
        "visibility": 2,
        "readback": 1 if failure == "visibility" else 2,
    }


@pytest.mark.parametrize(
    "foreign", ["source", "archive", "acceptance", "graph", "hash", "tag", "absent"]
)
def test_resume_refuses_foreign_or_changed_transfer_before_visibility(
    publication, monkeypatch, foreign
):
    state = _transport(monkeypatch, publication, "visibility")
    with pytest.raises(handoff.AdministratorHandoffRequired):
        _transfer(publication)
    _retry(publication)
    if foreign == "source":
        publication.build = {**publication.build, "source_sha": "b" * 40}
    elif foreign == "archive":
        publication.args.analysis_root.joinpath("build/image.oci.tar").write_bytes(
            b"other archive"
        )
    elif foreign == "acceptance":
        publication.args.acceptance.write_text('{"foreign_accepted_workload":true}')
    elif foreign == "graph":
        publication.graph = copy.deepcopy(publication.graph)
        publication.graph["receipt"]["blobs"][0]["size"] += 1
    elif foreign == "hash":
        publication.args.resume_transfer_sha256 = "0" * 64
    elif foreign == "tag":
        state["digest"] = "sha256:" + "b" * 64
    else:
        state["digest"] = None
    with pytest.raises(ValueError):
        _transfer(publication)
    assert state["copies"] == 1 and state["visibility"] == 1 and state["readback"] == 0


def test_resume_arguments_require_both_private_path_and_pinned_hash(publication):
    args = publication.args
    args.action = "publish"
    args.resume_transfer = args.acceptance = args.analysis_root / "synthetic.json"
    process.write_json(args.resume_transfer, {"fixture": True})
    args.resume_transfer_sha256 = None
    with pytest.raises(ValueError, match="invalid_resume"):
        cli._resume_input(args)
    args.resume_transfer_sha256 = process.file_sha(args.resume_transfer)
    cli._resume_input(args)
    args.action = "check"
    with pytest.raises(ValueError, match="invalid_resume"):
        cli._resume_input(args)
