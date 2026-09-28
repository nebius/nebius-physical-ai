"""Verify Content Agents physics evidence at the real publisher copy boundary."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from npa.deploy import images, publish_public


def _attestation_records(platform_digest):
    predicates = {
        "https://spdx.dev/Document": {"packages": [{"name": "fixture-package"}]},
        "https://slsa.dev/provenance/v0.2": {
            "buildType": "fixture-build",
            "materials": [{"uri": "https://source.example.invalid/repository"}],
        },
    }
    layers = []
    statements = {}
    for index, (predicate_type, predicate) in enumerate(predicates.items(), start=1):
        digest = "sha256:" + str(index) * 64
        layers.append(
            {
                "digest": digest,
                "annotations": {"in-toto.io/predicate-type": predicate_type},
            }
        )
        statements[digest] = {
            "subject": [
                {"digest": {"sha256": platform_digest.removeprefix("sha256:")}}
            ],
            "predicateType": predicate_type,
            "predicate": predicate,
        }
    return layers, statements


def _accepted_index(platform_digest, attestation_digest):
    return {
        "manifests": [
            {
                "digest": platform_digest,
                "platform": {"architecture": "amd64", "os": "linux"},
            },
            {
                "digest": attestation_digest,
                "annotations": {
                    "vnd.docker.reference.type": "attestation-manifest",
                    "vnd.docker.reference.digest": platform_digest,
                },
            },
        ]
    }


def _stub_registry_reads(mocker, accepted):
    # Bootstrap acceptance is a separately tested prerequisite to this physics gate.
    mocker.patch.object(
        publish_public,
        "verify_bootstrap_publication_source",
        return_value=(True, "fixture: bootstrap prerequisites accepted"),
    )
    layers, statements = _attestation_records(accepted["amd64_manifest"])
    attestation_digest = "sha256:" + "a" * 64
    index = _accepted_index(accepted["amd64_manifest"], attestation_digest)

    def manifest(args):
        assert args[0] == "manifest"
        if args[1].endswith("@" + attestation_digest):
            return {"layers": layers}
        assert args[1].endswith("@" + accepted["oci_digest"])
        return index

    mocker.patch.object(
        publish_public, "_crane_digest", return_value=(True, accepted["oci_digest"])
    )
    mocker.patch.object(
        publish_public,
        "_crane_manifest_readable",
        return_value=(True, "fixture-readable"),
    )
    mocker.patch.object(publish_public, "_crane_json", side_effect=manifest)
    mocker.patch.object(
        publish_public,
        "_crane_blob_json",
        side_effect=lambda _repository, digest: statements[digest],
    )


def _payload_counts(accepted):
    return {
        "specialized": {
            key: accepted["payload_scan"][key]
            for key in ("archives_scanned", "findings")
        },
        "general": {
            key: accepted["general_payload_scan"][key]
            for key in (
                "entries_scanned",
                "payload_hits",
                "history_hits",
                "weight_shaped_paths",
            )
        },
    }


@pytest.fixture
def content_agents_publisher(mocker, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    accepted = copy.deepcopy(images.content_agents_accepted_image_manifest())
    plan = [
        item
        for item in publish_public.build_publish_plan(
            target_registry="ghcr.io/example/workbench", development_git_sha="1" * 40
        )
        if item.tool == "content-agents"
    ]
    assert len(plan) == 1
    mocker.patch.object(publish_public, "build_publish_plan", return_value=plan)
    mocker.patch.object(
        images, "content_agents_accepted_image_manifest", return_value=accepted
    )
    _stub_registry_reads(mocker, accepted)
    payload = mocker.patch.object(
        publish_public,
        "_scan_content_agents_payload_exact_digest",
        return_value=_payload_counts(accepted),
    )
    trivy = mocker.patch.object(
        publish_public,
        "_scan_content_agents_trivy_exact_digest",
        return_value={
            key: accepted["vulnerability_scan"][key]
            for key in ("critical_total", "critical_with_fix", "secrets")
        },
    )
    copied = mocker.patch.object(publish_public, "_crane_copy", return_value=True)
    mocker.patch.object(
        publish_public, "anonymous_pull_ok", return_value=(True, "fixture-public")
    )
    return SimpleNamespace(
        accepted=accepted, payload=payload, trivy=trivy, copied=copied
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mass_or_density", float("nan")),
        ("mass_or_density", float("inf")),
        ("mass_or_density", True),
        ("mass_or_density", "1.0"),
        ("friction", float("nan")),
        ("friction", float("inf")),
        ("friction", True),
        ("friction", "0.5"),
    ],
)
def test_invalid_rigid_physics_blocks_real_publication_before_copy(
    content_agents_publisher, capsys, field, value
):
    fixture = content_agents_publisher
    fixture.accepted["rtx_proof"]["rigid_physics"][field] = value

    result = publish_public.main(["--target", "ghcr.io/example/workbench"])

    assert result == 1
    assert (
        "Content Agents accepted rigid-physics proof is invalid"
        in capsys.readouterr().err
    )
    fixture.payload.assert_not_called()
    fixture.trivy.assert_not_called()
    fixture.copied.assert_not_called()


def test_recorded_finite_physics_reaches_scans_and_copy(content_agents_publisher):
    fixture = content_agents_publisher

    result = publish_public.main(["--target", "ghcr.io/example/workbench"])

    assert result == 0
    fixture.payload.assert_called_once()
    fixture.trivy.assert_called_once()
    fixture.copied.assert_called_once()
    item = fixture.copied.call_args.args[0]
    assert item.source_ref.endswith("@" + fixture.accepted["oci_digest"])
