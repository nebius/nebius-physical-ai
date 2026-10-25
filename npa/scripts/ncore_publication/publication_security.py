"""Keep reviewed security populations unchanged across fresh gate executions."""

from image_byte_scan import core as W
from . import acceptance
from .process import file_sha


def _trivy(directory, relative, *, license_target=False):
    report = acceptance._json(directory / relative)
    # The retained unchanged-image executions prove these are invocation IDs and
    # creation timestamps, not scanner version, policy, database or findings.
    report.pop("CreatedAt", None)
    report.pop("ReportID", None)
    if license_target:
        W.require(
            report.get("ArtifactName") == str(directory / "components/licenses"),
            "publication_license_target_changed",
        )
        report["ArtifactName"] = "components/licenses"
    return report


def _selected_base(directory):
    receipt = acceptance._json(directory / "selected-base/selected.receipt.json")
    for key, relative in (
        ("sbom_sha256", "selected-base/selected.spdx.json"),
        ("report_sha256", "selected-base/selected.trivy.json"),
    ):
        W.require(
            receipt.pop(key) == file_sha(directory / relative),
            "publication_selected_base_binding",
        )
    sbom = acceptance._json(directory / "selected-base/selected.spdx.json")
    # SPDX generates a document UUID and timestamps per execution. All package,
    # file, relationship, annotation content and source hashes remain compared.
    sbom.pop("documentNamespace")
    sbom["creationInfo"].pop("created")
    for annotation in sbom.get("annotations", []):
        annotation.pop("annotationDate")
    return {
        "receipt": receipt,
        "sbom": sbom,
        "report": _trivy(directory, "selected-base/selected.trivy.json"),
    }


def _components(directory):
    receipt = acceptance._json(directory / "components/receipt.json")
    licenses = receipt["licenses"]
    W.require(
        licenses.pop("report_sha256")
        == file_sha(directory / "components/licenses.trivy.stdout"),
        "publication_component_license_binding",
    )
    for row in receipt["evaluations"]:
        if row["method"] != "grype-cpe":
            continue
        W.require(
            row.pop("report_sha256")
            == file_sha(
                directory / ("components/" + row["component"] + ".grype.stdout")
            )
            and row["database_status"].get("path")
            == str(directory / "components/db/6/vulnerability.db"),
            "publication_component_report_binding",
        )
        row["database_status"]["path"] = "components/db/6/vulnerability.db"
        # Database byte identity, scanner, query, every finding and disposition
        # remain exact. A new database or outcome needs a new explicit review.
    return {
        "receipt": receipt,
        "licenses": _trivy(
            directory, "components/licenses.trivy.stdout", license_target=True
        ),
    }


def _payload(directory):
    report = acceptance._json(directory / "payload.json")
    W.require(
        report.get("image") == str(directory / "inspection.tar"),
        "publication_payload_target_changed",
    )
    report["image"] = "inspection.tar"
    return {
        "report": report,
        "history": acceptance._json(directory / "payload-history.json"),
    }


def _population(directory):
    return {
        "trivy_all": _trivy(directory, "trivy-all.json"),
        "trivy_policy": _trivy(directory, "trivy-policy.json"),
        "selected_base": _selected_base(directory),
        "components": _components(directory),
        "payload": _payload(directory),
    }


def verify(original, fresh):
    """Reject changed reviewed findings without changing existing severity gates.

    Args:
        original: Hash-verified original reviewed gate directory.
        fresh: Directory whose complete fresh gates have passed.
    Returns:
        Canonical hash of the identical security decision populations.
    Raises:
        ValueError, OSError: Any decision, population, source or binding changed.
    """
    reviewed = _population(original)
    current = _population(fresh)
    W.require(reviewed == current, "publication_reviewed_security_population_changed")
    return W.sha(W.canonical(reviewed))
