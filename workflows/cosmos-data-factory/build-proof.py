"""Embed reviewed CDF evidence and the two existing specs into an offline HTML."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import yaml

WORKFLOW_SPECS = {
    "iaa": "paidf-image-attribute-augmentation.yaml",
    "evg": "paidf-event-video-generation.yaml",
}


def _embedded_asset(asset, evidence_directory):
    path = (evidence_directory / asset["path"]).resolve()
    if not path.is_relative_to(evidence_directory.resolve()):
        raise ValueError("Embedded media must stay within the evidence directory")
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != asset["sha256"]:
        raise ValueError(f"Asset hash mismatch: {asset['id']}")
    mime = asset["mime_type"]
    if mime not in {"image/jpeg", "image/png", "video/mp4"}:
        raise ValueError(f"Unsupported embedded media: {mime}")
    return {
        "id": asset["id"],
        "title": asset.get("title"),
        "alt": asset["alt"],
        "caption": asset["caption"],
        "mime_type": mime,
        "sha256": digest,
        "bytes": len(body),
        "src": f"data:{mime};base64,{base64.b64encode(body).decode('ascii')}",
    }


def _workflow_details(workflow, directory):
    path = directory / workflow["spec_filename"]
    if workflow["spec_filename"] != WORKFLOW_SPECS.get(workflow["id"]):
        raise ValueError("Evidence workflow ID must match its CDF spec filename")
    spec = yaml.safe_load(path.read_text())
    workflow["spec_path"] = f"workflows/cosmos-data-factory/{path.name}"
    workflow["spec_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    workflow["model"] = spec["config"]["generation_model"]
    workflow["model_revision"] = spec["config"]["generation_revision"]
    workflow["stages"] = []
    for name, state in spec["states"].items():
        resources = spec["resources"][state["resources"]]
        workflow["stages"].append(
            {
                "name": name,
                "component": state.get("toolRef", "Pinned upstream provenance record"),
                "description": state.get("description", ""),
                "accelerators": resources.get("accelerators", "CPU"),
                "outputs": state.get("outputs", {}),
            }
        )
    return workflow


def _media_references(workflow):
    evidence = workflow["evidence"]
    references = list(evidence["media_ids"])
    references.extend(evidence["source"].get("media_ids", []))
    for comparison in evidence.get("comparisons", []):
        references.extend(comparison["media_ids"])
    return references


def _validate_source_url(source):
    url = source.get("url")
    if url:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Source links must use an absolute HTTP or HTTPS URL")


def _validate_media_references(evidence):
    assets = {asset["id"]: asset for asset in evidence["assets"]}
    if len(assets) != len(evidence["assets"]):
        raise ValueError("Embedded asset IDs must be unique")
    references = list(evidence["robot_assets"])
    for workflow in evidence["workflows"]:
        references.extend(_media_references(workflow))
        details = workflow["evidence"]
        _validate_source_url(details["source"])
        if details.get("explanation", {}).get("watchpoints"):
            videos = [
                assets[id]
                for id in details["media_ids"]
                if id in assets and assets[id]["mime_type"] == "video/mp4"
            ]
            if not videos:
                raise ValueError("Video watchpoints require an embedded workflow video")
    missing = set(references) - assets.keys()
    if missing:
        raise ValueError(f"Unknown embedded media IDs: {sorted(missing)}")


def _report_data(evidence_path, directory):
    evidence = json.loads(evidence_path.read_text())
    workflows = evidence["workflows"]
    if {row["id"] for row in workflows} != {"iaa", "evg"} or len(workflows) != 2:
        raise ValueError("The proof must contain exactly IAA and EVG")
    _validate_media_references(evidence)
    evidence["assets"] = [
        _embedded_asset(asset, evidence_path.parent) for asset in evidence["assets"]
    ]
    evidence["workflows"] = [_workflow_details(row, directory) for row in workflows]
    return evidence


def _write_report(evidence_path, output_path, directory):
    report = _report_data(evidence_path, directory)
    data = json.dumps(report, ensure_ascii=True).replace("<", "\\u003c")
    template = (directory / "proof-template.html").read_text()
    if template.count("__REPORT_DATA__") != 1:
        raise ValueError("The template must contain one evidence placeholder")
    output_path.write_text(template.replace("__REPORT_DATA__", data))
    output_path.chmod(0o600)
    return {
        "bytes": output_path.stat().st_size,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "embedded_assets": len(report["assets"]),
    }


def main():
    """Build an operator-private standalone report from reviewed local evidence.

    Args:
        None. Reads evidence JSON and output paths from command-line arguments.
    Returns:
        None. Writes HTML and prints its size, hash, and embedded asset count.
    Raises:
        OSError: Evidence, media, specs, or template cannot be read or written.
        ValueError: Evidence or template does not satisfy the proof contract.
        KeyError: Required reviewed evidence fields are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Reviewed private evidence JSON")
    parser.add_argument("output", type=Path, help="Standalone HTML outside Git")
    arguments = parser.parse_args()
    directory = Path(__file__).resolve().parent
    print(json.dumps(_write_report(arguments.evidence, arguments.output, directory)))


if __name__ == "__main__":
    main()
