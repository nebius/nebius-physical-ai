#!/usr/bin/env python3
"""Run a twelve-variant managed-endpoint Cosmos3 campaign; build an offline gallery.

The hosted service owns every media stage: source preparation, guarded GPU
generation, border restoration, lossless RGB browser previews, checksums,
publication and the finishing report. This client only discovers the
contract, submits logical tasks, polls durable status, downloads
checksum-verified results and assembles HTML from the server's files. It runs
no FFmpeg, NumPy or other media decoding.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from functools import cache
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[3]
MODEL = "nvidia/Cosmos3-Super"
ROUTE = "/v1/augmentations"
API_VERSION = "paidf.cosmos3-managed-endpoint/v1"
KIND = "HostedCosmos3ManagedPaidfCampaign"
POLICY_VERSION = "managed-finishing-v1"
POLICY_FIELDS = {"border_preservation", "browser_preview"}
BORDER_MODES = {"restore_detected_source_borders", "disabled"}
PREVIEW_MODES = {"none", "lossless_rgb_webm"}
CRITERIA = {
    "motion_and_timing_plausible",
    "robot_object_identity_intact",
    "contacts_physically_plausible",
    "requested_appearance_visible",
}
# Customers submit model tasks and data/quality policy only.
COMPUTE_KEYS = re.compile(
    r"(^|_)(gpu|gpus|cpu|cpus|node|nodes|pod|pods|replica|replicas|container|containers|"
    r"vm|vms|worker|workers|placement|concurrency|scaling|command)(_|$)"
)
REQUIRED_ARTIFACTS = (
    "source.mp4",
    "generated.mp4",
    "model-raw.mp4",
    "finishing-report.json",
)
PREVIEW_ARTIFACTS = ("generated-preview.webm", "source-preview.webm")


@cache
def _client_module():
    path = REPOSITORY / "skills/tools/nebius-world-factory/scripts/cosmos_client.py"
    spec = importlib.util.spec_from_file_location("paidf_cosmos_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _reject_compute_controls(value, path="campaign"):
    if isinstance(value, dict):
        for key, child in value.items():
            if COMPUTE_KEYS.search(str(key).lower()):
                raise ValueError(
                    f"{path}.{key}: customers do not control service compute"
                )
            _reject_compute_controls(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_compute_controls(child, f"{path}[{index}]")


def _campaign(path):
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("campaign must be a YAML object")
    if document.get("apiVersion") != API_VERSION or document.get("kind") != KIND:
        raise ValueError("unsupported campaign schema")
    execution = document["execution"]
    if (
        execution.get("no_new_infrastructure") is not True
        or execution.get("customer_compute") != "none"
        or execution.get("customer_media_processing") != "none"
    ):
        raise ValueError(
            "campaign must declare managed execution without customer compute"
        )
    _reject_compute_controls({k: v for k, v in document.items() if k != "execution"})
    generation = document["generation"]
    required = {
        "route": ROUTE,
        "model": MODEL,
        "structural_control": "edge",
        "variants_per_request": 1,
        "guardrails": "required",
    }
    if any(generation.get(key) != value for key, value in required.items()):
        raise ValueError("campaign requires guarded Super edge augmentation requests")
    admissions = generation.get("client_admission_parallelism")
    if type(admissions) is not int or not 1 <= admissions <= 12:
        raise ValueError("client admission parallelism must be 1 to 12")
    evaluation = generation["evaluation"]
    if evaluation.get("mode") != "technical_report" or set(evaluation["criteria"]) != {
        "complete_decode",
        "timeline_coverage",
    }:
        raise ValueError("only decode/timeline technical reports are supported")
    sampling = generation["sampling"]
    if set(sampling) != {"num_inference_steps", "guidance_scale", "control_guidance"}:
        raise ValueError("unsupported sampling settings")
    if any(
        type(value) not in {int, float} or not math.isfinite(value)
        for value in sampling.values()
    ):
        raise ValueError("sampling settings must be finite numbers")
    if (
        type(sampling["num_inference_steps"]) is not int
        or sampling["num_inference_steps"] < 1
    ):
        raise ValueError("inference steps must be a positive integer")
    policy = document["output_policy"]
    if (
        not isinstance(policy, dict)
        or set(policy) != POLICY_FIELDS
        or policy["border_preservation"] not in BORDER_MODES
        or policy["browser_preview"] not in PREVIEW_MODES
    ):
        raise ValueError(
            "output_policy must name border_preservation and browser_preview"
        )
    _validate_variants(document["variants"])
    return document


def _validate_variants(variants):
    if not isinstance(variants, list) or len(variants) != 12:
        raise ValueError("exactly twelve variants are required")
    for row in variants:
        if not isinstance(row, dict) or not re.fullmatch(
            r"variant-[0-9]{2}-[a-z0-9-]+", row.get("id", "")
        ):
            raise ValueError("unsafe variant identity")
        if (
            type(row.get("seed")) is not int
            or not isinstance(row.get("prompt"), str)
            or not row["prompt"].strip()
        ):
            raise ValueError(
                "each variant requires an integer seed and nonempty prompt"
            )
    for field in ("id", "seed", "prompt"):
        if len({row[field] for row in variants}) != 12:
            raise ValueError("variant identities, seeds and prompts must be distinct")


def _config(module, path):
    config = module.read_json(path)
    required = {
        "wmh_base_url",
        "wmh_api_key_file",
        "input_connection_id",
        "output_connection_id",
        "source_uri",
        "output_prefix",
    }
    if set(config) != required or any(
        not isinstance(value, str) or not value for value in config.values()
    ):
        raise ValueError(
            "private config requires exactly the documented string bindings"
        )
    if not Path(config["wmh_api_key_file"]).is_absolute():
        raise ValueError("key file path must be absolute")
    return config


def _request(document, config, variant):
    bindings = {
        "${PAIDF_INPUT_CONNECTION_ID}": config["input_connection_id"],
        "${PAIDF_SOURCE_URI}": config["source_uri"],
        "${PAIDF_OUTPUT_CONNECTION_ID}": config["output_connection_id"],
        "${PAIDF_OUTPUT_PREFIX}": config["output_prefix"],
    }
    source, output = document["source"], document["output"]
    return {
        "model": MODEL,
        "input": {
            "connection_id": bindings[source["input_connection_id"]],
            "uri": bindings[source["uri"]],
        },
        "output": {
            "connection_id": bindings[output["connection_id"]],
            "prefix": bindings[output["prefix"]].rstrip("/") + "/" + variant["id"],
        },
        "augmentation": {
            "prompt": variant["prompt"],
            "structural_control": "edge",
            "variants": 1,
        },
        "generation": {"seed": variant["seed"], **document["generation"]["sampling"]},
        "evaluation": document["generation"]["evaluation"],
        "output_policy": dict(document["output_policy"]),
    }


def _workspace(module, path):
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise ValueError("work directory must not be a symlink")
    module.private_directory(path)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True
    )
    if result.returncode == 0 or path.resolve().is_relative_to(REPOSITORY):
        raise ValueError("work directory must be outside Git checkouts")
    return path


def _state(module, work, document, config):
    path = work / "campaign.json"
    state = (
        module.read_json(path)
        if path.exists()
        else {"workflow_sha256": _digest(document)}
    )
    if state["workflow_sha256"] != _digest(document):
        raise ValueError(
            "workflow changed; retain the original YAML and work directory"
        )
    if config:
        binding = _digest(
            {key: value for key, value in config.items() if key != "wmh_api_key_file"}
        )
        if state.get("binding_sha256", binding) != binding:
            raise ValueError("private campaign bindings changed")
        state["binding_sha256"] = binding
    module.save_private(path, state)
    return state


def _api_client(module, config):
    return module.Client(
        config["wmh_base_url"],
        config["wmh_api_key_file"],
        os.environ.get("WMH_CA_BUNDLE"),
    )


def _managed_capability(contract):
    model = next(
        (item for item in contract["models"]["data"] if item.get("id") == MODEL), None
    )
    finishing = (model or {}).get("managed_finishing") or {}
    if (
        finishing.get("status") != "available"
        or finishing.get("schema_version") != POLICY_VERSION
        or finishing.get("request_field") != "output_policy"
        or finishing.get("customer_compute_controls") != "none"
    ):
        raise ValueError(
            "the running service does not advertise managed finishing; no client media fallback is used"
        )
    rows = [
        row
        for row in model.get("capability_matrix", [])
        if row.get("task") == "full_source_v2v_augmentation"
    ]
    if not rows or rows[0].get("guardrails_required") is not True:
        raise ValueError("live full-source contract must require guardrails")
    return finishing


def _admit(module, work, document, config):
    # Validate every request and the managed contract before admitting paid work.
    requests = [_request(document, config, variant) for variant in document["variants"]]
    client = _api_client(module, config)
    contract = client.contract()
    _managed_capability(contract)
    for payload in requests:
        module.qualify_submission(client, contract, payload, ROUTE)

    def submit(item):
        variant, payload = item
        module.save_private(work / "requests" / (variant["id"] + ".json"), payload)
        return module.submit(
            _api_client(module, config),
            payload,
            work / "jobs" / (variant["id"] + ".json"),
            ROUTE,
        )

    workers = document["generation"]["client_admission_parallelism"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        states = list(pool.map(submit, zip(document["variants"], requests)))
    if len({state["job_id"] for state in states}) != 12:
        raise ValueError("service did not retain twelve distinct job identities")


def _resume(module, work, document, config):
    states = [
        work / "jobs" / (variant["id"] + ".json") for variant in document["variants"]
    ]
    if any(not state.exists() for state in states):
        raise ValueError(
            "missing admissions; rerun admit with the same bindings and work directory"
        )

    def recover(variant):
        # Polls durable status and downloads every published artifact with
        # SHA-256 verification. No generation is resubmitted.
        return module.resume(
            _api_client(module, config),
            work / "jobs" / (variant["id"] + ".json"),
            work / "downloads" / variant["id"],
        )

    workers = document["generation"]["client_admission_parallelism"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(recover, document["variants"]))


def _verify_variant(module, work, document, variant):
    """Check server receipts and file hashes only; no media is decoded."""
    directory = work / "downloads" / variant["id"]
    record = module.read_json(directory / "manifest.json")
    job = module.read_json(work / "jobs" / (variant["id"] + ".json"))
    published = record["manifest"]
    effective = published["effective"]
    expected = {"seed": variant["seed"], **document["generation"]["sampling"]}
    technical = published["technical_validation"]
    finishing = published.get("finishing") or {}
    policy = finishing.get("policy") or {}
    if (
        record["job_id"] != job["job_id"]
        or published["job_id"] != job["job_id"]
        or published["model"] != MODEL
        or published["prompt_sha256"]
        != hashlib.sha256(variant["prompt"].encode()).hexdigest()
        or effective.get("guardrails") is not True
        or effective.get("structural_control") != "edge"
        or any(
            effective["effective_generation"].get(key) != value
            for key, value in expected.items()
        )
        or technical.get("complete_decode") is not True
        or not technical.get("timeline_coverage")
    ):
        raise ValueError("service receipt does not match the guarded request")
    if (
        policy.get("version") != POLICY_VERSION
        or any(
            policy.get(key) != value for key, value in document["output_policy"].items()
        )
        or finishing.get("generation_dispatches") != 1
    ):
        raise ValueError("service did not apply the requested managed output policy")
    items = {item["name"]: item for item in published["artifacts"]}
    names = REQUIRED_ARTIFACTS + (
        PREVIEW_ARTIFACTS
        if document["output_policy"]["browser_preview"] == "lossless_rgb_webm"
        else ()
    )
    hashes = {}
    for name in names:
        if name not in items:
            raise ValueError("service publication lacks " + name)
        hashes[name] = _sha256(directory / name)
        if hashes[name] != items[name]["sha256"].lower():
            raise ValueError(
                "downloaded artifact no longer matches the service manifest"
            )
    report = json.loads((directory / "finishing-report.json").read_text())
    lineage = report["lineage"]
    if (
        hashes["finishing-report.json"] != finishing.get("report_sha256")
        or lineage["source_sha256"] != hashes["source.mp4"]
        or lineage["model_raw_sha256"] != hashes["model-raw.mp4"]
        or lineage["finished_sha256"] != hashes["generated.mp4"]
        or report["policy"].get("version") != POLICY_VERSION
        or report["timeline_preserved"] is not True
        or report["finished"]["decoded_frames"] != report["source"]["decoded_frames"]
        or report["finished"]["frame_rate"] != report["source"]["frame_rate"]
        or report["semantic_evaluation"]["status"] != "not_performed"
    ):
        raise ValueError(
            "finishing report does not bind the published lineage and timeline"
        )
    border = report["border_preservation"]
    if border["status"] == "verified":
        if not (border["scene_pixels_unchanged"] and border["padding_matches_source"]):
            raise ValueError("server border restoration was not verified")
    elif border["status"] not in {
        "no_padding_detected",
        "abstained",
        "disabled_by_policy",
    }:
        raise ValueError("unknown border preservation outcome")
    preview = report["browser_preview"]
    source_preview = report["source_browser_preview"]
    if document["output_policy"]["browser_preview"] == "lossless_rgb_webm":
        if (
            preview.get("status") != "verified"
            or source_preview.get("status") != "verified"
            or preview.get("rgb_matches_reference_every_frame") is not True
            or lineage["browser_preview_sha256"] != hashes["generated-preview.webm"]
            or lineage["source_browser_preview_sha256"] != hashes["source-preview.webm"]
        ):
            raise ValueError("server browser preview was not RGB-verified")
    metrics = (
        module.read_json(directory / "metrics.json")
        if (directory / "metrics.json").exists()
        else {}
    )
    bounds = (metrics.get("runner_stage_bounds") or {}).get("super_request") or {}
    return {
        "variant_id": variant["id"],
        "job_id": job["job_id"],
        "hashes": hashes,
        "finishing": finishing,
        "border": border,
        "bounds": (report.get("padding_detection") or {}).get("bounds"),
        "frames": report["finished"]["decoded_frames"],
        "frame_rate": report["finished"]["frame_rate"],
        "width": report["finished"]["width"],
        "height": report["finished"]["height"],
        "preview": preview,
        "source_preview": source_preview,
        "timing": published.get("timing") or {},
        "gpu_window": [
            bounds.get("started_at_unix_seconds"),
            bounds.get("completed_at_unix_seconds"),
        ],
    }


def _observed_concurrency(records):
    events = []
    for record in records:
        start, end = record["gpu_window"]
        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and end >= start
        ):
            events += [(start, 1), (end, -1)]
    current = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        peak = max(peak, current)
    return {"max_overlapping_model_requests": peak, "measured_jobs": len(events) // 2}


def _collect(module, work, document):
    records = [
        _verify_variant(module, work, document, variant)
        for variant in document["variants"]
    ]
    if len({record["hashes"]["source.mp4"] for record in records}) != 1:
        raise ValueError("variants must retain one identical source")
    summary = {
        "variants": records,
        "client_admission_parallelism": document["generation"][
            "client_admission_parallelism"
        ],
        "observed_service_concurrency": _observed_concurrency(records),
        "customer_media_processing": "none",
    }
    module.save_private(work / "technical.json", summary)
    return summary


def _review(module, path, document):
    rows = module.read_json(path)["variants"]
    expected = {variant["id"] for variant in document["variants"]}
    if len(rows) != 12 or {row["variant_id"] for row in rows} != expected:
        raise ValueError(
            "review must identify each of the twelve variants exactly once"
        )
    for row in rows:
        criteria = row["criteria"]
        if set(criteria) != CRITERIA or any(
            type(value) is not bool for value in criteria.values()
        ):
            raise ValueError("review must give four explicit boolean semantic criteria")
        if row["verdict"] not in {"accepted", "rejected"} or (
            row["verdict"] == "accepted" and not all(criteria.values())
        ):
            raise ValueError("accepted review requires every semantic criterion")
        if not isinstance(row["notes"], str) or not row["notes"].strip():
            raise ValueError("each review needs an evidence-based note")
    return {row["variant_id"]: row for row in rows}


def _video(path, label, samples):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    reference = html.escape(json.dumps(samples, separators=(",", ":")), quote=True)
    return (
        '<video controls muted preload="auto" data-label="'
        + html.escape(label, quote=True)
        + '" data-reference-frame0="'
        + reference
        + '" src="data:video/webm;base64,'
        + encoded
        + '"></video>'
    )


def _report(module, work, document, review_path):
    if document["output_policy"]["browser_preview"] != "lossless_rgb_webm":
        raise ValueError(
            "the offline gallery embeds the service's verified RGB WebM previews"
        )
    # Revalidate current server receipts and bytes rather than an older summary.
    summary = _collect(module, work, document)
    records = summary["variants"]
    reviews = _review(module, review_path, document)
    accepted = sum(review["verdict"] == "accepted" for review in reviews.values())
    concurrency = summary["observed_service_concurrency"]
    first = work / "downloads" / document["variants"][0]["id"]
    sections = [
        '<!doctype html><html><head><meta charset="utf-8">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; media-src data:; style-src 'unsafe-inline'\">",
        "<title>Cosmos3 managed-endpoint PAIDF proof</title><style>body{font:16px system-ui;margin:24px;max-width:1100px}"
        "video{width:100%;max-width:832px}section{margin:24px 0}code{overflow-wrap:anywhere}td,th{padding:2px 8px;text-align:left}</style></head><body>",
        "<h1>Twelve Cosmos3 appearance variants from a managed endpoint</h1>",
        "<p>The hosted service prepared the source, ran guarded generation, restored verified source borders, "
        "encoded lossless RGB WebM previews, wrote finishing reports and published checksummed results. "
        "This client submitted logical tasks, polled status and verified downloads; it ran no media processing "
        "and controlled no compute.</p>",
        "<p>Client admission parallelism: "
        + str(summary["client_admission_parallelism"])
        + ". Observed service model-request concurrency (overlap of service-reported generation windows): "
        + str(concurrency["max_overlapping_model_requests"])
        + " across "
        + str(concurrency["measured_jobs"])
        + " jobs.</p>",
        "<p><strong>Technical delivery</strong> (service receipts and SHA-256) is separate from "
        "<strong>semantic review</strong> (manual sampled frames): "
        + str(accepted)
        + " of 12 accepted. "
        "Neither establishes action fidelity or training readiness.</p>",
        "<h2>Source (service lossless RGB preview)</h2>",
        _video(
            first / "source-preview.webm",
            "source",
            records[0]["source_preview"]["browser_reference_frame0_samples"],
        ),
    ]
    for variant, record in zip(document["variants"], records):
        review = reviews[variant["id"]]
        border = record["border"]
        border_text = (
            "Source borders "
            + html.escape(json.dumps(record["bounds"]))
            + " restored exactly; every generated scene pixel retained (service verified)."
            if border["status"] == "verified"
            else "Service border outcome: " + html.escape(border["status"]) + "."
        )
        timing = record["timing"]
        sections.extend(
            [
                "<section><h2>"
                + html.escape(variant["id"])
                + "</h2><p>"
                + html.escape(variant["prompt"])
                + "</p>",
                _video(
                    work / "downloads" / variant["id"] / "generated-preview.webm",
                    variant["id"],
                    record["preview"]["browser_reference_frame0_samples"],
                ),
                "<table><tr><th>Technical delivery</th><td>"
                + str(record["width"])
                + "x"
                + str(record["height"])
                + ", "
                + str(record["frames"])
                + " frames at "
                + html.escape(record["frame_rate"])
                + " fps; timeline preserved; "
                + border_text
                + "</td></tr>",
                "<tr><th>Finished SHA-256</th><td><code>"
                + record["hashes"]["generated.mp4"]
                + "</code></td></tr>",
                "<tr><th>Raw model SHA-256</th><td><code>"
                + record["hashes"]["model-raw.mp4"]
                + "</code></td></tr>",
                "<tr><th>Service finishing</th><td>CPU "
                + html.escape(str(timing.get("finishing_cpu_seconds")))
                + " s after GPU release; "
                + "generation dispatches: "
                + str(record["finishing"].get("generation_dispatches"))
                + "</td></tr>",
                "<tr><th>Semantic review</th><td>"
                + html.escape(review["verdict"])
                + ": "
                + html.escape(review["notes"])
                + "</td></tr></table></section>",
            ]
        )
    sections.append("<h2>Workflow scope</h2><ul>")
    for name, value in document["canonical_main_audit"]["substitutions"].items():
        sections.append("<li>" + html.escape(name + ": " + value) + "</li>")
    sections.append(
        "</ul><p>Scoped PAIDF adaptation; the canonical Curator/evaluator/refinement graph is not executed.</p></body></html>"
    )
    module.save_private(
        work / "paidf-managed-endpoint-proof.html", "".join(sections).encode()
    )


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["validate", "admit", "resume", "collect", "report"]
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=Path(__file__).with_name("paidf-managed-endpoint.yaml"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="private directory outside Git; required except for validate",
    )
    parser.add_argument(
        "--private-config",
        type=Path,
        help="mode-0600 JSON; required for admit and resume",
    )
    parser.add_argument(
        "--quality-review",
        type=Path,
        help="mode-0600 manual review JSON; required for report",
    )
    args = parser.parse_args()
    if args.phase != "validate" and args.work_dir is None:
        parser.error("--work-dir is required")
    if args.phase in {"admit", "resume"} and args.private_config is None:
        parser.error("--private-config is required")
    if args.phase == "report" and args.quality_review is None:
        parser.error("--quality-review is required")
    return args


def main():
    """Execute one campaign phase using durable private state.

    Args:
        None; arguments are read from the command line.
    Returns:
        None.
    Raises:
        ValueError: Campaign configuration or service evidence is inconsistent.
    """
    args = _arguments()
    os.umask(0o077)
    document = _campaign(args.workflow)
    if args.phase == "validate":
        print("Validated twelve-variant managed-endpoint campaign; no API calls made.")
        return
    module = _client_module()
    work = _workspace(module, args.work_dir)
    config = _config(module, args.private_config) if args.private_config else None
    with module.state_lock(work / "campaign.json"):
        _state(module, work, document, config)
        if args.phase == "admit":
            _admit(module, work, document, config)
        elif args.phase == "resume":
            _resume(module, work, document, config)
        elif args.phase == "collect":
            _collect(module, work, document)
        else:
            _report(module, work, document, args.quality_review)
    print(args.phase + " complete; private evidence retained in work directory.")


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
        RuntimeError,
        yaml.YAMLError,
        _client_module().ClientError,
    ):
        sys.exit(
            "Campaign stopped; inspect private bindings and retained state. Reuse the same campaign identity."
        )
