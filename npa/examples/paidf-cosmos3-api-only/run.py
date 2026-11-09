#!/usr/bin/env python3
"""Run a twelve-variant hosted Cosmos3 campaign and build a private offline gallery."""

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
CRITERIA = {
    "motion_and_timing_plausible",
    "robot_object_identity_intact",
    "contacts_physically_plausible",
    "requested_appearance_visible",
}


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


def _campaign(path):
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("campaign must be a YAML object")
    if document.get("apiVersion") != "paidf.cosmos3-api-only/v1":
        raise ValueError("unsupported campaign schema")
    if (
        document.get("kind") != "HostedCosmos3PaidfCampaign"
        or document["execution"].get("no_new_infrastructure") is not True
    ):
        raise ValueError(
            "campaign must declare the standalone no-provisioning execution contract"
        )
    generation = document["generation"]
    required = {
        "route": ROUTE,
        "model": MODEL,
        "structural_control": "edge",
        "variants_per_request": 1,
        "guardrails": "required",
        "admission_parallelism": 12,
    }
    if any(generation.get(key) != value for key, value in required.items()):
        raise ValueError(
            "campaign requires twelve guarded Super edge augmentation requests"
        )
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


def _admit(module, work, document, config):
    # Validate every request before admitting any paid work.
    requests = [_request(document, config, variant) for variant in document["variants"]]
    client = _api_client(module, config)
    contract = client.contract()
    rows = [
        row
        for model in contract["models"]["data"]
        if model["id"] == MODEL
        for row in model.get("capability_matrix", [])
        if row.get("task") == "full_source_v2v_augmentation"
    ]
    if not rows or rows[0].get("guardrails_required") is not True:
        raise ValueError("live full-source contract must require guardrails")
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
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
        return module.resume(
            _api_client(module, config),
            work / "jobs" / (variant["id"] + ".json"),
            work / "downloads" / variant["id"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(recover, document["variants"]))


def _media(path):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    video = next(
        row
        for row in json.loads(probe.stdout)["streams"]
        if row["codec_type"] == "video"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return {
        "sha256": _sha256(path),
        "width": video["width"],
        "height": video["height"],
        "frames": int(video["nb_read_frames"]),
        "frame_rate": video["avg_frame_rate"],
        "decode": True,
    }


def _receipt(module, work, document, variant):
    directory = work / "downloads" / variant["id"]
    manifest = module.read_json(directory / "manifest.json")
    job = module.read_json(work / "jobs" / (variant["id"] + ".json"))
    published = manifest["manifest"]
    effective = published["effective"]
    expected = {"seed": variant["seed"], **document["generation"]["sampling"]}
    technical = published["technical_validation"]
    if (
        manifest["job_id"] != job["job_id"]
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
    items = {item["name"]: item for item in published["artifacts"]}
    for name in ("source.mp4", "generated.mp4"):
        if _sha256(directory / name) != items[name]["sha256"].lower():
            raise ValueError("canonical media no longer matches the service manifest")
    return directory


def _collect(module, work, document):
    records = []
    for variant in document["variants"]:
        directory = _receipt(module, work, document, variant)
        source, generated = (
            _media(directory / "source.mp4"),
            _media(directory / "generated.mp4"),
        )
        if any(
            source[key] != generated[key]
            for key in ("width", "height", "frames", "frame_rate")
        ):
            raise ValueError("generated geometry/timeline differs from the source")
        records.append(
            {"variant_id": variant["id"], "source": source, "generated": generated}
        )
    if len({record["source"]["sha256"] for record in records}) != 1:
        raise ValueError("variants must retain one identical source")
    module.save_private(work / "technical.json", {"variants": records})
    return records


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


def _video(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<video controls preload="metadata" src="data:video/mp4;base64,'
        + encoded
        + '"></video>'
    )


def _report(module, work, document, review_path):
    # Revalidate current bytes rather than trusting an older technical.json.
    records = _collect(module, work, document)
    reviews = _review(module, review_path, document)
    sections = [
        '<!doctype html><html><head><meta charset="utf-8">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; media-src data:; style-src 'unsafe-inline'\">",
        "<title>Cosmos3 API-only PAIDF proof</title><style>body{font:16px system-ui;margin:24px;max-width:1100px}video{width:100%;max-width:640px}section{margin:24px 0}code{overflow-wrap:anywhere}</style></head><body>",
        "<h1>Twelve Cosmos3 appearance variants</h1><p>Existing hosted inference and storage; local CPU verification. No infrastructure deployed by this runner.</p>",
        "<p>Technical checks do not establish semantic quality, action fidelity or training readiness. Manual review is a sampled assessment.</p>",
        "<h2>Source</h2>",
        _video(work / "downloads" / document["variants"][0]["id"] / "source.mp4"),
    ]
    for variant, record in zip(document["variants"], records):
        review = reviews[variant["id"]]
        facts = record["generated"]
        sections.extend(
            [
                "<section><h2>"
                + html.escape(variant["id"])
                + "</h2><p>"
                + html.escape(variant["prompt"])
                + "</p>",
                _video(work / "downloads" / variant["id"] / "generated.mp4"),
                "<p>SHA-256: <code>" + facts["sha256"] + "</code></p>",
                "<p>"
                + str(facts["frames"])
                + " frames; full decode passed. Manual verdict: "
                + review["verdict"]
                + "</p><p>"
                + html.escape(review["notes"])
                + "</p></section>",
            ]
        )
    sections.append("<h2>Workflow substitutions</h2><ul>")
    for name, value in document["canonical_main_audit"]["substitutions"].items():
        sections.append("<li>" + html.escape(name + ": " + value) + "</li>")
    sections.append(
        "</ul><p>No automated refinement or semantic evaluator is executed.</p></body></html>"
    )
    module.save_private(work / "paidf-cosmos3-proof.html", "".join(sections).encode())


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["validate", "admit", "resume", "collect", "report"]
    )
    parser.add_argument(
        "--workflow", type=Path, default=Path(__file__).with_name("campaign.yaml")
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
        ValueError: Campaign configuration or evidence is inconsistent.
    """
    args = _arguments()
    os.umask(0o077)
    document = _campaign(args.workflow)
    if args.phase == "validate":
        print("Validated twelve-variant standalone campaign; no API calls made.")
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
        StopIteration,
        subprocess.SubprocessError,
        yaml.YAMLError,
        _client_module().ClientError,
    ):
        sys.exit(
            "Campaign stopped; inspect private bindings, retained state and local media. Reuse the same campaign identity."
        )
