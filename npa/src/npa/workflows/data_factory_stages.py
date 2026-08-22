"""Real stage implementations for the Physical AI Data Factory blueprint.

These back the ``run.shell`` stages of ``physical-ai-data-factory.yaml`` so every
stage does real work against S3 instead of an ``echo`` stub:

- ``generate_configs``: sample appearance-only augmentation variables -> manifest.
- ``prepare_refinement``: turn the prior evaluator result into a different,
  auditable Cosmos control/guidance policy for the next render attempt.
- ``grade_gate``: read the real evaluator result and write a promote/loop decision.
- ``enforce_quality_disposition``: reject a run that exhausts refinement without
  satisfying the score threshold and required motion-integrity checks.
- ``curate``: build a real curation report over the augmented set (counts,
  per-attribute coverage, duplicate check).
- ``finalize``: aggregate the run's stage artifacts into a real final report.

All functions read/write real S3 objects (or local paths). ``npa`` is
pip-installed in the rendered task, so the blueprint invokes them inline via
``python3 -c "from npa.workflows.data_factory_stages import <fn>; <fn>(...)"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from botocore.exceptions import ClientError


class RefinementStateError(RuntimeError):
    """Fail-closed, operator-safe refinement state error."""


# Coherent appearance profiles for a replaceable physical scene. Sampling each
# field independently produced contradictions such as dim evening illumination
# with a sunlit window. Profiles keep every four-attribute request concrete and
# mutually compatible while the source remains authoritative for foreground
# identity, geometry, camera, timing, and motion.
APPEARANCE_PROFILES: tuple[dict[str, str], ...] = (
    {
        "lighting": "bright diffuse daylight",
        "background": "solid light-gray backdrop",
        "color_grade": "neutral balanced color palette",
        "surface_finish": "matte low-gloss backdrop finish",
    },
    {
        "lighting": "warm tungsten lamp illumination",
        "background": "solid beige backdrop",
        "color_grade": "warm amber color palette",
        "surface_finish": "satin soft-sheen backdrop finish",
    },
    {
        "lighting": "cool overhead industrial illumination",
        "background": "solid blue-gray backdrop",
        "color_grade": "cool blue color palette",
        "surface_finish": "polished reflective backdrop finish",
    },
    {
        "lighting": "dim soft evening illumination",
        "background": "solid charcoal backdrop",
        "color_grade": "high-contrast color palette",
        "surface_finish": "matte uniform backdrop finish",
    },
    {
        "lighting": "soft side-lit studio illumination",
        "background": "solid off-white backdrop",
        "color_grade": "neutral cool color palette",
        "surface_finish": "matte smooth backdrop finish",
    },
    {
        "lighting": "warm diffuse studio illumination",
        "background": "solid tan backdrop",
        "color_grade": "warm balanced color palette",
        "surface_finish": "low-gloss smooth backdrop finish",
    },
    {
        "lighting": "bright overcast daylight",
        "background": "solid pale-blue backdrop",
        "color_grade": "cool balanced color palette",
        "surface_finish": "matte low-texture backdrop finish",
    },
    {
        "lighting": "low-key even studio illumination",
        "background": "solid dark-gray backdrop",
        "color_grade": "muted neutral color palette",
        "surface_finish": "satin uniform backdrop finish",
    },
)
APPEARANCE_VARIABLES = {
    key: [profile[key] for profile in APPEARANCE_PROFILES]
    for key in APPEARANCE_PROFILES[0]
}

LEISAAC_SCENES = {
    "LeIsaac-SO101-PickOrange-v0": (
        "An SO101 robot arm demonstrating the same orange pick-and-place motion"
    ),
    "LeIsaac-SO101-LiftCube-v0": (
        "An SO101 robot arm demonstrating the same red-cube lift motion"
    ),
}


def prompt_from_combo(combo: dict[str, Any], *, scene: str = "") -> str:
    """Turn a sampled appearance combo into a natural-language Cosmos prompt.

    The clip defines the scene. This varies appearance only and explicitly
    protects object identity, geometry, camera, and motion.
    """
    lighting = str(combo.get("lighting") or "").strip()
    background = str(combo.get("background") or "").strip()
    color_grade = str(combo.get("color_grade") or "").strip()
    surface_finish = str(combo.get("surface_finish") or "").strip()
    subject = (
        scene or "Photorealistic input-conditioned physical robot manipulation scene"
    )
    return (
        f"Photorealistic {subject}. "
        "Apply all four visible appearance requirements consistently in every frame: "
        f"lighting is {lighting or 'bright diffuse daylight'}; "
        f"the background behind the foreground subject is a {background or 'solid light-gray backdrop'}; "
        f"the scene has a {color_grade or 'neutral balanced color palette'}; "
        f"the non-identity-bearing backdrop has a {surface_finish or 'matte low-gloss backdrop finish'}. "
        "Preserve the exact foreground objects, identities, geometry, camera, timing, and motion. "
        "Do not add, remove, recolor, or reshape foreground objects; change only lighting, "
        "global color treatment, and the non-identity-bearing backdrop appearance."
    )


def _leisaac_lineage_for_configs(configs_uri: str) -> dict[str, Any] | None:
    base = configs_uri.rstrip("/")
    if not base.endswith("/configs"):
        return None
    lineage_uri = base.removesuffix("configs") + "input/leisaac-lineage.json"
    try:
        payload = _download_json(lineage_uri)
    except Exception:  # noqa: BLE001 - ordinary PAIDF runs have no LeIsaac lineage
        return None
    if payload.get("schema") != "npa.leisaac.paidf-input.v1" or not isinstance(
        payload.get("source"), dict
    ):
        return None
    return payload


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _s3_client():
    # Reuse StorageClient so LIST uses the SAME validated endpoint as upload /
    # download. Building a raw boto3 client here with an unset endpoint would
    # silently fall back to real AWS S3 and make curate/finalize report 0 clips
    # instead of failing fast (StorageClient raises when the endpoint is unset).
    return _storage().s3


def _split(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def _list_keys(uri: str) -> list[str]:
    bucket, prefix = _split(uri if uri.endswith("/") else uri + "/")
    s3 = _s3_client()
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        keys.extend(o["Key"] for o in page.get("Contents", []) if o.get("Key"))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return keys


def _upload_json(payload: dict[str, Any], uri: str) -> str:
    if uri.startswith("s3://"):
        with tempfile.TemporaryDirectory(prefix="npa-df-stage-") as tmp:
            p = Path(tmp) / "out.json"
            p.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return _storage().upload_file(str(p), uri)
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    Path(uri).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return uri


def _download_key(bucket: str, key: str, dest: str) -> str:
    """Download a single S3 object (bucket/key) into ``dest`` and return the path."""
    return _storage().download_path(f"s3://{bucket}/{key}", dest)


def _read_json_key(bucket: str, key: str) -> dict[str, Any] | None:
    if not key:
        return None
    try:
        return _download_json(f"s3://{bucket}/{key}")
    except Exception:  # noqa: BLE001 - best-effort metadata read
        return None


def _download_json(uri: str) -> dict[str, Any]:
    if not uri.startswith("s3://"):
        return json.loads(Path(uri).read_text())
    from botocore.exceptions import ClientError

    want = uri.rstrip("/").split("/")[-1]
    with tempfile.TemporaryDirectory(prefix="npa-df-stage-") as tmp:
        target = Path(tmp) / (want or "payload.json")
        try:
            local = _storage().download_file(uri, str(target))
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError("requested JSON object does not exist") from None
            raise
        return json.loads(Path(local).read_text())


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_optional_refinement_json(uri: str, *, label: str) -> dict[str, Any] | None:
    """Read optional run state while treating only exact NotFound as absence.

    Authentication, authorization, transport, JSON, and schema-adjacent failures
    are fatal. The public error intentionally omits the URI and provider message,
    which can contain private object names or account metadata.
    """

    try:
        payload = _download_json(uri)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - classified and sanitized below
        raise RefinementStateError(
            f"{label} read failed ({type(exc).__name__}); retry after repairing storage access"
        ) from None
    if not isinstance(payload, dict):
        raise RefinementStateError(f"{label} must be a JSON object")
    return payload


def _put_immutable_json(payload: dict[str, Any], uri: str, *, label: str) -> str:
    """Create an immutable JSON object or accept an identical committed retry."""

    expected_sha = _payload_sha256(payload)
    existing = _read_optional_refinement_json(uri, label=label)
    if existing is not None:
        if _payload_sha256(existing) != expected_sha:
            raise RefinementStateError(
                f"{label} already exists with a conflicting immutable contract"
            )
        return uri

    body = _canonical_json_bytes(payload)
    if uri.startswith("s3://"):
        from botocore.exceptions import ClientError

        bucket, key = _split(uri)
        try:
            _s3_client().put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise RefinementStateError(
                    f"{label} write failed ({type(exc).__name__})"
                ) from None
        except Exception as exc:  # noqa: BLE001 - sanitized operator boundary
            raise RefinementStateError(
                f"{label} write failed ({type(exc).__name__})"
            ) from None
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(body)
        except FileExistsError:
            pass
        except Exception as exc:  # noqa: BLE001 - sanitized operator boundary
            raise RefinementStateError(
                f"{label} write failed ({type(exc).__name__})"
            ) from None

    committed = _read_optional_refinement_json(uri, label=label)
    if committed is None or _payload_sha256(committed) != expected_sha:
        raise RefinementStateError(
            f"{label} could not be verified after its immutable write"
        )
    return uri


def _quality_gate_contract(
    report: dict[str, Any], threshold: float
) -> dict[str, Any]:
    """Evaluate the one authoritative completed/hard-check/score contract."""

    score = float(report.get("score", 0.0))
    if not math.isfinite(score):
        raise ValueError("expected a finite quality score")
    status = str(report.get("status", "missing"))
    raw_passed = report.get("passed")
    if raw_passed is not None and not isinstance(raw_passed, bool):
        raise TypeError("expected 'passed' to be a boolean")
    hard_checks_passed = raw_passed is True
    decision = (
        "promote_checkpoint"
        if status == "completed" and hard_checks_passed and score >= threshold
        else "loop_back"
    )
    return {
        "decision": decision,
        "score": score,
        "threshold": threshold,
        "report_status": status,
        "hard_checks_passed": hard_checks_passed,
    }


def _quality_threshold(value: float | str) -> float:
    """Normalize malformed gate input consistently across every consumer."""

    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.5
    return (
        threshold
        if math.isfinite(threshold) and 0.0 <= threshold <= 1.0
        else 0.5
    )


def _validated_refinement_pointer(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a mutable pointer to one immutable, commit-marked policy."""

    if payload.get("schema") != "npa.data_factory.refinement.v1":
        raise RefinementStateError("refinement pointer has an unsupported schema")
    raw_attempt = payload.get("attempt")
    if isinstance(raw_attempt, bool) or not isinstance(raw_attempt, int):
        raise RefinementStateError("refinement pointer has an invalid attempt") from None
    attempt = raw_attempt
    if attempt < 0:
        raise RefinementStateError("refinement pointer has an invalid attempt")
    digest = str(payload.get("policy_sha256") or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RefinementStateError("refinement pointer is missing its policy digest")
    immutable = dict(payload)
    immutable.pop("policy_sha256", None)
    if _payload_sha256(immutable) != digest:
        raise RefinementStateError("refinement pointer digest does not match its contract")
    if not str(immutable.get("history_uri") or "") or not str(
        immutable.get("commit_uri") or ""
    ):
        raise RefinementStateError("refinement pointer is missing immutable provenance")
    settings = immutable.get("settings")
    if not isinstance(settings, dict):
        raise RefinementStateError("refinement pointer settings must be an object")
    try:
        control_weight = float(settings["control_weight"])
        guidance = settings["guidance"]
    except (KeyError, TypeError, ValueError):
        raise RefinementStateError("refinement pointer settings are invalid") from None
    if (
        not math.isfinite(control_weight)
        or not 0.0 <= control_weight <= 1.0
        or isinstance(guidance, bool)
        or not isinstance(guidance, int)
        or guidance < 0
    ):
        raise RefinementStateError("refinement pointer settings are invalid")
    return immutable


def _verify_committed_refinement(pointer: dict[str, Any]) -> dict[str, Any]:
    """Require the immutable history object and its commit marker to agree."""

    immutable = _validated_refinement_pointer(pointer)
    digest = str(pointer["policy_sha256"])
    history = _read_optional_refinement_json(
        str(immutable["history_uri"]), label="refinement attempt history"
    )
    if history is None or _payload_sha256(history) != digest or history != immutable:
        raise RefinementStateError(
            "refinement attempt history is missing or contradicts its pointer"
        )
    marker = _read_optional_refinement_json(
        str(immutable["commit_uri"]), label="refinement commit marker"
    )
    expected_marker = {
        "schema": "npa.data_factory.refinement.commit.v1",
        "attempt": immutable["attempt"],
        "history_uri": immutable["history_uri"],
        "policy_sha256": digest,
    }
    if marker != expected_marker:
        raise RefinementStateError(
            "refinement commit marker is missing or contradicts its attempt"
        )
    return immutable


def _committed_augment_manifest(
    augment_uri: str, *, listed_keys: list[str] | None = None
) -> dict[str, Any] | None:
    """Return an executed canonical augment manifest, or legacy ``None``."""

    uri = augment_uri.rstrip("/") + "/manifest.json"
    if listed_keys is not None and uri.startswith("s3://"):
        _bucket, manifest_key = _split(uri)
        if manifest_key not in listed_keys:
            _augment_bucket, augment_prefix = _split(
                augment_uri if augment_uri.endswith("/") else augment_uri + "/"
            )
            if any(
                key.startswith(augment_prefix + "_attempts/") for key in listed_keys
            ):
                raise RuntimeError(
                    "augment attempt objects exist without a canonical manifest; "
                    "refusing to infer a recovery generation"
                )
            return None
    try:
        manifest = _download_json(uri)
    except FileNotFoundError:
        return None
    except OSError:
        if listed_keys is None:
            return None
        raise
    if not isinstance(manifest, dict):
        raise RuntimeError(f"augment manifest at {uri} is not an object")
    from npa.workbench.cosmos.transfer import validate_committed_run_manifest

    try:
        validate_committed_run_manifest(manifest, augment_uri)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    return manifest


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_DEFAULT_INPUT_FRAME_COUNT = 8


def _derive_quality_anchor(anchor_uri: str) -> dict[str, Any] | None:
    """Derive the best independently passing settings from preserved reports."""

    source = str(anchor_uri or "").strip()
    if not source:
        return None
    reports: list[tuple[str, dict[str, Any]]] = []
    if source.endswith(".json"):
        payload = _download_json(source)
        if isinstance(payload, dict):
            reports.append((source, payload))
    else:
        bucket, _prefix = _split(source)
        for key in _list_keys(source):
            if not (
                re.search(
                    r"/grade/iteration-\d+/(?:ranking/)?cosmos_evaluator\.json$",
                    f"/{key}",
                )
            ):
                continue
            payload = _read_json_key(bucket, key)
            if isinstance(payload, dict):
                reports.append((f"s3://{bucket}/{key}", payload))
    passing: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    for report_uri, report in reports:
        if report.get("status") != "completed":
            continue
        for clip in report.get("clips", []):
            if not isinstance(clip, dict) or clip.get("passed") is not True:
                continue
            attributes = clip.get("attribute_verification")
            hallucination = clip.get("hallucination")
            if not isinstance(attributes, dict) or not isinstance(
                hallucination, dict
            ):
                continue
            if (
                attributes.get("passed") is not True
                or int(attributes.get("passed_checks") or 0) < 4
                or int(attributes.get("passed_checks") or 0)
                != int(attributes.get("total_checks") or 0)
                or hallucination.get("passed") is not True
            ):
                continue
            passing.append(
                (
                    float(clip.get("score") or 0.0),
                    report_uri,
                    report,
                    clip,
                )
            )
    if not passing:
        return None
    score, report_uri, report, clip = max(
        passing, key=lambda item: (item[0], str(item[3].get("clip_id") or ""))
    )
    augment_uri = str(report.get("augment_uri") or "")
    manifest = _committed_augment_manifest(augment_uri)
    if not isinstance(manifest, dict):
        raise RuntimeError("quality anchor report has no committed augment manifest")
    clip_id = str(clip.get("clip_id") or "")
    variant = next(
        (
            item
            for item in manifest.get("variants", [])
            if isinstance(item, dict) and str(item.get("clip") or "") == clip_id
        ),
        None,
    )
    if not isinstance(variant, dict):
        raise RuntimeError("quality anchor candidate is absent from its manifest")
    video_uri = str(variant.get("augmented_video_uri") or "")
    metadata = _download_json(video_uri.rsplit("/", 1)[0] + "/metadata.json")
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("variables"), dict
    ):
        raise RuntimeError("quality anchor candidate metadata is unavailable")
    variables = {
        key: str(metadata["variables"].get(key) or "").strip()
        for key in APPEARANCE_VARIABLES
    }
    if not all(variables.values()):
        raise RuntimeError("quality anchor omits a required appearance attribute")
    inference_seed = metadata.get("inference_seed")
    if isinstance(inference_seed, bool) or not isinstance(inference_seed, int):
        inference_seed = metadata["variables"].get("inference_seed")
    try:
        inference_seed = int(inference_seed)
    except (TypeError, ValueError):
        inference_seed = None
    return {
        "score": score,
        "report_uri": report_uri,
        "report_sha256": _payload_sha256(report),
        "clip_id": clip_id,
        "variables": variables,
        "inference_seed": inference_seed,
        "control_weight": metadata.get("effective_control_weight"),
        "guidance": metadata.get("effective_guidance"),
    }


def _seed_fixture_frames(
    input_uri: str, count: int = _DEFAULT_INPUT_FRAME_COUNT, seed: str = ""
) -> int:
    """Seed a small set of synthetic, captionable PNG frames into ``input_uri``.

    These repository-authored geometric placeholders are never a production
    default. Skip when any supported media exists so user input is never mixed
    with or clobbered by the fixture.
    """
    if not input_uri:
        return 0
    existing = [
        k
        for k in _list_keys(input_uri)
        if k.lower().endswith((".png", ".jpg", ".jpeg", ".mp4"))
    ]
    if existing:
        return 0

    from PIL import Image, ImageDraw

    rng = random.Random(f"{seed}:fixture-input")
    written = 0
    with tempfile.TemporaryDirectory(prefix="npa-df-seed-") as tmp:
        for i in range(max(1, count)):
            img = Image.new("RGB", (1280, 720), (18 + (i * 5) % 60, 24, 40))
            draw = ImageDraw.Draw(img)
            # A varying colored block ("cloth"/object) plus a fixed gripper-ish
            # bar, so frames are distinct and give the VLM something concrete.
            bx = 200 + (i * 90) % 700
            color = (rng.randint(60, 230), rng.randint(60, 230), rng.randint(60, 230))
            draw.rectangle([bx, 300, bx + 260, 520], fill=color)
            draw.rectangle([600, 120, 680, 320], fill=(200, 200, 200))
            local = Path(tmp) / f"frame_{i:04d}.png"
            img.save(local)
            _storage().upload_file(
                str(local), input_uri.rstrip("/") + f"/frame_{i:04d}.png"
            )
            written += 1
    return written


# Compatibility name for older callers that already opted in explicitly.
_seed_default_input_frames = _seed_fixture_frames


def generate_configs(
    configs_uri: str,
    n_augmentations: int | str = 2,
    seed: str = "",
    input_uri: str = "",
    seed_default_input: str | bool = "",
    seed_fixture: str | bool = "",
    augment_subject: str = "",
    augmentation_seed: str = "",
    quality_anchor_uri: str = "",
) -> dict[str, Any]:
    """Sample appearance-only augmentation combos and write a real config manifest.

    ``n_augmentations`` accepts a str (the blueprint interpolates a quoted config
    value) or int; a non-numeric value falls back to 2 rather than crashing.

    ``seed_fixture`` (or the compatibility ``seed_default_input`` spelling) is an
    explicit developer/test opt-in. Production provenance is read from the
    canonical ``input/provenance.json`` written by the prepare step.

    ``augmentation_seed`` optionally decouples appearance sampling from the run
    ID so controlled baseline/component comparisons receive identical prompts.
    """
    try:
        n = int(n_augmentations)
    except (TypeError, ValueError):
        n = 2
    lineage = _leisaac_lineage_for_configs(configs_uri)
    source = lineage.get("source", {}) if lineage else {}
    leisaac_scene = LEISAAC_SCENES.get(str(source.get("task") or ""), "")
    subject = (
        str(augment_subject or "").strip()
        or leisaac_scene
        or "input-conditioned physical robot manipulation"
    )
    anchor = _derive_quality_anchor(quality_anchor_uri)
    variables = {key: list(values) for key, values in APPEARANCE_VARIABLES.items()}
    if anchor:
        for key, value in anchor["variables"].items():
            if value not in variables[key]:
                variables[key].append(value)
    effective_augmentation_seed = str(augmentation_seed or "").strip() or seed
    rng = random.Random(effective_augmentation_seed or None)
    combos = []
    profiles: list[dict[str, str]] = []
    if anchor:
        profiles.append(dict(anchor["variables"]))
    while len(profiles) < max(1, n):
        cycle = [dict(profile) for profile in APPEARANCE_PROFILES]
        rng.shuffle(cycle)
        profiles.extend(cycle)
    for profile_index, profile in enumerate(profiles[: max(1, n)]):
        combo: dict[str, Any] = dict(profile)
        # Each candidate receives a stable, distinct diffusion seed. The field is
        # provenance/quality-search metadata, not a visual attribute question.
        combo["inference_seed"] = (
            anchor["inference_seed"]
            if anchor and profile_index == 0 and anchor["inference_seed"] is not None
            else rng.randrange(0, 2**31)
        )
        # The prompt is what actually conditions the Cosmos Transfer augmentation,
        # so the sampled appearance drives the pixels (not just a Rerun label).
        combo["prompt"] = prompt_from_combo(combo, scene=subject)
        combos.append(combo)
    manifest = {
        "schema": "npa.data_factory.configs.v1",
        "scene": subject,
        "n_augmentations": len(combos),
        "augmentation_seed": effective_augmentation_seed,
        "variables": variables,
        "augmentations": combos,
        "quality_anchor": (
            {
                key: anchor[key]
                for key in ("score", "report_uri", "report_sha256", "clip_id")
            }
            if anchor
            else None
        ),
    }
    # Seed before uploading: the manifest is this stage's declared artifact, and
    # downstream readers (the agent artifact browser, insights ingest) cannot tell
    # a synthetic-frame demo run from a real dataset run if the count only ever
    # reaches stdout.
    existing_provenance = None
    if input_uri:
        try:
            existing_provenance = _download_json(
                input_uri.rstrip("/") + "/provenance.json"
            )
        except Exception:  # noqa: BLE001 - absent until a fixture is generated
            existing_provenance = None
    seeded = 0
    fixture_requested = _is_truthy(seed_fixture) or _is_truthy(seed_default_input)
    if fixture_requested:
        try:
            seeded = _seed_default_input_frames(
                input_uri, seed=effective_augmentation_seed
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            # Explicitly requested seeding: swallowing this leaves the pipeline to
            # die two stages later with "No images found in .../input/", the exact
            # failure the flag exists to prevent, and buries the cause in an
            # earlier task's log.
            raise RuntimeError(
                f"seed_default_input was requested via explicit seed_fixture, but seeding "
                f"{input_uri or '<unset input_uri>'} "
                f"failed: {exc}"
            ) from exc
        if not seeded:
            existing_kind = (
                str(existing_provenance.get("source_kind") or "")
                if isinstance(existing_provenance, dict)
                else ""
            )
            if existing_kind != "synthetic_fixture":
                raise RuntimeError(
                    "seed_fixture was requested, but the canonical input prefix "
                    "already contains user media or is unavailable; refusing to "
                    "silently reuse or overwrite it. Use a new run id."
                )
    manifest["seeded_default_input_frames"] = seeded
    if seeded:
        from npa.workflows.data_factory_input import _fixture_provenance

        provenance = _fixture_provenance(seed, input_uri.rstrip("/") + "/")
        provenance["kind"] = "npa_seeded_fixture"
        provenance["frame_count"] = seeded
        provenance["derivation"]["source_frames"] = [
            f"frame_{index:04d}.png" for index in range(seeded)
        ]
        _upload_json(provenance, input_uri.rstrip("/") + "/provenance.json")
    elif isinstance(existing_provenance, dict):
        provenance = existing_provenance
    else:
        provenance = {
            "schema_version": "npa.paidf.input-provenance.v1",
            "source_kind": "user_supplied",
            "kind": "operator_provided",
            "input_origin": "operator_supplied",
            "input_origin_label": "User-supplied input",
            "staged_canonical_s3_uri": input_uri,
            "frame_count": 0,
            "description": "Pre-staged operator input; authenticity and license were not inferred.",
        }
    manifest["input_source"] = provenance
    uri = (
        configs_uri.rstrip("/") + "/manifest.json"
        if not configs_uri.endswith(".json")
        else configs_uri
    )
    if lineage:
        manifest["source_leisaac"] = source
    manifest["written_uri"] = _upload_json(manifest, uri)
    print(
        json.dumps(
            {
                "stage": "generate_configs",
                "augmentation_count": len(combos),
                "seeded_fixture_frame_count": seeded,
                "input_source_kind": str(provenance.get("source_kind") or "unknown"),
            }
        )
    )
    return manifest


def grade_gate(
    scores_uri: str,
    decision_uri: str,
    threshold: float | str = 0.5,
    refinement_uri: str = "",
) -> str:
    """Read the evaluator score and write a promote/loop decision.

    The blueprint's evaluate stage runs the real NVIDIA Cosmos Evaluator, which
    writes ``cosmos_evaluator.json``. Runs produced before that stage existed (and
    any spec still pointing the loop at ``workbench.vlm_eval.run``) wrote the
    vlm_eval tool's RESULT_FILENAME instead, so both locations are inspected,
    newest contract first. Promotion still requires the complete modern
    status/passed/score contract. Both filenames come from the producing tool's
    own constant rather than a literal here, so the gate cannot drift from its
    producer.

    ``threshold`` accepts a str (the blueprint interpolates a quoted config value)
    or float; a non-numeric value falls back to 0.5.

    Best-effort by design: an unreadable, malformed, or self-declared-degraded report
    yields ``loop_back`` rather than an exception, because a gate that raises takes
    the whole refinement loop down with it.
    """
    from npa.workbench.cosmos_evaluator import RESULT_FILENAME as COSMOS_EVALUATOR_RESULT
    from npa.workbench.vlm_eval import RESULT_FILENAME as VLM_EVAL_RESULT

    threshold = _quality_threshold(threshold)
    if scores_uri.endswith(".json"):
        candidates = [scores_uri]
    else:
        base = scores_uri.rstrip("/")
        candidates = [f"{base}/{COSMOS_EVALUATOR_RESULT}", f"{base}/{VLM_EVAL_RESULT}"]
    contract = {
        "decision": "loop_back",
        "score": 0.0,
        "threshold": threshold,
        "report_status": "missing",
        "hard_checks_passed": False,
    }
    report: dict[str, Any] = {}
    source = ""
    problems: list[str] = []
    for candidate in candidates:
        try:
            report = _download_json(candidate)
            if not isinstance(report, dict):
                raise TypeError(f"expected a JSON object, got {type(report).__name__}")
            # Parsed inside the try on purpose: malformed reports degrade to a
            # loop-back decision rather than aborting the refinement loop.
            candidate_contract = _quality_gate_contract(report, float(threshold))
        except FileNotFoundError:
            # Absence is the only condition under which a pre-Cosmos-Evaluator
            # legacy report may be authoritative.
            problems.append(f"candidate {len(problems) + 1}: FileNotFoundError")
            continue
        except Exception as exc:  # noqa: BLE001 - fail closed on the newest report
            problems.append(f"candidate {len(problems) + 1}: {type(exc).__name__}")
            break
        contract = candidate_contract
        source = candidate
        break
    if not source:
        print(
            json.dumps(
                {
                    "stage": "grade_gate",
                    "warn": f"could not read a score ({'; '.join(problems)})"[:300],
                }
            )
        )
    decision = str(contract["decision"])
    decision_payload: dict[str, Any] = {
        "schema": "npa.sim2real.threshold_decision.v1",
        **contract,
        "report_sha256": _payload_sha256(report) if source else "",
    }
    if refinement_uri:
        pointer = _read_optional_refinement_json(
            refinement_uri, label="refinement pointer"
        )
        if pointer is None:
            raise RefinementStateError(
                "refinement pointer is missing while recording a gate decision"
            )
        immutable = _verify_committed_refinement(pointer)
        decision_payload["evaluated_refinement_attempt"] = immutable["attempt"]
        decision_payload["evaluated_refinement_policy_sha256"] = pointer[
            "policy_sha256"
        ]
    gate_basis = {
        key: decision_payload.get(key)
        for key in (
            "decision",
            "report_sha256",
            "evaluated_refinement_attempt",
            "evaluated_refinement_policy_sha256",
        )
    }
    decision_payload["gate_id"] = _payload_sha256(gate_basis)
    _upload_json(decision_payload, decision_uri)
    print(
        json.dumps(
            {
                "stage": "grade_gate",
                "score": contract["score"],
                "threshold": contract["threshold"],
                "decision": decision,
                "report_contract": "selected" if source else "missing",
                "report_status": contract["report_status"],
                "hard_checks_passed": contract["hard_checks_passed"],
            }
        )
    )
    return decision


def prepare_refinement(
    scores_uri: str,
    refinement_uri: str,
    enabled: str | bool = "true",
    base_control_weight: float | str = 1.0,
    base_guidance: float | str = 3.0,
    control_weight_step: float | str = 0.25,
    max_control_weight: float | str = 1.0,
    guidance_step: float | str = 1.0,
    min_guidance: float | str = 1.0,
    decision_uri: str = "",
    grade_threshold: float | str = 0.75,
    loop_iteration: int | str = "",
    quality_anchor_uri: str = "",
) -> dict[str, Any]:
    """Write the effective Cosmos settings for this refinement attempt.

    The first loop iteration records the configured baseline. Later iterations
    consume the preceding iteration's immutable evaluator and gate artifacts and
    independently evaluate the same completed/status, hard-check, and
    score-threshold contract. Every real retry must change the effective
    control/guidance pair. The current pointer is mutable, but each attempt and its
    commit marker are immutable.

    This policy is intentionally generic: it reacts only to checker names and
    numeric scores, never to scene labels or deployment-specific semantics.
    """

    from npa.workbench.cosmos_evaluator import RESULT_FILENAME

    if str(loop_iteration).strip():
        try:
            current_iteration = int(loop_iteration)
        except (TypeError, ValueError) as exc:
            raise ValueError("loop_iteration must be a positive integer") from exc
        if current_iteration < 1:
            raise ValueError("loop_iteration must be a positive integer")
        prior_iteration = current_iteration - 1
        if scores_uri.endswith(".json"):
            score_parent, _, score_name = scores_uri.rpartition("/")
            score_prefix = f"{score_parent}/" if score_parent else ""
            scores_uri = (
                f"{score_prefix}iteration-{prior_iteration}/{score_name}"
            )
        else:
            scores_uri = (
                f"{scores_uri.rstrip('/')}/iteration-{prior_iteration}/"
            )
        if decision_uri:
            decision_parent, _, decision_name = decision_uri.rpartition("/")
            decision_prefix = f"{decision_parent}/" if decision_parent else ""
            decision_uri = (
                f"{decision_prefix}iteration-{prior_iteration}/{decision_name}"
            )

    def _number(name: str, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    def _guidance(name: str, value: Any) -> int:
        number = _number(name, value)
        if number < 0.0 or not number.is_integer():
            raise ValueError(f"{name} must be a non-negative integer")
        return int(number)

    quality_anchor = _derive_quality_anchor(quality_anchor_uri)
    base_control = _number("base_control_weight", base_control_weight)
    if quality_anchor and quality_anchor.get("control_weight") is not None:
        base_control = _number(
            "quality_anchor.control_weight", quality_anchor["control_weight"]
        )
    control_step = _number("control_weight_step", control_weight_step)
    control_ceiling = _number("max_control_weight", max_control_weight)
    if not 0.0 <= base_control <= 1.0:
        raise ValueError("base_control_weight must be between 0 and 1")
    if control_step < 0.0:
        raise ValueError("control_weight_step must be non-negative")
    if not base_control <= control_ceiling <= 1.0:
        raise ValueError(
            "max_control_weight must be between base_control_weight and 1"
        )

    base_cfg = _guidance("base_guidance", base_guidance)
    if quality_anchor and quality_anchor.get("guidance") is not None:
        base_cfg = _guidance("quality_anchor.guidance", quality_anchor["guidance"])
    cfg_step = _guidance("guidance_step", guidance_step)
    cfg_floor = _guidance("min_guidance", min_guidance)
    if cfg_floor > base_cfg:
        raise ValueError("min_guidance cannot exceed base_guidance")
    adaptive = _is_truthy(enabled)
    if adaptive and control_ceiling == base_control and cfg_floor == base_cfg:
        raise ValueError(
            "adaptive refinement must permit a control-weight or guidance change"
        )
    numeric_threshold = _quality_threshold(grade_threshold)

    previous_pointer = _read_optional_refinement_json(
        refinement_uri, label="refinement pointer"
    )
    previous = (
        _verify_committed_refinement(previous_pointer)
        if previous_pointer is not None
        else None
    )

    report_uri = (
        scores_uri
        if scores_uri.endswith(".json")
        else f"{scores_uri.rstrip('/')}/{RESULT_FILENAME}"
    )
    report = _read_optional_refinement_json(
        report_uri, label="prior evaluator report"
    )
    adaptation_report = report
    adaptation_report_uri = report_uri
    if report is not None and not report.get("clips"):
        ranking_report_uri = (
            report_uri.rsplit("/", 1)[0] + f"/ranking/{RESULT_FILENAME}"
        )
        ranking_report = _read_optional_refinement_json(
            ranking_report_uri, label="prior ranking evaluator report"
        )
        if ranking_report is not None:
            if ranking_report.get("status") != "completed" or not isinstance(
                ranking_report.get("clips"), list
            ):
                raise RefinementStateError(
                    "prior ranking evaluator report is not a completed clip report"
                )
            adaptation_report = ranking_report
            adaptation_report_uri = ranking_report_uri
    decision_payload = (
        _read_optional_refinement_json(decision_uri, label="quality gate decision")
        if decision_uri
        else None
    )
    if (report is None) != (decision_payload is None) and decision_uri:
        raise RefinementStateError(
            "refinement state is contradictory: evaluator report and gate decision must coexist"
        )

    if report is None:
        if previous_pointer is not None:
            if previous is None or int(previous["attempt"]) != 0:
                raise RefinementStateError(
                    "refinement state is contradictory before the first evaluation"
                )
            # Re-entering prepare before evaluation is an idempotent replay of the
            # already committed baseline, not a new attempt.
            print(
                json.dumps(
                    {
                        "stage": "prepare_refinement",
                        "attempt": 0,
                        "idempotent": True,
                        "settings": previous["settings"],
                    }
                )
            )
            return previous_pointer
        attempt = 0
        should_adapt = False
        gate_id = ""
        contract: dict[str, Any] | None = None
    else:
        if previous_pointer is None or previous is None:
            raise RefinementStateError(
                "refinement state is contradictory: an evaluator result has no committed policy"
            )
        try:
            contract = _quality_gate_contract(report, numeric_threshold)
        except (TypeError, ValueError):
            raise RefinementStateError(
                "prior evaluator report does not satisfy the quality-gate schema"
            ) from None

        from npa.orchestration.npa_workflow.decisions import normalize_decision

        computed_decision = normalize_decision(str(contract["decision"]))
        report_digest = _payload_sha256(report)
        evaluated_attempt_number = int(previous["attempt"])
        evaluated_digest = str(previous_pointer["policy_sha256"])
        if decision_payload is not None:
            schema = str(decision_payload.get("schema") or "")
            if schema and schema != "npa.sim2real.threshold_decision.v1":
                raise RefinementStateError(
                    "quality gate decision artifact has an unsupported schema"
                )
            try:
                recorded_decision = normalize_decision(
                    str(decision_payload["decision"])
                )
            except (KeyError, TypeError, ValueError):
                raise RefinementStateError(
                    "quality gate decision artifact is missing its decision"
                ) from None
            if recorded_decision != computed_decision:
                raise RefinementStateError(
                    "quality gate decision contradicts the evaluator contract"
                )
            recorded_report_digest = str(
                decision_payload.get("report_sha256") or ""
            )
            if recorded_report_digest and recorded_report_digest != report_digest:
                raise RefinementStateError(
                    "quality gate decision contradicts the evaluator report version"
                )
            evaluated_attempt = decision_payload.get(
                "evaluated_refinement_attempt", previous["attempt"]
            )
            evaluated_digest = str(
                decision_payload.get("evaluated_refinement_policy_sha256")
                or previous_pointer["policy_sha256"]
            )
            try:
                evaluated_attempt_number = int(evaluated_attempt)
            except (TypeError, ValueError):
                raise RefinementStateError(
                    "quality gate decision artifact has an invalid refinement attempt"
                ) from None
            gate_id = str(decision_payload.get("gate_id") or "")
        else:
            gate_id = ""
        if not gate_id:
            gate_id = _payload_sha256(
                {
                    "decision": computed_decision,
                    "report_sha256": report_digest,
                    "evaluated_refinement_attempt": evaluated_attempt_number,
                    "evaluated_refinement_policy_sha256": evaluated_digest,
                }
            )

        gate_matches_current = (
            evaluated_attempt_number == int(previous["attempt"])
            and evaluated_digest == str(previous_pointer["policy_sha256"])
        )
        gate_was_already_committed = (
            contract["decision"] == "loop_back"
            and str(previous.get("source_gate_id") or "") == gate_id
            and int(previous["attempt"]) == evaluated_attempt_number + 1
            and str(previous.get("prior_evaluator_report_sha256") or "")
            == report_digest
        )
        if gate_was_already_committed:
            print(
                json.dumps(
                    {
                        "stage": "prepare_refinement",
                        "attempt": previous["attempt"],
                        "idempotent": True,
                        "settings": previous["settings"],
                    }
                )
            )
            return previous_pointer
        if not gate_matches_current:
            raise RefinementStateError(
                "quality gate decision was produced for a different refinement attempt"
            )

        if contract["decision"] == "promote_checkpoint":
            print(
                json.dumps(
                    {
                        "stage": "prepare_refinement",
                        "attempt": previous["attempt"],
                        "promotion_observed": True,
                        "settings": previous["settings"],
                    }
                )
            )
            return previous_pointer
        if not adaptive:
            raise RefinementStateError(
                "quality gate requested a retry, but adaptive refinement is disabled"
            )
        attempt = int(previous["attempt"]) + 1
        should_adapt = True

    effective_control = base_control
    effective_guidance = base_cfg
    if should_adapt:
        effective_control = min(
            control_ceiling, base_control + control_step * max(1, attempt)
        )
        effective_guidance = max(
            cfg_floor, base_cfg - cfg_step * max(1, attempt)
        )
        previous_settings = previous.get("settings") if previous else {}
        previous_pair = (
            round(float(previous_settings.get("control_weight", base_control)), 6),
            int(previous_settings.get("guidance", base_cfg)),
        )
        effective_pair = (round(effective_control, 6), effective_guidance)
        if effective_pair == previous_pair:
            # Refuse once the declared monotonic policy saturates. Toggling back
            # to a prior pair would technically differ from the immediately
            # preceding attempt while still replaying byte-identical baseline or
            # retry settings, defeating the point of adaptive inference.
            raise RefinementStateError(
                "adaptive refinement schedule is exhausted; refusing to replay "
                "prior effective inference settings"
            )

    failed_checks: set[str] = set()
    failed_attributes: set[str] = set()
    report_clips = (
        adaptation_report.get("clips", [])
        if adaptation_report is not None
        else []
    )
    for clip in report_clips if isinstance(report_clips, list) else []:
        if not isinstance(clip, dict) or clip.get("passed") is True:
            continue
        for key in (
            "attribute_verification",
            "hallucination",
            "temporal_consistency",
            "appearance_fidelity",
        ):
            result = clip.get(key)
            if isinstance(result, dict) and result.get("passed") is False:
                failed_checks.add(key)
        attributes = clip.get("attribute_verification")
        if isinstance(attributes, dict):
            for check in attributes.get("checks", []):
                if not isinstance(check, dict) or check.get("passed") is True:
                    continue
                variable = str(check.get("variable") or "").strip()
                if variable:
                    failed_attributes.add(variable)

    prior_score = float(contract["score"]) if contract is not None else None
    prior_passed = (
        bool(contract["hard_checks_passed"]) if contract is not None else None
    )
    if refinement_uri.endswith(".json"):
        stem = refinement_uri[:-5]
    else:
        stem = refinement_uri.rstrip("/") + "/refinement"
    history_uri = f"{stem}-attempt-{attempt:02d}.json"
    commit_uri = f"{stem}-attempt-{attempt:02d}.commit.json"
    payload = {
        "schema": "npa.data_factory.refinement.v1",
        "attempt": attempt,
        "adaptive": adaptive,
        "adapted_from_prior_evaluation": should_adapt,
        "source_gate_id": gate_id,
        "prior_evaluator_report_uri": report_uri if report is not None else "",
        "prior_evaluator_report_sha256": (
            _payload_sha256(report) if report is not None else ""
        ),
        "adaptation_evaluator_report_uri": (
            adaptation_report_uri if adaptation_report is not None else ""
        ),
        "adaptation_evaluator_report_sha256": (
            _payload_sha256(adaptation_report)
            if adaptation_report is not None
            else ""
        ),
        "prior_score": prior_score,
        "prior_passed": prior_passed,
        "prior_report_status": (
            contract["report_status"] if contract is not None else ""
        ),
        "grade_threshold": numeric_threshold,
        "prior_gate_decision": contract["decision"] if contract is not None else "",
        "failed_checks": sorted(failed_checks),
        "failed_attributes": sorted(failed_attributes),
        "settings": {
            "control_weight": round(effective_control, 6),
            "guidance": effective_guidance,
        },
        "policy": {
            "base_control_weight": base_control,
            "control_weight_step": control_step,
            "max_control_weight": control_ceiling,
            "base_guidance": base_cfg,
            "guidance_step": cfg_step,
            "min_guidance": cfg_floor,
        },
        "quality_anchor": (
            {
                key: quality_anchor[key]
                for key in ("score", "report_uri", "report_sha256", "clip_id")
            }
            if quality_anchor
            else None
        ),
        "history_uri": history_uri,
        "commit_uri": commit_uri,
        "written_uri": refinement_uri,
    }
    digest = _payload_sha256(payload)
    marker = {
        "schema": "npa.data_factory.refinement.commit.v1",
        "attempt": attempt,
        "history_uri": history_uri,
        "policy_sha256": digest,
    }
    _put_immutable_json(payload, history_uri, label="refinement attempt history")
    _put_immutable_json(marker, commit_uri, label="refinement commit marker")
    pointer = {**payload, "policy_sha256": digest}
    try:
        _upload_json(pointer, refinement_uri)
    except Exception as exc:  # noqa: BLE001 - sanitized operator boundary
        raise RefinementStateError(
            f"refinement pointer write failed ({type(exc).__name__})"
        ) from None
    committed_pointer = _read_optional_refinement_json(
        refinement_uri, label="refinement pointer"
    )
    if committed_pointer != pointer:
        raise RefinementStateError(
            "refinement pointer could not be verified after its write"
        )
    _verify_committed_refinement(pointer)
    print(
        json.dumps(
            {
                "stage": "prepare_refinement",
                "attempt": attempt,
                "adapted_from_prior_evaluation": should_adapt,
                "settings": payload["settings"],
                "failed_checks": payload["failed_checks"],
            }
        )
    )
    return pointer


def _persist_quality_disposition(
    scores_uri: str,
    disposition_uri: str,
    threshold: float | str = 0.75,
) -> dict[str, Any]:
    """Persist and return the final accepted/rejected quality disposition."""

    from npa.workbench.cosmos_evaluator import RESULT_FILENAME

    numeric_threshold = _quality_threshold(threshold)
    report_uri = (
        scores_uri
        if scores_uri.endswith(".json")
        else f"{scores_uri.rstrip('/')}/{RESULT_FILENAME}"
    )
    reasons: list[str] = []
    report: dict[str, Any] = {}
    try:
        downloaded = _download_json(report_uri)
        if not isinstance(downloaded, dict):
            raise TypeError(f"expected a JSON object, got {type(downloaded).__name__}")
        report = downloaded
    except Exception as exc:  # noqa: BLE001 - rejection artifact must still be written
        # Keep ``report`` as the empty mapping. A valid-JSON list/scalar must take
        # the same persist-before-raise path as unreadable or invalid JSON rather
        # than escaping below through ``report.get``.
        reasons.append(
            "evaluator report unavailable or malformed "
            f"({type(exc).__name__})"
        )

    try:
        score = float(report.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
        reasons.append("evaluator score is not numeric")
    if not math.isfinite(score):
        score = 0.0
        reasons.append("evaluator score is not finite")
    evaluator_status = str(report.get("status", "missing"))
    hard_checks_passed = report.get("passed") is True
    if evaluator_status != "completed":
        reasons.append(f"evaluator status is {evaluator_status}")
    if score < numeric_threshold:
        reasons.append("aggregate score is below threshold")
    if not hard_checks_passed:
        reasons.append("one or more required checks did not pass")

    accepted = not reasons
    decision = "promote_checkpoint" if accepted else "loop_back"
    payload = {
        "schema": "npa.data_factory.quality_disposition.v1",
        "quality_status": "accepted" if accepted else "rejected",
        "decision": decision,
        "evaluator_status": evaluator_status,
        "score": score,
        "threshold": numeric_threshold,
        "hard_checks_passed": hard_checks_passed,
        "evaluator_report_uri": report_uri,
        "reasons": reasons,
    }
    payload["written_uri"] = _upload_json(payload, disposition_uri)
    return payload


def write_quality_disposition(
    scores_uri: str,
    disposition_uri: str,
    decision_uri: str,
    threshold: float | str = 0.75,
) -> dict[str, Any]:
    """Persist disposition and route accepted/rejected runs without raising.

    Rejected runs must reach the evidence-only Rerun stage before the workflow
    fails closed. This function therefore records the final disposition and a
    canonical transition decision; ``enforce_quality_disposition`` remains the
    rejecting action after that evidence has been materialized.
    """

    from npa.orchestration.npa_workflow.decisions import write_decision

    payload = _persist_quality_disposition(scores_uri, disposition_uri, threshold)
    decision = str(payload["decision"])
    write_decision(decision_uri, decision)
    print(
        json.dumps(
            {
                "stage": "write_quality_disposition",
                "quality_status": payload["quality_status"],
                "decision": decision,
                "score": payload["score"],
                "threshold": payload["threshold"],
            }
        )
    )
    return payload


def enforce_quality_disposition(
    scores_uri: str,
    disposition_uri: str,
    threshold: float | str = 0.75,
) -> dict[str, Any]:
    """Persist an accepted/rejected quality disposition and fail closed on reject.

    This state runs after the bounded refinement loop.  A loop can finish because
    it promoted or because it exhausted its iterations; only the evaluator report
    distinguishes those outcomes.  Persisting the disposition before raising keeps
    rejected output auditable while preventing downstream labeling and curation.
    """

    payload = _persist_quality_disposition(scores_uri, disposition_uri, threshold)
    print(
        json.dumps(
            {
                "stage": "enforce_quality_disposition",
                "quality_status": payload["quality_status"],
                "score": payload["score"],
                "threshold": payload["threshold"],
            }
        )
    )
    if payload["quality_status"] != "accepted":
        raise RuntimeError(
            "quality rejected after refinement; see quality disposition artifact"
        )
    return payload


def _inventory_rows(uri: str) -> list[dict[str, Any]]:
    bucket, prefix = _split(uri if uri.endswith("/") else uri + "/")
    rows: list[dict[str, Any]] = []
    paginator = _s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        rows.extend(
            {
                "key": str(item.get("Key") or ""),
                "size": int(item.get("Size") or 0),
                "etag": str(item.get("ETag") or ""),
            }
            for item in page.get("Contents", [])
            if item.get("Key")
        )
    return sorted(rows, key=lambda item: item["key"])


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    wire = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(wire).hexdigest()


def _terminal_review_source_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_prefix: str,
    report_key: str,
    workflow_prefix: str,
) -> list[dict[str, Any]]:
    """Return immutable source artifacts for terminal-review preservation checks.

    The review dataset and report are declared outputs. The durable workflow
    ledger is controller-owned and advances while a stage is running, so it is
    preserved but cannot participate in a byte-for-byte source comparison.
    Every other canonical object is immutable for the duration of publication.
    """

    return [
        row
        for row in rows
        if not str(row["key"]).startswith(dataset_prefix)
        and str(row["key"]) != report_key
        and not str(row["key"]).startswith(workflow_prefix)
    ]


def _assert_terminal_review_source_preserved(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    dataset_prefix: str,
    report_key: str,
    workflow_prefix: str,
) -> list[dict[str, Any]]:
    """Fail closed if review publication changed a canonical source artifact."""

    source_before = _terminal_review_source_rows(
        before,
        dataset_prefix=dataset_prefix,
        report_key=report_key,
        workflow_prefix=workflow_prefix,
    )
    source_after = _terminal_review_source_rows(
        after,
        dataset_prefix=dataset_prefix,
        report_key=report_key,
        workflow_prefix=workflow_prefix,
    )
    if source_before != source_after:
        raise RuntimeError("terminal review publication changed source inventory")
    before_keys = {str(row["key"]) for row in before}
    after_keys = {str(row["key"]) for row in after}
    workflow_before = {
        key for key in before_keys if key.startswith(workflow_prefix)
    }
    if not workflow_before.issubset(after_keys):
        raise RuntimeError("terminal review publication removed workflow evidence")
    unexpected = {
        key
        for key in after_keys - before_keys
        if not key.startswith(dataset_prefix)
        and key != report_key
        and not key.startswith(workflow_prefix)
    }
    if unexpected:
        raise RuntimeError("terminal review publication added undeclared run artifacts")
    return source_before


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_object(client: Any, bucket: str, key: str) -> str:
    digest = hashlib.sha256()
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        while chunk := body.read(1024 * 1024):
            digest.update(chunk)
    finally:
        body.close()
    return digest.hexdigest()


def _publish_terminal_review_directory_once(
    local_dir: Path, dataset_uri: str
) -> tuple[str, list[dict[str, Any]]]:
    """Conditionally publish or exactly resume a portable review archive."""

    bucket, prefix = _split(
        dataset_uri if dataset_uri.endswith("/") else dataset_uri + "/"
    )
    client = _s3_client()
    local_rows = {
        prefix + path.relative_to(local_dir).as_posix(): {
            "path": path,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(local_dir.rglob("*"))
        if path.is_file()
    }
    if not local_rows:
        raise RuntimeError("terminal review produced an empty archive")
    existing = {row["key"]: row for row in _inventory_rows(dataset_uri)}
    unexpected = set(existing) - set(local_rows)
    if unexpected:
        raise RuntimeError("terminal review archive contains unexpected objects")

    for key, expected in local_rows.items():
        current = existing.get(key)
        if current is not None:
            if (
                int(current["size"]) != int(expected["size"])
                or _sha256_object(client, bucket, key) != expected["sha256"]
            ):
                raise RuntimeError(
                    "terminal review refuses to overwrite a mismatched archive object"
                )
            continue
        try:
            with expected["path"].open("rb") as stream:
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=stream,
                    ContentLength=int(expected["size"]),
                    IfNoneMatch="*",
                )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"412", "PreconditionFailed"}:
                raise
        if _sha256_object(client, bucket, key) != expected["sha256"]:
            raise RuntimeError(
                "terminal review refuses to overwrite a mismatched archive object"
            )

    published = _inventory_rows(dataset_uri)
    published_by_key = {row["key"]: row for row in published}
    if set(published_by_key) != set(local_rows) or any(
        int(published_by_key[key]["size"]) != int(expected["size"])
        or _sha256_object(client, bucket, key) != expected["sha256"]
        for key, expected in local_rows.items()
    ):
        raise RuntimeError("terminal review archive verification failed")
    return f"s3://{bucket}/{prefix}", published


def _terminal_review_archive_metadata(
    *,
    dataset_uri: str,
    archive_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    dataset_name: str,
    run_disposition: str,
) -> dict[str, Any]:
    """Validate and describe an already-published portable FiftyOne archive."""

    bucket, prefix = _split(
        dataset_uri if dataset_uri.endswith("/") else dataset_uri + "/"
    )
    client = _s3_client()

    def read_json(relative: str) -> dict[str, Any]:
        body = client.get_object(Bucket=bucket, Key=prefix + relative)["Body"]
        try:
            value = json.loads(body.read())
        finally:
            body.close()
        if not isinstance(value, dict):
            raise RuntimeError("terminal review archive contains invalid JSON")
        return value

    rows_by_key = {str(row["key"]): row for row in archive_rows}
    required_json = {
        prefix + "metadata.json",
        prefix + "samples.json",
        prefix + "frames.json",
    }
    if not required_json.issubset(rows_by_key) or any(
        int(row["size"]) <= 0 for row in archive_rows
    ):
        raise RuntimeError("terminal review archive is incomplete")
    metadata = read_json("metadata.json")
    samples_payload = read_json("samples.json")
    frames_payload = read_json("frames.json")
    info = metadata.get("info") if isinstance(metadata.get("info"), dict) else {}
    samples = samples_payload.get("samples")
    frames = frames_payload.get("frames")
    if (
        info.get("schema") != "npa.paidf.fiftyone-terminal-review/v1"
        or str(info.get("dataset_name") or "") != dataset_name
        or str(info.get("quality_disposition") or "") != run_disposition
        or info.get("review_only") is not (run_disposition != "accepted")
        or not isinstance(samples, list)
        or not isinstance(frames, list)
    ):
        raise RuntimeError("terminal review archive identity does not match this run")
    expected = {str(item["candidate_id"]): item for item in candidates}
    observed = {
        str(item.get("candidate_id") or ""): item
        for item in samples
        if isinstance(item, dict) and item.get("candidate_id")
    }
    if len(observed) != len(samples) or set(observed) != set(expected):
        raise RuntimeError("terminal review archive candidates do not match this run")

    referenced_media: set[str] = set()
    for candidate_id, candidate in expected.items():
        sample = observed[candidate_id]
        filepath = str(sample.get("filepath") or "")
        if (
            not filepath.startswith("data/")
            or filepath.startswith("data/../")
            or Path(filepath).is_absolute()
        ):
            raise RuntimeError("terminal review archive contains an unsafe media path")
        archive_key = prefix + filepath
        referenced_media.add(archive_key)
        if archive_key not in rows_by_key:
            raise RuntimeError("terminal review archive is missing candidate media")
        expected_fields = {
            "iteration": int(candidate.get("iteration") or 0),
            "clip_id": str(candidate.get("clip_id") or ""),
            "quality_disposition": run_disposition,
            "candidate_passed": candidate.get("candidate_passed") is True,
            "promotion_eligible": candidate.get("promotion_eligible") is True,
            "score": float(candidate.get("score") or 0.0),
            "hard_checks_passed": candidate.get("hard_checks_passed") is True,
            "failed_attributes": [
                str(value) for value in candidate.get("failed_attributes", [])
            ],
            "hallucination_status": str(
                candidate.get("hallucination_status") or "unavailable"
            ),
        }
        if any(sample.get(key) != value for key, value in expected_fields.items()):
            raise RuntimeError("terminal review archive fields do not match this run")
        try:
            attribute_results = json.loads(
                str(sample.get("attribute_results_json") or "")
            )
            hard_check_results = json.loads(
                str(sample.get("hard_check_results_json") or "")
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "terminal review archive check fields are invalid"
            ) from exc
        if (
            attribute_results != candidate.get("attribute_results", [])
            or hard_check_results != candidate.get("hard_check_results", {})
        ):
            raise RuntimeError("terminal review archive checks do not match this run")
        source_key = str(candidate.get("media_key") or "")
        if (
            not source_key
            or int(rows_by_key[archive_key]["size"])
            <= 0
            or _sha256_object(client, bucket, archive_key)
            != _sha256_object(client, bucket, source_key)
        ):
            raise RuntimeError("terminal review archive media differs from its source")

    non_json_rows = set(rows_by_key) - required_json
    if non_json_rows != referenced_media:
        raise RuntimeError("terminal review archive contains undeclared media")
    fields = sorted(
        str(item.get("name") or "")
        for item in metadata.get("sample_fields", [])
        if isinstance(item, dict) and item.get("name")
    )
    return {
        "engine": "fiftyone",
        "fiftyone_version": str(metadata.get("version") or ""),
        "dataset_name": dataset_name,
        "candidate_count": len(samples),
        "quality_disposition": run_disposition,
        "review_only": run_disposition != "accepted",
        "promotion_eligible_count": sum(
            candidate.get("promotion_eligible") is True for candidate in candidates
        ),
        "fields": fields,
        "export_format": "FiftyOneDataset",
    }


def _publish_terminal_review_report_once(
    report: dict[str, Any], report_uri: str
) -> str:
    """Conditionally publish an exact report without overwriting prior evidence."""

    bucket, key = _split(report_uri)
    payload = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    client = _s3_client()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentLength=len(payload),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"412", "PreconditionFailed"}:
            raise
    existing = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        stored = existing.read()
    finally:
        existing.close()
    if stored != payload:
        raise RuntimeError("terminal review refuses to overwrite a mismatched report")
    return report_uri


def _review_candidate_from_evaluation(
    *,
    iteration: int,
    clip: str,
    media_key: str,
    evaluation: dict[str, Any],
    run_disposition: str,
) -> dict[str, Any]:
    attributes = (
        evaluation.get("attribute_verification", {})
        if isinstance(evaluation.get("attribute_verification"), dict)
        else {}
    )
    checks = [
        item for item in attributes.get("checks", []) if isinstance(item, dict)
    ]
    failed_attributes = [
        str(item.get("variable") or "unknown")
        for item in checks
        if item.get("passed") is not True
    ]
    hallucination = (
        evaluation.get("hallucination", {})
        if isinstance(evaluation.get("hallucination"), dict)
        else {}
    )
    temporal = (
        evaluation.get("temporal_consistency", {})
        if isinstance(evaluation.get("temporal_consistency"), dict)
        else {}
    )
    appearance = (
        evaluation.get("appearance_fidelity", {})
        if isinstance(evaluation.get("appearance_fidelity"), dict)
        else {}
    )
    candidate_passed = evaluation.get("passed") is True
    return {
        "candidate_id": f"iteration-{iteration}/{clip}",
        "iteration": iteration,
        "clip_id": clip,
        "media_key": media_key,
        "score": float(evaluation.get("score") or 0.0),
        "candidate_passed": candidate_passed,
        "hard_checks_passed": candidate_passed,
        "promotion_eligible": run_disposition == "accepted" and candidate_passed,
        "failed_attributes": failed_attributes,
        "attribute_results": checks,
        "hallucination_status": (
            "passed" if hallucination.get("passed") is True else "failed"
        ),
        "hard_check_results": {
            "attribute_verification": attributes,
            "hallucination": hallucination,
            "temporal_consistency": temporal,
            "appearance_fidelity": appearance,
        },
    }


def _terminal_review_candidates(
    run_root_uri: str, disposition: dict[str, Any]
) -> list[dict[str, Any]]:
    """Resolve every evaluated append-only candidate to its committed video."""

    run_root = run_root_uri.rstrip("/")
    keys = _list_keys(run_root + "/")
    manifest_keys = sorted(
        key
        for key in keys
        if re.search(r"/cosmos_augmented/iteration-\d+/manifest\.json$", key)
        and "/_attempts/" not in key
    )
    candidates: list[dict[str, Any]] = []
    quality_status = str(disposition.get("quality_status") or "")
    for manifest_key in manifest_keys:
        match = re.search(
            r"/cosmos_augmented/iteration-(\d+)/manifest\.json$", manifest_key
        )
        if match is None:
            continue
        iteration = int(match.group(1))
        augment_uri = f"{run_root}/cosmos_augmented/iteration-{iteration}/"
        manifest = _committed_augment_manifest(augment_uri, listed_keys=keys)
        if not isinstance(manifest, dict):
            raise RuntimeError(
                f"terminal review iteration {iteration} has no committed manifest"
            )
        report = _download_json(
            f"{run_root}/grade/iteration-{iteration}/ranking/cosmos_evaluator.json"
        )
        if not isinstance(report, dict) or report.get("status") != "completed":
            raise RuntimeError(
                f"terminal review iteration {iteration} has no completed evaluator report"
            )
        evaluations = {
            str(item.get("clip_id") or ""): item
            for item in report.get("clips", [])
            if isinstance(item, dict) and item.get("clip_id")
        }
        manifest_clips: set[str] = set()
        for variant in manifest.get("variants", []):
            if not isinstance(variant, dict):
                raise RuntimeError("terminal review manifest contains an invalid variant")
            clip = str(variant.get("clip") or "").strip()
            video_uri = str(variant.get("augmented_video_uri") or "").strip()
            if not clip or not video_uri or clip not in evaluations:
                raise RuntimeError(
                    "terminal review could not join a committed candidate to its evaluation"
                )
            video_bucket, video_key = _split(video_uri)
            run_bucket, _run_prefix = _split(run_root)
            if video_bucket != run_bucket or video_key not in keys:
                raise RuntimeError(
                    "terminal review candidate media is outside the canonical run inventory"
                )
            manifest_clips.add(clip)
            candidates.append(
                _review_candidate_from_evaluation(
                    iteration=iteration,
                    clip=clip,
                    media_key=video_key,
                    evaluation=evaluations[clip],
                    run_disposition=quality_status,
                )
            )
        if manifest_clips != set(evaluations):
            raise RuntimeError(
                "terminal review evaluator candidates differ from the committed manifest"
            )
    if not candidates:
        raise RuntimeError("terminal review found no evaluated committed candidates")
    return candidates


def review_terminal_candidates(
    run_root_uri: str,
    quality_disposition_uri: str,
    dataset_uri: str,
    report_uri: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Publish a portable, non-promoting real-FiftyOne terminal review set."""

    disposition = _download_json(quality_disposition_uri)
    quality_status = str(disposition.get("quality_status") or "")
    if quality_status not in {"accepted", "rejected"}:
        raise RuntimeError("terminal review requires an accepted/rejected disposition")
    before = _inventory_rows(run_root_uri)
    dataset_bucket, dataset_prefix = _split(
        dataset_uri if dataset_uri.endswith("/") else dataset_uri + "/"
    )
    report_bucket, report_key = _split(report_uri)
    run_bucket, run_prefix = _split(
        run_root_uri if run_root_uri.endswith("/") else run_root_uri + "/"
    )
    workflow_prefix = run_prefix + "npa-workflow/"
    if dataset_bucket != run_bucket or report_bucket != run_bucket:
        raise RuntimeError("terminal review must remain inside canonical run storage")

    candidates = _terminal_review_candidates(run_root_uri, disposition)
    from npa.workflows import data_factory_curate as dfc

    archive_rows = _inventory_rows(dataset_uri)
    if archive_rows:
        metadata = _terminal_review_archive_metadata(
            dataset_uri=dataset_uri,
            archive_rows=archive_rows,
            candidates=candidates,
            dataset_name=dataset_name,
            run_disposition=quality_status,
        )
        written_dataset_uri = dataset_uri.rstrip("/") + "/"
    else:
        with tempfile.TemporaryDirectory(prefix="npa-df-terminal-review-") as tmp:
            export_dir = Path(tmp) / "fiftyone-dataset"
            metadata = dfc.export_terminal_review_dataset(
                candidates=candidates,
                dataset_name=dataset_name,
                download_key=lambda key, dest: _download_key(run_bucket, key, dest),
                export_dir=str(export_dir),
                workdir=str(Path(tmp) / "media"),
                run_disposition=quality_status,
            )
            written_dataset_uri, archive_rows = (
                _publish_terminal_review_directory_once(export_dir, dataset_uri)
            )

    after_dataset = _inventory_rows(run_root_uri)
    source_before = _assert_terminal_review_source_preserved(
        before,
        after_dataset,
        dataset_prefix=dataset_prefix,
        report_key=report_key,
        workflow_prefix=workflow_prefix,
    )
    report = {
        "schema": "npa.paidf.fiftyone-terminal-review/v1",
        "status": "completed",
        **metadata,
        "dataset_uri": written_dataset_uri,
        "archive_object_count": len(archive_rows),
        "source_inventory_object_count": len(source_before),
        "source_inventory_sha256": _inventory_digest(source_before),
        "source_inventory_unchanged_after_publication": True,
        "candidate_results": [
            {
                key: value
                for key, value in candidate.items()
                if key != "media_key"
            }
            for candidate in candidates
        ],
    }
    report["written_uri"] = _publish_terminal_review_report_once(report, report_uri)
    _assert_terminal_review_source_preserved(
        before,
        _inventory_rows(run_root_uri),
        dataset_prefix=dataset_prefix,
        report_key=report_key,
        workflow_prefix=workflow_prefix,
    )
    print(
        json.dumps(
            {
                "stage": "review_terminal_candidates",
                "status": "completed",
                "quality_disposition": quality_status,
                "candidate_count": len(candidates),
                "promotion_eligible_count": metadata[
                    "promotion_eligible_count"
                ],
                "source_inventory_unchanged": True,
            }
        )
    )
    return report


def route_terminal_quality(
    quality_disposition_uri: str, decision_uri: str
) -> str:
    """Route after common review without rewriting the canonical disposition."""

    disposition = _download_json(quality_disposition_uri)
    quality_status = str(disposition.get("quality_status") or "")
    if quality_status == "accepted":
        decision = "promote_checkpoint"
    elif quality_status == "rejected":
        decision = "loop_back"
    else:
        raise RuntimeError("terminal quality disposition is unavailable")
    _upload_json(
        {
            "schema": "npa.sim2real.threshold_decision.v1",
            "decision": decision,
            "quality_status": quality_status,
            "source": quality_disposition_uri,
        },
        decision_uri,
    )
    return decision


def _independently_hard_passing_candidate(
    evaluation: dict[str, Any], threshold: float
) -> bool:
    """Require explicit per-check evidence, not only a summary ``passed`` bit."""

    if evaluation.get("status", "completed") != "completed":
        return False
    try:
        score = float(evaluation.get("score") or 0.0)
    except (TypeError, ValueError):
        return False
    if (
        evaluation.get("passed") is not True
        or not math.isfinite(score)
        or score < threshold
    ):
        return False
    attributes = evaluation.get("attribute_verification")
    if not isinstance(attributes, dict) or attributes.get("passed") is not True:
        return False
    try:
        total_checks = int(attributes.get("total_checks") or 0)
        passed_checks = int(attributes.get("passed_checks") or 0)
    except (TypeError, ValueError):
        return False
    if total_checks < len(APPEARANCE_VARIABLES) or passed_checks != total_checks:
        return False
    if evaluation.get("input_conditioned") is True:
        hallucination = evaluation.get("hallucination")
        if not isinstance(hallucination, dict) or hallucination.get("passed") is not True:
            return False
    for enforced_key, result_key in (
        ("temporal_enforced", "temporal_consistency"),
        ("appearance_enforced", "appearance_fidelity"),
    ):
        if evaluation.get(enforced_key) is True:
            result = evaluation.get(result_key)
            if not isinstance(result, dict) or result.get("passed") is not True:
                return False
    return True


def select_hard_passing_candidates(
    augment_uri: str,
    ranking_scores_uri: str,
    selection_uri: str,
    selection_report_uri: str,
    threshold: float | str = 0.75,
) -> dict[str, Any]:
    """Copy only independently hard-passing candidates into a final batch.

    The original ranking pool is immutable and remains the complete evidence set.
    Selection is an additive copy with a fresh manifest; it is never permission to
    delete or hide a failed candidate. A zero-selection manifest is truthful and
    lets final validation emit a fail-closed report instead of crashing the loop.
    """

    source_rows = _inventory_rows(augment_uri)
    destination_rows = _inventory_rows(selection_uri)
    if destination_rows:
        raise RuntimeError("candidate selection refuses to overwrite prior evidence")
    listed_keys = [str(row["key"]) for row in source_rows]
    manifest = _committed_augment_manifest(augment_uri, listed_keys=listed_keys)
    if not isinstance(manifest, dict):
        raise RuntimeError("candidate selection requires a committed augment manifest")
    ranking = _download_json(
        ranking_scores_uri
        if ranking_scores_uri.endswith(".json")
        else ranking_scores_uri.rstrip("/") + "/cosmos_evaluator.json"
    )
    if not isinstance(ranking, dict) or ranking.get("status") != "completed":
        raise RuntimeError("candidate selection requires a completed ranking report")
    numeric_threshold = _quality_threshold(threshold)
    evaluations = {
        str(item.get("clip_id") or ""): item
        for item in ranking.get("clips", [])
        if isinstance(item, dict) and item.get("clip_id")
    }
    variants = [
        item for item in manifest.get("variants", []) if isinstance(item, dict)
    ]
    if {str(item.get("clip") or "") for item in variants} != set(evaluations):
        raise RuntimeError(
            "candidate selection ranking differs from the committed augment manifest"
        )
    eligible = {
        clip
        for clip, evaluation in evaluations.items()
        if _independently_hard_passing_candidate(evaluation, numeric_threshold)
    }
    source_bucket, _source_prefix = _split(augment_uri)
    destination_bucket, destination_prefix = _split(
        selection_uri if selection_uri.endswith("/") else selection_uri + "/"
    )
    if source_bucket != destination_bucket:
        raise RuntimeError("candidate selection must remain in canonical run storage")
    source_keys = {str(row["key"]): row for row in source_rows}
    selected_variants: list[dict[str, Any]] = []
    for variant in variants:
        clip = str(variant.get("clip") or "")
        if clip not in eligible:
            continue
        video_uri = str(variant.get("augmented_video_uri") or "")
        video_bucket, video_key = _split(video_uri)
        if video_bucket != source_bucket or video_key not in source_keys:
            raise RuntimeError("selected candidate video is outside the source inventory")
        source_directory = video_key.rsplit("/", 1)[0] + "/"
        candidate_keys = sorted(
            key for key in source_keys if key.startswith(source_directory)
        )
        if not candidate_keys:
            raise RuntimeError("selected candidate has no preserved media directory")
        for key in candidate_keys:
            relative = key[len(source_directory) :]
            destination_key = f"{destination_prefix}{clip}/{relative}"
            _s3_client().copy_object(
                Bucket=destination_bucket,
                CopySource={"Bucket": source_bucket, "Key": key},
                Key=destination_key,
            )
        selected_variants.append(
            {
                "clip": clip,
                "variant_index": len(selected_variants),
                "augmented_video_uri": (
                    f"s3://{destination_bucket}/{destination_prefix}"
                    f"{clip}/{video_key.rsplit('/', 1)[1]}"
                ),
                "control_uris": {},
            }
        )
    selected_manifest = {
        "schema": "npa.cosmos2.transfer.v1",
        "mode": "cosmos_transfer2.5_gpu",
        "status": "executed",
        "node_count": 1,
        "variant_count": len(selected_variants),
        "variants": selected_variants,
        "selection_policy": "independent-hard-pass-only",
        "ranking_report_uri": (
            ranking_scores_uri
            if ranking_scores_uri.endswith(".json")
            else ranking_scores_uri.rstrip("/") + "/cosmos_evaluator.json"
        ),
        "source_manifest_sha256": _payload_sha256(manifest),
    }
    from npa.workbench.cosmos.transfer import validate_committed_run_manifest

    validate_committed_run_manifest(selected_manifest, selection_uri)
    manifest_uri = selection_uri.rstrip("/") + "/manifest.json"
    _upload_json(selected_manifest, manifest_uri)
    after_source = _inventory_rows(augment_uri)
    if source_rows != after_source:
        raise RuntimeError("candidate selection changed the preserved ranking pool")
    selected_rows = _inventory_rows(selection_uri)
    if not selected_rows or any(int(row["size"]) <= 0 for row in selected_rows):
        raise RuntimeError("candidate selection produced an incomplete final batch")
    result = {
        "schema": "npa.paidf.candidate-selection/v1",
        "status": "completed",
        "policy": "independent-hard-pass-only",
        "ranked_count": len(variants),
        "selected_count": len(selected_variants),
        "threshold": numeric_threshold,
        "selected_clip_ids": [item["clip"] for item in selected_variants],
        "ranking_pool_inventory_sha256": _inventory_digest(source_rows),
        "ranking_pool_unchanged_after_selection": True,
        "selection_manifest_uri": manifest_uri,
        "candidate_results": [
            {
                "clip_id": clip,
                "score": float(evaluation.get("score") or 0.0),
                "hard_checks_passed": evaluation.get("passed") is True,
                "selected": clip in eligible,
            }
            for clip, evaluation in sorted(evaluations.items())
        ],
    }
    result["written_uri"] = _upload_json(result, selection_report_uri)
    print(
        json.dumps(
            {
                "stage": "select_hard_passing_candidates",
                "ranked_count": result["ranked_count"],
                "selected_count": result["selected_count"],
                "ranking_pool_unchanged": True,
            }
        )
    )
    return result


def curate(
    augment_uri: str,
    report_uri: str,
    dedup_threshold: float | str = "",
    curator_report_uri: str = "",
) -> dict[str, Any]:
    """Curate the augmented set and write a real curation report.

    This stage runs *real* FiftyOne Brain curation over the augmented scenario
    variants -- ``compute_uniqueness`` + near-duplicate similarity + a 2D
    visualization -- and records which variants were kept vs dropped. A missing
    or failed FiftyOne runtime is a hard error so the pipeline cannot report a
    successful curation that never happened.

    ``curator_report_uri`` must point at the preceding real Cosmos Curator
    stage's completed summary. It is folded into this report under
    ``cosmos_curator``, so one
    document carries both the curator's clip catalog and the review decisions.
    """
    keys = _list_keys(augment_uri)
    committed = _committed_augment_manifest(augment_uri, listed_keys=keys)
    committed_variants = (
        committed.get("variants")
        if isinstance(committed, dict) and isinstance(committed.get("variants"), list)
        else None
    )
    if committed_variants is not None:
        prefixes: list[str] = []
        clips = []
        for item in committed_variants:
            if not isinstance(item, dict):
                raise RuntimeError("augment manifest has an invalid variant")
            clip = str(item.get("clip") or "").strip()
            video_uri = str(item.get("augmented_video_uri") or "").strip()
            if not clip or not video_uri:
                raise RuntimeError(
                    "augment manifest variant is missing its clip or generated video URI"
                )
            _bucket, video_key = _split(video_uri)
            prefixes.append(video_key.rsplit("/", 1)[0] + "/")
            clips.append(clip)
        keys = [
            key for key in keys if any(key.startswith(prefix) for prefix in prefixes)
        ]
        clips = sorted(clips)
    else:
        # Legacy direct layout. Never treat the recovery-attempt container as a
        # clip if a partial new publication is encountered.
        _, aug_prefix = _split(
            augment_uri if augment_uri.endswith("/") else augment_uri + "/"
        )
        rels = [k[len(aug_prefix) :] for k in keys if k.startswith(aug_prefix)]
        clips = sorted(
            {
                r.split("/", 1)[0]
                for r in rels
                if "/" in r and r.split("/", 1)[0] not in {"", "_attempts"}
            }
        )
    videos = [k for k in keys if k.endswith(".mp4")]
    frames = [k for k in keys if k.endswith(".png")]
    # Clip ids are the per-clip subdirectories under the augment prefix itself
    # (entries that have a further path segment); top-level files like
    # manifest.json are excluded. Deriving relative to the passed augment_uri
    # (rather than a hardcoded "/cosmos_augmented/") keeps this correct for any
    # prefix, including a bucket root. Matches publish_transfer_to_s3's layout.
    multi = len(clips) > 1
    report = {
        "schema": "npa.fiftyone.curation.v1",
        "augmented_clips": len(clips),
        "clip_ids": clips,
        "video_count": len(videos),
        "frame_count": len(frames),
        # Machine-readable "multiply" status: the augment stage runs one Cosmos
        # Transfer 2.5 inference per sampled appearance combo, so augmented_clips
        # reflects how many scenario variants the run actually produced.
        "multiply": {
            "mode": "multi-variant" if multi else "single-variant",
            "variant_count": len(clips),
            "note": (
                f"{len(clips)} Cosmos Transfer 2.5 scenario variants (one inference per sampled combo)"
                if multi
                else "one Cosmos Transfer 2.5 scenario variant for this run"
            ),
        },
        "status": "curated",
    }
    run_root = augment_uri.rstrip("/").rsplit("/", 1)[0]
    try:
        input_source = _download_json(f"{run_root}/input/provenance.json")
    except Exception:  # noqa: BLE001 - legacy runs predate provenance
        input_source = {}
    if isinstance(input_source, dict) and input_source:
        report["input_source"] = input_source
        report["dataset_groups"] = [
            {
                "name": "source",
                "label": input_source.get("input_origin_label") or "Source input",
                "role": input_source.get("source_kind") or "user_supplied",
            },
            {
                "name": "conditioning",
                "label": "Derived conditioning clip",
                "role": "derived_conditioning",
            },
            {
                "name": "augmented",
                "label": "Synthetic / augmented variants",
                "role": "cosmos_transfer_output",
            },
        ]

    report = _enrich_with_fiftyone_curation(report, augment_uri, keys, dedup_threshold)
    report = _merge_curator_report(report, curator_report_uri)

    report["written_uri"] = _upload_json(report, report_uri)
    # Keep customer-derived identifiers, object locations, and provenance in
    # the private report artifact. Task logs are aggregate-only release evidence.
    print(
        json.dumps(
            {
                "stage": "curate",
                "status": report["status"],
                "augmented_clips": report["augmented_clips"],
                "video_count": report["video_count"],
                "frame_count": report["frame_count"],
                "curation_engine": report.get("curation_engine", ""),
            }
        )
    )
    return report


def _merge_curator_report(
    report: dict[str, Any], curator_report_uri: str
) -> dict[str, Any]:
    """Fold the Cosmos Curator stage's summary into the curation report.

    Only the run-level fields are copied; the per-clip catalog stays in the
    curator's own report so this document does not grow with the clip count.
    """
    if not curator_report_uri:
        raise RuntimeError("Cosmos Curator report URI is required")
    try:
        curator = _download_json(curator_report_uri)
    except Exception as exc:
        raise RuntimeError("Cosmos Curator report could not be loaded") from exc
    if not isinstance(curator, dict):
        raise RuntimeError("Cosmos Curator report is not an object")
    status = str(curator.get("status") or "")
    engine = str(curator.get("engine") or "")
    if status != "completed" or not engine or engine == "unavailable":
        raise RuntimeError(
            "Cosmos Curator did not complete with the real upstream engine"
        )
    report["cosmos_curator"] = {
        "status": status,
        "engine": engine,
        "curated_uri": str(curator.get("curated_uri") or ""),
        "clip_count": int(curator.get("clip_count") or 0),
        "filtered_count": int(curator.get("filtered_count") or 0),
        "variant_count": int(curator.get("variant_count") or 0),
        "total_duration_s": float(curator.get("total_duration_s") or 0.0),
        "motion_filter": str(curator.get("motion_filter") or ""),
        "report_uri": curator_report_uri,
    }
    return report


def _enrich_with_fiftyone_curation(
    report: dict[str, Any],
    augment_uri: str,
    keys: list[str],
    dedup_threshold: float | str,
) -> dict[str, Any]:
    """Run real FiftyOne Brain curation and fail if it is unavailable."""
    from npa.workflows import data_factory_curate as dfc

    try:
        thresh = float(dedup_threshold)
    except (TypeError, ValueError):
        thresh = dfc.DEFAULT_DEDUP_THRESHOLD

    bucket, aug_prefix = _split(
        augment_uri if augment_uri.endswith("/") else augment_uri + "/"
    )
    with tempfile.TemporaryDirectory(prefix="npa-df-curate-") as tmp:
        result = dfc.run_curation(
            keys=keys,
            augment_prefix=aug_prefix,
            base_report=report,
            download_key=lambda key, dest: _download_key(bucket, key, dest),
            read_json=lambda key: _read_json_key(bucket, key),
            workdir=tmp,
            dedup_threshold=thresh,
        )
    if result.get("curation_engine") != dfc.CURATION_ENGINE_FIFTYONE:
        raise RuntimeError("FiftyOne Brain curation did not complete")
    return result


def finalize(run_root_uri: str, report_uri: str) -> dict[str, Any]:
    """Aggregate the run's stage artifacts into a real final report."""
    keys = _list_keys(run_root_uri)
    run_seg = run_root_uri.rstrip("/").split("/")[-1]
    marker = f"/{run_seg}/"
    stages: dict[str, int] = {}
    for k in keys:
        # Take the path *after* the run-id segment, then its first segment is the
        # stage. We prepend "/" to k so the run-id also matches when it is the
        # leading segment of the (bucket-relative) key; if the run id is absent
        # we fall back to the whole key.
        prefixed = f"/{k}"
        after_run = prefixed.split(marker, 1)[-1] if marker in prefixed else k
        stage = after_run.split("/", 1)[0] if "/" in after_run else after_run
        stages[stage] = stages.get(stage, 0) + 1
    # Count augmented scenario variants (per-clip subdirs under cosmos_augmented/,
    # excluding the top-level run manifest) so the final report reflects the real
    # "multiply" fan-out — one Cosmos Transfer 2.5 inference per sampled combo.
    aug_marker = "cosmos_augmented/"
    iteration_numbers: set[int] = set()
    for key in keys:
        match = re.search(r"cosmos_augmented/iteration-(\d+)/", key)
        if match:
            iteration_numbers.add(int(match.group(1)))
    selected_aug_marker = (
        f"{aug_marker}iteration-{max(iteration_numbers)}/"
        if iteration_numbers
        else aug_marker
    )
    selected_augment_uri = run_root_uri.rstrip("/") + f"/{selected_aug_marker}"
    committed = _committed_augment_manifest(
        selected_augment_uri, listed_keys=keys
    )
    if committed is not None:
        n_variants = int(committed.get("variant_count", 0) or 0)
    else:
        aug_clips: set[str] = set()
        for key in keys:
            if selected_aug_marker in key:
                rest = key.split(selected_aug_marker, 1)[1]
                segment = rest.split("/", 1)[0] if "/" in rest else ""
                if segment and segment != "_attempts":
                    aug_clips.add(segment)
        n_variants = len(aug_clips)
    bucket, _prefix = _split(run_root_uri)
    lineage_key = next(
        (key for key in keys if key.endswith("/input/leisaac-lineage.json")), ""
    )
    transfer_key = next(
        (key for key in keys if key.endswith("/cosmos_augmented/manifest.json")), ""
    )
    source_lineage = _read_json_key(bucket, lineage_key) if lineage_key else None
    transfer_manifest = _read_json_key(bucket, transfer_key) if transfer_key else None
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "status": "completed",
        "artifact_count": len(keys),
        "stages": stages,
        "has_rrd": any(k.endswith(".rrd") for k in keys),
        # Mirror the curate report so the final report is honest on its own: how
        # many Cosmos Transfer 2.5 scenario variants this run produced.
        "multiply_mode": "multi-variant" if n_variants > 1 else "single-variant",
        "variant_count": n_variants,
    }
    try:
        input_source = _download_json(
            run_root_uri.rstrip("/") + "/input/provenance.json"
        )
    except Exception:  # noqa: BLE001 - legacy runs predate provenance
        input_source = {}
    if isinstance(input_source, dict) and input_source:
        report["input_source"] = input_source
    if source_lineage:
        report["source_leisaac"] = source_lineage
    if transfer_manifest:
        report["augmentation_engine"] = str(transfer_manifest.get("mode") or "")
        report["input_conditioned"] = transfer_manifest.get("input_conditioned") is True
    report["written_uri"] = _upload_json(report, report_uri)
    print(
        json.dumps(
            {
                "stage": "finalize",
                "status": report["status"],
                "artifact_count": report["artifact_count"],
                "has_rrd": report["has_rrd"],
                "variant_count": report["variant_count"],
            }
        )
    )
    return report
