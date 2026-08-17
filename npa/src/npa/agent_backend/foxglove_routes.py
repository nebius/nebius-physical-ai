"""`/api/foxglove/*` routes for the agent backend (shipped module).

Uploaded to ``/opt/npa-agent/agent_backend/`` at bootstrap and imported by
``backend.py`` — the same "shipped, not inlined" mechanism as ``memory``,
``retrieval`` and ``trace``. Keeping the routes here (instead of inside the
backend f-string) keeps ``agent.py`` reviewable and lets the routes be tested
directly with a FastAPI ``TestClient`` and injected fakes.

Everything the routes need from the backend is injected through
:class:`FoxgloveDeps`, so this module has no import-time dependency on the agent
VM's filesystem, session state, or object storage.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


_EXPORT_LOCK = threading.RLock()


def _serialized_export(func: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize canonical conversion/publication state transitions."""

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _EXPORT_LOCK:
            return func(*args, **kwargs)

    return wrapped


try:  # agent VM: /opt/npa-agent is on sys.path and the package is flat there
    from agent_backend.foxglove_cloud import FoxgloveCloudError
    from agent_backend.foxglove import (
        convert_run_request,
        converted_recording_update,
        foxglove_data_source_link,
        foxglove_status_payload,
        foxglove_download_export,
        foxglove_recording_link,
        is_foxglove_artifact,
        live_source_update,
        prune_published,
    )
except ImportError:  # repo / local tests
    from npa.agent_backend.foxglove_cloud import FoxgloveCloudError
    from npa.agent_backend.foxglove import (
        convert_run_request,
        converted_recording_update,
        foxglove_data_source_link,
        foxglove_status_payload,
        foxglove_download_export,
        foxglove_recording_link,
        is_foxglove_artifact,
        live_source_update,
        prune_published,
    )


@dataclass
class FoxgloveDeps:
    """Everything the Foxglove routes need from the surrounding backend."""

    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    record_run: Callable[[dict, dict], Any]
    foxglove_config: Callable[..., dict]
    load_artifact: Callable[[dict], Any]
    convert_run: Callable[..., Any]
    now_iso: Callable[[], str]
    validate_run_id: Callable[[str], str]
    data_dir: Path
    runs_dir: Path
    keep_published: int = 3
    set_live_url: Callable[[str], None] | None = None
    ensure_cloud_recording: Callable[..., dict] | None = None
    ensure_cloud_layout: Callable[..., dict] | None = None
    prepare_canonical_mcap: Callable[..., dict] | None = None
    resolve_artifact: Callable[[dict], dict] | None = None
    apply_prepared_canonical: Callable[..., dict] | None = None


def _sim_viz(state: dict) -> dict:
    value = state.get("sim_viz")
    return value if isinstance(value, dict) else {}


def _reusable_canonical_transport(sim_viz: dict, path: Path | None) -> bool:
    """Return true only when the active public bytes match canonical state."""
    if path is None or not path.is_file():
        return False
    expected = str(sim_viz.get("canonical_mcap_sha256") or "").strip().lower()
    provenance = sim_viz.get("canonical_mcap_provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or not isinstance(provenance, dict):
        return False
    if str(provenance.get("sha256") or "").strip().lower() != expected:
        return False
    if not str(sim_viz.get("canonical_mcap_s3_uri") or "").startswith("s3://"):
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def _published_transport_path(
    data_dir: Path, sim_viz: dict, *, prefer_selected: bool = False
) -> Path | None:
    selected = sim_viz.get("foxglove_selected_artifact")
    selected_artifact = selected if isinstance(selected, dict) else {}
    raw = str(
        (
            selected_artifact.get("recording_url")
            if prefer_selected
            else sim_viz.get("foxglove_url")
        )
        or sim_viz.get("foxglove_url")
        or ""
    ).strip()
    if not raw:
        return None
    name = Path(urlparse(raw).path).name
    return data_dir / name if name else None


def _reusable_selected_transport(
    sim_viz: dict, path: Path | None, selected_artifact: dict
) -> bool:
    """Prove that the published bytes still represent the exact S3 object."""

    if path is None or not path.is_file():
        return False
    current = sim_viz.get("foxglove_selected_artifact")
    current = current if isinstance(current, dict) else {}
    expected_fingerprint = str(selected_artifact.get("source_fingerprint") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint):
        return False
    for field in (
        "run_id",
        "run_ref",
        "key",
        "s3_uri",
        "resource_bucket",
        "project_id",
        "resolved_prefix",
    ):
        if str(current.get(field) or "") != str(selected_artifact.get(field) or ""):
            return False
    if str(current.get("source_fingerprint") or "") != expected_fingerprint:
        return False
    expected_sha = str(current.get("sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        return False
    if int(current.get("size_bytes") or -1) != path.stat().st_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha


def register_foxglove_routes(app: Any, deps: FoxgloveDeps, http_error: Any) -> None:
    """Register the Foxglove viewer routes on ``app``.

    ``http_error`` is the backend's ``HTTPException`` class (injected so this
    module does not import FastAPI at module import time on the agent VM).
    """

    @app.get("/foxglove/config")
    def foxglove_config_route() -> dict:
        # Everything the UI needs to mount an MCAP viewer. "available" is False
        # (with a reason) when neither backend can render, so the pane can say so
        # instead of showing an empty viewer.
        return deps.foxglove_config()

    @app.get("/foxglove/status")
    def foxglove_status_route() -> dict:
        state = deps.load_state()
        return foxglove_status_payload(deps.foxglove_config(state), _sim_viz(state))

    @app.post("/foxglove/load-artifact")
    def foxglove_load_artifact_route(payload: dict | None = None) -> Any:
        # Thin wrapper over the shared artifact loader so the Foxglove pane and
        # the artifact browser publish through exactly one code path.
        body = payload if isinstance(payload, dict) else {}
        key_hint = str(body.get("key") or body.get("s3_uri") or "")
        if key_hint and not is_foxglove_artifact(key_hint):
            raise http_error(
                status_code=400,
                detail="Foxglove opens .mcap, .bag, .db3, .ulg and .ulog recordings",
            )
        result = deps.load_artifact(body)
        if isinstance(result, dict) and result.get("ok"):
            state = deps.load_state()
            result["foxglove"] = foxglove_status_payload(
                deps.foxglove_config(state), _sim_viz(state)
            )
        return result

    @app.post("/foxglove/convert-run")
    @_serialized_export
    def foxglove_convert_run_route(payload: dict | None = None) -> dict:
        # Resolve one canonical S3 artifact.  A valid native
        # reports/sim2real.mcap is authoritative; otherwise the callback stages
        # the run's real S3 artifacts, converts once, and persists that same key.
        state = deps.load_state()
        sim_viz = _sim_viz(state)
        request = convert_run_request(payload, sim_viz)
        run_id = request["run_id"]
        if not run_id:
            raise http_error(status_code=400, detail="no active run to convert")
        try:
            run_id = deps.validate_run_id(run_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            raise http_error(status_code=400, detail=str(exc))
        if deps.prepare_canonical_mcap is not None:
            try:
                canonical = deps.prepare_canonical_mcap(
                    run_id=run_id,
                    run_ref=request["run_ref"],
                    fps=request["fps"],
                    max_frames=request["max_frames"],
                )
            except Exception as exc:  # noqa: BLE001 - operator-triggered S3/conversion path
                status_code = int(getattr(exc, "status_code", 422) or 422)
                detail = str(getattr(exc, "detail", "") or exc)
                raise http_error(
                    status_code=status_code,
                    detail=f"canonical MCAP export failed: {detail}",
                )
            artifact_key = str(canonical.get("artifact_key") or "")
            s3_uri = str(canonical.get("s3_uri") or "")
            output_path = Path(str(canonical.get("local_path") or ""))
            summary = dict(canonical.get("summary") or {})
            if not artifact_key or not s3_uri or not output_path.is_file():
                raise http_error(
                    status_code=502,
                    detail="canonical MCAP persistence returned an incomplete contract",
                )
            if deps.apply_prepared_canonical is not None:
                loaded = deps.apply_prepared_canonical(
                    canonical=canonical,
                    run_id=run_id,
                    run_ref=request["run_ref"],
                )
            else:
                load_request = {"run_id": run_id, "key": artifact_key}
                if request["run_ref"]:
                    load_request["run_ref"] = request["run_ref"]
                loaded = deps.load_artifact(load_request)
            if not isinstance(loaded, dict) or not loaded.get("ok"):
                raise http_error(
                    status_code=502,
                    detail="canonical MCAP was stored but could not be loaded",
                )
            state = deps.load_state()
            sim_viz = _sim_viz(state)
            sim_viz.update(
                {
                    "canonical_mcap_s3_uri": s3_uri,
                    "canonical_mcap_key": artifact_key,
                    "canonical_mcap_sha256": str(canonical.get("sha256") or ""),
                    "canonical_mcap_size_bytes": int(canonical.get("size_bytes") or 0),
                    "canonical_mcap_provenance": dict(
                        canonical.get("provenance") or {}
                    ),
                    "canonical_mcap_source": str(canonical.get("source") or ""),
                    "transport_state": "published-local-cache",
                }
            )
            state["sim_viz"] = sim_viz
            deps.record_run(state, sim_viz)
            deps.save_state(state)
            return {
                "ok": True,
                "summary": summary,
                "canonical": {k: v for k, v in canonical.items() if k != "local_path"},
                "sim_viz": sim_viz,
                "foxglove": foxglove_status_payload(
                    deps.foxglove_config(state), sim_viz
                ),
            }

        # Compatibility path for callers embedding the route module without S3.
        source_dir = deps.runs_dir / run_id
        if not source_dir.is_dir():
            raise http_error(
                status_code=404, detail=f"no local artifacts for run {run_id}"
            )
        name = f"{secrets.token_hex(8)}-{run_id}.mcap"
        output_path = deps.data_dir / name
        try:
            deps.data_dir.mkdir(parents=True, exist_ok=True)
            summary = deps.convert_run(
                input_path=source_dir,
                output_path=output_path,
                fps=request["fps"],
                max_frames=request["max_frames"],
                run_id=run_id,
            ).to_dict()
            output_path.chmod(0o644)
        except Exception as exc:  # noqa: BLE001 - conversion is operator-triggered
            raise http_error(status_code=422, detail=f"MCAP conversion failed: {exc}")
        prune_published(deps.data_dir, keep=deps.keep_published)
        sim_viz = converted_recording_update(
            sim_viz, run_id=run_id, name=name, summary=summary, now=deps.now_iso()
        )
        state["sim_viz"] = sim_viz
        deps.record_run(state, sim_viz)
        deps.save_state(state)
        return {
            "ok": True,
            "summary": summary,
            "sim_viz": sim_viz,
            "foxglove": foxglove_status_payload(deps.foxglove_config(state), sim_viz),
        }

    @app.post("/foxglove/export")
    @_serialized_export
    def foxglove_export_route(payload: dict | None = None) -> dict:
        """Export an exact selected MCAP through the public recording path.

        Artifact-card requests carry their server-issued run reference and exact
        S3 key.  The shared loader authorizes and publishes those bytes before a
        destination is returned.  The reserved canonical recording additionally
        passes through canonical preparation so a stale format cannot survive a
        user selecting an existing native ``reports/sim2real.mcap``.
        """
        route_started = time.perf_counter()
        body = payload if isinstance(payload, dict) else {}
        state = deps.load_state()
        sim_viz = _sim_viz(state)
        requested_run = str(body.get("run_id") or sim_viz.get("run_id") or "").strip()
        requested_run_ref = str(body.get("run_ref") or "").strip()
        selected_key = str(body.get("key") or "").strip()
        if not requested_run:
            raise http_error(status_code=400, detail="no active run to export")
        try:
            deps.validate_run_id(requested_run)
        except Exception as exc:  # noqa: BLE001 - validator message is operator-facing
            raise http_error(status_code=400, detail=str(exc))

        exact_selection = bool(selected_key)
        selected_is_canonical = exact_selection and selected_key.endswith(
            "/reports/sim2real.mcap"
        )
        selected_artifact: dict[str, Any] = {}
        load_request: dict[str, Any] = {}
        loaded: dict[str, Any] = {}
        if exact_selection:
            if not selected_key.lower().endswith(".mcap"):
                raise http_error(
                    status_code=400,
                    detail="View in Foxglove requires an exact .mcap artifact",
                )
            load_request = {
                "run_id": requested_run,
                "key": selected_key,
            }
            if requested_run_ref:
                load_request["run_ref"] = requested_run_ref
            selected_bucket = str(
                body.get("resource_bucket") or body.get("bucket") or ""
            ).strip()
            selected_project = str(body.get("project_id") or "").strip()
            selected_prefix = str(body.get("resolved_prefix") or "").strip()
            if selected_bucket:
                load_request["bucket"] = selected_bucket
            if selected_project:
                load_request["project_id"] = selected_project
            if "resolved_prefix" in body:
                load_request["resolved_prefix"] = selected_prefix
            selected_uri = str(body.get("s3_uri") or "").strip()
            if selected_uri:
                load_request["s3_uri"] = selected_uri
            if deps.resolve_artifact is not None:
                try:
                    selected_artifact = dict(deps.resolve_artifact(load_request) or {})
                except Exception as exc:  # noqa: BLE001 - authorization is operator-facing
                    status_code = int(getattr(exc, "status_code", 400) or 400)
                    detail = str(getattr(exc, "detail", "") or exc)
                    raise http_error(status_code=status_code, detail=detail)
                selected_artifact["bucket"] = str(
                    selected_artifact.get("bucket")
                    or selected_artifact.get("resource_bucket")
                    or ""
                )
                selected_artifact["resource_bucket"] = selected_artifact["bucket"]
            else:
                loaded = dict(deps.load_artifact(load_request) or {})
                loaded_viz = dict(loaded.get("sim_viz") or {})
                loaded_key = str(loaded_viz.get("artifact_key") or "").strip()
                if not loaded.get("ok") or not loaded_key:
                    raise http_error(
                        status_code=502,
                        detail="the selected MCAP could not be prepared for Foxglove",
                    )
                selected_artifact = {
                    "run_id": str(loaded_viz.get("run_id") or requested_run),
                    "run_ref": str(
                        loaded_viz.get("artifact_run_ref")
                        or loaded.get("run_ref")
                        or requested_run_ref
                    ),
                    "key": loaded_key,
                    "s3_uri": str(
                        loaded_viz.get("artifact_uri")
                        or loaded.get("artifact_uri")
                        or ""
                    ),
                    "bucket": str(loaded_viz.get("bucket") or selected_bucket),
                    "resource_bucket": str(loaded_viz.get("bucket") or selected_bucket),
                    "project_id": str(loaded_viz.get("project_id") or ""),
                    "resolved_prefix": str(loaded_viz.get("resolved_prefix") or ""),
                }
                state = deps.load_state()
                sim_viz = _sim_viz(state)
            if str(selected_artifact.get("key") or "") != selected_key:
                raise http_error(
                    status_code=409,
                    detail="artifact selection changed while Foxglove preparation was running",
                )
            for field, requested, actual, enforce in (
                ("run id", requested_run, selected_artifact["run_id"], True),
                (
                    "run reference",
                    requested_run_ref,
                    selected_artifact["run_ref"],
                    bool(requested_run_ref),
                ),
                (
                    "resource bucket",
                    selected_bucket,
                    selected_artifact["bucket"],
                    "resource_bucket" in body or "bucket" in body,
                ),
                (
                    "project id",
                    selected_project,
                    selected_artifact["project_id"],
                    "project_id" in body,
                ),
                (
                    "resolved prefix",
                    selected_prefix,
                    selected_artifact["resolved_prefix"],
                    "resolved_prefix" in body,
                ),
                (
                    "S3 URI",
                    selected_uri,
                    selected_artifact["s3_uri"],
                    "s3_uri" in body,
                ),
            ):
                if enforce and requested != actual:
                    raise http_error(
                        status_code=409,
                        detail=(
                            f"the prepared Foxglove artifact {field} does not match "
                            "the selected artifact card"
                        ),
                    )
        resolution_finished = time.perf_counter()
        active_url = str(sim_viz.get("foxglove_url") or "").strip()
        active_run = str(sim_viz.get("run_id") or "").strip()
        active_path = _published_transport_path(
            deps.data_dir,
            sim_viz,
            prefer_selected=deps.resolve_artifact is not None,
        )
        # A selected canonical transport is reusable only when its local bytes,
        # persisted SHA, provenance SHA, S3 identity, and run identity all agree.
        # This keeps the popup-safe action responsive without trusting a stale
        # previous-run publication. Missing/corrupt evidence still goes through
        # the authoritative S3 resolver, as does an explicit force request.
        has_active_transport = bool(
            active_url
            and requested_run == active_run
            and active_path is not None
            and active_path.is_file()
        )
        canonical_reusable = (
            _reusable_canonical_transport(sim_viz, active_path)
            if deps.prepare_canonical_mcap is not None
            else has_active_transport
        )
        selected_cache_reused = bool(
            exact_selection
            and not body.get("force_convert")
            and _reusable_selected_transport(sim_viz, active_path, selected_artifact)
        )
        if (
            exact_selection
            and deps.resolve_artifact is not None
            and not selected_cache_reused
            and not selected_is_canonical
        ):
            loaded = dict(deps.load_artifact(load_request) or {})
            loaded_viz = dict(loaded.get("sim_viz") or {})
            if (
                not loaded.get("ok")
                or str(loaded_viz.get("artifact_key") or "") != selected_key
            ):
                raise http_error(
                    status_code=502,
                    detail="the selected MCAP could not be prepared for Foxglove",
                )
            # Do not attach the pre-download object identity to bytes fetched
            # after a concurrent overwrite. A retry will resolve and download
            # the new version from one stable selection.
            resolved_after_load = dict(deps.resolve_artifact(load_request) or {})
            if any(
                str(resolved_after_load.get(field) or "")
                != str(selected_artifact.get(field) or "")
                for field in (
                    "run_id",
                    "run_ref",
                    "key",
                    "s3_uri",
                    "bucket",
                    "project_id",
                    "resolved_prefix",
                    "source_fingerprint",
                )
            ):
                raise http_error(
                    status_code=409,
                    detail="the selected MCAP identity changed while its bytes were being published; retry",
                )
            selected_artifact = resolved_after_load
            selected_artifact["bucket"] = str(
                selected_artifact.get("bucket")
                or selected_artifact.get("resource_bucket")
                or ""
            )
            selected_artifact["resource_bucket"] = selected_artifact["bucket"]
            state = deps.load_state()
            sim_viz = _sim_viz(state)
            active_url = str(sim_viz.get("foxglove_url") or "").strip()
            active_run = str(sim_viz.get("run_id") or "").strip()
            active_path = _published_transport_path(deps.data_dir, sim_viz)
        if selected_is_canonical and not (selected_cache_reused and canonical_reusable):
            converted = foxglove_convert_run_route(body)
            sim_viz = dict(converted.get("sim_viz") or {})
            state = deps.load_state()
            summary = converted.get("summary")
            canonical = dict(converted.get("canonical") or {})
            active_url = str(sim_viz.get("foxglove_url") or "").strip()
            active_path = _published_transport_path(deps.data_dir, sim_viz)
            canonical_key = str(sim_viz.get("artifact_key") or "").strip()
            if canonical_key != selected_key:
                raise http_error(
                    status_code=409,
                    detail="canonical preparation did not preserve the selected artifact key",
                )
            selected_artifact.update(
                {
                    "key": canonical_key,
                    "s3_uri": str(sim_viz.get("canonical_mcap_s3_uri") or ""),
                    "source_fingerprint": str(
                        sim_viz.get("artifact_source_fingerprint")
                        or selected_artifact.get("source_fingerprint")
                        or ""
                    ),
                }
            )
        elif selected_is_canonical:
            summary = None
            canonical = {}
        elif exact_selection:
            summary = None
            canonical = {}
        elif (
            bool(body.get("force_convert"))
            or not active_url
            or requested_run != active_run
            or active_path is None
            or not active_path.is_file()
            or not canonical_reusable
        ):
            converted = foxglove_convert_run_route(body)
            sim_viz = dict(converted.get("sim_viz") or {})
            state = dict(state)
            state["sim_viz"] = sim_viz
            summary = converted.get("summary")
            canonical = dict(converted.get("canonical") or {})
            active_url = str(sim_viz.get("foxglove_url") or "").strip()
            active_path = _published_transport_path(deps.data_dir, sim_viz)
        else:
            summary = None
            canonical = {}
        preparation_finished = time.perf_counter()
        config_state = state
        if exact_selection:
            # The loader has just made this selected artifact the active local
            # transport. Ignore a prior same-run pinned artifact while deriving
            # the response URL; otherwise a fast second click could pair the new
            # key/SHA with the first click's recording URL.
            config_viz = dict(sim_viz)
            config_viz["foxglove_selected_artifact"] = {}
            config_state = dict(state)
            config_state["sim_viz"] = config_viz
        config = deps.foxglove_config(config_state)
        export = foxglove_download_export(str(config.get("recording_url") or ""))
        if not export["available"]:
            raise http_error(status_code=409, detail=export["reason"])
        if active_path is None or not active_path.is_file():
            raise http_error(
                status_code=404, detail="exported MCAP is missing on the agent VM"
            )
        export["size_bytes"] = active_path.stat().st_size
        if exact_selection and not selected_is_canonical:
            digest = hashlib.sha256()
            with active_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            export["sha256"] = digest.hexdigest()
            export["canonical_s3_uri"] = str(selected_artifact.get("s3_uri") or "")
            export["provenance"] = {}
        else:
            export["canonical_s3_uri"] = str(sim_viz.get("canonical_mcap_s3_uri") or "")
            export["sha256"] = str(sim_viz.get("canonical_mcap_sha256") or "")
            export["provenance"] = dict(sim_viz.get("canonical_mcap_provenance") or {})
        if exact_selection:
            selected_artifact.update(
                {
                    "sha256": export["sha256"],
                    "size_bytes": active_path.stat().st_size,
                    "recording_url": str(export.get("recording_url") or ""),
                }
            )
            export["selected_artifact"] = selected_artifact
            # Persist the selected transport identity independently of the
            # canonical-cache identity. A generic/native MCAP in the same run
            # must not inherit another artifact's SHA, provenance, or layout.
            sim_viz["foxglove_selected_artifact"] = dict(selected_artifact)
            sim_viz["transport_state"] = (
                "published-selected-cache"
                if selected_cache_reused
                else "published-selected-artifact"
            )
            state["sim_viz"] = sim_viz
            deps.save_state(state)
        if bool(body.get("open_web")):
            provenance = dict(export.get("provenance") or {})
            layout: dict[str, Any] = {}
            if deps.ensure_cloud_layout is not None and (
                not exact_selection or provenance.get("schemas")
            ):
                try:
                    layout = dict(deps.ensure_cloud_layout(provenance=provenance) or {})
                except FoxgloveCloudError as exc:
                    layout = {
                        "available": False,
                        "layout_id": "",
                        "reason": str(exc),
                    }
            elif exact_selection and not selected_is_canonical:
                layout = {
                    "available": False,
                    "layout_id": "",
                    "reason": (
                        "This native MCAP has no canonical NPA layout metadata; "
                        "Foxglove will open it with its default topic browser."
                    ),
                }
            web = foxglove_data_source_link(
                config.get("data_source"),
                layout_id=str(layout.get("layout_id") or "")
                if layout.get("available")
                else "",
                start_time_ns=int(provenance.get("start_time_ns") or 0),
                end_time_ns=int(provenance.get("end_time_ns") or 0),
            )
            if not web["available"]:
                raise http_error(status_code=409, detail=web["reason"])
            export.update(web)
            export["layout"] = layout
            export["layout_note"] = (
                "Foxglove Web was opened with the canonical shared NPA layout."
                if web.get("layout_id")
                else (
                    str(layout.get("reason") or "")
                    or "Foxglove Web remote-file links cannot carry an inline layout; "
                    "use the rich topics or select a saved layout after signing in."
                )
            )
            sim_viz["foxglove_cloud_layout"] = layout
            state["sim_viz"] = sim_viz
            deps.save_state(state)
        if bool(body.get("cloud_import")):
            if deps.ensure_cloud_recording is None:
                raise http_error(
                    status_code=503,
                    detail="Foxglove Cloud upload is not configured on this agent.",
                )
            try:
                cloud = deps.ensure_cloud_recording(
                    active_path,
                    requested_run,
                    provenance=dict(export.get("provenance") or {}),
                )
            except FoxgloveCloudError as exc:
                raise http_error(
                    status_code=int(getattr(exc, "status_code", 502)), detail=str(exc)
                )
            provenance = dict(export.get("provenance") or {})
            layout = dict(cloud.get("layout") or {})
            web = foxglove_recording_link(
                str(cloud.get("recording_id") or ""),
                layout_id=str(layout.get("layout_id") or "")
                if layout.get("available")
                else "",
                start_time_ns=int(provenance.get("start_time_ns") or 0),
                end_time_ns=int(provenance.get("end_time_ns") or 0),
            )
            if not web["available"]:
                raise http_error(status_code=502, detail=web["reason"])
            export.update(web)
            export["cloud"] = cloud
            expected_key = f"npa-{str(export.get('sha256') or '')}"
            cloud_key = str(cloud.get("recording_key") or "")
            if expected_key != "npa-" and cloud_key != expected_key:
                raise http_error(
                    status_code=502,
                    detail="Foxglove Cloud indexed content does not match the canonical MCAP hash",
                )
            sim_viz["foxglove_cloud"] = dict(cloud)
            sim_viz["foxglove_cloud"]["web_url"] = str(web.get("web_url") or "")
            sim_viz["foxglove_cloud"]["sha256"] = str(export.get("sha256") or "")
            state["sim_viz"] = sim_viz
            deps.save_state(state)
        final_config = deps.foxglove_config(state)
        completed_at = time.perf_counter()
        return {
            "ok": True,
            "cache_reused": selected_cache_reused,
            "converted": bool(canonical.get("created"))
            if canonical
            else summary is not None,
            "summary": summary,
            "run_id": str(sim_viz.get("run_id") or ""),
            "artifact_key": str(sim_viz.get("artifact_key") or ""),
            "selected_artifact": selected_artifact,
            "size_bytes": active_path.stat().st_size,
            "sim_viz": sim_viz,
            "export": export,
            "foxglove": final_config,
            "timings_ms": {
                "resolve_authorize": round(
                    (resolution_finished - route_started) * 1000, 3
                ),
                "prepare_transport": round(
                    (preparation_finished - resolution_finished) * 1000, 3
                ),
                "response_finalize": round(
                    (completed_at - preparation_finished) * 1000, 3
                ),
                "total": round((completed_at - route_started) * 1000, 3),
            },
        }

    @app.post("/foxglove/live")
    def foxglove_live_route(payload: dict | None = None) -> dict:
        # Point the embedded viewer at a live Foxglove/ROS-bridge WebSocket.
        state = deps.load_state()
        result = live_source_update(payload, _sim_viz(state), now=deps.now_iso())
        if result is None:
            raise http_error(
                status_code=400,
                detail=(
                    "Provide a public ws:// or wss:// URL (loopback, private, "
                    "link-local and metadata targets are refused)"
                ),
            )
        source, sim_viz = result
        # Session state, not process environment: the URL belongs to this agent
        # session and must survive a backend restart with the rest of the state.
        sim_viz["foxglove_live_url"] = source["url"]
        sim_viz["foxglove_live_protocol"] = source["protocol"]
        state["sim_viz"] = sim_viz
        deps.save_state(state)
        if deps.set_live_url is not None:
            deps.set_live_url(source["url"])
        return {
            "ok": True,
            "data_source": source,
            "foxglove": foxglove_status_payload(deps.foxglove_config(state), sim_viz),
        }
