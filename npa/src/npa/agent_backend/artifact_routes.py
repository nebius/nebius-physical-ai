"""S3-only run discovery routes shipped with the NPA agent backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ArtifactRouteDeps:
    """Runtime dependencies supplied by the generated backend."""

    s3_client: Callable[[], tuple[Any, dict]]
    discovery_prefix: Callable[[dict, str], str]
    list_runs_cached: Callable[..., Any]
    list_runs_cached_multi: Callable[..., Any]
    list_buckets: Callable[[Any, dict], list[str]]
    validate_run_id: Callable[[str], str]
    find_artifacts: Callable[..., tuple[str, list]]
    resolve_run: Callable[..., Any]
    summarize_run: Callable[..., Any]
    discovery_excludes: Callable[[], Any]
    list_artifacts: Callable[..., list]
    select_preferred: Callable[[list], Any]
    http_exception: type[Exception]
    json_response: Callable[..., Any]


def register_artifact_routes(
    app: Any, deps: ArtifactRouteDeps
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Register and return S3-backed run-list and run-artifact handlers."""

    @app.get("/artifacts/runs")
    def artifacts_runs(prefix: str = "", limit: int = 50, q: str = ""):
        try:
            s3, settings = deps.s3_client()
            query = str(q or "").strip()
            if prefix:
                effective_prefix = deps.discovery_prefix(settings, prefix)
                page = deps.list_runs_cached(
                    settings["bucket"],
                    prefix=effective_prefix,
                    base_prefix=settings.get("prefix", ""),
                    limit=limit,
                    contains=query,
                    s3=s3,
                )
                return {
                    "ok": True,
                    "bucket": settings["bucket"],
                    "prefix": effective_prefix,
                    "base_prefix": settings.get("prefix", ""),
                    "query": query,
                    **page.to_dict(),
                }

            base = settings.get("prefix", "")
            buckets = deps.list_buckets(s3, settings)
            # Exact run lookup is the interactive restore/switch/export path.
            # It avoids a broad bucket scan while preserving substring search.
            if len(query) >= 20:
                try:
                    exact_run = deps.validate_run_id(query)
                except Exception:
                    exact_run = ""
                if exact_run:
                    try:
                        bucket, artifacts = deps.find_artifacts(
                            buckets, base_prefix=base, run_id=exact_run, s3=s3
                        )
                    except Exception:
                        # Duplicate basenames are intentionally ambiguous for a
                        # direct lookup. Fall through to exhaustive discovery so
                        # every source-qualified run_ref remains selectable.
                        bucket, artifacts = "", []
                    if artifacts:
                        summary = deps.summarize_run(
                            exact_run, artifacts, bucket=bucket
                        )
                        return {
                            "ok": True,
                            "bucket": settings["bucket"],
                            "buckets": buckets,
                            "prefix": base,
                            "base_prefix": base,
                            "query": query,
                            "runs": [summary.to_dict()],
                            "truncated": False,
                            "total_runs": 1,
                            "limit": limit,
                        }

            # Seed cold first paint from the canonical category, then let the
            # exhaustive multi-bucket cache refresh complete in the background.
            canonical_seed = deps.list_runs_cached(
                settings["bucket"],
                prefix=deps.discovery_prefix(settings, ""),
                base_prefix=base,
                limit=limit,
                contains=query,
                s3=s3,
            )
            page = deps.list_runs_cached_multi(
                buckets,
                base_prefix=base,
                limit=limit,
                exclude=deps.discovery_excludes(),
                contains=query,
                s3=s3,
                cold_seed=canonical_seed,
            )
            return {
                "ok": True,
                "bucket": settings["bucket"],
                "buckets": buckets,
                "prefix": base,
                "base_prefix": base,
                "query": query,
                **page.to_dict(),
            }
        except deps.http_exception:
            raise
        except Exception as exc:
            return deps.json_response(
                status_code=502,
                content={"ok": False, "error": str(exc), "source": "s3"},
            )

    @app.get("/artifacts/run/{run_id:path}")
    def artifacts_for_run(run_id: str, prefix: str = "", run_ref: str = ""):
        try:
            normalized_run = deps.validate_run_id(run_id)
        except Exception as exc:
            raise deps.http_exception(status_code=400, detail=str(exc)) from exc
        try:
            s3, settings = deps.s3_client()
            effective_prefix = deps.discovery_prefix(settings, prefix)
            artifacts = []
            run_bucket = settings["bucket"]
            resolved_ref = str(run_ref or "").strip()
            if resolved_ref:
                resolution = deps.resolve_run(
                    deps.list_buckets(s3, settings),
                    base_prefix=settings.get("prefix", ""),
                    run_ref_or_id=resolved_ref,
                    s3=s3,
                )
                if resolution is not None:
                    if resolution.run_id != normalized_run:
                        raise deps.http_exception(
                            status_code=400, detail="run_ref does not identify run_id"
                        )
                    run_bucket = resolution.bucket
                    artifacts = resolution.artifacts
                    resolved_ref = resolution.run_ref
            elif prefix:
                artifacts = deps.list_artifacts(
                    settings["bucket"],
                    normalized_run,
                    prefix=effective_prefix,
                    s3=s3,
                )
            if not artifacts:
                run_bucket, artifacts = deps.find_artifacts(
                    deps.list_buckets(s3, settings),
                    base_prefix=settings.get("prefix", ""),
                    run_id=normalized_run,
                    s3=s3,
                )
                run_bucket = run_bucket or settings["bucket"]
            preferred = deps.select_preferred(artifacts)
            return {
                "ok": True,
                "bucket": run_bucket,
                "prefix": effective_prefix,
                "base_prefix": settings.get("prefix", ""),
                "run_id": normalized_run,
                "run_ref": resolved_ref,
                "count": len(artifacts),
                "artifacts": [item.to_dict() for item in artifacts],
                "preferred": preferred.to_dict() if preferred else None,
            }
        except deps.http_exception:
            raise
        except Exception as exc:
            return deps.json_response(
                status_code=502,
                content={"ok": False, "error": str(exc), "source": "s3"},
            )

    # Chat intents and grounded action tools call the same handlers directly;
    # returning them prevents a second, divergent artifact-discovery path.
    return artifacts_runs, artifacts_for_run
