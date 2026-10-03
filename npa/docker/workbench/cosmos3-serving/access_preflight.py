"""Fail early unless the operator can read every gated runtime repository."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _model_info(repository: str, token: str) -> dict:
    # Hugging Face's route is /api/models/{owner}/{repo}. Encoding the ownership
    # slash produces HTTP 400 rather than addressing the repository.
    encoded = urllib.parse.quote(repository, safe="/")
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{encoded}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.load(response)


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "[npa-cosmos3-serving] ERROR: HF_TOKEN is required: the image contains "
            "no model or guardrail payload and cannot prove gated access anonymously.",
            file=sys.stderr,
        )
        return 2

    repositories = [
        (
            os.environ.get("NPA_COSMOS3_SERVE_MODEL", "nvidia/Cosmos3-Super"),
            os.environ.get("NPA_COSMOS3_SERVE_MODEL_REVISION"),
        )
    ]
    if os.environ.get("NPA_COSMOS3_SERVE_GUARDRAILS", "on") == "on":
        repositories.append(
            (
                os.environ.get(
                    "NPA_COSMOS3_SERVE_GUARDRAIL_MODEL",
                    "nvidia/Cosmos-1.0-Guardrail",
                ),
                os.environ.get("NPA_COSMOS3_SERVE_GUARDRAIL_REVISION"),
            )
        )

    for repository, expected_revision in repositories:
        try:
            info = _model_info(repository, token)
        except urllib.error.HTTPError as exc:
            print(
                f"[npa-cosmos3-serving] ERROR: authenticated access probe failed for "
                f"{repository} (HTTP {exc.code}). The token may lack this repository's "
                "separate upstream entitlement; credentials do not accept terms.",
                file=sys.stderr,
            )
            return 3
        except urllib.error.URLError as exc:
            print(
                f"[npa-cosmos3-serving] ERROR: access probe could not reach Hugging "
                f"Face for {repository}: {exc.reason}",
                file=sys.stderr,
            )
            return 4
        actual_revision = str(info.get("sha") or "")
        if expected_revision and actual_revision != expected_revision:
            print(
                f"[npa-cosmos3-serving] ERROR: {repository} resolved to "
                f"{actual_revision or 'unknown'}, expected pinned revision "
                f"{expected_revision}.",
                file=sys.stderr,
            )
            return 5
        print(
            f"[npa-cosmos3-serving] access confirmed: "
            f"{repository}@{actual_revision or 'unknown'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
