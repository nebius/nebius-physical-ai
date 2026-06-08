"""SDK helpers for Cosmos3 reason workflow contracts."""

from __future__ import annotations

from npa.workflows.cosmos_split import Cosmos3ReasonConfig, build_cosmos3_reason_manifest


def reason(
    *,
    input_uri: str,
    output_uri: str,
    model: str = "nvidia/Cosmos-Reason1-7B",
    image: str = "",
    prompt: str = "",
    run_id: str = "",
) -> dict[str, object]:
    """Return a Cosmos3 reason manifest."""

    return build_cosmos3_reason_manifest(
        Cosmos3ReasonConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            model=model,
            image=image,
            prompt=prompt,
            run_id=run_id,
        )
    )


__all__ = ["Cosmos3ReasonConfig", "reason"]
