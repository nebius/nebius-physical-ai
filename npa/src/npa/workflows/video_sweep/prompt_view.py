"""Show how source descriptions and hints become shared Cosmos prompt inputs."""

from html import escape


def render(summary: dict) -> str:
    """Render prompt preparation and its connection to parameter fanout.

    Args:
        summary: Public-safe matrix summary, without raw source or prompt text.
    Returns:
        An escaped flow diagram and prompt-group membership table.
    Raises:
        None.
    """
    groups = summary.get("prompt_groups", [])
    if not groups:
        return ""
    augmented = sum(group["mode"] == "llm-augmented" for group in groups)
    stages = [
        ("1 · Observe", f"{summary['sources']} source videos → VLM scene descriptions"),
        ("2 · Augment", f"Descriptions + hints → {augmented} LLM-enhanced prompts"),
        ("3 · Expand", "Each final prompt + every configured parameter combination"),
        (
            "4 · Generate",
            f"{len(summary['jobs'])} Cosmos3 inputs → {summary['workers']} GPU workers",
        ),
    ]
    cards = "".join(
        f'<div class="prompt-stage"><b>{escape(title)}</b><p>{escape(body)}</p></div>'
        for title, body in stages
    )
    rows = "".join(_group(group) for group in groups)
    return f"""<section aria-label="Prompt augmentation flow">
<h3>Source + user hint → LLM augmentation → parameter fanout</h3>
<div class="prompt-flow">{cards}</div>
<p>Enhanced prompt = source-video authority + LLM-expanded appearance + protected preserve constraints + avoid constraints.
Sampling comparisons reuse the exact same final prompt. Direct prompts bypass the LLM explicitly.</p>
<ul class="prompt-groups">{rows}</ul></section>"""


def _group(group):
    mode = (
        "LLM augmentation"
        if group["mode"] == "llm-augmented"
        else "Direct prompt · LLM bypassed"
    )
    members = ", ".join(str(candidate) for candidate in group["candidates"])
    return f"<li><strong>{escape(group['id'])}</strong> · Source {group['source']} · {mode} → candidates {members}</li>"
