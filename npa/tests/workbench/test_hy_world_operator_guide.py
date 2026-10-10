"""Keep the HY-World candidate guide executable by a new operator or agent."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs" / "workbench" / "byof-hy-world.md"


def _bash_blocks() -> list[str]:
    return re.findall(
        r"```bash\n(.*?)\n```", GUIDE.read_text(encoding="utf-8"), flags=re.DOTALL
    )


def _unquoted_angle_placeholder(block: str) -> str | None:
    """Return an angle placeholder that Bash would not see as quoted data."""

    quote = ""
    escaped = False
    for index, character in enumerate(block):
        if escaped:
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if character in {"'", '"'}:
            if not quote:
                quote = character
            elif quote == character:
                quote = ""
            continue
        if not quote and character == "<" and index + 1 < len(block):
            if block[index + 1].isalpha():
                return block[index : block.find(">", index) + 1]
    return None


def test_hy_world_operator_guide_has_parseable_copy_paste_shell_snippets() -> None:
    blocks = _bash_blocks()
    assert blocks
    for block in blocks:
        parsed = subprocess.run(
            ["bash", "-n"], input=block, text=True, capture_output=True, check=False
        )
        assert parsed.returncode == 0, parsed.stderr
        # Angle-bracket shell placeholders are data, never a redirect/operator.
        assert _unquoted_angle_placeholder(block) is None, block


def test_hy_world_guide_links_the_private_delivery_and_vllm_contracts() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert "npa/docker/workbench/hy-world/build.sh" in guide
    assert "--push --operator-private" in guide
    assert '--receipt-dir "$PRIVATE_RECEIPTS"' in guide
    assert 'json.load(open(sys.argv[1], encoding="utf-8"))["immutable_image"]' in guide
    assert 'vllm serve "$MODEL" --revision "$REVISION"' in guide
    assert '--served-model-name "$MODEL"' in guide
    assert "NPA_HY_WORLD_LLM_ADDR" in guide
    assert "metadata.executionMode: runtime" in guide
