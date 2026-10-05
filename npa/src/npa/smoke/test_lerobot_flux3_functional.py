"""Import and pinned recipe smoke; live LoRA remains a separate GPU gate."""

import json
from pathlib import Path


def main() -> None:
    import lerobot
    import natten
    import torch

    assert lerobot and natten
    assert torch.__version__.startswith("2.11.")
    assert torch.cuda.is_available()
    recipe = json.loads(
        Path(__file__)
        .parents[1]
        .joinpath("workbench/lerobot/flux3_so101/recipe.json")
        .read_text()
    )
    assert recipe["dataset"]["fps"] == 30
    assert recipe["dataset"]["cameras"] == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert Path("/opt/lerobot/examples/flux3/lora.json").is_file()
    print(
        json.dumps({"status": "imports-and-recipe-valid", "torch": torch.__version__})
    )


if __name__ == "__main__":
    main()
