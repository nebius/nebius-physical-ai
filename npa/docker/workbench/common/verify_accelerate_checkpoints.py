"""Verify real Accelerate loaders with tensors, traversal inputs, and named pipes."""

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import tempfile
import threading


def _index(folder, entry, *, wrapped=True):
    path = folder / "pytorch_model.bin.index.json"
    mapping = {"weight": entry}
    path.write_text(json.dumps({"weight_map": mapping} if wrapped else mapping))
    return path


@contextmanager
def _fifo_observer(path):
    os.mkfifo(path)
    finished = threading.Event()
    opened = threading.Event()

    def writer():
        while not finished.is_set():
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as error:
                if error.errno != errno.ENXIO:
                    raise
                finished.wait(0.01)
                continue
            opened.set()
            os.close(descriptor)
            return

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        yield opened
    finally:
        finished.set()
        thread.join()


def _load(folder, *, dispatch=False):
    import accelerate
    import torch

    model = torch.nn.Linear(2, 2, bias=False)
    if dispatch:
        return accelerate.load_checkpoint_and_dispatch(
            model, str(folder), device_map={"": "cpu"}
        )
    accelerate.load_checkpoint_in_model(model, str(folder), strict=True)
    return model


def _reject(folder):
    for dispatch in (False, True):
        try:
            _load(folder, dispatch=dispatch)
        except ValueError as error:
            assert "checkpoint" in str(error), str(error)
        else:
            raise AssertionError("unsafe index accepted by a real checkpoint loader")


def _regular_checkpoints(root, folder):
    import torch
    from safetensors.torch import save_file

    weights = {"weight": torch.full((2, 2), 63.0)}
    nested = folder / "nested"
    nested.mkdir()
    torch.save(weights, nested / "part.bin")
    blob = root / "blobs" / "safe-weights"
    blob.parent.mkdir()
    save_file(weights, str(blob), metadata={"format": "pt"})
    (folder / "cached.safetensors").symlink_to(blob)
    for entry, wrapped in (
        ("nested/part.bin", True),
        ("nested/part.bin", False),
        ("cached.safetensors", True),
    ):
        _index(folder, entry, wrapped=wrapped)
        for dispatch in (False, True):
            model = _load(folder, dispatch=dispatch)
            assert torch.equal(model.weight, weights["weight"])
            assert torch.equal(model(torch.ones(1, 2)), torch.full((1, 2), 126.0))


def _traversal_checkpoints(root, folder):
    import torch

    outside = root / "outside.bin"
    torch.save({"weight": torch.full((2, 2), 17.0)}, outside)
    for entry in (
        "../outside.bin",
        str(outside),
        "../../outside.bin",
        "../checkpoint-other/part.bin",
        "",
        None,
        42,
        [],
    ):
        _index(folder, entry)
        _reject(folder)


def _nonregular_checkpoints(folder):
    fifo = folder / "pipe.bin"
    with _fifo_observer(fifo) as opened:
        _index(folder, fifo.name)
        _reject(folder)
        assert not opened.is_set(), "a checkpoint loader opened the shard FIFO"
        link = folder / "pipe-link.bin"
        link.symlink_to(fifo)
        _index(folder, link.name)
        _reject(folder)
        assert not opened.is_set(), "a checkpoint loader followed the FIFO symlink"
    index = folder / "pytorch_model.bin.index.json"
    index.unlink()
    with _fifo_observer(index) as opened:
        _reject(folder)
        assert not opened.is_set(), "a checkpoint loader opened the index FIFO"


def main() -> None:
    """Exercise the installed CPU loaders without accessing models or the network.

    Args:
        None.
    Returns:
        None; prints the installed versions and successful behavior checks.
    Raises:
        AssertionError: A loader accepts an unsafe input or corrupts valid weights.
        OSError: Temporary files cannot be created.
    """
    import accelerate
    import torch

    with tempfile.TemporaryDirectory(prefix="npa-checkpoint-proof-") as temporary:
        root = Path(temporary)
        folder = root / "checkpoint"
        folder.mkdir()
        _regular_checkpoints(root, folder)
        _traversal_checkpoints(root, folder)
        _nonregular_checkpoints(folder)
    print(
        json.dumps(
            {
                "accelerate": accelerate.__version__,
                "torch": torch.__version__,
                "regular_nested_shards": "passed",
                "flat_index": "passed",
                "safetensors_cache_symlink": "passed",
                "traversal": "rejected",
                "fifo_shard_and_index": "rejected-before-open",
                "loaders": ["load_checkpoint_in_model", "load_checkpoint_and_dispatch"],
            }
        )
    )


if __name__ == "__main__":
    main()
