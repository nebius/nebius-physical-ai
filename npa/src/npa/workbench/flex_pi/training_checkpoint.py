"""Join node-local Accelerate RNG files into the sole published checkpoint."""

from pathlib import Path

from npa.workbench.flex_pi.training_topology import training_topology


def collect_rank_checkpoint(directory):
    """Collect every saved rank RNG file without changing live generator state.

    Args:
        directory: This rank's completed Accelerate state directory.
    Returns:
        None; the leader owns all RNG files and every node owns the saved cursor.
    Raises:
        RuntimeError: A rank's state or the leader's training cursor is absent.
    """
    if training_topology()["nodes"] == 1:
        return
    import torch.distributed as distributed

    rank = distributed.get_rank()
    root = Path(directory)
    rng = root / f"random_states_{rank}.pkl"
    cursor = root / "trainer_state.json"
    payload = {
        "rank": rank,
        "rng": rng.read_bytes() if rng.is_file() else None,
        "cursor": cursor.read_bytes() if rank == 0 and cursor.is_file() else None,
    }
    peers = [None] * 4
    distributed.all_gather_object(peers, payload)
    if not peers[0]["cursor"] or any(
        row["rank"] != index or not row["rng"] for index, row in enumerate(peers)
    ):
        raise RuntimeError("distributed checkpoint lacks a rank RNG or training cursor")
    cursor.write_bytes(peers[0]["cursor"])
    if rank == 0:
        for index, row in enumerate(peers):
            (root / f"random_states_{index}.pkl").write_bytes(row["rng"])
