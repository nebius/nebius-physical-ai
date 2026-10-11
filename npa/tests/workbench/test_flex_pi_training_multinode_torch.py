"""Exercise the real CPU rendezvous and node-local checkpoint join on four processes."""

from datetime import timedelta
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


def _peer_environment(rank):
    os.environ.update(
        NPA_FLEX_PI_NODE_COUNT="4",
        SKYPILOT_NODE_RANK=str(rank),
        SKYPILOT_NODE_IPS="127.0.0.1 127.0.0.2 127.0.0.3 127.0.0.4",
        NPA_FLEX_PI_SESSION="cpu-coordination-test",
    )


def _coordinate_peer(rank, port, directory, fail_rank, listener):
    from npa.workbench.flex_pi import training_multinode as coordinator
    from npa.workbench.flex_pi.runtime import FlexPiError

    _peer_environment(rank)
    native_tcp_store = torch.distributed.TCPStore

    def open_reserved_store(*args, **kwargs):
        if listener is not None:
            kwargs["master_listen_fd"] = listener.detach()
        return native_tcp_store(*args, **kwargs)

    # Keep the real production rendezvous; transfer its already-bound socket.
    with patch.object(torch.distributed, "TCPStore", open_reserved_store):
        store = coordinator._connect_store(
            leader=rank == 0, coordinator=True, port=port
        )
    parent = coordinator._Coordinator(store, rank)
    parent.watch.thread.start()
    parent._execute = lambda command: _cpu_phase(
        parent, command, directory, fail_rank, port
    )
    try:
        if rank == 0:
            result = _lead_cpu_phases(parent, directory)
            parent.finish(result)
        else:
            result = parent.follow()
        Path(directory, f"result-{rank}.json").write_text(json.dumps(result))
    except FlexPiError as error:
        coordinator._signal_failure(store, rank, error)
        Path(directory, f"failure-{rank}").write_text(type(error).__name__)
    finally:
        parent.watch.close()


def _cpu_phase(parent, command, directory, fail_rank, port):
    from npa.workbench.flex_pi import training_multinode as coordinator
    from npa.workbench.flex_pi.runtime import FlexPiError

    sequence = command["sequence"]
    if parent.rank == fail_rank:
        raise FlexPiError("injected local worker failure")
    assert bytes.fromhex(command["normalization"]) == b"{}" + b" " * 207849
    marker = Path(directory, f"phase-{sequence}-rank-{parent.rank}")
    script = """import pathlib,sys
from npa.workbench.flex_pi import training_multinode as coordinator
store = coordinator._connect_store(port=int(sys.argv[2]))
prefix = sys.argv[3]
store.set(prefix + '/' + sys.argv[4], 'ready')
coordinator._wait_keys(store, [prefix + '/' + str(rank) for rank in range(4)])
pathlib.Path(sys.argv[1]).write_text('completed')
"""
    coordinator._run_worker(
        [
            sys.executable,
            "-c",
            script,
            str(marker),
            str(port),
            f"phase-client/{sequence}",
            str(parent.rank),
        ],
        os.environ.copy(),
        parent.store,
    )
    parent.store.set(f"done/{sequence}/{parent.rank}", "complete")
    if parent.rank == 0:
        Path(command["root"], "result.json").write_text(
            json.dumps({"sequence": sequence})
        )


def _lead_cpu_phases(parent, directory):
    normalization = Path(directory, "stats.json")
    normalization.write_bytes(b"{}" + b" " * 207849)
    plan = {"normalization_file": str(normalization)}
    for sequence in range(1, 3):
        root = Path(directory, f"phase-{sequence}")
        root.mkdir()
        result = parent.dispatch(plan, root)
        assert result == {"sequence": sequence}
    return result


def _checkpoint_peer(rank, port, directory, missing_rank, listener):
    from npa.workbench.flex_pi.training_checkpoint import collect_rank_checkpoint

    _peer_environment(rank)
    store = torch.distributed.TCPStore(
        "127.0.0.1",
        port,
        world_size=4,
        is_master=rank == 0,
        timeout=timedelta(seconds=30),
        master_listen_fd=listener.detach() if listener is not None else None,
        use_libuv=False,
    )
    torch.distributed.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    root = Path(directory, f"node-{rank}")
    root.mkdir()
    if rank != missing_rank:
        (root / f"random_states_{rank}.pkl").write_bytes(f"saved-rng-{rank}".encode())
    if rank == 0:
        (root / "trainer_state.json").write_bytes(b'{"global_step":30}')
    before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    try:
        collect_rank_checkpoint(root)
        assert (
            before
            == hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        )
        (root / "passed").touch()
    except RuntimeError:
        if missing_rank < 0:
            raise
        (root / "rejected").touch()
    finally:
        torch.distributed.destroy_process_group()


def _four_processes(target, tmp_path, failure):
    context = multiprocessing.get_context("spawn")
    processes = []
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            # A concurrent worker must not claim the port before native adoption.
            with socket.socket() as contender:
                with pytest.raises(OSError) as refused:
                    contender.bind(("127.0.0.1", port))
                assert refused.value.errno == errno.EADDRINUSE
            processes = [
                context.Process(
                    target=target,
                    args=(
                        rank,
                        port,
                        str(tmp_path),
                        failure,
                        listener if rank == 0 else None,
                    ),
                )
                for rank in range(4)
            ]
            for process in processes:
                process.start()
        for process in processes:
            process.join(timeout=45)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


@pytest.mark.parametrize("failure", [-1, 3])
def test_real_four_process_control_protocol(tmp_path, failure):
    _four_processes(_coordinate_peer, tmp_path, failure)
    if failure < 0:
        assert [
            json.loads(path.read_text())
            for path in sorted(tmp_path.glob("result-*.json"))
        ] == [{"sequence": 2}] * 4
        assert len(list(tmp_path.glob("phase-*-rank-*"))) == 8
    else:
        assert len(list(tmp_path.glob("failure-*"))) == 4
        assert not list(tmp_path.glob("result-*.json"))


@pytest.mark.parametrize("missing_rank", [-1, 3])
def test_real_four_process_rng_checkpoint_join(tmp_path, missing_rank):
    _four_processes(_checkpoint_peer, tmp_path, missing_rank)
    marker = "passed" if missing_rank < 0 else "rejected"
    assert len(list(tmp_path.glob(f"node-*/{marker}"))) == 4
    if missing_rank < 0:
        assert [
            (tmp_path / f"node-0/random_states_{rank}.pkl").read_bytes()
            for rank in range(4)
        ] == [f"saved-rng-{rank}".encode() for rank in range(4)]
