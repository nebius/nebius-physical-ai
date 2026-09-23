"""Coordinate fresh four-node training phases with one durable artifact publisher."""

from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training_topology import training_topology

ACTIVE = None


def launch_training(request):
    """Enter the vendor interpreter for the CPU-only distributed coordinator.

    Args:
        request: Validated immutable training request.
    Returns:
        The leader's verified result on every participating node.
    Raises:
        FlexPiError: Any coordinator or training phase fails.
    """
    if not request.normalization_path:
        raise FlexPiError(
            "multi-node training requires verified original normalization"
        )
    with tempfile.TemporaryDirectory(prefix="flex-pi-distributed-") as directory:
        path = Path(directory) / "request.json"
        path.write_text(json.dumps(asdict(request)))
        path.chmod(0o600)
        command = [
            os.environ.get("FLEX_PI_PYTHON", "/opt/conda/bin/python"),
            "-m",
            "npa.workbench.flex_pi.training_worker",
            "--distributed-request",
            str(path),
        ]
        from npa.workbench.flex_pi.training import _phase_environment

        process = subprocess.run(command, env=_phase_environment(), stdout=sys.stderr)
        if process.returncode:
            raise FlexPiError(f"distributed training exited {process.returncode}")
        return json.loads(path.with_name("result.json").read_text())


def _connect_store(*, leader=False, coordinator=False):
    import torch.distributed as distributed

    topology = training_topology()
    store = distributed.TCPStore(
        topology["master_address"],
        29501,
        world_size=4 if coordinator else None,
        is_master=leader,
        timeout=timedelta(minutes=5),
        wait_for_workers=coordinator,
    )
    return distributed.PrefixStore(os.environ["NPA_FLEX_PI_SESSION"], store)


def _check_failure(store):
    if ACTIVE is not None and ACTIVE.watch.error is not None:
        raise FlexPiError("training control connection failed") from ACTIVE.watch.error
    if store.check(["failure"]):
        raise FlexPiError(store.get("failure").decode())


def _wait_keys(store, keys):
    while not store.check(keys):
        _check_failure(store)
        time.sleep(0.25)
    _check_failure(store)


def phase_barrier(label):
    """Join node-local preparation before starting a fresh distributed process.

    Args:
        label: Preparation milestone within the current phase.
    Returns:
        None; every node has completed the milestone.
    Raises:
        FlexPiError: Another node reported a phase failure.
    """
    topology = training_topology()
    if topology["nodes"] == 1:
        return
    store = _connect_store()
    prefix = f"phase/{os.environ['NPA_FLEX_PI_PHASE']}/{label}"
    store.set(f"{prefix}/{topology['node_rank']}", "ready")
    _wait_keys(store, [f"{prefix}/{rank}" for rank in range(4)])


def _terminate_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_worker(command, env, store):
    process = subprocess.Popen(
        command, env=env, stdout=sys.stderr, start_new_session=True
    )
    try:
        while process.poll() is None:
            _check_failure(store)
            time.sleep(0.25)
        if process.returncode:
            raise FlexPiError(f"training worker exited {process.returncode}")
    finally:
        _terminate_group(process)


class _Coordinator:
    def __init__(self, store, rank):
        self.store = store
        self.rank = rank
        self.sequence = 0
        self.restored_checkpoint = None
        self.watch = _PeerWatch(store, rank)

    def dispatch(self, plan, root):
        self.sequence += 1
        command = {"plan": plan, "root": str(root), "sequence": self.sequence}
        normalization = Path(plan["normalization_file"])
        command["normalization"] = normalization.read_bytes().hex()
        self.store.set(f"command/{self.sequence}", json.dumps(command))
        self._execute(command)
        _wait_keys(self.store, [f"done/{self.sequence}/{rank}" for rank in range(4)])
        return json.loads((root / "result.json").read_text())

    def follow(self):
        while True:
            self.sequence += 1
            key = f"command/{self.sequence}"
            _wait_keys(self.store, [key])
            command = json.loads(self.store.get(key))
            if "result" in command:
                self.store.set(f"finished/{self.rank}", "acknowledged")
                return command["result"]
            self._execute(command)

    def _execute(self, command):
        from npa.workbench.flex_pi.training import _vendor_command, _phase_environment

        plan = command["plan"]
        root = Path(command["root"])
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        Path(plan["work_directory"]).mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.rank:
            self._prepare_inputs(plan, command["normalization"])
        path = root / "request.json"
        path.write_text(json.dumps(plan))
        path.chmod(0o600)
        env = _phase_environment()
        env["NPA_FLEX_PI_PHASE"] = str(command["sequence"])
        _run_worker(_vendor_command(path), env, self.store)
        self.store.set(f"done/{command['sequence']}/{self.rank}", "complete")

    def _prepare_inputs(self, plan, normalization):
        from npa.clients.storage import StorageClient
        from npa.workbench.flex_pi.training_artifacts import restore_checkpoint

        checkpoint = plan.get("distributed_checkpoint")
        if checkpoint and checkpoint != self.restored_checkpoint:
            restore_checkpoint(
                checkpoint["manifest"],
                checkpoint["source"],
                Path(checkpoint["directory"]),
                StorageClient.from_environment(),
            )
            self.restored_checkpoint = checkpoint
        path = Path(plan["normalization_file"])
        payload = bytes.fromhex(normalization)
        expected = plan["execution"]["normalization_sha256"]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise FlexPiError("distributed normalization differs from the original")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(payload)
        path.chmod(0o600)

    def finish(self, result):
        self.store.set(f"command/{self.sequence + 1}", json.dumps({"result": result}))
        _wait_keys(self.store, [f"finished/{rank}" for rank in range(1, 4)])


class _PeerWatch:
    def __init__(self, store, rank):
        self.store, self.rank = store, rank
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_seen = {peer: (None, time.monotonic()) for peer in range(4)}

    def _run(self):
        counter = 0
        try:
            while not self.stop.is_set():
                counter += 1
                self.store.set(f"heartbeat/{self.rank}", str(counter))
                self._check_peers()
                self.stop.wait(1)
        except Exception as error:
            self.error = error

    def _check_peers(self):
        for peer, (previous, changed_at) in self.last_seen.items():
            key = f"heartbeat/{peer}"
            value = self.store.get(key) if self.store.check([key]) else None
            if value != previous:
                self.last_seen[peer] = (value, time.monotonic())
            elif time.monotonic() - changed_at > 90:
                # This detects a dead node, including during asset preparation;
                # training and downloads themselves have no elapsed-time limit.
                self.store.set("failure", f"training node {peer} lost its heartbeat")

    def close(self):
        self.stop.set()
        self.thread.join()


def _signal_failure(store, rank, error):
    try:
        store.set("failure", f"training node {rank} failed: {type(error).__name__}")
        store.set(f"failed/{rank}", "acknowledged")
        if rank == 0:
            deadline = time.monotonic() + 5
            keys = [f"failed/{peer}" for peer in range(4)]
            while not store.check(keys) and time.monotonic() < deadline:
                time.sleep(0.1)
    except RuntimeError as reporting_error:
        print(
            "training failure notification unavailable: "
            + type(reporting_error).__name__,
            file=sys.stderr,
        )


def distributed_main(request_path):
    """Run the sole publisher or a node-local phase worker in the vendor runtime.

    Args:
        request_path: Private serialized training request from the NPA interpreter.
    Returns:
        None; writes the verified final result for the invoking NPA process.
    Raises:
        Exception: Propagates local failures and signals every surviving peer.
    """
    from npa.workbench.flex_pi.training import TrainingRequest, _execute_training, _plan

    global ACTIVE
    request = TrainingRequest(**json.loads(request_path.read_text()))
    rank = training_topology()["node_rank"]
    identity = request.output_path + "\n" + request.run_id
    os.environ["NPA_FLEX_PI_SESSION"] = hashlib.sha256(identity.encode()).hexdigest()
    store = _connect_store(leader=rank == 0, coordinator=True)
    ACTIVE = _Coordinator(store, rank)
    ACTIVE.watch.thread.start()
    try:
        result = (
            _execute_training(request, _plan(request)) if rank == 0 else ACTIVE.follow()
        )
        if rank == 0:
            ACTIVE.finish(result)
        request_path.with_name("result.json").write_text(
            json.dumps(result, allow_nan=False)
        )
    except BaseException as error:
        _signal_failure(store, rank, error)
        raise
    finally:
        ACTIVE.watch.close()
        ACTIVE = None
