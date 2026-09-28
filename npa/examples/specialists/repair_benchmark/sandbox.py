"""Build fixed, network-isolated commands for candidate checks and native simulation."""

from pathlib import Path

TARGETS = (
    "npa/workbench/token_factory/robot_sim.py",
    "npa/workbench/token_factory/robot_artifacts.py",
    "npa/adapter/sim_to_lerobot.py",
)
TESTS = (
    "/tests/test_robot_physics_trace_contract.py",
    "/tests/test_robot_export_transaction.py",
    "/tests/test_robot_export_provenance.py",
    "/tests/test_token_factory_robot_sdg.py",
    "/tests/test_sim_episode_stream_lengths.py",
    "/tests/test_adapter.py::TestConvertErrors",
)


def _runtime_mounts(interpreter):
    paths = [
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/etc/alternatives",
        "/etc/ld.so.cache",
        str(interpreter.parent.parent),
        str(interpreter.resolve().parents[2]),
    ]
    return [
        arg
        for path in paths
        if Path(path).exists()
        for arg in ("--ro-bind", path, path)
    ]


def _input_mounts(config, root):
    mounts = {
        "/workbench/npa/src": config["source"],
        "/tests": config["tests"],
        "/driver": config["driver"],
        "/input/matrix.json": config["matrix"],
    }
    mounts.update(
        {"/workbench/npa/src/" + name: str(root / "source" / name) for name in TARGETS}
    )
    return [
        arg
        for destination, source in mounts.items()
        for arg in ("--ro-bind", source, destination)
    ]


def _environment():
    values = {
        "HOME": "/home/worker",
        "TMPDIR": "/scratch",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": "/workbench/npa/src",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    return [arg for key, value in values.items() for arg in ("--setenv", key, value)]


def _writable_runtime(root):
    return [
        "--bind",
        str(root / "output"),
        "/output",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/scratch",
        "--dir",
        "/home/worker",
        "--chdir",
        "/output",
    ]


def _command(config, root, kind):
    interpreter = Path(config["test_python"] if kind == "tests" else config["python"])
    argv = [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--clearenv",
    ]
    argv += _runtime_mounts(interpreter) + _input_mounts(config, root)
    argv += _writable_runtime(root) + _environment() + [str(interpreter)]
    if kind == "tests":
        return argv + [
            "-m",
            "pytest",
            *TESTS,
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ]
    return argv + [
        "/driver/workload.py",
        "--matrix",
        "/input/matrix.json",
        "--output",
        "/output/run",
    ]
