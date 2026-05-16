"""SONIC serve command."""

from __future__ import annotations

from npa.cli.workbench.sonic.helpers import (
    DEFAULT_MODEL_REPO,
    InputType,
    OutputFormat,
    ServeMode,
    WorkbenchRuntime,
    enum_value,
    output,
    require_real_confirmation,
    validate_port,
)
from npa.serverless_common import validate_output_path
import typer


def serve_cmd(
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.container, "--runtime", help="Runtime."),
    mode: ServeMode = typer.Option(ServeMode.sim, "--mode", help="Serve mode."),
    input_type: InputType = typer.Option(InputType.keyboard, "--input-type", help="Input source."),
    model_repo: str = typer.Option(DEFAULT_MODEL_REPO, "--model-repo", help="Hugging Face model repo."),
    zmq_host: str = typer.Option("127.0.0.1", "--zmq-host", help="ZMQ source host."),
    zmq_port: int = typer.Option(5556, "--zmq-port", help="ZMQ source port."),
    zmq_topic: str = typer.Option("pose", "--zmq-topic", help="ZMQ topic."),
    realtime_debug_port: int = typer.Option(5557, "--realtime-debug-port", help="Realtime debug port."),
    headless: bool = typer.Option(True, "--headless/--no-headless", help="Run without an interactive viewer."),
    smoke: bool = typer.Option(False, "--smoke", help="Run the minimal smoke path and exit."),
    output_path: str = typer.Option("", "--output-path", help="S3 output URI for serverless serve smoke."),
    confirm_real: bool = typer.Option(False, "--confirm-real", help="Required to acknowledge real robot mode."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "--output", help="Output format."
    ),
) -> None:
    """Launch or describe a SONIC serving path."""

    runtime_value = enum_value(runtime)
    mode_value = enum_value(mode)
    input_value = enum_value(input_type)
    validate_port(zmq_port, "--zmq-port")
    validate_port(realtime_debug_port, "--realtime-debug-port")
    require_real_confirmation(mode_value, confirm_real)
    if input_value in {"zmq", "zmq_manager"} and not zmq_host:
        from npa.cli.workbench.sonic.helpers import fail

        fail("--zmq-host is required when --input-type uses ZMQ.")
    if runtime_value == "serverless":
        if not output_path:
            from npa.cli.workbench.sonic.helpers import fail

            fail("SONIC serve --runtime serverless requires --output-path.")
        try:
            validate_output_path(output_path)
        except ValueError as exc:
            from npa.cli.workbench.sonic.helpers import fail

            fail(str(exc))

    endpoint = f"tcp://{zmq_host}:{zmq_port}"
    payload = {
        "status": "smoke-ready" if smoke else "planned",
        "runtime": runtime_value,
        "mode": mode_value,
        "input_type": input_value,
        "model_repo": model_repo,
        "endpoint": endpoint,
        "zmq_topic": zmq_topic,
        "realtime_debug_port": realtime_debug_port,
        "headless": headless,
        "output_path": output_path,
        "container_command": (
            f"docker run --rm --gpus all -e SONIC_MODE=serve -e SONIC_INPUT_TYPE={input_value} "
            f"-p {zmq_port}:{zmq_port} -p {realtime_debug_port}:{realtime_debug_port} npa-sonic:0.1.0"
        ),
    }
    output(payload, output_format)
