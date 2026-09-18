"""Keep an Antioch evaluation attached until completion without a command deadline."""

from importlib.metadata import version
import sys


def run(command: list[str]) -> int:
    """Execute literal argv through the pinned Antioch client's managed attachment.

    Args:
        command: Remote argv, executed in the current Antioch project's service.
    Returns:
        Remote exit status, or 130 when the operator interrupts execution.
    Raises:
        ValueError: No command was provided or the installed SDK is incompatible.
        RuntimeError: The Antioch client cannot start or follow the command.
    """
    if not command:
        raise ValueError("An attached remote command is required")
    if version("antioch-sim") != "0.4.236":
        raise ValueError("This attachment adapter requires antioch-sim==0.4.236")
    from antioch.cli.commands.service import default_service, run_door_session, run_service_command
    from antioch.cli.options import StreamMode

    session = run_door_session(())
    service = default_service(session, None, operation="execute")
    interrupted, exit_code = run_service_command(
        session, service, command, stream=StreamMode.OFF, timeout_s=None, tty=False,
    )
    return 130 if interrupted else exit_code


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    raise SystemExit(run(arguments))
