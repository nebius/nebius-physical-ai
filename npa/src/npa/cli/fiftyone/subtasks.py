"""Register the FiftyOne-to-LeRobot subtask export command."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from typing import Any

import typer

from npa.lifecycle_intent import json_stdout_contract


@dataclass(frozen=True)
class _CommandBindings:
    output_format_type: type
    fail: Callable[[str], Any]
    emit: Callable[[object, Any], None]


_COMMAND_BINDINGS: _CommandBindings | None = None


def register_lerobot_subtask_export(
    app: typer.Typer,
    *,
    output_format_type: type,
    fail: Callable[[str], Any],
    emit: Callable[[object, Any], None],
) -> None:
    """Register the reviewed temporal-tag export command.

    Args:
        app: FiftyOne Typer application.
        output_format_type: CLI output enum used by the parent command group.
        fail: Parent error renderer.
        emit: Parent structured output renderer.
    Returns:
        None.

    Raises:
        None.
    """

    global _COMMAND_BINDINGS
    _COMMAND_BINDINGS = _CommandBindings(output_format_type, fail, emit)
    app.command("export-lerobot-subtasks")(export_lerobot_subtasks_cmd)


@json_stdout_contract
def export_lerobot_subtasks_cmd(
    dataset_name: str = typer.Option(
        ..., "--dataset-name",
        help="Persistent FiftyOne LeRobot dataset reviewed in the App.",
    ),
    output_path: str = typer.Option(
        ..., "--output-path",
        help="Empty S3 prefix for the derived LeRobot dataset.",
    ),
    output_format: str = typer.Option(
        "text", "--output-format",
        help="Output format: text or json.",
    ),
) -> None:
    """Export ``subtask:`` timeline tags into a derived LeRobot dataset.

    Args:
        dataset_name: Persistent FiftyOne dataset reviewed in the App.
        output_path: Empty S3 prefix for the derived LeRobot dataset.
        output_format: Human-readable text or machine-readable JSON.

    Returns:
        None.

    Raises:
        None. CLI failures are normalized by the parent command group.
    """
    bindings = _command_bindings()
    _validate_export_options(dataset_name, output_path, output_format, bindings.fail)
    try:
        report = _export_lerobot_subtasks_remote(dataset_name.strip(), output_path.strip())
    except Exception as exc:  # noqa: BLE001 - normalize the remote boundary
        bindings.fail(f"LeRobot subtask export failed: {exc}")
        return
    if output_format == "json":
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    bindings.emit(report, bindings.output_format_type(output_format))


def _command_bindings() -> _CommandBindings:
    if _COMMAND_BINDINGS is None:
        raise RuntimeError("FiftyOne subtask export command is not registered")
    return _COMMAND_BINDINGS


def _validate_export_options(
    dataset_name: str,
    output_path: str,
    output_format: str,
    fail: Callable[[str], Any],
) -> None:
    if not dataset_name.strip():
        fail("--dataset-name must not be empty.")
    if not output_path.strip().startswith("s3://"):
        fail("--output-path must be an s3:// URI.")
    if output_format not in {"text", "json"}:
        fail("--output-format must be text or json.")


def _subtask_adapter_source() -> str:
    return resources.files("npa").joinpath("fiftyone_lerobot_subtasks.py").read_text()


def bundle_lerobot_importer(importer_source: str) -> str:
    """Embed the subtask adapter in the standalone remote LeRobot importer.

    Args:
        importer_source: Source code of ``npa.fiftyone_lerobot``.

    Returns:
        Importer source that registers its bundled subtask adapter at startup.

    Raises:
        ValueError: If the importer lacks its expected future-import marker.
    """
    marker = "from __future__ import annotations\n"
    if marker not in importer_source:
        raise ValueError("LeRobot importer is missing its future-import marker")
    adapter_source = json.dumps(_subtask_adapter_source())
    bootstrap = f"""{marker}
import sys as _npa_sys
import types as _npa_types

_NPA_SUBTASK_MODULE = "_npa_fiftyone_lerobot_subtasks"
_npa_subtask_module = _npa_types.ModuleType(_NPA_SUBTASK_MODULE)
_npa_subtask_module.__file__ = "<bundled-npa-fiftyone-lerobot-subtasks>"
_npa_sys.modules[_NPA_SUBTASK_MODULE] = _npa_subtask_module
exec(compile({adapter_source}, _npa_subtask_module.__file__, "exec"), _npa_subtask_module.__dict__)
"""
    return importer_source.replace(marker, bootstrap, 1)


def _subtask_export_python_script(dataset_name: str, output_path: str) -> str:
    module_source = json.dumps(_subtask_adapter_source())
    return f"""\
from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

MODULE_SOURCE = {module_source}
DATASET_NAME = {json.dumps(dataset_name)}
OUTPUT_PATH = {json.dumps(output_path)}

with tempfile.TemporaryDirectory(prefix="npa-fiftyone-subtasks-module-") as private_directory:
    module_path = Path(private_directory) / "npa_fiftyone_lerobot_subtasks.py"
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_npa_fiftyone_lerobot_subtasks_export", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the bundled LeRobot subtask adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        report = module.export_fiftyone_subtasks_to_s3(DATASET_NAME, OUTPUT_PATH)
    finally:
        sys.modules.pop(spec.name, None)
print(json.dumps(report))
"""


def _build_export_lerobot_subtasks_command(dataset_name: str, output_path: str) -> str:
    from npa.cli import fiftyone as fiftyone_cli

    script = _subtask_export_python_script(dataset_name, output_path)
    command = f"""\
set -euo pipefail
source {fiftyone_cli.FIFTYONE_VENV}/bin/activate
{fiftyone_cli._source_storage_env_script()}
export FIFTYONE_DATABASE_DIR={fiftyone_cli.FIFTYONE_HOME}/db
python - <<'PY'
{script}
PY
"""
    return fiftyone_cli._remote_bash(command)


def _build_container_export_lerobot_subtasks_command(dataset_name: str, output_path: str) -> str:
    from npa.cli import fiftyone as fiftyone_cli

    script = _subtask_export_python_script(dataset_name, output_path)
    container_command = _container_subtask_export_command(fiftyone_cli, script)
    host_command = _container_host_command(fiftyone_cli, container_command)
    return fiftyone_cli._remote_bash(host_command)


def _container_subtask_export_command(fiftyone_cli: Any, script: str) -> str:
    return f"""\
set -euo pipefail
source {fiftyone_cli.FIFTYONE_VENV}/bin/activate
{fiftyone_cli.load_env_file_script('/etc/npa-fiftyone/env')}
export FIFTYONE_DATABASE_DIR={fiftyone_cli.FIFTYONE_CONTAINER_DB_DIR}
python - <<'PY'
{script}
PY
"""


def _container_host_command(fiftyone_cli: Any, container_command: str) -> str:
    return f"""\
set -euo pipefail
sudo docker inspect {fiftyone_cli.FIFTYONE_CONTAINER_NAME} >/dev/null
if ! sudo docker inspect -f '{{{{.State.Running}}}}' {fiftyone_cli.FIFTYONE_CONTAINER_NAME} | grep -q true; then
  sudo docker start {fiftyone_cli.FIFTYONE_CONTAINER_NAME} >/dev/null
fi
sudo docker exec -i {fiftyone_cli.FIFTYONE_CONTAINER_NAME} bash -lc {shlex.quote(container_command)}
"""


def _export_lerobot_subtasks_remote(dataset_name: str, output_path: str) -> dict[str, Any]:
    from npa.cli import fiftyone as fiftyone_cli

    cfg = fiftyone_cli._get_ssh_config()
    ssh = fiftyone_cli.SSHClient(cfg.ssh)
    if fiftyone_cli._is_container_runtime(cfg):
        command = _build_container_export_lerobot_subtasks_command(dataset_name, output_path)
    else:
        command = _build_export_lerobot_subtasks_command(dataset_name, output_path)
    _, stdout, _ = fiftyone_cli._run_fiftyone_command(
        ssh,
        command,
        label="FiftyOne LeRobot subtask export",
    )
    report = fiftyone_cli._parse_first_json_object(stdout)
    if report is None:
        raise RuntimeError("FiftyOne subtask export returned no JSON report")
    return report
