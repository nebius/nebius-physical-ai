"""Verify and supervise a pinned published RLC checkpoint on the 2026 evaluator."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .protocol import file_digest

SOURCE_COMMIT = "ca556f74a455cef7987a2be4537b5ac85cc56dd7"
OPENPI_COMMIT = "01177e0242a1c7e8fad2547caa0e987def614cda"
BEHAVIOR_COMMIT = "684a83050ddd398de231e6aa7fc605bc34458d4b"
MODEL_REVISION = "89545bc1b7aa7f2e687bc0032d091f132d715d4e"
NORMALIZATION = "assets/IliaLarchenko/behavior_224_rgb/norm_stats.json"


def _verify_checkout(root: Path, expected: str) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if revision != expected or changes:
        raise ValueError("RLC source must match the clean pinned checkout")


def _task_checkpoint(root: Path, upstream: Path, task: str) -> tuple[int, str]:
    task_id = _task_id(root, upstream, task)
    mapping = json.loads((root / "task_checkpoint_mapping.json").read_text())
    matches = [
        name for name, row in mapping["checkpoints"].items() if task_id in row["tasks"]
    ]
    if len(matches) != 1:
        raise ValueError("RLC task must map to exactly one checkpoint")
    return task_id, matches[0]


def _task_id(root: Path, upstream: Path, task: str) -> int:
    old = json.loads((root / "BEHAVIOR-1K/docs/challenge/task_data.json").read_text())
    new = json.loads((upstream / "docs/challenge/task_data.json").read_text())
    names = [item["id"] for item in old["tasks"]]
    current = [item["id"] for item in new["tasks"]]
    if len(names) != 50 or names != current[:50] or task not in names:
        raise ValueError("Published RLC checkpoints support the original 50 tasks only")
    return names.index(task)


def _verify_published_files(root: Path, checkpoint: str, files: dict) -> None:
    manifest = json.loads(Path(__file__).with_name("rlc-checkpoints.json").read_text())
    expected = manifest["checkpoints"][checkpoint]
    if manifest["revision"] != MODEL_REVISION or set(files) != set(expected):
        raise ValueError("Checkpoint file set differs from the pinned published model")
    for relative, identity in expected.items():
        path = root / relative
        if path.stat().st_size != identity["size"]:
            raise ValueError("Published checkpoint file size differs")
        if "sha256" in identity:
            matches = files[relative] == identity["sha256"]
        else:
            digest = hashlib.sha1(
                f"blob {identity['size']}\0".encode(), usedforsecurity=False
            )
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            matches = digest.hexdigest() == identity["git_blob_sha1"]
        if not matches:
            raise ValueError("Checkpoint bytes differ from the pinned published model")


def _adapter_files(
    output: Path, *, selected: bool, specialist: bool = False
) -> dict[str, str]:
    adapters = {}
    names = [
        "rlc_server.py",
        "rlc_observations.py",
        "rlc_execution.py",
        "rlc_correlation.py",
    ]
    if selected:
        names.extend(("rlc_selected.py", "rlc_selected_server.py", "rlc_transition.py"))
    if specialist:
        names.extend(("rlc_specialist.py", "rlc-specialist-checkpoint.json"))
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), output / name)
        adapters[name] = file_digest(output / name)
    return adapters


def _record(
    output: Path,
    command: list[str],
    files: dict,
    checkpoint: str,
    plan: dict,
    stock_correlation: Path | None = None,
):
    adapters = _adapter_files(output, selected=False)
    shutil.copyfile(
        Path(__file__).with_name("POLICY_LICENSE"), output / "rlc-adapter.LICENSE"
    )
    evidence = {
        "schema": "npa.behavior.policy.v1",
        "kind": "rlc",
        "source_commit": SOURCE_COMMIT,
        "openpi_commit": OPENPI_COMMIT,
        "model_repository": "IliaLarchenko/behavior_submission",
        "published_model_revision": MODEL_REVISION,
        "checkpoint": checkpoint,
        "checkpoint_files": files,
        "checkpoint_archive_sha256": plan["recipe"]["policy_checkpoint_sha256"],
        "adapters": adapters,
        "command": command,
        "execution_variant": _execution_variant_record(command, selected=False),
        "stock_correlation": _stock_correlation_record(stock_correlation),
        "normalization_asset": NORMALIZATION,
        "memory_compliance": "unverified",
        "control": "published stage voting, rolling inpainting, compression and recovery",
        "transfer_limitation": "2025 training base velocity differs from 2026 robot-local velocity",
    }
    (output / "policy-provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


def _stock_correlation_record(path: Path | None) -> dict | None:
    if path is None:
        return None
    return {
        "path": path.name,
        "sha256": file_digest(path),
        "bytes": path.stat().st_size,
        "installation": "pre_policy_fp32_intermediate",
    }


def _verify_task(args, plan):
    tasks = plan["recipe"]["tasks"]
    if (
        plan["recipe"]["split"] not in {"development", "report"}
        or not isinstance(tasks, list)
        or len(tasks) != 1
    ):
        raise ValueError("RLC transfer requires one development or report task")
    for relative, revision in (
        (".", SOURCE_COMMIT),
        ("openpi", OPENPI_COMMIT),
        ("BEHAVIOR-1K", BEHAVIOR_COMMIT),
    ):
        _verify_checkout(args.policy_root / relative, revision)
    task_id, checkpoint = _task_checkpoint(
        args.policy_root, args.upstream_root, tasks[0]
    )
    kind = getattr(args, "policy_kind", "rlc")
    if kind == "rlc-specialist":
        from .rlc_specialist import SUPPORTED_TASK_IDS
        from .rlc_specialist_admission import _report_paths_present

        if plan["recipe"]["split"] == "report":
            from .rlc_specialist_admission import verify_specialist_policy_scope

            verify_specialist_policy_scope(args, plan)
        elif any(_report_paths_present(args)):
            raise ValueError("Specialist development does not accept report receipts")
        if task_id not in SUPPORTED_TASK_IDS:
            raise ValueError("Released RLC specialist does not support this task")
        return task_id, "shawn-task-specialist"
    if kind == "rlc-selected":
        if checkpoint != "checkpoint_2":
            raise ValueError("Selected RLC export supports checkpoint_2 tasks only")
        return task_id, "selected"
    return task_id, checkpoint


def _verify_weights(args, plan, checkpoint):
    from .policy import _verify_checkpoint

    files = _verify_checkpoint(
        args.policy_archive,
        args.policy_checkpoint,
        plan["recipe"]["policy_checkpoint_sha256"],
        prefix=checkpoint + "/",
        normalization=NORMALIZATION,
    )
    _verify_published_files(args.policy_checkpoint, checkpoint, files)
    return files


def _verify_specialist_weights(args, plan):
    from .policy import _verify_checkpoint
    from .rlc_specialist import NORMALIZATION, verify_checkpoint_files

    files = _verify_checkpoint(
        args.policy_archive,
        args.policy_checkpoint,
        plan["recipe"]["policy_checkpoint_sha256"],
        prefix="shawn-task-specialist/",
        normalization=NORMALIZATION,
    )
    verify_checkpoint_files(args.policy_checkpoint, files)
    return files


def _verify_selected_export(args, files: dict[str, str]) -> dict:
    path = args.policy_selected_export_receipt
    if path.is_symlink() or not path.is_file():
        raise ValueError("Selected export receipt must be a regular file")
    receipt = json.loads(path.read_text())
    expected = receipt.get("files")
    if (
        receipt.get("schema") != "npa.behavior.rlc-selected-export.v1"
        or receipt.get("status") != "holdout_selected_not_rollout_evaluated"
        or not isinstance(receipt.get("selected_step"), int)
        or isinstance(receipt.get("selected_step"), bool)
        or receipt["selected_step"] <= 0
        or not isinstance(expected, dict)
        or set(files) != set(expected)
    ):
        raise ValueError("Selected export receipt contract differs")
    for name, identity in expected.items():
        path = args.policy_checkpoint / name
        if path.is_symlink() or identity != {
            "sha256": files[name],
            "bytes": path.stat().st_size,
        }:
            raise ValueError("Selected checkpoint bytes differ from its export receipt")
    return receipt


def _verify_selected_weights(args, plan):
    from .policy import _verify_checkpoint
    from .rlc_selected import validate_selected_correlation

    files = _verify_checkpoint(
        args.policy_archive,
        args.policy_checkpoint,
        plan["recipe"]["policy_checkpoint_sha256"],
        prefix="selected-model/",
        normalization=NORMALIZATION,
    )
    receipt = _verify_selected_export(args, files)
    validate_selected_correlation(
        args.policy_correlation_manifest,
        args.policy_validation_receipt,
        args.policy_selected_export_receipt,
        Path(__file__).parent,
    )
    return files, receipt


def _stage_selected_artifacts(args, output: Path) -> dict[str, Path]:
    manifest = json.loads(args.policy_correlation_manifest.read_text())
    artifact = args.policy_correlation_manifest.parent / manifest["artifact"]["path"]
    sources = {
        "selected_export": args.policy_selected_export_receipt,
        "correlation_manifest": args.policy_correlation_manifest,
        "validation_receipt": args.policy_validation_receipt,
        "correlation_artifact": artifact,
    }
    staged = {}
    for name, source in sources.items():
        target = output / source.name
        if target in staged.values():
            raise ValueError("Selected RLC artifact basenames must be distinct")
        shutil.copyfile(source, target)
        staged[name] = target
    return staged


def _command(
    args,
    task_id,
    output,
    selected_artifacts=None,
    stock_correlation: Path | None = None,
):
    server = "rlc_selected_server.py" if selected_artifacts else "rlc_server.py"
    command = [
        str(args.policy_python),
        str(output / server),
        "--source-root",
        str(args.policy_root),
        "--checkpoint",
        str(args.policy_checkpoint),
        "--task-id",
        str(task_id),
        "--port",
        str(args.port),
    ]
    if selected_artifacts:
        _append_selected_arguments(command, output, selected_artifacts)
    if stock_correlation is not None:
        command.extend(
            [
                "--correlation-asset",
                str(stock_correlation),
                "--correlation-sha256",
                getattr(args, "policy_stock_correlation_sha256"),
            ]
        )
    if getattr(args, "policy_kind", "rlc") == "rlc-specialist":
        command.append("--specialist-state-contract")
    command.extend(
        [
            "--execution-variant",
            getattr(args, "policy_execution_variant", "native"),
        ]
    )
    return command


def _append_selected_arguments(
    command: list[str], output: Path, artifacts: dict
) -> None:
    command[4:4] = ["--adapter-root", str(output)]
    command.extend(
        [
            "--selected-export-receipt",
            str(artifacts["selected_export"]),
            "--correlation-manifest",
            str(artifacts["correlation_manifest"]),
            "--validation-receipt",
            str(artifacts["validation_receipt"]),
        ]
    )


def _stage_stock_correlation(args, output: Path) -> Path | None:
    source = getattr(args, "policy_stock_correlation_asset", None)
    expected = getattr(args, "policy_stock_correlation_sha256", None)
    if (source is None) != (expected is None):
        raise ValueError("stock correlation requires both artifact and SHA-256")
    if source is None:
        return None
    from .rlc_correlation import load_fp32_correlation

    load_fp32_correlation(source, expected)
    target = output / "stock-correlation.float32.bin"
    shutil.copyfile(source, target)
    if file_digest(target) != expected:
        raise ValueError("staged stock correlation identity differs")
    return target


def _execution_variant_record(command: list[str], *, selected: bool) -> dict:
    variant = command[command.index("--execution-variant") + 1]
    if variant == "transition-refresh":
        from .rlc_transition import TRANSITION_REFRESH_PROVENANCE

        provenance = dict(TRANSITION_REFRESH_PROVENANCE)
    else:
        from .rlc_execution import execution_provenance

        value = execution_provenance(variant, selected=selected)
        provenance = dict(value) if value is not None else None
    return {
        "name": variant,
        "transition_refresh_provenance": (
            provenance if variant == "transition-refresh" else None
        ),
        "provenance": provenance,
    }


def _record_selected(output, command, files, receipt, plan, staged):
    variant_record = _execution_variant_record(command, selected=True)
    variant = variant_record["name"]
    adapters = _adapter_files(output, selected=True)
    shutil.copyfile(
        Path(__file__).with_name("POLICY_LICENSE"), output / "rlc-adapter.LICENSE"
    )
    evidence = {
        "schema": "npa.behavior.policy.v1",
        "kind": "rlc-selected",
        "source_commit": SOURCE_COMMIT,
        "openpi_commit": OPENPI_COMMIT,
        "selected_step": receipt["selected_step"],
        "checkpoint_files": files,
        "checkpoint_archive_sha256": plan["recipe"]["policy_checkpoint_sha256"],
        "adapters": adapters,
        "selected_artifacts": {
            name: {"sha256": file_digest(path), "bytes": path.stat().st_size}
            for name, path in staged.items()
        },
        "command": command,
        "normalization_asset": NORMALIZATION,
        "execution_variant": variant_record,
        "status": (
            "serving_validated_not_rollout_evaluated"
            if variant == "native"
            else "selected_state_validated_execution_variant_unevaluated"
        ),
    }
    (output / "policy-provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


def _record_specialist(output, command, files, plan, *, args):
    from .rlc_specialist import (
        MODEL_REPOSITORY,
        MODEL_REVISION,
        NORMALIZATION,
        SOURCE_COMMIT as SPECIALIST_SOURCE_COMMIT,
        SOURCE_REPOSITORY,
        SUPPORTED_TASK_IDS,
    )

    adapters = _adapter_files(output, selected=False, specialist=True)
    from .rlc_specialist_admission import verify_staged_specialist_runtime

    verify_staged_specialist_runtime(args, output, adapters, command)
    shutil.copyfile(
        Path(__file__).with_name("POLICY_LICENSE"), output / "rlc-adapter.LICENSE"
    )
    evidence = {
        "schema": "npa.behavior.policy.v1",
        "kind": "rlc-specialist",
        "source_commit": SOURCE_COMMIT,
        "openpi_commit": OPENPI_COMMIT,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_source_repository": SOURCE_REPOSITORY,
        "model_source_commit": SPECIALIST_SOURCE_COMMIT,
        "supported_task_ids": sorted(SUPPORTED_TASK_IDS),
        "checkpoint_files": files,
        "checkpoint_archive_sha256": plan["recipe"]["policy_checkpoint_sha256"],
        "normalization_asset": NORMALIZATION,
        "adapters": adapters,
        "command": command,
        "execution_variant": "native",
        "memory_compliance": "unverified",
        "status": "local_development_only_memory_unverified_not_rollout_ranked",
    }
    _add_specialist_report_provenance(evidence, args, plan)
    (output / "policy-provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


def _add_specialist_report_provenance(evidence, args, plan) -> None:
    if plan["recipe"].get("split", "development") != "report":
        return
    from .rlc_specialist_admission import specialist_report_provenance

    evidence["report_authorization"] = specialist_report_provenance(args, plan)
    evidence["status"] = "local_report_admitted_memory_unverified"


def prepare_policy(args, plan: dict, output: Path) -> list[str]:
    """Validate the RLC source, task, archive and launcher before policy execution.

    Args:
        args: Managed policy paths and loopback endpoint.
        plan: Frozen one-task development or report recipe.
        output: Evidence directory receiving launcher sources and provenance.
    Returns:
        Policy interpreter command using the published control settings.
    Raises:
        ValueError: Source, checkpoint, task, endpoint or development scope is invalid.
        OSError: A required source or checkpoint cannot be read.
    """
    from .policy import _healthy

    task_id, checkpoint = _verify_task(args, plan)
    kind = getattr(args, "policy_kind", "rlc")
    selected = kind == "rlc-selected"
    specialist = kind == "rlc-specialist"
    if selected:
        files, receipt = _verify_selected_weights(args, plan)
    elif specialist:
        files = _verify_specialist_weights(args, plan)
    else:
        files = _verify_weights(args, plan, checkpoint)
    if _healthy(args.port) or any(
        case.get("policy_port") not in {None, args.port} for case in plan["cases"]
    ):
        raise ValueError("Managed RLC policy requires its own matching loopback port")
    staged = _stage_selected_artifacts(args, output) if selected else None
    stock_correlation = None if selected else _stage_stock_correlation(args, output)
    command = _command(args, task_id, output, staged, stock_correlation)
    if selected:
        _record_selected(output, command, files, receipt, plan, staged)
    elif specialist:
        _record_specialist(output, command, files, plan, args=args)
    else:
        _record(output, command, files, checkpoint, plan, stock_correlation)
    return command
