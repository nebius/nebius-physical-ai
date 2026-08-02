"""Guardrail helpers: a ``toolRef`` argv template must match a real CLI command.

Why this exists
---------------
An ``npa.workflow`` spec invokes a workbench tool by ``toolRef``. The engine turns
that reference into an argv list from
:data:`npa.orchestration.npa_workflow.catalog.TOOL_CATALOG` and runs it inside the
task pod. Nothing in the plan/render/submit path ever *checks* that the flags in an
argv template are flags the target CLI command actually accepts, so a template can
validate, plan and render perfectly and still crash the moment it runs on real
infrastructure.

That is not hypothetical. ``workbench.rl.policy_train`` renders
``npa workbench isaac-lab train --learning-rate ... --batch-size ... --input-path
...`` and none of those three options exist on that command (it takes
``--override``, ``--num-envs``, ``--steps``, ``--output-path``). The repo's own
``DESIGN.md`` §7 records the mismatch and deliberately left it unfixed; this module
turns it from tribal knowledge into a pinned, shrink-only guardrail.

This check replaces what the SkyPilot side of the three-tier contract used to give
us. While a raw SkyPilot YAML shipped for every tool, "the YAML declares an ``envs``
key per CLI flag" was a proxy for "the documented way to run this tool at scale
exposes its parameters". Once the npa.workflow spec is the only workflow surface,
the equivalent — and strictly sharper — question is "does the toolRef argv name real
CLI options?".
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
import re
import shlex
from typing import Any, Sequence


@dataclass(frozen=True)
class ResolvedCommand:
    """A CLI command reached by walking a Typer app tree with an argv prefix."""

    #: Dotted command path as a user types it, e.g. ``workbench sonic train``.
    path: str
    #: Fully qualified callback, e.g. ``npa.cli.workbench.sonic.train:train_cmd``.
    callback_ref: str
    #: Every long option the command accepts (``--flag`` forms only).
    flags: frozenset[str]


class ArgvResolutionError(RuntimeError):
    """Raised when an argv template cannot be mapped onto a CLI command."""


def _root_app() -> Any:
    from npa.cli.main import app

    return app


def _command_names(command_info: Any) -> tuple[str, ...]:
    """Return the names a Typer ``CommandInfo`` is reachable by."""

    explicit = getattr(command_info, "name", None)
    if explicit:
        return (str(explicit),)
    callback = getattr(command_info, "callback", None)
    if callback is None:
        return ()
    # Typer's default: the callback name with underscores turned into dashes.
    return (callback.__name__.lower().replace("_", "-"),)


def option_flags_for_callback(callback: Any) -> frozenset[str]:
    """Return every ``--long-option`` a Typer callback declares."""

    flags: set[str] = set()
    for param in inspect.signature(callback).parameters.values():
        for decl in getattr(param.default, "param_decls", ()) or ():
            for part in str(decl).split("/"):
                if part.startswith("--"):
                    flags.add(part)
    return frozenset(flags)


def resolve_argv_command(argv: Sequence[str]) -> ResolvedCommand:
    """Walk the ``npa`` Typer tree along ``argv`` and return the command it names.

    ``argv`` is a catalog ``argv_template``: ``["npa", "workbench", "sonic",
    "train", "--checkpoint", "{{config.x}}", ...]``. Only the leading
    non-option tokens are used for resolution; everything from the first ``-``
    onwards is treated as arguments.
    """

    tokens = [str(token) for token in argv]
    if not tokens or tokens[0] != "npa":
        raise ArgvResolutionError(
            f"argv template must start with 'npa', got {tokens[:1]!r}"
        )

    node = _root_app()
    walked: list[str] = []
    for token in tokens[1:]:
        if token.startswith("-"):
            break
        group = next(
            (
                info
                for info in node.registered_groups
                if str(getattr(info, "name", "")) == token
            ),
            None,
        )
        if group is not None:
            node = group.typer_instance
            walked.append(token)
            continue
        command = next(
            (
                info
                for info in node.registered_commands
                if token in _command_names(info)
            ),
            None,
        )
        if command is None:
            raise ArgvResolutionError(
                f"'npa {' '.join([*walked, token])}' is not a registered command or "
                "group; the toolRef argv names a CLI path that does not exist"
            )
        callback = command.callback
        walked.append(token)
        return ResolvedCommand(
            path=" ".join(walked),
            callback_ref=f"{callback.__module__}:{callback.__name__}",
            flags=option_flags_for_callback(callback),
        )

    raise ArgvResolutionError(
        f"'npa {' '.join(walked)}' resolved to a command group, not a command; the "
        "toolRef argv is missing a subcommand"
    )


def argv_template_flags(argv: Sequence[str]) -> tuple[str, ...]:
    """Return the long options a catalog argv template passes, in order."""

    return tuple(str(token) for token in argv if str(token).startswith("--"))


def argv_flag_drift(tool_ref: str, argv: Sequence[str]) -> tuple[str, ...]:
    """Return the argv flags that the target CLI command does not accept.

    An empty tuple means the template can actually run. Raises
    :class:`ArgvResolutionError` when the argv does not name a CLI command at all
    (which is a harder failure than flag drift).
    """

    del tool_ref  # kept for a readable call site / future per-tool exemptions
    command = resolve_argv_command(argv)
    return tuple(flag for flag in argv_template_flags(argv) if flag not in command.flags)


#: Words that are output *formats*, never file paths.
FORMAT_WORDS = frozenset({"json", "text", "yaml", "table"})

#: Flags whose value is a destination path/URI, not a format.
PATH_LIKE_FLAG_HINTS = ("output", "path", "uri", "dir", "file")


def _cli_parameters(callback_ref: str) -> dict[str, inspect.Parameter]:
    module_name, _, callback_name = callback_ref.partition(":")
    callback = import_callback(module_name, callback_name)
    return dict(inspect.signature(callback).parameters)


def _resolved_annotations(callback_ref: str) -> dict[str, Any]:
    """Resolve string annotations (``from __future__ import annotations``) to types.

    Without this, an option annotated ``OutputFormat`` is only ever the *string*
    ``"OutputFormat"``, so its members cannot be checked.
    """

    module_name, _, callback_name = callback_ref.partition(":")
    callback = import_callback(module_name, callback_name)
    try:
        from typing import get_type_hints

        return dict(get_type_hints(callback))
    except Exception:  # noqa: BLE001 - unresolvable hints fall back to the raw ones
        return {
            name: param.annotation
            for name, param in inspect.signature(callback).parameters.items()
        }


def _parameter_for_flag(
    parameters: dict[str, inspect.Parameter], flag: str
) -> inspect.Parameter | None:
    for param in parameters.values():
        for decl in getattr(param.default, "param_decls", ()) or ():
            for part in str(decl).split("/"):
                if part == flag:
                    return param
    return None


def _enum_class(annotation: Any) -> Any | None:
    """Return the Enum class an annotation names, or ``None``."""

    import enum

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation
    return None


def _is_enum_annotation(param: inspect.Parameter, resolved: Any = None) -> bool:
    return _enum_class(resolved if resolved is not None else param.annotation) is not None


def argv_literal_value_mismatches(argv: Sequence[str]) -> tuple[str, ...]:
    """Report literal argv values that cannot be right for the option they follow.

    Catches the two directions of the same mistake:

    * a bare format word (``json``) handed to an option whose value is a **path**
      (``sonic eval --output`` is ``output_path: str``, so ``--output json`` silently
      wrote the eval result to a relative ``json/`` directory inside the pod and the
      spec's declared ``eval.json`` artifact never appeared — found live, runs
      ``npa-wf-gpu-sonic-eval-*``);
    * a value that is not a member of an option's **Enum** type.

    Only literal tokens are checked; ``{{...}}`` templates are resolved per run.
    """

    tokens = [str(token) for token in argv]
    command = resolve_argv_command(tokens)
    parameters = _cli_parameters(command.callback_ref)
    hints = _resolved_annotations(command.callback_ref)
    problems: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if not token.startswith("--"):
            continue
        value = tokens[index + 1]
        if value.startswith("--") or "{{" in value:
            continue
        param = _parameter_for_flag(parameters, token)
        if param is None:
            continue
        enum_class = _enum_class(hints.get(param.name, param.annotation))
        if enum_class is not None:
            allowed = {str(member.value) for member in enum_class}
            if value not in allowed:
                problems.append(f"{token} {value!r} is not one of {sorted(allowed)}")
            continue
        if value.lower() in FORMAT_WORDS and any(
            hint in token for hint in PATH_LIKE_FLAG_HINTS
        ):
            problems.append(
                f"{token} takes a path/URI ({param.name}), but the argv passes the "
                f"format word {value!r}; the declared output artifact will never be "
                "written where the spec says"
            )
    return tuple(problems)


def embedded_npa_commands(argv: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Extract each ``npa …`` invocation from a ``bash -c`` toolRef script.

    Several toolRefs wrap a loop or a sequence of CLI calls in ``["bash", "-c", "…"]``,
    which slipped past the flag audit entirely — the check bailed out whenever
    ``argv_template[0] != "npa"``. That hole shipped a real defect: the BDD100K
    ``create_failure_views`` toolRef passed ``--table`` to ``lancedb create-mv``, whose option
    is ``--source-table``, so the stage could only fail with "No such option '--table'".
    Found by driving the pipeline's real argv against mock endpoints; it should have been
    caught offline.

    Each returned tuple is one ``npa …`` command with `{{…}}` placeholders left intact, ready
    for :func:`argv_flag_drift`. Shell operators, loop keywords and `$var` words are dropped;
    a command whose flags cannot be split reliably (quoted values containing `;`) still yields
    its flag names, which is what the audit needs.
    """

    if len(argv) < 3 or str(argv[0]) != "bash":
        return ()
    script = str(argv[-1])
    commands: list[tuple[str, ...]] = []
    # Statements are separated by `;`, `&&` or newlines; `npa` may also appear after `do`.
    for statement in re.split(r";|&&|\n", script):
        words = shlex.split(statement, comments=False, posix=True) if statement.strip() else []
        if "npa" not in words:
            continue
        command = words[words.index("npa") :]
        # A trailing shell keyword (`done`, `fi`) is not part of the command.
        while command and command[-1] in {"done", "fi", "esac"}:
            command = command[:-1]
        if len(command) > 1:
            commands.append(tuple(command))
    return tuple(commands)


def catalog_argv_literal_mismatches() -> dict[str, tuple[str, ...]]:
    """Map every non-stub ``npa ...`` toolRef to its literal-value problems."""

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    out: dict[str, tuple[str, ...]] = {}
    for tool_ref, entry in TOOL_CATALOG.items():
        if entry.stub:
            continue
        if not entry.argv_template or str(entry.argv_template[0]) != "npa":
            continue
        try:
            problems = argv_literal_value_mismatches(entry.argv_template)
        except ArgvResolutionError:
            continue  # reported by catalog_argv_drift()
        if problems:
            out[tool_ref] = problems
    return out


def catalog_argv_drift() -> dict[str, tuple[str, ...]]:
    """Map every non-stub catalog toolRef to its unaccepted flags.

    Stub entries are excluded: a ``stub=True`` tool is documented as not yet
    executing real work, so holding its argv to a live CLI signature would pin
    placeholder shapes.
    """

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    drift: dict[str, tuple[str, ...]] = {}
    for tool_ref, entry in TOOL_CATALOG.items():
        if entry.stub:
            continue
        if not entry.argv_template:
            continue
        if str(entry.argv_template[0]) != "npa":
            # A `bash -c` wrapper still contains real `npa …` calls; audit each of them.
            # Anything else (a bare interpreter call) is out of scope.
            for command in embedded_npa_commands(entry.argv_template):
                try:
                    unaccepted = argv_flag_drift(tool_ref, command)
                except ArgvResolutionError as exc:
                    drift[tool_ref] = drift.get(tool_ref, ()) + (f"<unresolvable: {exc}>",)
                    continue
                if unaccepted:
                    drift[tool_ref] = drift.get(tool_ref, ()) + unaccepted
            continue
        try:
            unaccepted = argv_flag_drift(tool_ref, entry.argv_template)
        except ArgvResolutionError as exc:
            drift[tool_ref] = (f"<unresolvable: {exc}>",)
            continue
        if unaccepted:
            drift[tool_ref] = unaccepted
    return drift


def import_callback(module_name: str, callback_name: str) -> Any:
    """Import a CLI callback by module and attribute name."""

    return getattr(import_module(module_name), callback_name)


__all__ = [
    "ArgvResolutionError",
    "FORMAT_WORDS",
    "PATH_LIKE_FLAG_HINTS",
    "ResolvedCommand",
    "argv_flag_drift",
    "argv_literal_value_mismatches",
    "catalog_argv_literal_mismatches",
    "argv_template_flags",
    "catalog_argv_drift",
    "embedded_npa_commands",
    "import_callback",
    "option_flags_for_callback",
    "resolve_argv_command",
]
