"""Versioned job descriptor schema (manifest/v0.1).

A descriptor is the single source of truth for a tool: what container to run,
how to invoke it, what it produces, and how to tell it succeeded. The runtime
validates every request against the descriptor; nothing tool-specific lives
in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_API_VERSION = "manifest/v0.1"


class DescriptorError(ValueError):
    """Raised when a descriptor is invalid."""


@dataclass(frozen=True)
class ImageRef:
    repository: str
    tag: str
    digest: str  # sha256:... — immutable; the only thing the runtime pulls

    def pinned(self) -> str:
        return f"{self.repository}@{self.digest}"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # integer | number | string | boolean
    default: Any = None
    required: bool = False
    min_value: float | None = None
    max_value: float | None = None

    def validate(self, value: Any) -> Any:
        if self.type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise DescriptorError(f"param {self.name!r}: expected integer")
        elif self.type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DescriptorError(f"param {self.name!r}: expected number")
        elif self.type == "string":
            if not isinstance(value, str):
                raise DescriptorError(f"param {self.name!r}: expected string")
        elif self.type == "boolean":
            if not isinstance(value, bool):
                raise DescriptorError(f"param {self.name!r}: expected boolean")
        else:
            raise DescriptorError(f"param {self.name!r}: unknown type {self.type!r}")
        if self.min_value is not None and value < self.min_value:
            raise DescriptorError(f"param {self.name!r}: below minimum")
        if self.max_value is not None and value > self.max_value:
            raise DescriptorError(f"param {self.name!r}: above maximum")
        return value


@dataclass(frozen=True)
class OutputSpec:
    name: str
    path: str  # absolute path inside the container (ignored when source=stdout)
    format: str  # json | text | binary
    required: bool = True
    source: str = "file"  # file | stdout


@dataclass(frozen=True)
class S3Input:
    name: str
    s3_uri: str  # s3://bucket/key
    container_path: str  # absolute path inside the container


@dataclass(frozen=True)
class ArtifactStore:
    type: str  # s3
    bucket: str
    prefix: str
    endpoint_url: str = ""


@dataclass(frozen=True)
class Environment:
    pip: tuple[str, ...] = ()  # pip packages installed in-pod before argv
    apt: tuple[str, ...] = ()  # apt packages installed in-pod before pip


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: list[str]  # may contain {{param}} placeholders; never a shell string
    params: dict[str, ParamSpec] = field(default_factory=dict)
    outputs: dict[str, OutputSpec] = field(default_factory=dict)

    def render_argv(self, inputs: dict[str, Any]) -> list[str]:
        """Validate inputs and render the argv list. No shell involved."""
        rendered: list[str] = []
        for token in self.argv:
            rendered.append(_substitute(token, inputs, self.params, self.name))
        # Any supplied param not consumed by a placeholder is appended as
        # --kebab-case value (booleans render as bare --flag when true).
        # Params left at their None default are omitted: rendering the
        # literal string "None" would silently corrupt the command line.
        consumed = {p for token in self.argv for p in _placeholders(token)}
        for pname, value in inputs.items():
            if pname in consumed or pname not in self.params:
                continue
            if value is None:
                continue
            spec = self.params[pname]
            flag = "--" + pname.replace("_", "-")
            if spec.type == "boolean":
                if value:
                    rendered.append(flag)
            else:
                rendered.extend([flag, str(value)])
        return rendered

    def resolve_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        unknown = set(inputs) - set(self.params)
        if unknown:
            raise DescriptorError(
                f"command {self.name!r}: unknown params {sorted(unknown)}"
            )
        resolved: dict[str, Any] = {}
        for pname, spec in self.params.items():
            if pname in inputs:
                resolved[pname] = spec.validate(inputs[pname])
            elif spec.required:
                raise DescriptorError(f"command {self.name!r}: missing {pname!r}")
            else:
                resolved[pname] = spec.default
        return resolved


@dataclass(frozen=True)
class Resources:
    gpu: int = 0
    memory_gb: int = 0


@dataclass(frozen=True)
class SuccessCheck:
    artifact: str
    json_path: str  # dotted path into the artifact JSON
    equals: Any = None
    greater_than: float | None = None

    def evaluate(self, artifacts: dict[str, Any]) -> tuple[bool, str]:
        if self.artifact not in artifacts:
            return False, f"artifact {self.artifact!r} missing"
        node: Any = artifacts[self.artifact]
        for part in self.json_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return False, f"{self.artifact}.{self.json_path}: path missing"
            node = node[part]
        if self.equals is not None and node != self.equals:
            return (
                False,
                f"{self.artifact}.{self.json_path}={node!r} != {self.equals!r}",
            )
        if self.greater_than is not None and not (
            isinstance(node, (int, float)) and node > self.greater_than
        ):
            return False, f"{self.artifact}.{self.json_path} not > {self.greater_than}"
        return True, "ok"


@dataclass(frozen=True)
class Descriptor:
    api_version: str
    name: str
    version: str
    description: str
    image: ImageRef
    commands: dict[str, CommandSpec]
    resources: Resources
    success_checks: tuple[SuccessCheck, ...] = ()
    payload_files: tuple[tuple[str, str], ...] = ()  # (container_path, host_path)
    inputs: tuple[S3Input, ...] = ()
    artifact_store: ArtifactStore | None = None
    environment: Environment = Environment()

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def _placeholders(token: str) -> set[str]:
    out: set[str] = set()
    i = 0
    while True:
        s = token.find("{{", i)
        if s < 0:
            return out
        e = token.find("}}", s)
        if e < 0:
            return out
        out.add(token[s + 2 : e].strip())
        i = e + 2


def _substitute(
    token: str, inputs: dict[str, Any], params: dict[str, ParamSpec], cmd: str
) -> str:
    for pname in _placeholders(token):
        if pname not in params:
            raise DescriptorError(f"command {cmd!r}: placeholder {pname!r} not a param")
        value = inputs[pname]
        if value is None:
            raise DescriptorError(
                f"command {cmd!r}: placeholder {pname!r} has no value"
            )
        text = str(value).lower() if isinstance(value, bool) else str(value)
        token = token.replace("{{" + pname + "}}", text)
    return token


def load_descriptor(path: str | Path) -> Descriptor:
    raw = yaml.safe_load(Path(path).read_text())
    return parse_descriptor(raw, source=str(path))


def parse_descriptor(raw: dict[str, Any], source: str = "<dict>") -> Descriptor:
    if raw.get("apiVersion") != SUPPORTED_API_VERSION:
        raise DescriptorError(
            f"{source}: unsupported apiVersion (want {SUPPORTED_API_VERSION})"
        )
    for key in ("name", "version", "image", "commands"):
        if key not in raw:
            raise DescriptorError(f"{source}: missing {key!r}")
    img = raw["image"]
    if not img.get("digest", "").startswith("sha256:"):
        raise DescriptorError(f"{source}: image.digest must be a sha256: pin")
    commands: dict[str, CommandSpec] = {}
    for cname, cdef in raw["commands"].items():
        params = {}
        for pname, pdef in (cdef.get("params") or {}).items():
            ptype = pdef.get("type", "string")
            if ptype not in ("integer", "number", "string", "boolean"):
                raise DescriptorError(
                    f"{source}: command {cname!r} param {pname!r}: "
                    f"unknown type {ptype!r}"
                )
            params[pname] = ParamSpec(
                name=pname,
                type=ptype,
                default=pdef.get("default"),
                required=bool(pdef.get("required", False)),
                min_value=pdef.get("min"),
                max_value=pdef.get("max"),
            )
        outputs = {}
        for oname, odef in (cdef.get("outputs") or {}).items():
            source = odef.get("source", "file")
            path = odef.get("path", "")
            if source == "file" and not path:
                raise DescriptorError(
                    f"{source}: command {cname!r} output {oname!r}: "
                    "file-source outputs require a path"
                )
            if source not in ("file", "stdout"):
                raise DescriptorError(
                    f"{source}: command {cname!r} output {oname!r}: "
                    f"unknown source {source!r}"
                )
            fmt = odef.get("format", "json")
            if fmt not in ("json", "text", "binary"):
                raise DescriptorError(
                    f"{source}: command {cname!r} output {oname!r}: "
                    f"unknown format {fmt!r}"
                )
            outputs[oname] = OutputSpec(
                name=oname,
                path=path,
                format=fmt,
                required=bool(odef.get("required", True)),
                source=source,
            )
        commands[cname] = CommandSpec(
            name=cname, argv=list(cdef["argv"]), params=params, outputs=outputs
        )
    # Success checks must name artifacts some command actually declares;
    # otherwise they can only fail inside the pod.
    declared_outputs = {oname for cmd in commands.values() for oname in cmd.outputs}
    res = raw.get("resources") or {}
    checks = tuple(
        SuccessCheck(
            artifact=c["artifact"],
            json_path=c["json_path"],
            equals=c.get("equals"),
            greater_than=c.get("greater_than"),
        )
        for c in (raw.get("success", {}).get("checks") or [])
    )
    for c in checks:
        if c.artifact not in declared_outputs:
            raise DescriptorError(
                f"{source}: success check names undeclared artifact {c.artifact!r}"
            )
    payload = tuple(
        (p["container_path"], p["host_path"]) for p in (raw.get("payload_files") or [])
    )
    for cpath, hpath in payload:
        if not cpath.startswith("/"):
            raise DescriptorError(
                f"{source}: payload container_path must be absolute: {cpath!r}"
            )
        if Path(hpath).is_absolute() or ".." in Path(hpath).parts:
            raise DescriptorError(
                f"{source}: payload host_path must be relative without '..': {hpath!r}"
            )
    inputs = tuple(
        S3Input(name=i["name"], s3_uri=i["s3_uri"], container_path=i["container_path"])
        for i in (raw.get("inputs") or [])
    )
    for i in inputs:
        if not i.s3_uri.startswith("s3://"):
            raise DescriptorError(
                f"{source}: input {i.name!r}: s3_uri must start with s3://"
            )
        if not i.container_path.startswith("/"):
            raise DescriptorError(
                f"{source}: input {i.name!r}: container_path must be absolute"
            )
    store = None
    if raw.get("artifact_store"):
        s = raw["artifact_store"]
        if s.get("type") != "s3":
            raise DescriptorError(
                f"{source}: artifact_store.type must be 's3', got {s.get('type')!r}"
            )
        if not s.get("bucket"):
            raise DescriptorError(f"{source}: artifact_store.bucket is required")
        store = ArtifactStore(
            type="s3",
            bucket=s["bucket"],
            prefix=s.get("prefix", ""),
            endpoint_url=s.get("endpoint_url", ""),
        )
    env_raw = raw.get("environment") or {}
    pip_pkgs = env_raw.get("pip") or []
    apt_pkgs = env_raw.get("apt") or []
    for field_name, pkgs in (("pip", pip_pkgs), ("apt", apt_pkgs)):
        if not isinstance(pkgs, list) or not all(isinstance(p, str) for p in pkgs):
            raise DescriptorError(
                f"{source}: environment.{field_name} must be a list of strings"
            )
    return Descriptor(
        api_version=raw["apiVersion"],
        name=raw["name"],
        version=str(raw["version"]),
        description=raw.get("description", ""),
        image=ImageRef(
            repository=img["repository"], tag=img.get("tag", ""), digest=img["digest"]
        ),
        commands=commands,
        resources=Resources(
            gpu=int(res.get("gpu", 0)), memory_gb=int(res.get("memory_gb", 0))
        ),
        success_checks=checks,
        payload_files=payload,
        inputs=inputs,
        artifact_store=store,
        environment=Environment(pip=tuple(pip_pkgs), apt=tuple(apt_pkgs)),
    )
