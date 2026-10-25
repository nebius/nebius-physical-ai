"""Preserve RoboCasa's runtime contract while excluding superseded parent layers."""

import copy
import hashlib
import json
import re
import shlex
from ipaddress import IPv4Address
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/robocasa"
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)")
SOURCE_SHA = "0123456789abcdef0123456789abcdef01234567"


def _instructions(text: str) -> list[tuple[str, str]]:
    result = []
    pending = ""
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pending += line.rstrip().removesuffix("\\") + " "
        if line.rstrip().endswith("\\"):
            continue
        instruction, _, value = pending.strip().partition(" ")
        result.append((instruction, value))
        pending = ""
    assert not pending
    return result


def _stages(text: str) -> dict[str, list[tuple[str, str]]]:
    stages = {}
    current = None
    for instruction, value in _instructions(text):
        if instruction == "FROM":
            current = value.split()[-1]
            assert current not in stages
            stages[current] = [(instruction, value)]
        else:
            assert current is not None
            stages[current].append((instruction, value))
    return stages


def _expand(value: str, variables: dict[str, str]) -> str:
    return VARIABLE.sub(lambda match: variables[match[1] or match[2]], value)


def _apply_metadata(config, arguments, instruction, value):
    variables = {**arguments, **config["Env"]}
    if instruction == "ARG":
        key, separator, default = value.partition("=")
        if key != "NPA_SOURCE_SHA":
            assert separator
            arguments[key] = default
    elif instruction in {"ENV", "LABEL"}:
        field = "Env" if instruction == "ENV" else "Labels"
        updates = {}
        for token in shlex.split(value):
            key, separator, raw = token.partition("=")
            assert separator and key not in updates
            updates[key] = _expand(raw, variables)
        config[field].update(updates)
    elif instruction in {"USER", "WORKDIR"}:
        config[instruction] = _expand(value, variables)
    elif instruction == "EXPOSE":
        config["EXPOSE"] = value
    elif instruction in {"CMD", "ENTRYPOINT"}:
        config[instruction] = json.loads(value)
    elif instruction == "HEALTHCHECK":
        config[instruction] = value


def _metadata(instructions, base=None):
    config = {"Env": {}, "Labels": {}}
    if base:
        config["Env"] = dict(item.split("=", 1) for item in base["Env"])
        config["Labels"] = copy.deepcopy(base["Labels"])
        config["CMD"] = base["Cmd"]
    arguments = {"NPA_SOURCE_SHA": SOURCE_SHA}
    for instruction, value in instructions:
        _apply_metadata(config, arguments, instruction, value)
    return config


def _assert_clean_root_contract(text: str) -> None:
    stages = _stages(text)
    assert list(stages) == ["secure-pip-builder", "runtime-builder", "runtime"]
    final = stages["runtime"]
    assert final[0] == ("FROM", "scratch AS runtime")
    assert [item for item in final if item[0] in {"COPY", "ADD", "RUN"}] == [
        ("COPY", "--from=runtime-builder / /")
    ]
    base = json.loads((IMAGE / "cuda-base-runtime.json").read_text())["config"]
    expected = _metadata(stages["runtime-builder"], base)
    actual = _metadata(final)
    assert actual == expected
    assert actual["USER"] == "ubuntu"
    assert actual["WORKDIR"] == "/opt/robocasa"
    assert actual["EXPOSE"] == "8791"
    assert actual["Env"]["NPA_IMAGE_SOURCE_SHA"] == SOURCE_SHA
    assert actual["Labels"]["org.opencontainers.image.revision"] == SOURCE_SHA
    assert actual["CMD"][:3] == [
        "uvicorn",
        "npa.workbench.robocasa.service:app",
        "--host",
    ]
    # This reads metadata, not a socket. Container ingress requires the IPv4
    # unspecified address; loopback, hostnames and IPv6 are not equivalent.
    assert IPv4Address(actual["CMD"][3]).is_unspecified
    assert actual["CMD"][4:] == ["--port", "8791"]
    assert (
        "--interval=30s --timeout=5s --start-period=15s --retries=3"
        in actual["HEALTHCHECK"]
    )


def test_clean_final_root_preserves_exact_runtime_configuration():
    _assert_clean_root_contract((IMAGE / "Dockerfile").read_text())


@pytest.mark.parametrize("host", ["127.0.0.1", "::", "localhost"])
def test_clean_root_contract_rejects_non_ingress_host_in_both_stages(host):
    text = (IMAGE / "Dockerfile").read_text()
    changed, count = re.subn(r'("--host",\s*)"[^"]+"', rf'\1"{host}"', text)
    assert count == 2
    with pytest.raises((AssertionError, ValueError)):
        _assert_clean_root_contract(changed)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("FROM scratch AS runtime", "FROM runtime-builder AS runtime"),
        ("COPY --from=runtime-builder / /", "COPY --from=secure-pip-builder / /"),
        ("COPY --from=runtime-builder / /", "COPY --from=runtime-builder /opt /opt"),
        (
            "COPY --from=runtime-builder / /",
            "COPY --from=runtime-builder / /\nRUN true",
        ),
        ("USER ubuntu", "USER root"),
        ("EXPOSE 8791", "EXPOSE 8792"),
        ("WORKDIR /opt/robocasa", "WORKDIR /app"),
        ('NVIDIA_VISIBLE_DEVICES="all"', 'NVIDIA_VISIBLE_DEVICES="none"'),
        (
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        ),
        ('NV_CUDA_CUDART_VERSION="12.9.79-1"', 'NV_CUDA_CUDART_VERSION="12.9.78-1"'),
        ('LD_LIBRARY_PATH="/usr/local/cuda/lib64"', 'LD_LIBRARY_PATH="/wrong"'),
        ("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA=1", "ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA=0"),
        ("--start-period=15s", "--start-period=14s"),
        ('"--port", "8791"', '"--port", "8792"'),
    ],
    ids=[
        "parent-layer",
        "wrong-stage",
        "partial-root",
        "extra-layer",
        "root-user",
        "port",
        "workdir",
        "gpu-visibility",
        "graphics",
        "cudart",
        "linker",
        "source-gate",
        "healthcheck",
        "service-command",
    ],
)
def test_clean_root_contract_rejects_runtime_or_layer_regressions(old, new):
    builder, final = (IMAGE / "Dockerfile").read_text().rsplit("\nFROM ", 1)
    final = "FROM " + final
    assert old in final
    with pytest.raises(AssertionError):
        _assert_clean_root_contract(builder + "\n" + final.replace(old, new, 1))


def test_cuda_contract_binds_the_exact_public_base_identity():
    contract = json.loads((IMAGE / "cuda-base-runtime.json").read_text())
    assert contract["image"] == (
        "nvidia/cuda:12.9.1-base-ubuntu22.04@sha256:"
        "59436e8ac61921052d8f420be8b8cb8b117f1d7cc643397a289e37aaa0b83ea1"
    )
    assert contract["config_digest"] == (
        "sha256:2e1e354f0a7e32186e843c1cacda3fd1932f3c04da790c5fd50dc6af8bd7a1fb"
    )
    assert set(contract["config"]) == {"Env", "Cmd", "Labels", "OnBuild"}
    assert contract["config"]["OnBuild"] is None
    encoded = json.dumps(
        contract["config"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "6c173355c274fb2aba23858bd17fb71db9b6c759063712506b8202893bf2d4b0"
    )
