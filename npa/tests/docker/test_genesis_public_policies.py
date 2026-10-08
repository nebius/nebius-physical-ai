"""Keep ACT source branches intact and reject unknown public policy source."""

import ast
import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/genesis/prepare_public_policies.py"
)
PACKAGE = """# retained Apache notice
from .act.configuration_act import ACTConfig as ACTConfig
from .groot.configuration_groot import GrootConfig as GrootConfig
__all__ = ['ACTConfig', 'GrootConfig']
"""
FACTORY = '''# retained Apache notice
from __future__ import annotations
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
def get_policy_class(name):
    """native class selection"""
    if name == 'groot':
        return GrootPolicy
    elif name == 'act':
        return native_act_policy
    else:
        raise ValueError(name)
def make_policy_config(policy_type, **kwargs):
    """native config selection"""
    if policy_type == 'groot':
        return GrootConfig(**kwargs)
    elif policy_type == 'act':
        return ACTConfig(**kwargs)
    else:
        raise ValueError(policy_type)
def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
    """native processor factory"""
    if pretrained_path:
        if isinstance(policy_cfg, GrootConfig):
            kwargs['excluded'] = True
        return native_checkpoint_loader(pretrained_path, **kwargs)
    if isinstance(policy_cfg, GrootConfig):
        processors = excluded_processors(policy_cfg)
    elif isinstance(policy_cfg, ACTConfig):
        processors = native_act_processors(policy_cfg, **kwargs)
    else:
        raise ValueError(policy_cfg)
    return processors
def make_policy(cfg):
    """native model factory"""
    return native_policy_factory(cfg)
'''


@pytest.fixture
def source_module():
    spec = importlib.util.spec_from_file_location("public_policy_source", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_tree(tmp_path, module, monkeypatch):
    files = {
        "lerobot/policies/__init__.py": PACKAGE,
        "lerobot/policies/factory.py": FACTORY,
    }
    rows = []
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest())
            .decode()
            .rstrip("=")
        )
        rows.append([name, "sha256=" + digest, str(len(text.encode()))])
    rows.append(["lerobot-0.4.4.dist-info/LICENSE", "sha256=retained", "23"])
    record = tmp_path / "lerobot-0.4.4.dist-info/RECORD"
    record.parent.mkdir()
    with record.open("w") as stream:
        csv.writer(stream).writerows(rows)
    monkeypatch.setattr(
        module,
        "SOURCE_HASHES",
        {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in files.items()
        },
    )


def test_native_act_bodies_and_checkpoint_loader_are_retained(source_module):
    tree = ast.parse(
        source_module._replacement("lerobot/policies/factory.py", FACTORY.encode())
    )
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert all(
        node.module != "lerobot.policies.groot.configuration_groot" for node in imports
    )
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    originals = {
        node.name: node
        for node in ast.parse(FACTORY).body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("get_policy_class", "make_policy_config"):
        original = source_module._act_branch(originals[name].body[1])
        updated = functions[name].body[1]
        assert ast.dump(original.test) == ast.dump(updated.test)
        assert ast.dump(ast.Module(body=original.body, type_ignores=[])) == ast.dump(
            ast.Module(body=updated.body, type_ignores=[])
        )
        assert ast.unparse(updated.orelse[0]).startswith("raise NotImplementedError")
    processor = functions["make_pre_post_processors"]
    assert (
        ast.unparse(processor.body[1].test) == "not isinstance(policy_cfg, ACTConfig)"
    )
    assert ast.dump(processor.body[2].body[-1]) == ast.dump(
        originals["make_pre_post_processors"].body[1].body[-1]
    )
    assert (
        ast.unparse(functions["make_policy"].body[1].test)
        == "cfg.type != 'act' or getattr(cfg, 'use_peft', False)"
    )
    assert ast.dump(functions["make_policy"].body[-1]) == ast.dump(
        originals["make_policy"].body[-1]
    )


def test_policy_source_record_and_notice_rows_are_preserved(
    source_module, tmp_path, monkeypatch
):
    _source_tree(tmp_path, source_module, monkeypatch)
    receipt = source_module.prepare(tmp_path, check_only=False)
    assert len(receipt) == 2 and all(row["record_updated"] for row in receipt)
    for name in source_module.SOURCE_HASHES:
        assert (tmp_path / name).read_text().startswith("# retained Apache notice")
    rows = list(
        csv.reader(
            io.StringIO((tmp_path / "lerobot-0.4.4.dist-info/RECORD").read_text())
        )
    )
    notice = [row for row in rows if row[0].endswith("LICENSE")]
    assert notice == [["lerobot-0.4.4.dist-info/LICENSE", "sha256=retained", "23"]]
    assert len([row for row in rows if row[0].endswith(".pyc")]) == 2
    for row in rows:
        if row[0].endswith("LICENSE"):
            continue
        content = (tmp_path / row[0]).read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .decode()
            .rstrip("=")
        )
        assert row[1:] == ["sha256=" + digest, str(len(content))]


def test_unknown_second_source_fails_before_first_source_mutation(
    source_module, tmp_path, monkeypatch
):
    _source_tree(tmp_path, source_module, monkeypatch)
    first = tmp_path / "lerobot/policies/__init__.py"
    original = first.read_bytes()
    second = tmp_path / "lerobot/policies/factory.py"
    second.write_text(second.read_text() + "# unknown source\n")
    with pytest.raises(RuntimeError, match="unreviewed policy source"):
        source_module.prepare(tmp_path, check_only=False)
    assert first.read_bytes() == original


def test_duplicate_bytecode_record_fails_before_source_mutation(
    source_module, tmp_path, monkeypatch
):
    _source_tree(tmp_path, source_module, monkeypatch)
    source = tmp_path / "lerobot/policies/factory.py"
    original = source.read_bytes()
    bytecode = Path(source_module.cache_from_source(str(source))).relative_to(tmp_path)
    record = tmp_path / "lerobot-0.4.4.dist-info/RECORD"
    with record.open("a") as stream:
        stream.write(f"{bytecode.as_posix()},,\n" * 2)
    with pytest.raises(RuntimeError, match="duplicate policy bytecode RECORD"):
        source_module.prepare(tmp_path, check_only=False)
    assert source.read_bytes() == original
