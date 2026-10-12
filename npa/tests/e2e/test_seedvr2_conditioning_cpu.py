"""Opt-in CPU checks of hash-pinned upstream methods; no weights or GPU calls.

Set NPA_SEEDVR2_UPSTREAM_SOURCE to a retained unmodified upstream source tree.
Synthetic posteriors exercise actual wrapper/selection/RNG control flow. This
does not establish real encoder tensors, CUDA reproducibility, or quality.
"""

from __future__ import annotations

import ast
from collections import namedtuple
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace

import pytest

PINS = {
    "projects/inference_seedvr2_3b.py": "089de47cd576bfd51b63b77b8f430146ae85bdd98bc2076011f869e54e2922ee",
    "projects/video_diffusion_sr/infer.py": "8a4034c343cb8bdc816dc6e5686f8843be2812d7cc77786160a12af272d885ae",
    "models/video_vae_v3/modules/attn_video_vae.py": "981faf238b040a9f62b29f4f10e3886ba7152ff24ece03713129f6f19ea44568",
}

METHOD_PINS = {
    "generation_loop": "9189e64c43aec5800355911b6b4281afa1a6bb23c7e01dbb38415d781e146692",
    "vae_encode": "36086d32007820cba6446c840e31383e88642a819f6a9677f573b7a2b0206be5",
    "Wrapper": "d3cb1105d07ef3d617b1b3098e39d72825222ab26f749dd478ec0b13445607b8",
}


@pytest.fixture
def upstream():
    directory = os.environ.get("NPA_SEEDVR2_UPSTREAM_SOURCE")
    if not directory:
        pytest.skip("requires retained exact upstream source; CPU-only opt-in")
    return _verified_trees(Path(directory))


def _verified_trees(directory):
    result = {}
    for name, digest in PINS.items():
        source = directory / name
        assert not source.is_symlink() and source.is_file(), "regular upstream file"
        assert directory.resolve() in source.resolve().parents, "upstream containment"
        payload = source.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        result[name] = ast.parse(payload)
    return result


def _function(tree, name):
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _execute(node, namespace):
    digest = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    assert METHOD_PINS.get(node.name) == digest, "unapproved upstream method"
    assert not any(name.startswith("__") for name in namespace), "reserved namespace"
    payload = (ast.unparse(node) + "\n").encode()
    with tempfile.TemporaryDirectory(prefix="seedvr-pinned-method-") as directory:
        root = Path(directory)
        path = root / "method.py"
        with path.open("xb") as output:
            output.write(payload)
        path.chmod(0o600)
        module = _import_pinned_module(root, path, payload, namespace)
        return getattr(module, node.name)


def _import_pinned_module(root, path, payload, namespace):
    assert not root.is_symlink() and root.stat().st_mode & 0o077 == 0, "private root"
    assert path.parent == root and path.name == "method.py", "module containment"
    mode = path.lstat().st_mode
    assert stat.S_ISREG(mode) and mode & 0o077 == 0, "private regular module"
    assert path.read_bytes() == payload, "module bytes changed"
    spec = importlib.util.spec_from_file_location("seedvr_pinned_method", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(namespace)
    spec.loader.exec_module(module)
    return module


class _Config(dict):
    __getattr__ = dict.__getitem__


def _encoder(upstream, torch, skip_sample=False):
    class Posterior:
        def __init__(self, value):
            self.mean = value * 0.25

        def sample(self):
            return self.mean if skip_sample else self.mean + torch.randn_like(self.mean)

        def mode(self):
            return self.mean

    class Base:
        def encode(self, value):
            return SimpleNamespace(latent_dist=Posterior(value))

    wrapper = _wrapper_class(upstream)
    namespace = {
        "Base": Base,
        "torch": torch,
        "CausalEncoderOutput": namedtuple("CausalEncoderOutput", "latent posterior"),
    }
    return _execute(wrapper, namespace)()


def _wrapper_class(upstream):
    tree = upstream["models/video_vae_v3/modules/attn_video_vae.py"]
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(child, ast.FunctionDef)
            and child.name == "encode"
            and "p.sample" in ast.unparse(child)
            for child in node.body
        )
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "encode"
    )
    wrapper = ast.ClassDef(
        name="Wrapper",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    return wrapper


def _selection(upstream, torch):
    from einops import rearrange

    method = _function(upstream["projects/video_diffusion_sr/infer.py"], "vae_encode")
    namespace = {
        "torch": torch,
        "List": list,
        "Tensor": torch.Tensor,
        "ListConfig": list,
        "get_device": lambda: "cpu",
        "rearrange": rearrange,
    }
    return _execute(method, namespace)


def _arm(upstream, use_sample, skip_sample=False):
    import torch

    encoder = _encoder(upstream, torch, skip_sample=skip_sample)
    config = _Config(
        use_sample=use_sample, dtype="bfloat16", grouping=False, scaling_factor=0.9152
    )
    runner = SimpleNamespace(config=SimpleNamespace(vae=config), vae=encoder)
    select = _selection(upstream, torch)
    sample = torch.arange(96, dtype=torch.float32).reshape(3, 2, 4, 4) / 96
    torch.manual_seed(666)
    before = torch.get_rng_state().clone()
    latents = select(runner, [sample])
    after = torch.get_rng_state().clone()
    noise = torch.randn_like(latents[0])
    augmentation = torch.randn_like(latents[0])
    return before, after, latents[0], noise, augmentation


def _assert_pair(sample, mode):
    import torch

    assert torch.equal(sample[0], mode[0]), "initial RNG mismatch"
    assert torch.equal(sample[1], mode[1]), "post-encode RNG mismatch"
    assert not torch.equal(sample[2], mode[2]), "conditioning did not change"
    assert sample[2].dtype == mode[2].dtype == torch.bfloat16
    assert sample[2].shape == mode[2].shape
    assert torch.equal(sample[3], mode[3]), "diffusion noise mismatch"
    assert torch.equal(sample[4], mode[4]), "augmentation noise mismatch"


def test_real_upstream_selection_preserves_rng_consumption(upstream):
    _assert_pair(_arm(upstream, True), _arm(upstream, False))


def test_removed_wrapper_sample_is_detected(upstream):
    with pytest.raises(AssertionError, match="post-encode RNG mismatch"):
        _assert_pair(_arm(upstream, True), _arm(upstream, False, skip_sample=True))


def test_ignored_conditioning_selection_is_detected(upstream):
    with pytest.raises(AssertionError, match="conditioning did not change"):
        _assert_pair(_arm(upstream, True), _arm(upstream, True))


def test_actual_entrypoint_overrides_yaml_before_diffusion(upstream):
    class Observed(Exception):
        pass

    def configured():
        assert config.diffusion.cfg.scale == 1.0
        assert config.diffusion.cfg.rescale == 0.0
        assert config.diffusion.timesteps.sampling.steps == 1
        raise Observed

    config = SimpleNamespace(
        diffusion=SimpleNamespace(
            cfg=SimpleNamespace(scale=7.5, rescale=9),
            timesteps=SimpleNamespace(sampling=SimpleNamespace(steps=50)),
        )
    )
    runner = SimpleNamespace(config=config, configure_diffusion=configured)
    entry = _function(upstream["projects/inference_seedvr2_3b.py"], "generation_loop")
    function = _execute(entry, {})
    with pytest.raises(Observed):
        function(runner)


def test_actual_noise_calls_remain_shape_only(upstream):
    tree = _function(upstream["projects/inference_seedvr2_3b.py"], "generation_step")
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for name in ("noises", "aug_noises"):
        assert (
            ast.unparse(assignments[name])
            == "[torch.randn_like(latent) for latent in cond_latents]"
        )
    assert ast.literal_eval(assignments["cond_noise_scale"]) == 0.0


@pytest.mark.parametrize("name", ["generation_loop", "vae_encode", "Wrapper", "other"])
def test_unapproved_method_is_rejected_before_import(name, monkeypatch):
    def forbidden(*args):
        pytest.fail("unapproved method reached module import")

    monkeypatch.setattr(importlib.util, "spec_from_file_location", forbidden)
    node = ast.parse(f"def {name}():\n    return 7\n").body[0]
    with pytest.raises(AssertionError, match="unapproved upstream method"):
        _execute(node, {})


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "outside", "public"])
def test_module_boundary_refuses_mutation(tmp_path, monkeypatch, mutation):
    def forbidden(*args):
        pytest.fail("hostile module reached import")

    monkeypatch.setattr(importlib.util, "spec_from_file_location", forbidden)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "method.py"
    payload = b"value = 1\n"
    path.write_bytes(payload)
    path.chmod(0o600)
    if mutation == "bytes":
        path.write_bytes(b"value = 2\n")
    elif mutation == "symlink":
        path.rename(root / "retained.py")
        path.symlink_to(root / "retained.py")
    elif mutation == "outside":
        path.rename(tmp_path / "method.py")
        path = tmp_path / "method.py"
    else:
        path.chmod(0o644)
    with pytest.raises(AssertionError):
        _import_pinned_module(root, path, payload, {})


@pytest.mark.parametrize("mutation", ["bytes", "symlink"])
def test_upstream_source_rejects_mutation(tmp_path, mutation):
    name = next(iter(PINS))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    if mutation == "bytes":
        path.write_bytes(b"# altered upstream source\n")
    else:
        retained = tmp_path / "retained.py"
        retained.write_bytes(b"# altered upstream source\n")
        path.symlink_to(retained)
    with pytest.raises(AssertionError):
        _verified_trees(tmp_path)
