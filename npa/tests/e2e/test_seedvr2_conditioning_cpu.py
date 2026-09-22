"""Opt-in CPU checks of hash-pinned upstream methods; no weights or GPU calls.

Set NPA_SEEDVR2_UPSTREAM_SOURCE to a retained unmodified upstream source tree.
Synthetic posteriors exercise actual wrapper/selection/RNG control flow. This
does not establish real encoder tensors, CUDA reproducibility, or quality.
"""

from __future__ import annotations

import ast
from collections import namedtuple
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

PINS = {
    "projects/inference_seedvr2_3b.py": "089de47cd576bfd51b63b77b8f430146ae85bdd98bc2076011f869e54e2922ee",
    "projects/video_diffusion_sr/infer.py": "8a4034c343cb8bdc816dc6e5686f8843be2812d7cc77786160a12af272d885ae",
    "models/video_vae_v3/modules/attn_video_vae.py": "981faf238b040a9f62b29f4f10e3886ba7152ff24ece03713129f6f19ea44568",
}


@pytest.fixture
def upstream():
    directory = os.environ.get("NPA_SEEDVR2_UPSTREAM_SOURCE")
    if not directory:
        pytest.skip("requires retained exact upstream source; CPU-only opt-in")
    result = {}
    for name, digest in PINS.items():
        payload = (Path(directory) / name).read_bytes()
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
    module = ast.Module(body=[node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<pinned-upstream-method>", "exec"),
        namespace,
    )
    return namespace[node.name]


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
