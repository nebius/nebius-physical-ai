# LeRobot VLA-JEPA candidate notices

This image is an operator-private validation candidate. It is not a generic
LeRobot release and is not published to a public registry by this change.

## Baked software

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot), exact source
  `7e241bd630a3719a56157a497ce5d08f244784f1` (`v0.6.1`), copyright 2024 and
  2026 Hugging Face Inc. team, [Apache-2.0](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/LICENSE).
  The image changes only packaging: it installs the documented
  `training,evaluation,libero,vla_jepa` extras and disables optional W&B and
  Hugging Face telemetry. The implementation remains upstream LeRobot.
- The inherited generic LeRobot 0.6.0 base is pinned by OCI digest in the
  Dockerfile. Its own notices and package terms remain applicable.
- NPA's Apache-2.0 workflow adapter is copied into the candidate at build time
  solely to run the five sealed stages. Its exact source commit is recorded in
  the `npa.adapter.source-revision` OCI label; the adapter changes packaging,
  runtime-fetch provenance, task-disjoint processing, artifact checksums, and
  Rerun reporting, not the upstream VLA-JEPA method or weights.

## Runtime-fetched artifacts (not image layers)

- [LeRobot VLA-JEPA Pretrain](https://huggingface.co/lerobot/VLA-JEPA-Pretrain),
  revision `e946c3e5b538d760f4b4ff239d1b1c12090c041d`, Apache-2.0, published by
  `lerobot`.
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
  revision `89644892e4d85e24eaac8bacfd4f463576704203`, Apache-2.0, published by
  Qwen.
- [V-JEPA2 ViT-L](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256),
  revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, MIT, published by
  Meta/Facebook.
- [HuggingFaceVLA LIBERO](https://huggingface.co/datasets/HuggingFaceVLA/libero),
  revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`, Apache-2.0, published by
  HuggingFaceVLA. This candidate uses it only as simulation demonstrations.

Runtime fetch under the operator's own access does not transfer, broaden, or
replace any upstream rights. Populated caches, datasets, checkpoints, rollout
videos, and credentials are excluded from image layers and public publication.

## Research attribution

The upstream model card attributes the method to **Jingwen Sun, Wenyao Zhang,
Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and
Zhibo Chen**, *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent
World Model*, arXiv:2602.10098 (2026), and to the original
[ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) repository at
`db32b4a4457c39921869f667ac76cf8a8fc39da8` (Apache-2.0).

The LeRobot model card supplies the LeRobot citation:

```bibtex
@misc{cadene2024lerobot,
  author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
  title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  year = {2024}
}
```

NPA contributes the workflow adapter, artifact provenance, and validation
plumbing only; it does not claim the upstream method, model, benchmark, or
research results as original NPA work.
