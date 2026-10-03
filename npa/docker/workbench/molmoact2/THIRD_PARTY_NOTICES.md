# MolmoAct2 official-policy BYOF image notices

This image includes source code from [AllenAI MolmoAct2](https://github.com/allenai/molmoact2),
revision `6070080a20321b4f498ab30f28e1d09ac465edb7`, licensed under Apache-2.0.
Copyright and attribution remain with the upstream authors. See the upstream
[`LICENSE`](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/LICENSE),
[`NOTICE`](https://github.com/allenai/molmoact2/tree/6070080a20321b4f498ab30f28e1d09ac465edb7),
and citation material in its README.

It also includes the LeRobot-derived `experiments/lerobot` tree committed in
that exact MolmoAct2 source revision, licensed Apache-2.0 and attributed to the
Hugging Face LeRobot contributors. The upstream root also records a separate
`lerobot/` gitlink at `80633827176a0203064cb141383664fba024e050`; the runnable
MolmoAct2 trainer imports the committed `experiments/lerobot` tree instead, so
the two are not represented as interchangeable runtime revisions.

The official LIBERO benchmark source is [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
licensed MIT.  LIBERO simulation is installed via the upstream LeRobot extra,
not copied from this repository.

No AllenAI checkpoint, adapter, Hugging Face dataset, cache, robot capture, or
operator credential is present in this image. Those artifacts are fetched at
runtime under the invoking operator's existing provider access and preserve
their own upstream provenance and terms.
