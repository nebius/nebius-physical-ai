# DM05 LeRobot validation runtime attribution

This operator-private validation image derives its DM05 policy implementation
from [`hbzfeng/lerobot`](https://github.com/hbzfeng/lerobot) at
`6eede4f7d2efe6b4f6a58ddb7b13ed55e2346b9c`, the source supplied with
Hugging Face LeRobot pull request [#4051](https://github.com/huggingface/lerobot/pull/4051).
That pull request was closed/superseded rather than merged. The source tree,
including its `LICENSE`, is retained at `/opt/lerobot/dm05-source`.

- Copyright 2024 The Hugging Face team. All rights reserved. Individual DM05
  files also credit Dexmal and Hugging Face contributors.
- Source implementation license: Apache License 2.0. This image retains the
  exact upstream license; no separate NPA license or acceptance mechanism is
  introduced.
- The runtime fetches `Dexmal/DM05-Lerobot` and
  `Dexmal/DM05-Lerobot-LIBERO` only when the operator runs a workflow. No
  checkpoint, processor cache, Gemma payload, LIBERO data, or model-derived
  output is baked into the image. The model cards label the weights `gemma`;
  their terms govern those separate weights and any redistribution/service use.
- LIBERO benchmark source and simulator assets are
  `Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01`,
  MIT (Copyright 2023 Lifelong Robot Learning), retained with its `LICENSE` at
  `/opt/lerobot/libero-benchmark`. Its dataset notice identifies demonstration
  data as CC BY 4.0; demonstrations are runtime-provided and not included here.

Credits and citations are recorded in
`docs/workbench/dm05-lerobot-libero.md`. This image is an operator-private
qualification artifact only and is not an official NPA public image release.
