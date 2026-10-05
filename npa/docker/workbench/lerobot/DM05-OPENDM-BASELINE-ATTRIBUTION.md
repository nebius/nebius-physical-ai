# DM05 OpenDM LIBERO baseline runtime attribution

This operator-private validation image contains source-only runtime support for
the released [`Dexmal/DM05-libero`](https://huggingface.co/Dexmal/DM05-libero)
checkpoint. The checkpoint is Gemma-labeled and is fetched only at operator
runtime; it is not copied into this image, public registry, or source tree.

- [`dexmal/opendm`](https://github.com/dexmal/opendm) at
  `7d52f1591437332cb0157be3303c1c46da811344` supplies
  `playground/dm05_libero.py`, the released inference entry point. Its license
  is Apache-2.0 and its upstream `LICENSE` is retained at `/opt/opendm`.
- [`dexmal/dexbotic-benchmark`](https://github.com/dexmal/dexbotic-benchmark)
  at `789b87f50d9fadc7663d2e8bac057941221aab81` supplies the native closed-loop
  evaluator. Its license is MIT and its upstream `LICENSE` is retained at
  `/opt/dexbotic-benchmark`.
- [`Lifelong-Robot-Learning/LIBERO`](https://github.com/Lifelong-Robot-Learning/LIBERO)
  at `8f1084e3132a39270c3a13ebe37270a43ece2a01` supplies the benchmark closure.
  Its license is MIT (Copyright 2023 Lifelong Robot Learning) and is retained
  at `/opt/dexbotic-benchmark/libero`.

The image keeps no checkpoint, model cache, LIBERO demonstrations, credentials,
telemetry opt-in, or model-derived output. Gemma weight, service, output, and
redistribution terms remain separate from the Apache/MIT source licenses. This
is an operator-private qualification artifact only; it is not an NPA public
release or a grant to redistribute the checkpoint or a hosted service.

Credits and the exact absolute-action/relative-controller comparison contract
are recorded in `docs/workbench/dm05-lerobot-libero.md`.
