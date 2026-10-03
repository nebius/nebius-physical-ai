# Third-party notices: OpenVLA-OFT runtime fetch

The image's minimal NPA control path is Apache-2.0. Its hash-pinned artifact
transfer dependencies are: Boto3, Botocore, and S3Transfer (Apache-2.0);
Jmespath, Six, and Urllib3 (MIT); and python-dateutil (Apache-2.0/BSD-3-Clause).
Their installed distributions retain their license files. The exact wheel set is
`control-requirements.lock`; it remains subject to built-byte review before any
publication determination.

When materialized, retain the upstream MIT notice for:

- Moo Jin Kim, Chelsea Finn, Percy Liang, `moojink/openvla-oft`, commit
  `e4287e94541f459edc4feabc4e181f537cd569a8`.
- Lifelong Robot Learning, `Lifelong-Robot-Learning/LIBERO`, commit
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`.

Credit the OFT paper:

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```

The base model, adapters, and LIBERO data are not container bytes. Their exact
runtime revision, license/term source, and any required notices are recorded in
the run's immutable artifact manifests; see `docs/workbench/openvla-oft.md`.
