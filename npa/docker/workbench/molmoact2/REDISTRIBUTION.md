# MolmoAct2 official-policy BYOF redistribution decision

Status: `unvalidated`; private operator BYOF build only.

The recipe ships pinned Apache-2.0 MolmoAct2 source and the Apache-2.0 upstream
LeRobot integration. The CUDA base and every resolved build/runtime dependency
must be reviewed from the built bytes before a public-redistribution decision.
This repository has not performed that review or published this image.

The build removes LeRobot test-only fixture bytes, torchmetrics' LPIPS and DISTS
reference weights, and imageio-ffmpeg's bundled static executable in the layers
where they would otherwise enter the image. Video runtime uses the Ubuntu
`ffmpeg` package instead; that package remains subject to the base distribution's
dependency and license review before any public redistribution decision.

MolmoAct2 checkpoints/adapters, the LIBERO dataset, DROID data, SO100/101
captures, Hugging Face caches, operator credentials, and run outputs are not
baked. They are runtime artifacts under their own upstream terms. Keeping them
outside the image does not itself grant redistribution permission or provider
access; it only prevents this image from distributing those payloads.

No public registry tag, public image catalog entry, or supported release claim
is authorized by this recipe. A future public image requires the full
secure-image and source/notice/SBOM/license/vulnerability/secret/payload scan
chain, an anonymous exact-digest pull, and an independently inspected real GPU
workflow result.
