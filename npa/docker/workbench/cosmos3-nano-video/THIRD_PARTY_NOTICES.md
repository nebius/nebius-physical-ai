# Third-party components and runtime delivery

The image inherits the accepted public `npa-cosmos3-serving` bootstrap at
`sha256:3342bbe44bd1c00ebf05ab4c9d7286058a94bb5ce90b49b164b23604d3acf180`.
Python uses the PSF License. Debian packages retain their component license and
copyright files under `/usr/share/doc`. This includes FFmpeg and operating-system
libraries. NPA adapter and bootstrap source use Apache-2.0.

The reviewed replacement hash-locked recipe describes CUDA and serving dependencies fetched
at runtime under their own terms. vLLM-Omni source revision
`eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c` uses Apache-2.0; that license does not
replace NVIDIA component licenses. No executable CUDA/vLLM dependency, checkpoint,
vendor fixture, or sample media is inherited from the former vendor image.

Model, data, cache, and output licenses are separate from the image license.
See `REDISTRIBUTION.md` in this image's source directory. Runtime caches and
outputs are operator storage, never inputs to image builds or publication.

Ray uses Apache-2.0; its pinned wheel and dependencies are fetched at runtime.
