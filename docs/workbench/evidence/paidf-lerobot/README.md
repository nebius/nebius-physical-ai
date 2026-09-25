# LeRobot augmentation review evidence

The `cups.jpg`, `coffee.jpg` and `lift.jpg` sheets compare decoded frames from a LeRobot
episode and two Cosmos3-Nano outputs. Columns are source, native CFG normalization
disabled, then enabled. Labels contain zero-based frame indices and timestamps.
Images were resized and arranged for review; no pixels were retouched or blended
between source and output. All six outputs were rejected by the configured gate.

See the [comparison guide](../../guides/paidf-lerobot-realism.md) for controls,
observations and limitations, and [results.json](results.json) for measurements
and artifact hashes. Source metadata and downloaded-file hashes are in the
[source manifest](../../examples/paidf-lerobot-realism-sources.json).

## Twelve-scenario cup fanout

[cups-fanout.jpg](cups-fanout.jpg) shows the prepared cup-opening source and
twelve actual Cosmos3-Nano outputs at frame 128 (5.333 seconds). Only resizing,
arrangement and labels were applied. [cups-fanout-results.json](cups-fanout-results.json)
records the complete videos' hashes, profiles, strict gate results and sampled
visual observations. [cups-fanout.mp4](cups-fanout.mp4) contains the complete
eight-second source-plus-twelve overview, with resized decoded frames and static
labels. Its hash, frame count and dimensions are in the result manifest.
The cup source attribution and MIT permission below apply
to these additional source-frame and source-video reproductions as well.

## Source attribution

The source columns reproduce frames from these datasets, whose pinned dataset
cards declare the MIT license:

- [LeRobot ALOHA cup opening](https://huggingface.co/datasets/lerobot/aloha_static_cups_open/blob/d793c969cf716001dcca18a0842c3d7e9de9e41b/README.md), associated with [ALOHA](https://tonyzhaozh.github.io/aloha/).
- [LeRobot ALOHA coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/b144896feb1f37398a862927b22cd3abdf005a6b/README.md), associated with ALOHA.
- [LeRobot xArm lift medium](https://huggingface.co/datasets/lerobot/xarm_lift_medium/blob/79efb0e3cef0e530ddec4b8569b190966ab45808/README.md), associated with [TD-MPC](https://www.nicklashansen.com/td-mpc/).

Credit belongs to the respective dataset authors and LeRobot contributors. The
downloaded cards do not provide a separate copyright notice. The MIT permission
text is retained here for the reproduced source frames:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
