# A preview whose provenance claims were invented

These files are kept because they were produced and inspected, not because they support
anything. `mkrrd-invalid-provenance.py` fed the visualizer a 64-zero `fused_sha256` and a
single fragment named `f0` carrying an identity pose, built from the already-fused cloud. Three
claims in that recording — fragment identity, fragment pose, and content hash — were invented to
satisfy required fields. An independent audit caught it before publication.

The pixels are real geometry, so `ordinary-window-4x3.png` is citable for what the viewer looked
like and for the fact that the panes framed their contents. It is not citable for fragment
identity, pose, registration, or any hash.

The replacement is `../preview-aggregate-fused.rrd` with `../ordinary-native.png`: an aggregate
fused-cloud view that logs zero fragments, because zero is the true count once the fragment
clouds were not retained. Its bindings are in
`../../capability-record-provenance-bound-preview.json`.
