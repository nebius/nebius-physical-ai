# Actual hosted visual review

Descriptive inspection of one native rendered frame. Actual hosted provider API calls; this is not a validation of the shipped workbench vlm-eval CLI. No numeric score, quality threshold, geometry accuracy or safety gate.

Four calls completed normally. No overall VLM pass is accepted. The empty-scene control was recognized; identity prompts changed the certainty of bridging comments. The original freeze said thinking was disabled, but no thinking parameter was sent and the provider reported reasoning tokens.

## subject

Model: `MiniMaxAI/MiniMax-M3`; HTTP 200; finish `stop`.

1. **Readability and framing.** The geometry is legible in both panes at this size. The left pane shows a larger, closer view of the structure, while the right pane shows the same object at a smaller scale, fitting comfortably within its frame. Nothing appears clipped by pane edges, and neither pane is empty.

2. **Point data versus surface.** Discrete sampled points are visible as the bumpy, granular orange texture covering most of the structure, particularly on the angled roof-like surfaces. Smoother, continuous reconstructed surfaces appear as the gray/white regions, especially on the lower portions and what looks like a protruding element on the right side of the structure.

3. **Completeness cues.** There are visible ragged edges and open boundaries, most notably on the right side of the structure where the geometry tapers off into scattered, sparse points. The left edge also shows irregular boundaries where the surface ends abruptly. I cannot tell from this view whether any surface is bridging across gaps in the underlying data.

4. **What to inspect next.** I would suggest opening the right-side region of the structure first, where the geometry clearly thins out into sparse points and ragged edges — this is where data coverage appears weakest and any bridging artifacts or missing surface would be most apparent.

## control_blank

Model: `MiniMaxAI/MiniMax-M3`; HTTP 200; finish `stop`.

1. **Readability and framing.** Both panes are framed normally with no clipping at the edges, but neither pane contains any visible geometry — only an empty perspective grid floor receding into a dark purple/blue background. There is nothing to assess because no points, surfaces, or meshes are rendered in either view.

2. **Point data versus surface.** I cannot distinguish point data from a reconstructed surface because neither is visible. The panes show only the empty grid; no discrete points and no continuous surface appear anywhere in the image.

3. **Completeness cues.** No holes, open boundaries, ragged edges, or bridging artifacts can be observed, since there is no geometry present to exhibit them. The image contains no surface or point data to evaluate for completeness.

4. **What to inspect next.** I would tell the engineer that the image itself shows no geometry, so the first thing to check is whether the data has actually loaded into the viewer — the "Observed scan only" / "Supported surface only" toggles in the right pane, the scene selection, and the log_time indicator at the bottom left are the only leads visible. If the data is confirmed loaded elsewhere, the next step would be to verify the visibility toggles and camera framing, since the panes appear empty rather than showing a clipped or off-screen model.

## control_identity_truthful

Model: `MiniMaxAI/MiniMax-M3`; HTTP 200; finish `stop`.

1. **Readability and framing.** The geometry is legible in both panes but rendered fairly small relative to the available viewport, with the right pane showing the same scene at a noticeably reduced scale. Nothing is clipped by a pane edge and no pane is empty, though fine surface detail is hard to assess at this zoom level.

2. **Point data versus surface.** In the left pane you can see a granular, bumpy texture consistent with discrete sampled points alongside smoother shaded regions consistent with a continuous reconstructed surface; the right pane shows only the discrete point cloud, with no smooth surface visible. The orange/brown walls and gray interior elements appear in both, but only the left pane exhibits the smoother Poisson-style fill.

3. **Completeness cues.** The reconstruction has clearly open boundaries — the structure does not close into a full room, and the right-hand and rear edges fade into ragged, fragmented points rather than a clean surface. There are regions where the smoother surface appears to span across gaps in the underlying points, particularly along the upper wall/roof area and where the gray interior mass meets the orange walls.

4. **What to inspect next.** I would tell the engineer to first open the per-scan alignment view (or the individual fragment streams) to check whether the ragged edges and bridging surfaces correspond to misregistered scans rather than true geometry. After that, isolating the Poisson mesh alone would help confirm whether the bridging regions are reconstruction artifacts versus real surfaces supported by the points.

## control_identity_false

Model: `MiniMaxAI/MiniMax-M3`; HTTP 200; finish `stop`.

1. **Readability and framing.** Both panes contain visible geometry; neither is empty. The left pane shows the structure at a larger scale with readable detail, while the right pane shows the same object at a smaller, more distant framing where fine features become hard to assess. No pane edges appear to clip the geometry, though the right pane's reduced size makes surface texture and edge quality difficult to judge.

2. **Point data versus surface.** Discrete sampled points are clearly visible across the structure, especially along the sloped face and the upper flat region, where individual returns can be resolved. A continuous reconstructed surface appears in patches — most noticeably on the lighter, smoother-looking areas on the right side of the left pane and on portions of the upper deck — where the points blend into a more uniform appearance.

3. **Completeness cues.** The right-hand edge of the structure in the left pane shows a ragged, irregular boundary where points thin out and terminate abruptly, suggesting an open boundary of the scan rather than a clean edge of the object. There are also visible gaps and holes within the sloped face where the point density drops, and the lighter reconstructed regions appear to bridge across some of these sparser areas, which could mask missing data underneath.

4. **What to inspect next.** I would tell the engineer to first examine the ragged right-hand boundary and the lighter reconstructed patches in the left pane, toggling between "Observed scan only" and "Supported surface only" to see whether the smooth areas are genuine surfaces or interpolation bridging real gaps in the returns. The right pane is too small to be useful for this check and should be zoomed in or replaced with a closer view of the same region.

## Independent interpretation

NO_OVERALL_VLM_PASS. The blank control succeeds narrowly; identity-conditioned descriptions are prompt-sensitive. These answers suggest what to inspect, not verified geometry defects or correctness.

The subject explicitly cannot tell whether a surface bridges gaps. With truthful identity context, the same pixels are described as containing smoother patches spanning gaps. This is a change in certainty and interpretation; no identity-consistency pass is accepted.

The answer does not explicitly assert an outdoor construction site or intensity colorization, which were the concrete false assertions in the frozen expectation. It also does not rebut the false framing and adopts some scan/return vocabulary. Absence of a rebuttal alone is not a newly invented failure criterion. Treat identity robustness as unestablished and retain the full answer, rather than claiming a pass.

These are inspection suggestions from one frame. Objective geometric measurements, negative controls, and the downloadable recording are separate evidence.
