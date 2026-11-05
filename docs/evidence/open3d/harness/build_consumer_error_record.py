"""Assemble `consumer-error-structure.json` from the measurement, not by hand.

Every figure in the committed record and in its prose is read out of the harness
output rather than typed. The lane has already had one record whose prose
sentence disagreed with its own rows because a human copied a number, so the
narrative strings here are formatted from the measured values.

    python build_consumer_error_record.py <harness-output.json> <record.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Stated in the record and bound to the committed suite by a test, which counts
#: the entries in `harness/consumer_error_guard_mutations.py`. A count in prose
#: that nothing checks is how a guard suite decays while still looking tested.
MUTATIONS_CLAIMED = 11


def narrative(measured: dict) -> dict:
    """The record's prose, formatted from the measured values and nothing else.

    Kept separate from `main` so that the guard suite can call it with the
    committed record as input and compare the result to the committed prose.
    That makes the narrative reproducible rather than merely plausible: a
    hand-touched sentence, or a regenerated measurement whose prose was left
    behind, stops matching. Consistency is all this establishes -- the figures
    are true because the harness gated on reproducing the published counts, not
    because they appear here.
    """
    base = measured["variants"]["baseline"]
    cand = measured["variants"]["candidate"]
    bp = base["phantom_obstacle_structure"]
    cp = cand["phantom_obstacle_structure"]
    bm = base["missed_structure_structure"]
    cm = cand["missed_structure_structure"]
    threshold = measured["structured_threshold_cells"]

    record: dict = {}
    record["recorded_utc"] = measured.get("recorded_utc")
    record["what_this_closes"] = (
        "`downstream-consumer.json` reports the mapping consumer's residual "
        f"errors as counts: {cp['cells_total']} phantom cells and "
        f"{cm['cells_total']} missed-structure cells on the cropped candidate. "
        "A count cannot separate the resolution limit of a 0.1 m grid from an "
        "unmapped wall, and those two readings would lead a mapping consumer to "
        "opposite conclusions. This record decomposes both error classes into "
        "6-connected components and reports every component size and the "
        "largest component's shape."
    )
    record["the_finding"] = {
        "what_the_published_rate_hides": (
            "The published summary is a rate falling from "
            f"{base['phantom_obstacle_cells'] / base['occupied_cells']:.4f} to "
            f"{cand['phantom_obstacle_cells'] / cand['occupied_cells']:.4f}. The "
            "rate is correct and it is not the interesting part. The baseline's "
            f"fabrication is overwhelmingly coherent: {bp['cells_in_structured_components']} "
            f"of its {bp['cells_total']} phantom cells sit in "
            f"{bp['structured_components']} components of {threshold} cells or "
            f"more, the largest {bp['largest_component_cells']} cells spanning "
            f"{bp['largest_component_extent_cells']} cells "
            f"({bp['largest_component_volume_m3']:.3f} m3 of invented solid). "
            "For this self-scored capture the crop removes those cells from the "
            "measured phantom mask; this is not independent evidence that the "
            "remaining surface is accurate."
        ),
        "what_the_residual_error_is_instead": (
            f"The candidate's {cp['cells_total']} phantom cells are spread over "
            f"{cp['components']} components, {cp['single_cell_components']} of "
            f"them single cells, with {cp['cells_in_structured_components']} "
            f"cells in {cp['structured_components']} component at or above the "
            f"threshold. That component's extent is "
            f"{cp['largest_component_extent_cells']} cells -- "
            f"{cp['largest_component_min_thickness_cells']} cell thick. A "
            "one-cell-thick sheet is an expected consequence of scoring a "
            "distance-band crop against the same cloud used to construct it. "
            "The shape describes this capture; it does not independently prove "
            "that the mesh is in the right place."
        ),
        "the_error_class_that_would_actually_be_dangerous": (
            "A missed cell is the safety-relevant class for a mapping consumer. "
            "Against this capture's own fused point cloud, both variants report: "
            f"{bm['cells_total']} cells in {bm['components']} components on the "
            f"baseline and {cm['cells_total']} in {cm['components']} on the "
            "candidate. Their largest components are "
            f"{bm['largest_component_cells']} cells with extent "
            f"{bm['largest_component_extent_cells']} and "
            f"{cm['largest_component_cells']} cells with extent "
            f"{cm['largest_component_extent_cells']}; "
            f"{cm['structured_components']} components at or above the threshold "
            "on either. Nearest-surface distances reach "
            f"{base['missed_distance_in_radii']['max']:.2f} baseline and "
            f"{cand['missed_distance_in_radii']['max']:.2f} candidate support "
            "radii (medians "
            f"{base['missed_distance_in_radii']['median']:.2f} and "
            f"{cand['missed_distance_in_radii']['median']:.2f}); "
            f"{base['missed_distance_in_radii']['beyond_five_radii']} and "
            f"{cand['missed_distance_in_radii']['beyond_five_radii']} lie beyond "
            "five. This only shows proximity to the training cloud. It cannot "
            "detect holes over real structure absent from fused.ply."
        ),
        "the_recall_cost_of_the_crop_in_these_terms": (
            "Against the same-cloud reference, the crop's measured recall cost is "
            f"{cm['cells_total'] - bm['cells_total']} cells. Neither variant has "
            "an object-scale component in this mask, against "
            f"{bp['cells_in_structured_components'] - cp['cells_in_structured_components']} "
            "cells of structured fabrication removed."
        ),
    }
    if cp["largest_component_cells"] == threshold:
        threshold_reading = (
            f"The threshold of {threshold} cells was frozen in the claim before "
            "any component size was known. The candidate's largest phantom "
            f"component is {cp['largest_component_cells']} cells -- exactly the "
            "threshold -- so by the frozen gate the candidate does still carry "
            "one structured component, and this record says so rather than "
            "rounding it away. Moving the threshold to 9 would erase that "
            "sentence and is precisely the move the frozen gate exists to "
            "prevent. The shape reading is reported next to the size, not "
            "instead of it: that component is "
            f"{cp['largest_component_min_thickness_cells']} cell thick where the "
            f"baseline's largest is {bp['largest_component_min_thickness_cells']}, "
            "so the two components the gate calls structured are not the same "
            "kind of error. A reader who wants one number should use the "
            "structured-cell count, not the component count."
        )
    else:
        relation = "above" if cp["largest_component_cells"] > threshold else "below"
        threshold_reading = (
            f"The threshold of {threshold} cells was frozen before measurement. "
            f"The candidate's largest phantom component is "
            f"{cp['largest_component_cells']} cells, {relation} that threshold; "
            f"the measured distribution therefore contains "
            f"{cp['structured_components']} structured components totalling "
            f"{cp['cells_in_structured_components']} cells."
        )

    record["the_verdict"] = {
        "for_this_retained_capture": (
            "The component structure is characterised for this retained "
            "synthetic capture at this cell size. Against the same-cloud "
            "reference, structured phantom cells fall from "
            f"{bp['cells_in_structured_components']} cells to "
            f"{cp['cells_in_structured_components']}, and neither variant's "
            "same-cloud missed mask contains an object-scale component."
        ),
        "what_this_does_not_say": (
            "This record does not establish occupancy-mapping, navigation or "
            "collision fitness. `capability-record-support-crop.json` already "
            "records that the surface is not closed and that G5 held-out "
            "coverage falls after cropping (0.883 to 0.851, 0.961 to 0.933 and "
            "0.934 to 0.855). A same-cloud distance test cannot overrule that "
            "independent limitation."
        ),
        "the_frozen_threshold_was_not_moved": threshold_reading,
    }
    record["how_this_was_validated"] = {
        "the_reproduction_gate_came_first": (
            "The harness re-derives occupied, phantom and missed counts with the "
            "occupancy definition in `consumer_occupancy.py` and exits non-zero "
            "unless all six match `downstream-consumer.json`. They match, so the "
            "decomposition below describes the same cells the published record "
            "counts."
        ),
        "the_labeller_is_an_instrument_and_was_controlled": (
            "Component sizes are only meaningful if the labeller separates what "
            "it should. The control that matters is two blocks touching at a "
            "single corner: 26-connectivity reports one component, 6 reports "
            "two, and the run asserts both so the control cannot pass under the "
            "wrong setting. All "
            f"{len(measured['labeller_validation']['checks'])} checks pass."
        ),
        "the_guards_were_tested_by_corrupting_the_record": (
            f"The guard suite is exercised against {MUTATIONS_CLAIMED} "
            "mutations of this record in "
            "`harness/consumer_error_guard_mutations.py`, each applied to the "
            "real file and reverted. All are caught. The one worth reading is "
            "the first: the candidate's largest phantom component is exactly "
            "the frozen threshold, so moving the threshold to 9 would turn one "
            "structured component into none and make this record read better. "
            "One mutation found a real weakness rather than confirming a "
            "strength: a prose check that accepted the false sentence "
            "'only 3 of the 79 components exceed 6 cells', because 3, 79 "
            "and 6 are all figures this record holds elsewhere. That check "
            "was replaced by regenerating the narrative from the "
            "measurements and comparing, which the same mutation then "
            "failed. Running the suite also surfaced two defects in the "
            "suite itself -- it ran pytest from the wrong directory, "
            "collected nothing, and scored every mutation as missed; and it "
            "restored the record with `git checkout`, which reverts to the "
            "last commit rather than to the bytes under test. Both were "
            "caught by its own assertions, which is the reason to write "
            "them."
        ),
        "the_baseline_is_a_phantom_class_positive_control": (
            "The baseline variant of the same run carries a "
            f"{bp['largest_component_cells']}-cell fabricated region and this "
            "analysis reports it at that size and shape. That controls the "
            "phantom-component labeller only; it is not a positive control for "
            "detecting missed real structure."
        ),
    }
    return record


def main() -> int:
    measured = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    record = {**measured, **narrative(measured)}
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
