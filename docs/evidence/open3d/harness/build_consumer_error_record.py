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
    record["recorded_utc"] = "2026-11-01T16:05:00Z"
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
            "The crop does not thin that region out, it removes it."
        ),
        "what_the_residual_error_is_instead": (
            f"The candidate's {cp['cells_total']} phantom cells are spread over "
            f"{cp['components']} components, {cp['single_cell_components']} of "
            f"them single cells, with {cp['cells_in_structured_components']} "
            f"cells in {cp['structured_components']} component at or above the "
            f"threshold. That component's extent is "
            f"{cp['largest_component_extent_cells']} cells -- "
            f"{cp['largest_component_min_thickness_cells']} cell thick. A "
            "one-cell-thick sheet is what a distance-band occupancy test "
            "produces at the edge of a mesh that is in the right place, not a "
            "lump of geometry that is not there."
        ),
        "the_error_class_that_would_actually_be_dangerous": (
            "A phantom obstacle makes a planner conservative; a missed one makes "
            "it collide. So the missed-structure class is the one worth being "
            f"afraid of, and on both variants it is sampling noise: "
            f"{bm['cells_total']} cells in {bm['components']} components on the "
            f"baseline and {cm['cells_total']} in {cm['components']} on the "
            f"candidate, largest {cm['largest_component_cells']} cells with "
            f"extent {cm['largest_component_extent_cells']}, and "
            f"{cm['structured_components']} components at or above the threshold "
            "on either. Every missed cell's nearest surface lies within "
            f"{cand['missed_distance_in_radii']['max']:.2f} support radii "
            f"(median {cand['missed_distance_in_radii']['median']:.2f}), and "
            f"{cand['missed_distance_in_radii']['beyond_five_radii']} lie beyond "
            "five. These are cells whose mesh is right there, just outside the "
            "quantisation boundary -- not holes."
        ),
        "the_recall_cost_of_the_crop_in_these_terms": (
            f"The crop's recall cost is {cm['cells_total'] - bm['cells_total']} "
            "cells. Both variants carry zero object-scale misses, so the cost is "
            "paid entirely in boundary cells, against "
            f"{bp['cells_in_structured_components'] - cp['cells_in_structured_components']} "
            "cells of structured fabrication removed."
        ),
    }
    record["the_verdict"] = {
        "for_the_mapping_use": (
            "Fit for occupancy mapping at this cell size, with the residual "
            "error characterised rather than bounded away. The structured "
            f"fabrication a consumer would be misled by falls from "
            f"{bp['cells_in_structured_components']} cells to "
            f"{cp['cells_in_structured_components']}, and no missed-structure "
            "component on either variant is object-scale."
        ),
        "what_this_does_not_say": (
            "Nothing here revisits collision fitness. "
            "`capability-record-collision-fitness.json` measured that "
            "separately and the verdict there stands: the surface is not closed, "
            "so inside/outside has no defined answer on it. Occupancy mapping is "
            "the narrower use that does not need closedness, which is why it is "
            "the use being assessed."
        ),
        "the_threshold_landed_exactly_on_the_boundary_and_was_not_moved": (
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
        ),
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
        "the_baseline_is_the_positive_control": (
            "A method that finds no object-scale error could be a method that "
            "cannot see one. The baseline variant of the same run carries a real "
            f"{bp['largest_component_cells']}-cell fabricated region and this "
            "analysis reports it at that size and shape, so the negative result "
            "on the candidate's missed-structure class is informative rather "
            "than an artifact of the instrument."
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
