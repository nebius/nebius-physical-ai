# Reference for synthetic instruction training

This example corpus concerns text instructions about colored objects and small
discrete reasoning problems. Every described object, container, and observation
is synthetic. The resulting text is training material, not an executed robot
plan, a sensor observation, or evidence of a successful physical action.

## Meaning of movement instructions

An instruction to put an object into a container describes a change in its
location. A paraphrase must preserve the object, its color, the destination,
and the spatial relation. For example, moving a red cube into a blue tray does
not mean moving the tray, changing the cube's color, or placing the cube beside
the tray. It is valid to replace “put” with “place” when the remaining meaning
is preserved. Adding speed, a gripper, a grasp location, a starting pose, or a
claim that the action succeeded introduces facts absent from a simple seed.

“On” and “in” are different relations. Placing a cylinder on a shelf means the
shelf supports it. Placing a cylinder in a box means the box contains it.
“Beside” specifies adjacency without containment or support; a paraphrase must
not turn it into “inside,” “above,” or “under.” Colors identify the objects in
these examples, so color substitutions change the meaning even when the shape
and action are otherwise unchanged. Keep both object and destination colors.

Simple paraphrase training examples should have a self-contained question that
quotes or clearly states the source instruction. The answer should give a
single faithful alternative. A short answer is sufficient when the question
asks for one paraphrase. It does not need an explanation of the paraphrasing
process or an evaluation of its own quality. A reviewer can compare each noun,
color, and spatial relation directly with the seed to detect changed meaning.

## Drawing colored balls without replacement

A sealed box in these reasoning examples contains exactly the stated numbers
of two kinds of balls. The balls differ only in the property named by the
question, such as their colors. A draw removes one ball, and that ball is not
returned before subsequent draws. The question asks for a guarantee in the
worst possible ordering, not a high probability or an average number of draws.
Assume the target color has at least one ball; otherwise no number of permitted
draws can guarantee obtaining a ball of that color.

If there are k balls of the non-target color, an unfavorable ordering can begin
with all k of them. Thus k draws are not sufficient for a guarantee. Once those
k balls have been removed, every remaining ball has the target color, so draw
k + 1 must produce it. This proves both sufficiency and minimality. The count
of target-colored balls must be positive, but increasing it does not change the
worst-case guarantee while the non-target count stays fixed. Do not mistake the
total number of balls for the minimum number of draws needed for a guarantee.

A complete training answer should state the required number of draws, exhibit
the worst-case initial ordering, and explain why the next draw has the target
color. The instruction should include both counts, the target color, and the
fact that drawing occurs without replacement. The answer should not assume
that the box is shaken, that each color alternates, that the draws are fair, or
that an observer can select a particular color before drawing. No probability
calculation is needed to prove a worst-case guarantee.

Drawing with replacement is a different problem. Returning the ball allows
the same non-target outcome to occur repeatedly, so a finite guarantee usually
does not follow from these counts. Do not silently add replacement to a seed
that explicitly says “without replacement.” Likewise, a question asking for
at least two target-colored balls needs a different argument; do not answer
that different question when the seed asks for the first target-colored ball.

## What belongs in a training pair

The instruction should contain all information a reader needs to solve the
task. The answer should respond to that instruction directly and preserve the
seed's intended task. It should not refer to an unavailable image, external
document, previous turn, hidden rubric, or an unnamed reference. Background
from this reference can inform the wording, but the pair must make sense when
the reference file is not included in the later training example.

For the simple tasks, unnecessary embellishment can introduce errors. For the
reasoning tasks, an unexplained number can omit the requested justification.
Match the amount of explanation to the task. A concise proof of a counting
claim is more useful than a long discussion that never establishes the worst
case. Avoid unsupported statements about real robot performance, dataset
accuracy, physical safety, or model capabilities: none is established by a
generated instruction/answer pair.

Review the candidate itself, including whether its answer satisfies its own
instruction. Schema validity only establishes the shape of the output. It
does not establish correctness, fidelity to the seed, or suitability for a
particular training objective. A generated review is another model judgment
and can be wrong. Retaining the candidate, review, selected model, actual
generation model, and provider usage allows a human to audit both successes
and failures before using the exported data in training.
