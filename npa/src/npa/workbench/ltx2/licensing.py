"""Facts about the LTX-2.x licence that the workbench needs to state accurately.

LTX-2.5 is *not* OSI-licensed open source, despite being described as such on
Lightricks' marketing pages. Everything Lightricks publishes for it — weights
*and* the ``ltx-core`` / ``ltx-pipelines`` / ``ltx-trainer`` code — is governed
by the LTX-2.x Community License Agreement (Section 1.9 folds
"inference-enabling code, training-enabling code ... accompanying source code"
into the licensed subject matter). That is why the ``npa-ltx2`` image bakes none
of it and fetches everything at run time from Lightricks' own channels.

Compliance with that agreement is the operator's own responsibility, and this
module deliberately does not try to model it:

* The Agreement forms **by conduct** — its opening line is "By downloading,
  using, accessing or distributing any portion or element of LTX-2.x, you agree
  that you have read and accepted to be bound by this Agreement." A local
  ``ACCEPT=YES`` variable never formed that contract and added nothing to it.
* The weights repository is **gated** on Hugging Face, and access is granted
  only after a human accepts Lightricks' terms there. A working ``HF_TOKEN``
  with access is therefore *checkable evidence* of acceptance — strictly
  stronger than a variable the operator typed themselves.
* Section 2.1 (a paid Commercial Use Agreement above $10M annual revenue) and
  Attachment A(18) (no training other models on Outputs for commercial use)
  depend on facts about the operator that Nebius cannot verify. We ship zero LTX
  bytes and are not a distributor, so we do not ask customers to self-certify
  their revenue; we state the terms and name the vendor contact.

So what remains here is the record of *which* text applies and where to read it.
It is stated once, in one place, so the CLI, the container's ``terms`` output,
and the documentation cannot drift apart.

This is a record of published licence facts, not legal advice.
"""

from __future__ import annotations

# Pinned upstream identity. The licence is versioned by date and Lightricks has
# already reissued it once (the LTX-2 agreement of 2026-01-05 was superseded by
# the LTX-2.x agreement of 2026-08-11, released with LTX-2.5), so name which
# text applies rather than just "the LTX licence".
LICENSE_NAME = "LTX-2.x Community License Agreement"
LICENSE_DATE = "2026-08-11"
ACCEPTABLE_USE_POLICY_URL = (
    "https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf"
)
COMMERCIAL_LICENSE_CONTACT = "ltxv-licensing@lightricks.com"
COMMERCIAL_REVENUE_THRESHOLD_USD = 10_000_000

# Upstream source and weights. Neither is baked into the image; both are fetched
# at run time from Lightricks' own distribution channels, against the operator's
# own gated-repository entitlement. See npa/docker/workbench/ltx2/REDISTRIBUTION.md.
SOURCE_REPO = "https://github.com/Lightricks/LTX-2"
SOURCE_REF = "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca"

# Pinned to the ref we fetch, not to `main`: the licence is versioned by date and
# Lightricks has already reissued it once, so a mutable URL would stop naming the
# text a given run was accepted against. REDISTRIBUTION.md says the same.
LICENSE_URL = f"https://github.com/Lightricks/LTX-2/blob/{SOURCE_REF}/LICENSE.md"
WEIGHTS_REPO = "Lightricks/LTX-2.5"
WEIGHTS_REPO_URL = f"https://huggingface.co/{WEIGHTS_REPO}"

__all__ = [
    "ACCEPTABLE_USE_POLICY_URL",
    "COMMERCIAL_LICENSE_CONTACT",
    "COMMERCIAL_REVENUE_THRESHOLD_USD",
    "LICENSE_DATE",
    "LICENSE_NAME",
    "LICENSE_URL",
    "SOURCE_REF",
    "SOURCE_REPO",
    "WEIGHTS_REPO",
    "WEIGHTS_REPO_URL",
]
