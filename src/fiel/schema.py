"""Core data models for Fiel.

These types are the corpus-agnostic contract of the harness. A corpus is a set of
`Passage`s; a labeled eval set is a set of `EvalItem`s. SEC filings are only the
shipped example — anything that can be expressed as (passage, question, answer,
ground-truth label) flows through unchanged.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Label(str, Enum):
    """Ground-truth grounding label for an answer. This is what we CONTROL."""

    GROUNDED = "grounded"  # every claim in the answer is supported by the passage
    UNGROUNDED = "ungrounded"  # the answer asserts something the passage does not support


class FailureMode(str, Enum):
    """Taxonomy of *realistic* ungrounding failure modes (spec §2).

    NONE is used for grounded answers. The rest are deliberately subtle — the whole
    point is to catch judges that miss hard cases, not cartoonish ones.
    """

    NONE = "none"  # grounded answer; no injected failure
    SUBTLE_NUMBER_ERROR = "subtle_number_error"  # right structure, wrong figure
    UNSUPPORTED_ADDITION = "unsupported_addition"  # fluent extra claim the source never makes
    OVERGENERALIZATION = "overgeneralization"  # drops qualifiers (period/geo/segment)
    UNSUPPORTED_CAUSATION = "unsupported_causation"  # asserts a "because" not in the source
    TEMPORAL_SCOPE_DRIFT = "temporal_scope_drift"  # applies a figure to the wrong period/segment
    HEDGE_REMOVAL = "hedge_removal"  # source says "may/expects"; answer states as fact


class Difficulty(str, Enum):
    """How hard the item is for a judge. The easy/hard contrast IS the finding."""

    EASY = "easy"
    HARD = "hard"


class Complexity(str, Enum):
    """Structural shape of the item — the analysis axis for *where* judges degrade.

    The hardness probe showed frontier judges are near-perfect on single-claim
    lookups. These structures target the two documented weaknesses: a long, fluent,
    mostly-correct answer with one buried error, and an answer whose error only
    surfaces if you actually do the arithmetic (lookup alone passes it).
    """

    SINGLE = "single"  # one claim, verifiable by direct lookup
    MULTI_CLAIM = "multi_claim"  # several claims, at most one buried error
    COMPUTATION = "computation"  # correctness requires arithmetic/derivation


class Passage(BaseModel):
    """A source passage the answer must be grounded in. Corpus = list of these."""

    passage_id: str
    text: str
    # Free-form provenance; SEC uses company/filing/section/url, other corpora differ.
    metadata: dict[str, str] = Field(default_factory=dict)


class EvalItem(BaseModel):
    """One labeled test case for the judge: (question, source, answer) + ground truth.

    The judge sees question + passage text + answer and must decide grounded/ungrounded.
    Everything else here is held out and used only for validation and breakdowns.
    """

    item_id: str
    passage_id: str
    question: str
    answer: str
    label: Label
    failure_mode: FailureMode
    difficulty: Difficulty
    # Structural shape (single-claim lookup vs multi-claim vs derived). Defaults to
    # "single" so the pre-existing items need no rewrite. Used only for breakdowns.
    complexity: Complexity = Complexity.SINGLE
    # Why this label is correct — human-authored justification, for auditability.
    rationale: str = ""

    @model_validator(mode="after")
    def _check_label_consistency(self) -> EvalItem:
        # Invariant: grounded <=> no failure mode. Keeps the labeled set internally honest.
        if self.label is Label.GROUNDED and self.failure_mode is not FailureMode.NONE:
            raise ValueError(
                f"{self.item_id}: grounded item must have failure_mode=none, "
                f"got {self.failure_mode.value}"
            )
        if self.label is Label.UNGROUNDED and self.failure_mode is FailureMode.NONE:
            raise ValueError(
                f"{self.item_id}: ungrounded item must declare a failure_mode (not none)"
            )
        return self


class Verdict(BaseModel):
    """A judge's decision on a single item, parsed from the model's output."""

    verdict: Label
    rationale: str = ""
    # True when we could not parse a clean verdict from the model output.
    parse_failed: bool = False
