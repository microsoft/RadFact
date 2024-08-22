#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from radfact.data_utils.grounded_phrase_list import GroundedPhrase, NormalizedBox
from radfact.llm_utils.nli.schema import EvidencedPhrase
from radfact.llm_utils.text_utils import find_best_match, normalise_text_for_comparison

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OneWayNLIFractions:
    entailed_fraction: float
    full_box_fraction: float
    entailed_box_fraction: float
    num_phrases: int
    num_phrases_with_boxes: int


@dataclass(frozen=True)
class RadFactScore:
    logical_precision: float
    logical_recall: float
    spatial_precision: float
    spatial_recall: float
    grounding_precision: float
    grounding_recall: float
    num_candidate_phrases: int | float
    num_reference_phrases: int | float
    num_candidate_phrases_with_boxes: int | float
    num_reference_phrases_with_boxes: int | float

    @staticmethod
    def _compute_f1_score(precision: float, recall: float) -> float:
        return 2 * (precision * recall) / (precision + recall)

    @property
    def logical_f1(self) -> float:
        return self._compute_f1_score(self.logical_precision, self.logical_recall)

    @property
    def spatial_f1(self) -> float:
        return self._compute_f1_score(self.spatial_precision, self.spatial_recall)

    @property
    def grounding_f1(self) -> float:
        return self._compute_f1_score(self.grounding_precision, self.grounding_recall)

    @classmethod
    def from_candidate_and_reference_fractions(
        cls, candidate: OneWayNLIFractions, reference: OneWayNLIFractions
    ) -> "RadFactScore":
        """Create a score from the candidate and reference fractions."""
        return cls(
            logical_precision=candidate.entailed_fraction,
            logical_recall=reference.entailed_fraction,
            spatial_precision=candidate.full_box_fraction,
            spatial_recall=reference.full_box_fraction,
            grounding_precision=candidate.entailed_box_fraction,
            grounding_recall=reference.entailed_box_fraction,
            num_candidate_phrases=candidate.num_phrases,
            num_reference_phrases=reference.num_phrases,
            num_candidate_phrases_with_boxes=candidate.num_phrases_with_boxes,
            num_reference_phrases_with_boxes=reference.num_phrases_with_boxes,
        )

    @classmethod
    def from_aggregate(cls, scores: list["RadFactScore"], only_factual_scores: bool = False) -> "RadFactScore":
        """Aggregate the scores from a list of samples. If only_factual_scores is True, we only aggregate the logical
        scores. The spatial and grounding scores are set to 0.0.
        """

        def _nanmean(values: list[float | int]) -> float:
            """
            Compute the mean of the values, ignoring NaNs.
            This is mostly for mypy convenience.
            """
            return float(np.nanmean(values))

        n = len(scores)
        if n == 0:
            return cls(
                logical_precision=0.0,
                logical_recall=0.0,
                spatial_precision=0.0,
                spatial_recall=0.0,
                grounding_precision=0.0,
                grounding_recall=0.0,
                num_candidate_phrases=0.0,
                num_reference_phrases=0.0,
                num_candidate_phrases_with_boxes=0.0,
                num_reference_phrases_with_boxes=0.0,
            )
        return cls(
            # If no predicted or reference phrases, these can be NaN
            logical_precision=_nanmean([x.logical_precision for x in scores]),
            logical_recall=_nanmean([x.logical_recall for x in scores]),
            # Box metrics can be NaN if there are no boxes, either direction
            spatial_precision=0.0 if only_factual_scores else _nanmean([x.spatial_precision for x in scores]),
            spatial_recall=0.0 if only_factual_scores else _nanmean([x.spatial_recall for x in scores]),
            grounding_precision=0.0 if only_factual_scores else _nanmean([x.grounding_precision for x in scores]),
            grounding_recall=0.0 if only_factual_scores else _nanmean([x.grounding_recall for x in scores]),
            # Numbers of phrases etc. should never have NaN
            num_candidate_phrases=sum(x.num_candidate_phrases for x in scores) / n,
            num_reference_phrases=sum(x.num_reference_phrases for x in scores) / n,
            # These can be nan if we are running the metric on data without boxes so we set it to 0.0 when
            # only_factual_scores is True
            num_candidate_phrases_with_boxes=(
                0.0 if only_factual_scores else _nanmean([x.num_candidate_phrases_with_boxes for x in scores])
            ),
            num_reference_phrases_with_boxes=(
                0.0 if only_factual_scores else _nanmean([x.num_reference_phrases_with_boxes for x in scores])
            ),
        )


class SpatialEntailmentStatus(str, Enum):
    NO_BOXES = "no_boxes"
    SPATIAL_ENTAILMENT = "spatial_entailment"
    NO_SPATIAL_ENTAILMENT = "no_spatial_entailment"


@dataclass(frozen=True, kw_only=True)
class GroundedPhraseEvidenced(GroundedPhrase):
    status: str
    spatial_entailment_status: SpatialEntailmentStatus | None = None
    evidence: list[GroundedPhrase]
    evidence_indices: list[int] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.evidence_indices is not None:
            assert len(self.evidence_indices) == len(self.evidence)

    def get_all_evidence_boxes(self) -> list[NormalizedBox]:
        all_evidence_boxes = []
        for premise in self.evidence:
            if premise.boxes is not None:
                all_evidence_boxes.extend(premise.boxes)
        return all_evidence_boxes

    @staticmethod
    def attach_evidence_to_hypothesis(
        *,
        evidenced_phrase: EvidencedPhrase,
        hypothesis_grounded_phrase: GroundedPhrase,
        premise_grounded_phrases: list[GroundedPhrase],
    ) -> 'GroundedPhraseEvidenced':
        """
        Attach the evidence to the hypothesis phrase based on NLI output.

        We need to do this because `GroundedPhrase` includes boxes, whereas the NLI processor operates only on strings.

        :param evidenced_phrase: `EvidencedPhrase` for the given hypothesis, as generated by the NLI processor.
        :param hypothesis_grounded_phrase: `GroundedPhrase` corresponding to the hypothesis.
        :param premise_grounded_phrases: List of `GroundedPhrase` premises, containing at least the evidence phrases.
        :return: `GroundedPhraseEvidenced` corresponding to the hypothesis with NLI status and evidence.
        """
        if normalise_text_for_comparison(hypothesis_grounded_phrase.text) != normalise_text_for_comparison(
            evidenced_phrase.phrase
        ):
            raise ValueError(
                f"Evidenced phrase ({evidenced_phrase.phrase}) does not match "
                f"hypothesis ({hypothesis_grounded_phrase.text})."
            )
        evidence_indices = [
            find_best_match(premise, [x.text for x in premise_grounded_phrases])[0]
            for premise in evidenced_phrase.evidence
        ]
        evidence_grounded_phrases = [premise_grounded_phrases[i] for i in evidence_indices]
        return GroundedPhraseEvidenced(
            text=hypothesis_grounded_phrase.text,
            boxes=hypothesis_grounded_phrase.boxes,
            status=evidenced_phrase.status,
            evidence=evidence_grounded_phrases,
            evidence_indices=evidence_indices,
        )

    @staticmethod
    def attach_evidence_to_all_hypotheses(
        *,
        evidenced_phrases: list[EvidencedPhrase],
        hypothesis_grounded_phrases: list[GroundedPhrase],
        premise_grounded_phrases: list[GroundedPhrase],
    ) -> list['GroundedPhraseEvidenced']:
        """
        Attach evidence to all hypothesis phrase based on NLI output.

        We need to do this because `GroundedPhrase` includes boxes, whereas the NLI processor operates only on strings.

        :param evidenced_phrases: List of `EvidencedPhrase` as generated by the NLI processor.
            All phrases and evidence must be contained in the given hypothesis and premise lists, respectively.
        :param hypothesis_grounded_phrases: List of `GroundedPhrase` hypotheses.
        :param premise_grounded_phrases: List of `GroundedPhrase` premises.
        :return: List of `GroundedPhraseEvidenced` corresponding to the hypotheses with NLI status and evidence.
        """

        def retrieve_evidenced_phrase(phrase: str) -> EvidencedPhrase:
            """Given the phrase (text), retrieve the EvidencedPhrase object from the list."""
            phrase_idx, _ = find_best_match(phrase, [x.phrase for x in evidenced_phrases])
            return evidenced_phrases[phrase_idx]

        return [
            GroundedPhraseEvidenced.attach_evidence_to_hypothesis(
                hypothesis_grounded_phrase=hypothesis_grounded_phrase,
                evidenced_phrase=retrieve_evidenced_phrase(hypothesis_grounded_phrase.text),
                premise_grounded_phrases=premise_grounded_phrases,
            )
            for hypothesis_grounded_phrase in hypothesis_grounded_phrases
        ]


@dataclass(frozen=True)
class PerSampleNLIResult:
    study_id: str
    scores: RadFactScore | None = None
    candidate_phrases: list[GroundedPhraseEvidenced] = field(default_factory=list)
    reference_phrases: list[GroundedPhraseEvidenced] = field(default_factory=list)
