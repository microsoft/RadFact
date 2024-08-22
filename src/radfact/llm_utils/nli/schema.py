#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import json
from enum import Enum
from pathlib import Path
from typing import Any
import logging
from pydantic import BaseModel, Field, root_validator

from radfact.data_utils.grounded_phrase_list import GroundedPhraseList
from radfact.llm_utils.processor.base_processor import BaseModelWithId
from radfact.llm_utils.text_utils import normalise_text_for_comparison

logger = logging.getLogger(__name__)


class DirectionOptions(str, Enum):
    """
    Keys for the direction of the comparison.
    """

    B_TO_A = "B_to_A"  # Given B, is A true?
    A_TO_B = "A_to_B"  # Given A, is B true?


class NLIState(str, Enum):
    """State of phrase from report."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class EVState(str, Enum):
    """Entailment verification state."""

    ENTAILMENT = "entailment"
    NOT_ENTAILMENT = "not_entailment"

    @classmethod
    def from_nli_state(cls, nli_state: NLIState) -> "EVState":
        """
        Convert an NLIState to an EVState.
        ENTAILMENT --> ENTAILMENT, else NOT_ENTAILMENT.

        Returns a ValueError if the NLIState is not recognized.
        """
        if nli_state == NLIState.ENTAILMENT:
            return EVState.ENTAILMENT
        elif nli_state in [NLIState.CONTRADICTION, NLIState.NEUTRAL]:
            return EVState.NOT_ENTAILMENT
        else:
            raise ValueError(f"Unrecognized NLIState: {nli_state}.")


class ComparisonQuery(BaseModel):
    phrases_A: list[str] = Field(description="Phrases from report A.")
    phrases_B: list[str] = Field(description="Phrases from report B.")


class ComparisonQuerySinglePhrase(BaseModel):
    reference: list[str] = Field(description="Reference report.")
    hypothesis: str = Field(description="Phrase to assess.")


class EvidencedPhrase(BaseModel):
    phrase: str = Field(description="Phrase from report.")
    evidence: list[str] = Field(description="Phrase(s) from reference report supporting the logical state.")
    # Note that the status could either be NLIState or EVState
    status: str = Field(description="Logical state of phrase given reference report.")

    def convert_to_binary(self) -> "EvidencedPhrase":
        """Convert the status to binary."""
        try:
            _ = EVState(self.status)
            # If this succeeds, we are already binary.
            return self
        except ValueError:
            new_status = EVState.from_nli_state(NLIState(self.status))
            return self.copy(update={"status": new_status.value})

    @root_validator
    @classmethod
    def evidence_exists_or_not(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Entailed phrases always need evidence.
        Neutral phrases should have no evidence.
        Contradicted phrases should have evidence.
        Not-entailed phrases can either have or not have evidence (since they are contradiction OR neutral).
        """
        status = values["status"]
        evidence = values["evidence"]
        # Entailment --> evidence
        if status == NLIState.ENTAILMENT or status == EVState.ENTAILMENT:
            if len(evidence) == 0:
                raise ValueError(f"Entailed phrases should have evidence. {values['phrase']=}")
        # Neutral --> no evidence
        elif status == NLIState.NEUTRAL:
            if len(evidence) > 0:
                raise ValueError(f"Neutral phrases should not have evidence. {values['phrase']=}")
        # Contradiction --> evidence
        elif status == NLIState.CONTRADICTION:
            if len(evidence) == 0:
                raise ValueError(f"Contradicted phrases should have evidence. {values['phrase']=}")
        elif status == EVState.NOT_ENTAILMENT:
            # Not-entailed phrases can either have or not have evidence (since they are contradiction OR neutral).
            pass
        else:
            raise ValueError(f"Unrecognized status: {status}.")
        return values

    def pretty_format(self) -> str:
        return f"{self.status.ljust(15)}|{self.phrase}|{self.evidence}"


class BidirectionalEvidence(BaseModel):
    phrases_A_evidenced: list[EvidencedPhrase] = Field(
        description="Phrases from report A with logical state and supporting evidence."
    )
    phrases_B_evidenced: list[EvidencedPhrase] = Field(
        description="Phrases from report B with logical state and supporting evidence."
    )

    def pretty_format(self) -> str:
        def _format_single_direction(evidenced_phrases: list[EvidencedPhrase]) -> str:
            return "\n".join(phrase.pretty_format() for phrase in evidenced_phrases)

        overall_output = "=== Phrases A ===\n"
        overall_output += _format_single_direction(self.phrases_A_evidenced)
        overall_output += "\n=== Phrases B ===\n"
        overall_output += _format_single_direction(self.phrases_B_evidenced)
        return overall_output


class NLIQuerySample(BaseModelWithId):
    """
    A single sample for the NLI task.
    Does not include an output as it is not necessarily for demonstration/testing.
    """

    example_id: str
    input: ComparisonQuery

    @classmethod
    def from_grounded_phrases_list_pair(
        cls, example_id: str, candidate: GroundedPhraseList, reference: GroundedPhraseList
    ) -> "NLIQuerySample":
        """
        Create an NLIQuerySample from a pair of `GroundedPhraseList` instances (candidate and reference).
        """
        candidate_grounded_phrase = [
            phrase.text for phrase in candidate.get_all_grounded_phrases(fail_if_non_grounded_phrase=True)
        ]
        reference_grounded_phrase = [
            phrase.text for phrase in reference.get_all_grounded_phrases(fail_if_non_grounded_phrase=True)
        ]
        return NLIQuerySample(
            example_id=example_id,
            input=ComparisonQuery(phrases_A=candidate_grounded_phrase, phrases_B=reference_grounded_phrase),
        )


class NLISample(NLIQuerySample):
    """
    A single sample with output for the NLI task.
    Enables validation of the output.
    """

    output: BidirectionalEvidence

    @classmethod
    def from_pair_of_unidirectional_lists(
        cls, example_id: str, A_to_B: list[EvidencedPhrase], B_to_A: list[EvidencedPhrase]
    ) -> "NLISample":
        """
        Create an NLISample from a pair of one-way samples.
        """
        return NLISample(
            example_id=example_id,
            input=ComparisonQuery(phrases_A=[x.phrase for x in B_to_A], phrases_B=[x.phrase for x in A_to_B]),
            output=BidirectionalEvidence(phrases_A_evidenced=B_to_A, phrases_B_evidenced=A_to_B),
        )

    @root_validator(skip_on_failure=True)
    @classmethod
    def input_and_output_phrases_match(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Every original phrase must appear in the output. Every evidenced phrase must appear in the input."""
        output: BidirectionalEvidence = values["output"]
        input: ComparisonQuery = values["input"]
        phrase_pairings = {
            "phrases_A": (input.phrases_A, output.phrases_A_evidenced),
            "phrases_B": (input.phrases_B, output.phrases_B_evidenced),
        }
        missing_phrases: dict[str, list[str]] = {}
        added_phrases: dict[str, list[str]] = {}
        for phrase_list_name, (phrases, evidenced_phrases) in phrase_pairings.items():
            missing_phrases[phrase_list_name] = [x for x in phrases if x not in [y.phrase for y in evidenced_phrases]]
            added_phrases[phrase_list_name] = [x.phrase for x in evidenced_phrases if x.phrase not in phrases]
        if any(missing_phrases.values()):
            raise ValueError(f"Phrases should have been classified but are not in output: {missing_phrases}.")
        if any(added_phrases.values()):
            raise ValueError(f"Phrases should not have been classified but are in output: {added_phrases}.")
        return values

    @root_validator(skip_on_failure=True)
    @classmethod
    def evidence_from_correct_report(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check that all evidence comes from the appropriate report."""
        output: BidirectionalEvidence = values["output"]
        input: ComparisonQuery = values["input"]
        cleaned_phrases_A = [normalise_text_for_comparison(phrase) for phrase in input.phrases_A]
        cleaned_phrases_B = [normalise_text_for_comparison(phrase) for phrase in input.phrases_B]

        def _confirm_evidence_is_in_expected_report(
            phrase_with_evidence: EvidencedPhrase, report_phrases: list[str]
        ) -> None:
            """All the evidence from a phrase should be in the expected report.
            If phrase is from A, evidence is from B.
            """
            for supporting_phrase in phrase_with_evidence.evidence:
                if normalise_text_for_comparison(supporting_phrase) not in report_phrases:
                    raise ValueError(
                        f"Evidence for {phrase_with_evidence.phrase} comes from {supporting_phrase}, "
                        f"which is not in the expected report. Expected it to be one of {report_phrases}."
                    )

        for phrase_with_evidence in output.phrases_A_evidenced:
            _confirm_evidence_is_in_expected_report(phrase_with_evidence, cleaned_phrases_B)
        for phrase_with_evidence in output.phrases_B_evidenced:
            _confirm_evidence_is_in_expected_report(phrase_with_evidence, cleaned_phrases_A)
        return values

    @root_validator
    @classmethod
    def evidenced_phrases_exist_in_original_list(cls, values: dict[str, Any]) -> dict[str, Any]:
        """All evidenced phrases must be found in the original phrase list."""
        if "output" not in values:
            raise ValueError("No BidirectionalEvidence? Must have failed validation.")
        output: BidirectionalEvidence = values["output"]
        input: ComparisonQuery = values["input"]
        for phrase_with_evidence in output.phrases_A_evidenced:
            if phrase_with_evidence.phrase not in input.phrases_A:
                raise ValueError(
                    f"Evidenced phrase {phrase_with_evidence.phrase} not found in phrase A list ({input.phrases_A})."
                )
        for phrase_with_evidence in output.phrases_B_evidenced:
            if phrase_with_evidence.phrase not in input.phrases_B:
                raise ValueError(
                    f"Evidenced phrase {phrase_with_evidence.phrase} not found in phrase B list ({input.phrases_B})."
                )
        return values


class NLIQuerySampleSinglePhrase(BaseModel):
    """A single sample for the NLI task, assuming we only go one phrase at a time."""

    example_id: str
    input: ComparisonQuerySinglePhrase

    @classmethod
    def from_nli_query_sample(
        cls, nli_query_sample: NLIQuerySample
    ) -> dict[DirectionOptions, list["NLIQuerySampleSinglePhrase"]]:
        """
        Create a dict list of NLIQuerySampleSinglePhrase instances from an NLIQuerySample.
        We return a dict so we can do A-to-B and B-to-A separately.
        """
        B_to_A = [
            NLIQuerySampleSinglePhrase(
                example_id=nli_query_sample.example_id,
                input=ComparisonQuerySinglePhrase(reference=nli_query_sample.input.phrases_B, hypothesis=phrase_A),
            )
            for phrase_A in nli_query_sample.input.phrases_A
        ]
        A_to_B = [
            NLIQuerySampleSinglePhrase(
                example_id=nli_query_sample.example_id,
                input=ComparisonQuerySinglePhrase(reference=nli_query_sample.input.phrases_A, hypothesis=phrase_B),
            )
            for phrase_B in nli_query_sample.input.phrases_B
        ]
        return {DirectionOptions.B_TO_A: B_to_A, DirectionOptions.A_TO_B: A_to_B}


class NLISampleSinglePhrase(BaseModel):
    """A single sample with output for the NLI task, assuming  we only go one phrase at a time."""

    example_id: str
    input: ComparisonQuerySinglePhrase
    output: EvidencedPhrase

    @classmethod
    def from_nli_sample(cls, nli_sample: NLISample) -> dict[DirectionOptions, list["NLISampleSinglePhrase"]]:
        """
        Create a dict list of NLISampleSinglePhrase instances from an NLISample.
        We return a dict so we can do A-to-B and B-to-A separately.
        """
        B_to_A = [
            NLISampleSinglePhrase(
                example_id=nli_sample.example_id,
                input=ComparisonQuerySinglePhrase(
                    reference=nli_sample.input.phrases_B, hypothesis=evidenced_phrase_A.phrase
                ),
                output=evidenced_phrase_A,
            )
            for evidenced_phrase_A in nli_sample.output.phrases_A_evidenced
        ]
        A_to_B = [
            NLISampleSinglePhrase(
                example_id=nli_sample.example_id,
                input=ComparisonQuerySinglePhrase(
                    reference=nli_sample.input.phrases_A, hypothesis=evidenced_phrase_B.phrase
                ),
                output=evidenced_phrase_B,
            )
            for evidenced_phrase_B in nli_sample.output.phrases_B_evidenced
        ]
        return {DirectionOptions.B_TO_A: B_to_A, DirectionOptions.A_TO_B: A_to_B}


def load_examples_from_json(json_path: Path, binary: bool = True) -> list[NLISample]:
    """
    Helper function to load NLISamples from a json.

    :param json_path: Path to the json we wish to load from.
    :param binary: Whether to convert NLIState to EVState (ENTAILMENT v. NOT_ENTAILMENT).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        examples_json = json.load(f)
    if binary:
        # Convert NLIState to EVState (ENTAILMENT v. NOT_ENTAILMENT)
        samples = []
        for example_json in examples_json:
            nli_sample = NLISample.parse_obj(example_json)
            nli_sample.output.phrases_A_evidenced = [
                phrase.convert_to_binary() for phrase in nli_sample.output.phrases_A_evidenced
            ]
            nli_sample.output.phrases_B_evidenced = [
                phrase.convert_to_binary() for phrase in nli_sample.output.phrases_B_evidenced
            ]
            samples.append(nli_sample)
            for evidenced_phrase in nli_sample.output.phrases_A_evidenced:
                _ = EVState(evidenced_phrase.status)
            for evidenced_phrase in nli_sample.output.phrases_B_evidenced:
                _ = EVState(evidenced_phrase.status)
    else:
        samples = [NLISample.parse_obj(example_json) for example_json in examples_json]
    logger.info(f"Loaded {len(samples)} examples.")
    return samples
