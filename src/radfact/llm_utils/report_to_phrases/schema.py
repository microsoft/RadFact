#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, root_validator

from radfact.data_utils.grounded_phrase_list import GroundedPhrase, GroundedPhraseList
from radfact.llm_utils.processor.base_processor import BaseModelWithId


class SentenceWithRephrases(BaseModel):
    """Dataclass for a sentence with rephrases. The source sentence is 'orig' and the rephrased sentences are 'new'."""

    orig: str
    new: list[str]


class ParsedReport(BaseModelWithId):
    """
    How we represent a parsed report. This applies to the model output and the examples.
    The `ParsedReport` is a list of `SentenceWithRephrases`.
    Each `SentenceWithRephrases` has an 'orig' sentence and a list of 'new' rephrased ('phrasified') sentences.
    """

    sentence_list: list[SentenceWithRephrases]

    def phrases_as_list(self) -> List[str]:
        """Collect all the rephrased sentences from a model output.

        :return: List of rephrased sentences.
        """
        rephrases = []
        for sentence in self.sentence_list:
            for new_sentence in sentence.new:
                if len(new_sentence) > 0:
                    rephrases.append(new_sentence)
        return rephrases

    def pretty_print_rephrased(self, print_fn: Callable[[str], None] = print) -> None:
        """Pretty print the rephrased sentences."""
        for sentence in self.sentence_list:
            print_fn(f"{sentence.orig}")
            for new_sentence in sentence.new:
                print_fn(f"  -->\t{new_sentence}")

    def get_sentence_mappings(self) -> Dict[str, List[str]]:
        """Return a dictionary mapping original sentences to rephrased sentences.

        :return: Dictionary with keys being original sentences and values corresponding to rephrased sentences.
        """
        sentence_mappings = {x.orig: x.new for x in self.sentence_list}
        return sentence_mappings

    def to_grounded_phrases_list(self, rephrased: bool = True) -> GroundedPhraseList:
        """Convert the parsed report to a `GroundedPhraseList`. Specifically a list of `GroundedPhrase` objects.
        If rephrased (default), we use the 'new' phrases. Otherwise we use 'orig'.
        """
        sequence = GroundedPhraseList()
        if rephrased:
            rephrased_sentences = self.phrases_as_list()
            for sentence in rephrased_sentences:
                sequence.append(GroundedPhrase(text=sentence))
        else:
            for sentence_with_rephrases in self.sentence_list:
                sequence.append(GroundedPhrase(text=sentence_with_rephrases.orig))
        return sequence


class PhraseParsingExample(BaseModel):
    """Dataclass for a single example."""

    example_id: int | str
    findings_text: str
    parsed_report: ParsedReport | None = None
    study_id: str | None = None
    example_rationale: str | None = None

    @property
    def input(self) -> str:
        return self.findings_text

    @property
    def output(self) -> ParsedReport | None:
        return self.parsed_report

    @root_validator
    @classmethod
    def no_unnecessarily_duplicated_sentences(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Make sure the same 'orig' sentence doesn't appear twice."""
        if values["parsed_report"] is None:
            # Nothing to do here
            return values
        sentence_list = values["parsed_report"].sentence_list
        findings_text = values["findings_text"]
        duplicated_sentences = [sentence.orig for sentence in sentence_list if sentence_list.count(sentence) > 1]
        # A duplicated sentence is acceptable if it also appears in the original report twice.
        real_duplications = []
        for duplication_candidate in duplicated_sentences:
            if findings_text.count(duplication_candidate) == 1:
                real_duplications.append(duplication_candidate)
        if len(real_duplications) > 0:
            raise ValueError(
                "Duplicate sentences found in ParsedReport."
                f"Duplicated sentences: {real_duplications}. Original report: {findings_text}."
            )
        return values


def load_examples_from_json(file_path: Path) -> list[PhraseParsingExample]:
    """
    Given a path to a json file, load the examples into a list of PhraseParsingExample objects.

    This is implemented to be "backwards compatible" with the old json format, where the parsed_report was a list of
    strings.
    """
    if file_path.suffix != ".json":
        file_path = file_path.with_suffix(".json")
    examples = json.load(open(file_path, "r", encoding="utf-8"))
    examples_list: list[PhraseParsingExample] = []
    for example in examples:
        parsed_example = PhraseParsingExample.parse_obj(example)
        examples_list.append(parsed_example)
    return examples_list
