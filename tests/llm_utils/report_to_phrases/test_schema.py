#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from radfact.data_utils.grounded_phrase_list import GroundedPhrase, GroundedPhraseList
from radfact.llm_utils.report_to_phrases.schema import ParsedReport, SentenceWithRephrases


def test_to_grounded_phrases_list() -> None:
    # Create a ParsedReport instance
    sentence_with_rephrases = ParsedReport(
        sentence_list=[
            SentenceWithRephrases(orig="Original sentence", new=["Rephrased sentence 1", "Rephrased sentence 2"]),
            SentenceWithRephrases(orig="Another original sentence", new=["Another rephrased sentence"]),
        ]
    )

    # Convert ParsedReport to GroundedPhraseList
    sequence = sentence_with_rephrases.to_grounded_phrases_list()

    expected_sequence = GroundedPhraseList(
        [
            GroundedPhrase(text="Rephrased sentence 1"),
            GroundedPhrase(text="Rephrased sentence 2"),
            GroundedPhrase(text="Another rephrased sentence"),
        ]
    )

    assert sequence == expected_sequence
