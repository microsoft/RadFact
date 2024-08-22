#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from radfact.llm_utils.text_utils import find_best_match


def test_find_best_match() -> None:
    text_options = ["There is a small pleural effusion", "No acute cardiopulmonary process", "Widespread consolidation"]

    # Case 1: Exact match, with or without normalisation
    text = "There is a small pleural effusion"
    index, match = find_best_match(text, text_options)
    assert index == 0
    assert match == "There is a small pleural effusion"

    # Case 2: Normalisation
    text = "there is a small pleural effusion."
    index, match = find_best_match(text, text_options)
    assert index == 0
    assert match == "There is a small pleural effusion"

    # Case 3: Not exact match. Use the longest common substring to select.
    text = "No acute cardio"
    index, match = find_best_match(text, text_options)
    assert index == 1
    assert match == "No acute cardiopulmonary process"

    # Case 4: Not exact match. Use the longest common substring to select, but text is longer.
    text = "There is consolidation seen throughout the lungs."
    index, match = find_best_match(text, text_options)
    assert index == 2
    assert match == "Widespread consolidation"
