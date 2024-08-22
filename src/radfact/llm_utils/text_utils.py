#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import re
from difflib import SequenceMatcher


logger = logging.getLogger(__name__)


def normalise_text_for_comparison(text: str) -> str:
    """
    Normalise a string for comparison by removing whitespace, dashes, and punctuation.

    :param text: The text to normalise.
    :return: The normalised text.
    """
    text = text.lower().strip()  # Normalise case and trim whitespace
    text = re.sub(r"\s+", " ", text)  # Remove duplicated spaces
    text = re.sub(r"[\-–—]", "", text)  # Remove dashes
    text = re.sub(r"\s*(?<=[\.\:\!\?])", "", text)  # Remove spaces before punctuation
    text = re.sub(r"[\.\:\!\?]$", "", text)  # Remove final punctuation
    return text


def find_best_match(text: str, candidate_texts: list[str]) -> tuple[int, str]:
    """
    Given "text" and a list of possible matches ("candidate_texts"), return the
    index of the best match and the match itself. We assume there is a match.

    :param text: The text to match.
    :param candidate_texts: The list of candidate texts to match against.
    :return: A tuple of the index of the best match and the match itself.
    """
    candidates_normalised = [normalise_text_for_comparison(candidate) for candidate in candidate_texts]
    text_normalised = normalise_text_for_comparison(text)

    # Case 1: Exact match, with or without normalisation
    for i, candidate in enumerate(candidates_normalised):
        if candidate == text_normalised:
            return i, candidate_texts[i]

    # Case 2: Not exact match. Use the longest common substring to select.
    logger.info("No good match! Using substring matching...")
    best_match = ""
    match_index = -1
    best_substring_length = 0
    for i, candidate in enumerate(candidates_normalised):
        substring_match = SequenceMatcher(None, text_normalised, candidate).find_longest_match(
            0, len(text_normalised), 0, len(candidate)
        )
        substring = text[substring_match.a : substring_match.a + substring_match.size]
        if len(substring) > best_substring_length:
            best_match = candidate_texts[i]
            match_index = i
            best_substring_length = len(substring)
    assert match_index != -1, f"Failed to find a match for {text} in {candidate_texts}."
    return match_index, best_match
