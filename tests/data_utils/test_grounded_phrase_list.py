#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from radfact.data_utils.grounded_phrase_list import GroundedPhrase, GroundedPhraseList, NormalizedBox


def test_box() -> None:
    box = NormalizedBox(0.1, 0.2, 0.3, 0.4)
    assert box.x_min == 0.1
    assert box.y_min == 0.2
    assert box.x_max == 0.3
    assert box.y_max == 0.4
    assert tuple(box) == (0.1, 0.2, 0.3, 0.4)


def test_box_invalid_coords() -> None:
    for invalid_box_coords in [
        (-0.1, 0.2, 0.3, 0.4),  # x_min < 0
        (0.6, 0.2, 0.3, 0.4),  # x_min > x_max
        (0.1, 0.2, 1.3, 0.4),  # x_max > 1
        (0.1, -0.2, 0.3, 0.4),  # y_min < 0
        (0.1, 0.6, 0.8, 0.4),  # y_min > y_max
        (0.1, 0.2, 0.3, 1.4),  # y_max > 1
    ]:
        with pytest.raises(ValueError, match="Invalid . coordinates"):
            NormalizedBox(*invalid_box_coords)


def test_box_has_zero_area() -> None:
    assert NormalizedBox(0, 0, 0, 0).has_zero_area()
    assert NormalizedBox(0, 0, 0, 0.5).has_zero_area()
    assert NormalizedBox(0, 0, 0.5, 0).has_zero_area()
    assert not NormalizedBox(0, 0, 0.5, 0.5).has_zero_area()


def test_get_all_text() -> None:
    grounded_phrase_list = GroundedPhraseList(
        [
            "Plain str ",  # Note there is whitespace at the end
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4)],
            ),
        ]
    )
    assert grounded_phrase_list.get_all_text() == "Plain str Grounded str"
    assert grounded_phrase_list.get_all_text(sep="|") == "Plain str|Grounded str"


def test_get_all_boxes() -> None:
    grounded_phrase_list = GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4)],
            ),
        ]
    )
    assert grounded_phrase_list.get_all_boxes() == [NormalizedBox(0.1, 0.2, 0.3, 0.4)]
    with pytest.raises(ValueError, match="Encountered a non-box while extracting boxes"):
        grounded_phrase_list.get_all_boxes(fail_if_non_box=True)

    grounded_phrase_list_no_boxes = GroundedPhraseList([GroundedPhrase(text="Phrase str with no box", boxes=None)])
    assert grounded_phrase_list_no_boxes.get_all_boxes() == []
    with pytest.raises(ValueError, match="Encountered a non-box while extracting boxes"):
        assert grounded_phrase_list_no_boxes.get_all_boxes(fail_if_non_box=True) == []

    box_only_sequence = GroundedPhraseList([NormalizedBox(0.1, 0.2, 0.3, 0.4)])
    assert box_only_sequence.get_all_boxes() == [NormalizedBox(0.1, 0.2, 0.3, 0.4)]
    assert box_only_sequence.get_all_boxes(fail_if_non_box=True) == [NormalizedBox(0.1, 0.2, 0.3, 0.4)]


def test_get_all_grounded_phrases() -> None:
    grounded_phrase_list = GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4)],
            ),
        ]
    )
    assert grounded_phrase_list.get_all_grounded_phrases() == [
        GroundedPhrase(text="Grounded str", boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4)])
    ]
    with pytest.raises(ValueError, match="Encountered a non-GroundedPhrase while extracting phrases"):
        grounded_phrase_list.get_all_grounded_phrases(fail_if_non_grounded_phrase=True)

    phrase_only_sequence = GroundedPhraseList([GroundedPhrase(text="Phrase str with no box", boxes=None)])
    assert phrase_only_sequence.get_all_grounded_phrases() == [
        GroundedPhrase(text="Phrase str with no box", boxes=None)
    ]
    assert phrase_only_sequence.get_all_grounded_phrases(fail_if_non_grounded_phrase=True) == [
        GroundedPhrase(text="Phrase str with no box", boxes=None)
    ]


def test_grounded_phrase_from_dict() -> None:
    grounded_phrase_dict = {
        "text": "Grounded str",
        "boxes": [
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
        ],
    }
    grounded_phrase = GroundedPhrase.from_dict(grounded_phrase_dict)  # type: ignore
    assert grounded_phrase.text == "Grounded str"
    assert grounded_phrase.boxes == [NormalizedBox(0.1, 0.2, 0.3, 0.4)]

    grounded_phrase_dict = {
        "text": "Grounded str",
        "boxes": [
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
            {"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8},
        ],
    }
    grounded_phrase = GroundedPhrase.from_dict(grounded_phrase_dict)  # type: ignore
    assert grounded_phrase.text == "Grounded str"
    assert grounded_phrase.boxes == [NormalizedBox(0.1, 0.2, 0.3, 0.4), NormalizedBox(0.5, 0.6, 0.7, 0.8)]

    # malformed dicts
    with pytest.raises(KeyError, match="text"):
        GroundedPhrase.from_dict({"boxes": []})
    with pytest.raises(KeyError, match="boxes"):
        GroundedPhrase.from_dict({"text": ""})
    with pytest.raises(ValueError, match="boxes is not a list:"):
        GroundedPhrase.from_dict({"text": "", "boxes": {}})  # type: ignore


def test_grounded_phrase_list_to_dict() -> None:
    grounded_phrase_list = GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Ungrounded str",
                boxes=None,
            ),
        ]
    )
    assert grounded_phrase_list.to_list_of_dicts() == [
        {
            "text": "Plain str",
        },
        {
            "text": "Ungrounded str",
            "boxes": None,
        },
    ]

    grounded_phrase_list = GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4)],
            ),
        ]
    )
    assert grounded_phrase_list.to_list_of_dicts() == [
        {
            "text": "Plain str",
        },
        {
            "text": "Grounded str",
            "boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
            ],
        },
    ]

    grounded_phrase_list = GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4), NormalizedBox(0.5, 0.6, 0.7, 0.8)],
            ),
        ]
    )

    assert grounded_phrase_list.to_list_of_dicts() == [
        {
            "text": "Plain str",
        },
        {
            "text": "Grounded str",
            "boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                {"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8},
            ],
        },
    ]


def test_grounded_phrase_list_from_dict() -> None:
    list_of_dicts = [
        {
            "text": "Plain str",
        },
        {
            "text": "Ungrounded str",
            "boxes": None,
        },
    ]
    grounded_phrase_list = GroundedPhraseList.from_list_of_dicts(list_of_dicts)  # type: ignore
    assert grounded_phrase_list == GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Ungrounded str",
                boxes=None,
            ),
        ]
    )

    list_of_dicts = [
        {
            "text": "Plain str",
        },
        {
            "text": "Grounded str",
            "boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
            ],
        },
    ]
    grounded_phrase_list = GroundedPhraseList.from_list_of_dicts(list_of_dicts)  # type: ignore
    assert grounded_phrase_list == GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4)],
            ),
        ]
    )

    list_of_dicts = [
        {
            "text": "Plain str",
        },
        {
            "text": "Grounded str",
            "boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                {"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8},
            ],
        },
    ]
    grounded_phrase_list = GroundedPhraseList.from_list_of_dicts(list_of_dicts)  # type: ignore
    assert grounded_phrase_list == GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4), NormalizedBox(0.5, 0.6, 0.7, 0.8)],
            ),
        ]
    )

    list_of_dicts = [
        {
            "text": "Plain str",
        },
        {
            "text": "Grounded str",
            "boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                {"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8},
            ],
        },
    ]

    grounded_phrase_list = GroundedPhraseList.from_list_of_dicts(list_of_dicts)  # type: ignore
    assert grounded_phrase_list == GroundedPhraseList(
        [
            "Plain str",
            GroundedPhrase(
                text="Grounded str",
                boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4), NormalizedBox(0.5, 0.6, 0.7, 0.8)],
            ),
        ]
    )

    # Malformed sequences
    list_of_dicts = [
        "Plain str",
        GroundedPhrase(
            text="Grounded str",
            boxes=[NormalizedBox(0.1, 0.2, 0.3, 0.4)],
        ),
    ]

    with pytest.raises(ValueError, match="Expected dictionary, got:"):
        GroundedPhraseList.from_list_of_dicts(list_of_dicts)  # type: ignore
    list_of_dicts = [
        {
            "text": "Grounded str",
            "boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
            ],
            "other_key": True,
        },
    ]
    with pytest.raises(ValueError, match="Unknown member of grounded phrase list"):
        GroundedPhraseList.from_list_of_dicts(list_of_dicts)  # type: ignore
