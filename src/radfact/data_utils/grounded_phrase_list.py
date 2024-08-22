#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Generic, Iterator, Mapping, TypeVar

TypeT = TypeVar("TypeT", float, int)


@dataclass(frozen=True)
class GenericBox(Generic[TypeT]):
    """Bounding box class with coordinates of type TypeT. Allows for looping and unpacking of coordinates."""

    x_min: TypeT
    y_min: TypeT
    x_max: TypeT
    y_max: TypeT

    def __post_init__(self) -> None:
        if not 0 <= self.x_min <= self.x_max:
            raise ValueError(f"Invalid x coordinates: {self}")
        if not 0 <= self.y_min <= self.y_max:
            raise ValueError(f"Invalid y coordinates: {self}")

    def __iter__(self) -> Iterator[TypeT]:
        yield from (self.x_min, self.y_min, self.x_max, self.y_max)

    def has_zero_area(self) -> bool:
        return self.x_min == self.x_max or self.y_min == self.y_max


@dataclass(frozen=True)
class NormalizedBox(GenericBox[float]):
    """Bounding box normalized to the image size, with coordinates in the range [0, 1]."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.x_max <= 1:
            raise ValueError(f"Invalid x coordinates: {self}")
        if not self.y_max <= 1:
            raise ValueError(f"Invalid y coordinates: {self}")


BoxDictType = dict[str, float]
GroundedPhraseDictType = Mapping[str, str | list[BoxDictType]]


@dataclass(frozen=True)
class GroundedPhrase:
    """A grounded phrase consists of a string with an (optional) list of normalized bounding boxes."""

    text: str
    boxes: list[NormalizedBox] | None = None

    def __post_init__(self) -> None:
        if self.boxes is not None and len(self.boxes) == 0:
            raise ValueError(f"Empty boxes for grounded text: {self}, this should be set to None")

    @classmethod
    def from_dict(cls, grounded_phrase_dict: GroundedPhraseDictType) -> GroundedPhrase:
        text = grounded_phrase_dict["text"]
        if not isinstance(text, str):
            raise ValueError(f"text is not a string: {text}")
        box_list = grounded_phrase_dict["boxes"]
        if box_list is None:
            return cls(text=text, boxes=None)
        if isinstance(box_list, list):
            return cls(text=text, boxes=[NormalizedBox(**box) for box in box_list])
        else:
            raise ValueError(f"boxes is not a list: {box_list}")


GroundedPhraseListType = list[str | NormalizedBox | GroundedPhrase]

GroundedPhraseListDictType = list[Mapping[str, str | BoxDictType | list[BoxDictType]]]


class GroundedPhraseList(GroundedPhraseListType):
    def get_all_text(self, sep: str = " ") -> str:
        """Extract all text segments from the sequence as a continuous string.

        :param sep: Separator between joined substrings, defaults to " ".
        :return: Single string containing whitespace-stripped text segments joined by `sep`.
        """
        text_parts = []
        for part in self:
            if isinstance(part, str):
                text_parts.append(part.strip())
            elif isinstance(part, GroundedPhrase):
                text_parts.append(part.text.strip())
        return sep.join(text_parts)

    def get_all_boxes(self, fail_if_non_box: bool = False) -> list[NormalizedBox]:
        """Extract all bounding boxes from the sequence as a single list."""
        box_list = []
        for part in self:
            if isinstance(part, NormalizedBox):
                box_list.append(part)
            elif fail_if_non_box:
                raise ValueError(f"Encountered a non-box while extracting boxes: {part}")
            if isinstance(part, GroundedPhrase) and part.boxes is not None:
                box_list.extend(part.boxes)

        return box_list

    def get_all_grounded_phrases(self, fail_if_non_grounded_phrase: bool = False) -> list[GroundedPhrase]:
        """Extract all GroundedPhrase from the sequence as a single list.

        If there are any non-GroundedPhrase in the sequence, it will raise a ValueError if fail_if_non_phrase is True.
        This can occur if we expect a sequence to contain only GroundedPhrase.
        """
        phrases = [part for part in self if isinstance(part, GroundedPhrase)]
        if fail_if_non_grounded_phrase and len(phrases) != len(self):
            raise ValueError(f"Encountered a non-GroundedPhrase while extracting phrases: {self}")
        return phrases

    def to_list_of_dicts(self) -> GroundedPhraseListDictType:
        """Convert the sequence to a list of dictionaries."""

        list_of_dicts: GroundedPhraseListDictType = []
        for part in self:
            if isinstance(part, str):
                list_of_dicts.append({"text": part})
            elif isinstance(part, NormalizedBox):
                box_as_dict: dict[str, float] = dataclasses.asdict(part)
                list_of_dicts.append({"box": box_as_dict})
            elif isinstance(part, GroundedPhrase):
                list_of_dicts.append(dataclasses.asdict(part))
            else:
                raise ValueError(f"Unknown member of grounded phrase list: {part}")
        return list_of_dicts

    @classmethod
    def from_list_of_dicts(cls, list_of_dicts: GroundedPhraseListDictType) -> GroundedPhraseList:
        """Convert a list of dictionaries to a grounded phrase list.

        :param list_of_dicts: List of dictionaries.
        """
        if not isinstance(list_of_dicts, list):
            raise ValueError(f"Expected list of dictionaries, got: {list_of_dicts}")
        grounded_phrase_list: GroundedPhraseListType = []
        for part in list_of_dicts:
            if not isinstance(part, dict):
                raise ValueError(f"Expected dictionary, got: {part}")
            part_keys = part.keys()
            if part_keys == {"text"}:
                assert isinstance(part["text"], str), f"Expected string, got: {part['text']}"
                grounded_phrase_list.append(part["text"])
            elif part_keys == {"box"}:
                box = part["box"]
                grounded_phrase_list.append(NormalizedBox(**box))
            elif part_keys == {"text", "boxes"}:
                grounded_phrase_list.append(GroundedPhrase.from_dict(part))
            else:
                raise ValueError(f"Unknown member of grounded phrase list: {part}")
        return cls(grounded_phrase_list)
