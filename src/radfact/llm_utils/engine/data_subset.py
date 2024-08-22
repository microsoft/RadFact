#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataSubset:
    """Class to represent a subset of a dataset. Includes methods to save progress and skipped IDs."""

    df: pd.DataFrame
    start_index: int
    end_index: int
    index_col: str
    output_folder: Path
    processed_ids: set[str] = field(default_factory=set)
    skipped_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        assert (
            len(self.df) == self.end_index - self.start_index
        ), f"Dataframe {len(self.df)=} does not match {(self.end_index - self.start_index)=}"
        self.progress_folder = self.output_folder / "progress"
        self.progress_folder.mkdir(parents=True, exist_ok=True)
        self.skip_folder = self.output_folder / "skipped"
        self.skip_folder.mkdir(parents=True, exist_ok=True)

    @property
    def filename(self) -> str:
        """Return the file stem for this subset of the dataset."""
        return f"subset_{self.start_index}_{self.end_index}.csv"

    @property
    def progress_file(self) -> Path:
        """Status file for this subset of the dataset saving the progress so far."""
        return self.progress_folder / self.filename

    @property
    def skipped_file(self) -> Path:
        """File for IDs that have been skipped."""
        return self.skip_folder / self.filename

    @property
    def relative_progress(self) -> float:
        """Return the relative progress of this subset of the dataset."""
        return len(self.processed_ids) / len(self.df)

    @property
    def progress_stats(self) -> dict[str, str | float]:
        """Return the name of the progress metric for this subset of the dataset."""
        return {"name": f"progress_{self.start_index}_{self.end_index}", "value": self.relative_progress}

    @property
    def skipped_stats(self) -> dict[str, str | int]:
        """Return the name of the skipped metric for this subset of the dataset."""
        return {"name": f"skipped_{self.start_index}_{self.end_index}", "value": len(self.skipped_ids)}

    @property
    def indices(self) -> set[str]:
        """Return the indices of this subset of the dataset."""
        return set(self.df[self.index_col])

    def save_progress(self) -> None:
        """Save progress to progress file"""
        if len(self.processed_ids) > 0:
            processed_df = pd.DataFrame({self.index_col: list(self.processed_ids)})
            processed_df.to_csv(self.progress_file, index=False)

    def save_skipped(self) -> None:
        """Save skipped IDs to skipped file"""
        if len(self.skipped_ids) > 0:
            skipped_df = pd.DataFrame({self.index_col: list(self.skipped_ids)})
            skipped_df.to_csv(self.skipped_file, index=False)

    def __len__(self) -> int:
        return len(self.df)
