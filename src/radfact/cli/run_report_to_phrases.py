#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from radfact.llm_utils.report_to_phrases.processor import FINDINGS_SECTION, get_report_to_phrases_engine
from radfact.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)
http_logger = logging.getLogger("httpx")
http_logger.setLevel(logging.WARNING)


def _validate_column(df: pd.DataFrame, col_name: str) -> None:
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in dataset. Available columns: {df.columns}")


def get_dataset_dataframe(cfg: DictConfig) -> pd.DataFrame:
    """Read the dataset dataframe and drop duplicates based on the index column and findings column."""
    dataset_dataframe_path = cfg.dataset.csv_path
    dataset_name = cfg.dataset.name
    df = pd.read_csv(dataset_dataframe_path)
    findings_col = FINDINGS_SECTION
    _validate_column(df, findings_col)
    id_col = cfg.processing.index_col
    _validate_column(df, id_col)
    df.drop_duplicates(subset=[id_col, findings_col], inplace=True)
    logger.info(f"Loaded {len(df)} rows from {dataset_name} dataset")
    df.dropna(subset=[findings_col], inplace=True)
    logger.info(f"Processing {len(df)} rows with non-null findings")
    return df


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="report_to_phrases")
def main(cfg: DictConfig) -> None:
    dataset_df = get_dataset_dataframe(cfg)
    engine = get_report_to_phrases_engine(cfg=cfg, dataset_df=dataset_df)
    engine.run()


if __name__ == "__main__":
    main()
