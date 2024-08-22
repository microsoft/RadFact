#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from radfact.llm_utils.engine.engine import LLMEngine, get_subfolder
from radfact.llm_utils.processor.structured_processor import StructuredProcessor
from radfact.llm_utils.report_to_phrases.schema import ParsedReport, load_examples_from_json
from radfact.paths import OUTPUT_DIR, get_prompts_dir

FINDINGS_SECTION = "FINDINGS"
PARSING_TASK = "report_to_phrases"
PROMPTS_DIR = get_prompts_dir(task=PARSING_TASK)
StudyIdType = str | int


def get_report_to_phrases_processor(log_dir: Path | None = None) -> StructuredProcessor[str, ParsedReport]:
    """Return a processor for converting reports to phrases.

    :param log_dir: The directory to save logs.
    :return: The processor for report to phrase conversion.
    """
    system_message_path = PROMPTS_DIR / "system_message.txt"
    few_shot_examples_path = PROMPTS_DIR / "few_shot_examples.json"
    system_prompt = system_message_path.read_text()
    few_shot_examples = load_examples_from_json(few_shot_examples_path)
    processor = StructuredProcessor(
        query_type=str,
        result_type=ParsedReport,
        system_prompt=system_prompt,
        format_query_fn=lambda x: x,  # Our query is simply the findings text
        few_shot_examples=few_shot_examples,  # type: ignore[arg-type]
        log_dir=log_dir,
    )
    return processor


def get_findings_from_row(row: "pd.Series[Any]") -> str:
    """Get the findings from a row in a DataFrame."""
    findings = row[FINDINGS_SECTION]
    assert isinstance(findings, str), f"Findings should be a string, got {findings}"
    return findings


def get_report_to_phrases_engine(cfg: DictConfig, dataset_df: pd.DataFrame) -> LLMEngine:
    """
    Create the processing engine for converting reports to phrases.

    :param cfg: The configuration for the processing engine.
    :param dataset_df: The dataset DataFrame.
    :param subfolder: The subfolder to save the processing output.
    :return: The processing engine.
    """
    subfolder = cfg.dataset.name
    root = OUTPUT_DIR / PARSING_TASK
    output_folder = get_subfolder(root, subfolder)
    final_output_folder = get_subfolder(root, subfolder)
    log_dir = get_subfolder(root, "logs")

    report_to_phrases_processor = get_report_to_phrases_processor(log_dir=log_dir)
    id_col = cfg.processing.index_col
    dataset_df = dataset_df[[id_col, FINDINGS_SECTION]]
    engine = LLMEngine(
        cfg=cfg,
        processor=report_to_phrases_processor,
        dataset_df=dataset_df,
        progress_output_folder=output_folder,
        final_output_folder=final_output_folder,
        row_to_query_fn=get_findings_from_row,
    )
    return engine
