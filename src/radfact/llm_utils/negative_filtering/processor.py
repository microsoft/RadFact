#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
from radfact.llm_utils.prompt_tasks import NegativeFilteringTaskOptions, ReportType
from omegaconf import DictConfig

from radfact.llm_utils.engine.engine import LLMEngine, get_subfolder
from radfact.llm_utils.processor.structured_processor import StructuredProcessor, parse_examples_from_json
from radfact.llm_utils.report_to_phrases.schema import (
    ParsedReport,
    PhraseList,
    PhraseListExample,
    SentenceWithRephrases,
)
from radfact.paths import OUTPUT_DIR

NEGATIVE_FILTERING_SUBFOLDER = "negative_report_filtering"
ORIG = "orig"
NEW = "new"


def get_negative_filtering_phrase_processor(
    report_type: ReportType, log_dir: Path | None = None
) -> StructuredProcessor[list[str], PhraseList]:
    """Return a processor for filtering negative findings from a list of phrases.

    :param report_type: The type of report, e.g., "ReportType.CXR" or "ReportType.CT".
    :param log_dir: The directory to save logs.
    :return: The processor for negative finding filtering.
    """
    task = NegativeFilteringTaskOptions[report_type.name].value
    system_prompt = task.system_message_path.read_text()
    few_shot_examples = parse_examples_from_json(task.few_shot_examples_path, PhraseListExample)
    processor = StructuredProcessor(
        query_type=list[str],
        result_type=PhraseList,
        system_prompt=system_prompt,
        format_query_fn=lambda x: json.dumps(x),
        few_shot_examples=few_shot_examples,
        log_dir=log_dir,
    )
    return processor


def load_filtering_queries_from_parsed_reports(
    reports: list[ParsedReport],
    index_col: str,
) -> pd.DataFrame:
    """
    Load queries for filtering from a list of parsed reports. Queries consist of all the
    newly parsed phrases from phrasification, along with metadata including the study ID
    and original phrase.
    :param reports: A list of ParsedReport objects.
    :param index_col: The column containing the index
    :return: A list of queries.
    """
    queries = []
    for report in reports:
        for i, sentence in enumerate(report.sentence_list):
            queries.append([f"{report.id}_{i}", sentence.orig, sentence.new])
    query_df = pd.DataFrame(queries, columns=[index_col, ORIG, NEW])
    return query_df


def get_negative_filtering_engine(
    cfg: DictConfig, parsed_reports: list[ParsedReport], subfolder_prefix: str, report_type: ReportType
) -> LLMEngine:
    """
    Create the processing engine for filtering negative findings from parsed reports.

    :param cfg: The configuration for the processing engine.
    :param parsed_reports: A list of ParsedReport objects to filter.
    :param subfolder_prefix: The prefix for the metric folder
    :param report_type: The type of report, e.g., CT.
    :return: The processing engine.
    """
    OUTPUT_FOLDER = OUTPUT_DIR / NEGATIVE_FILTERING_SUBFOLDER
    output_folder = get_subfolder(OUTPUT_FOLDER, subfolder_prefix)
    final_output_folder = get_subfolder(OUTPUT_FOLDER, subfolder_prefix)
    log_dir = get_subfolder(OUTPUT_FOLDER, "logs")

    query_df = load_filtering_queries_from_parsed_reports(parsed_reports, cfg.processing.index_col)
    negative_filtering_processor = get_negative_filtering_phrase_processor(report_type=report_type, log_dir=log_dir)

    engine = LLMEngine(
        cfg=cfg,
        processor=negative_filtering_processor,
        dataset_df=query_df,
        row_to_query_fn=lambda row: row[NEW],
        progress_output_folder=output_folder,
        final_output_folder=final_output_folder,
    )
    return engine


def process_filtered_reports(engine: LLMEngine, cfg: DictConfig) -> tuple[list[ParsedReport], int]:
    """
    Process the filtered reports using the provided engine.

    :param engine: The LLMEngine used for processing.
    :param cfg: The configuration for negative filtering processing.
    :return: A tuple containing a list of ParsedReport objects and the number of rewritten sentences.
    """
    outputs = engine.return_raw_outputs
    metadata = engine.return_dataset_subsets

    parsed_report_dict = defaultdict(list)
    num_rewritten_sentences = 0

    for k in outputs.keys():
        phrase_list = outputs[k]
        metadata_df = metadata[k].df

        for idx, row in metadata_df.iterrows():
            study_id = row[cfg.processing.index_col].rsplit("_", 1)[0]
            orig = row[ORIG]
            unfiltered_phrases = set(row[NEW])
            filtered_phrases = set(phrase_list[idx].phrases)

            if not filtered_phrases.issubset(unfiltered_phrases):
                rewritten_phrases = filtered_phrases - unfiltered_phrases
                print(
                    f"New phrases {rewritten_phrases} not in original phrases {unfiltered_phrases}. Reverting back to original phrases."
                )
                filtered_phrases = unfiltered_phrases
                num_rewritten_sentences += 1

            parsed_report_dict[study_id].append(SentenceWithRephrases(orig=orig, new=list(filtered_phrases)))

    parsed_reports = [
        ParsedReport(id=study_id, sentence_list=sentences) for study_id, sentences in parsed_report_dict.items()
    ]
    return parsed_reports, num_rewritten_sentences
