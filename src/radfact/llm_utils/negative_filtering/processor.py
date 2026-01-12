#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
from radfact.llm_utils.report_to_phrases.processor import StudyIdType
from radfact.llm_utils.prompt_tasks import NegativeFilteringTaskOptions, ReportType
from omegaconf import DictConfig

from radfact.llm_utils.engine.engine import LLMEngine, get_subfolder
from radfact.llm_utils.processor.structured_processor import StructuredProcessor, parse_examples_from_json
from radfact.llm_utils.report_to_phrases.schema import ParsedReport, Rephrases, RephrasesExample, SentenceWithRephrases
from radfact.paths import OUTPUT_DIR


def get_negative_filtering_phrase_processor(
    report_type: ReportType, log_dir: Path | None = None
) -> StructuredProcessor[list[str], Rephrases]:
    """Return a processor for filtering negative findings from a list of phrases.

    :param report_type: The type of report, e.g., "ReportType.CXR" or "ReportType.CT".
    :param log_dir: The directory to save logs.
    :return: The processor for negative finding filtering.
    """
    task = NegativeFilteringTaskOptions[report_type.name].value
    system_prompt = task.system_message_path.read_text()
    few_shot_examples = parse_examples_from_json(task.few_shot_examples_path, RephrasesExample)
    processor = StructuredProcessor(
        query_type=list[str],
        result_type=Rephrases,
        system_prompt=system_prompt,
        format_query_fn=lambda x: json.dumps(x),
        few_shot_examples=few_shot_examples,
        log_dir=log_dir,
    )
    return processor


def load_filtering_queries_from_parsed_reports(
    reports: list[ParsedReport],
) -> pd.DataFrame:
    """
    Load queries for filtering from a list of parsed reports. Queries consist of all the
    newly parsed phrases from phrasification, along with metadata including the study ID
    and original phrase.
    :param reports: A list of ParsedReport objects.
    :return: A list of queries.
    """
    queries = []
    report_ids: dict[StudyIdType, int] = defaultdict(int)
    for report in reports:
        for sentence in report.sentence_list:
            assert report.id is not None
            queries.append([f"{str(report.id)}_{report_ids[report.id]}", sentence.orig, sentence.new])
            report_ids[str(report.id)] += 1
    query_df = pd.DataFrame(queries, columns=["study_id", "orig", "new_phrases"])
    return query_df


def get_negative_filtering_engine(cfg: DictConfig, parsed_reports: list[ParsedReport]) -> LLMEngine:
    """
    Create the processing engine for filtering negative findings from parsed reports.

    :param cfg: The configuration for the processing engine.
    :return: The processing engine.
    """
    subfolder = cfg.dataset.name
    root = OUTPUT_DIR / "negative_report_filtering"
    output_folder = get_subfolder(root, subfolder)
    final_output_folder = get_subfolder(root, subfolder)
    log_dir = get_subfolder(root, "logs")

    report_type_value = cfg.get("report_type")
    try:
        report_type = ReportType(report_type_value)
    except ValueError as e:
        raise ValueError(
            f"Invalid report_type '{report_type_value}'. Valid options are: {[rt.value for rt in ReportType]}"
        ) from e

    query_df = load_filtering_queries_from_parsed_reports(parsed_reports)
    negative_filtering_processor = get_negative_filtering_phrase_processor(report_type=report_type, log_dir=log_dir)

    engine = LLMEngine(
        cfg=cfg,
        processor=negative_filtering_processor,
        dataset_df=query_df,
        row_to_query_fn=lambda row: row["new_phrases"],
        progress_output_folder=output_folder,
        final_output_folder=final_output_folder,
    )
    return engine


def process_filtered_reports(engine: LLMEngine) -> tuple[list[ParsedReport], int]:
    """
    Process the filtered reports using the provided engine.

    :param engine: The LLMEngine used for processing.
    :return: A tuple containing a list of ParsedReport objects and the number of rewritten sentences.
    """
    outputs = engine.return_raw_outputs
    metadata = engine.return_dataset_subsets

    parsed_report_dict = defaultdict(list)
    num_rewritten_sentences = 0
    for k in outputs.keys():
        rephrases = outputs[k]
        metadata_df = metadata[k].df

        for idx, row in metadata_df.iterrows():
            study_id = row["study_id"].split("_")[0]
            orig = row["orig"]
            unfiltered_phrases = set(row["new_phrases"])
            filtered_phrases = set(rephrases[idx].new)

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
