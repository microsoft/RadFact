#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from omegaconf import DictConfig
from pydantic import ValidationError

from radfact.data_utils.grounded_phrase_list import GroundedPhraseList
from radfact.llm_utils.engine.engine import LLMEngine, get_subfolder
from radfact.llm_utils.nli.schema import (
    ComparisonQuerySinglePhrase,
    DirectionOptions,
    EvidencedPhrase,
    EVState,
    NLIQuerySample,
    NLIQuerySampleSinglePhrase,
    NLISample,
    NLISampleSinglePhrase,
    load_examples_from_json,
)
from radfact.llm_utils.processor.base_processor import BaseProcessor
from radfact.llm_utils.processor.structured_processor import (
    FormatStyleOptions,
    ProcessorStats,
    StructuredProcessor,
    simple_formatter,
)
from radfact.paths import OUTPUT_DIR, get_prompts_dir

logger = logging.getLogger(__name__)
PARSING_TASK = "nli"
PROMPTS_DIR = get_prompts_dir(task=PARSING_TASK)
RADFACT_SUBFOLDER = "radfact"


class MetricDataframeKeys(str, Enum):
    CANDIDATE = "candidate"
    REFERENCE = "reference"
    STUDY_ID = "study_id"


def get_ev_processor_singlephrase(log_dir: Path) -> StructuredProcessor[ComparisonQuerySinglePhrase, EvidencedPhrase]:
    """
    Helper function to load the NLI processor with the correct system prompt and few-shot examples.

    The setting here is to classify a SINGLE PHRASE at a time given the reference report.
    Further, we do entailment verification, aka the binary version of NLI.

    :param api_arguments: API arguments for the LLM.
    :param log_dir: Directory to save logs.
    :return: Processor for entailment verification.
    """

    system_prompt_path = PROMPTS_DIR / "system_message_ev_singlephrase.txt"
    few_shot_examples_path = PROMPTS_DIR / "few_shot_examples.json"
    system_prompt = system_prompt_path.read_text()
    few_shot_examples = load_examples_from_json(json_path=few_shot_examples_path, binary=True)
    # The few-shots are in the bidirectional format, we need to convert them to single-phrase.
    few_shot_examples_single_phrase: list[NLISampleSinglePhrase] = []
    for few_shot_example in few_shot_examples:
        one_way_dict = NLISampleSinglePhrase.from_nli_sample(few_shot_example)
        for single_phrase_sample in one_way_dict.values():
            few_shot_examples_single_phrase.extend(single_phrase_sample)

    formatter = partial(simple_formatter, style=FormatStyleOptions.YAML)

    processor = StructuredProcessor(
        query_type=ComparisonQuerySinglePhrase,
        result_type=EvidencedPhrase,
        system_prompt=system_prompt,
        few_shot_examples=few_shot_examples_single_phrase,
        format_query_fn=formatter,
        format_output_fn=formatter,
        log_dir=log_dir,
        use_schema_in_system_prompt=False,
        output_format=FormatStyleOptions.YAML,
    )
    logger.info("Initialized the processor for single phrase entailment verification.")
    return processor


class ReportGroundingNLIProcessor(BaseProcessor[NLIQuerySample, NLISample]):
    NUM_LLM_FAILURES = "num_llm_failures"
    NUM_LLM_SUCCESS = "num_llm_success"
    NUM_LLM_PHRASE_REWRITES = "num_llm_phrase_rewrites"

    def __init__(self, format_query_fn: Callable[..., Any] | None = None) -> None:
        super().__init__()
        self.format_query_fn = format_query_fn
        self.phrase_processor = get_ev_processor_singlephrase(log_dir=OUTPUT_DIR / "ev_processor_logs")
        # Logging errors
        self.num_llm_failures = 0
        self.num_llm_success = 0
        self.num_llm_phrase_rewrites = 0  # Sometimes it rewrites the phrase, which is not ideal
        self.response_dict: dict[DirectionOptions, list[EvidencedPhrase]] = {}

    def run_processor_on_single_phrase(
        self, single_phrase: NLIQuerySampleSinglePhrase, query_id: str
    ) -> EvidencedPhrase:
        """
        Run the processor on a single phrase.

        If LLM fails to respond, we return a default NOT_ENTAILMENT with no evidence.
        If LLM tries to rephrase the input, we log a warning and correct it.
        """
        single_response = self.phrase_processor.run(query=single_phrase.input, query_id=query_id)

        if single_response is None:
            logger.warning(f"WARNING: No response for example {query_id}. Setting as NOT ENTAILED.")
            single_response = EvidencedPhrase(
                phrase=single_phrase.input.hypothesis, status=EVState.NOT_ENTAILMENT, evidence=[]
            )
            self.num_llm_failures += 1
        else:
            self.num_llm_success += 1
            # There is a chance that the LLM rewrites the original input phrase somehow, so we need to check.
            # If it does rewrite it, we log a warning and correct it.
            phrase_from_llm = single_response.phrase
            if phrase_from_llm != single_phrase.input.hypothesis:
                self.num_llm_phrase_rewrites += 1
                logger.warning(
                    "WARNING: LLM has rewritten the input phrase. "
                    f"Original: '{single_phrase.input.hypothesis}' Rewritten: '{phrase_from_llm}'"
                )
                single_response = single_response.copy(update={"phrase": single_phrase.input.hypothesis})
        return single_response

    def set_model(self, model: BaseLanguageModel[str] | BaseLanguageModel[BaseMessage]) -> None:
        self.phrase_processor.set_model(model)

    def run(self, query: NLIQuerySample | Any, query_id: str) -> NLISample | None:
        if self.format_query_fn is not None:
            query = self.format_query_fn(query)
        assert isinstance(
            query, NLIQuerySample
        ), f"Query must be an NLIQuerySample, got {type(query)}. Provide a format_query_fn to convert it."
        phrase_level_examples = NLIQuerySampleSinglePhrase.from_nli_query_sample(query)
        for direction, phrase_list in phrase_level_examples.items():
            processed_list: list[EvidencedPhrase] = []
            for single_phrase in phrase_list:
                single_response = self.run_processor_on_single_phrase(single_phrase, query_id=query_id)
                processed_list.append(single_response)
            self.response_dict[direction] = processed_list
        try:
            output = NLISample.from_pair_of_unidirectional_lists(
                example_id=query_id,
                A_to_B=self.response_dict[DirectionOptions.A_TO_B],
                B_to_A=self.response_dict[DirectionOptions.B_TO_A],
            )
            return output
        except ValidationError as e:
            logger.warning(f"WARNING: Validation error for example {query_id}. Skipping.")
            logger.warning(e)
            return None

    def get_processor_stats(self) -> ProcessorStats:
        return {
            self.NUM_LLM_FAILURES: self.num_llm_failures,
            self.NUM_LLM_SUCCESS: self.num_llm_success,
            self.NUM_LLM_PHRASE_REWRITES: self.num_llm_phrase_rewrites,
        }

    def aggregate_processor_stats(self, stats_per_processor: dict[str, ProcessorStats]) -> ProcessorStats:
        result: ProcessorStats = {}
        for _, stats in stats_per_processor.items():
            for key, value in stats.items():
                result[key] = result.get(key, 0) + value
        return result


def format_row_to_nli_query_sample(row: "pd.Series[Any]") -> NLIQuerySample:
    return NLIQuerySample.from_grounded_phrases_list_pair(
        example_id=row[MetricDataframeKeys.STUDY_ID],
        candidate=row[MetricDataframeKeys.CANDIDATE],
        reference=row[MetricDataframeKeys.REFERENCE],
    )


def get_report_nli_engine(
    cfg: DictConfig, candidates: dict[str, GroundedPhraseList], references: dict[str, GroundedPhraseList]
) -> LLMEngine:
    output_folder = get_subfolder(root=OUTPUT_DIR, subfolder=RADFACT_SUBFOLDER)
    nli_report_processor = ReportGroundingNLIProcessor(format_query_fn=format_row_to_nli_query_sample)
    dataset_df = pd.DataFrame(
        {
            MetricDataframeKeys.STUDY_ID: study_id,
            MetricDataframeKeys.CANDIDATE: candidates[study_id],
            MetricDataframeKeys.REFERENCE: references[study_id],
        }
        for study_id in candidates.keys()
    )
    engine = LLMEngine(
        cfg=cfg,
        processor=nli_report_processor,
        dataset_df=dataset_df,
        progress_output_folder=output_folder,
        final_output_folder=output_folder,
    )
    return engine
