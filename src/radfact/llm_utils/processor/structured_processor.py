#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Protocol, TypeVar

import yaml
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser, YamlOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import BaseChatPromptTemplate
from pydantic import BaseModel

from radfact.llm_utils.processor.base_processor import BaseProcessor, QueryT

logger = logging.getLogger(__name__)

_QUERY_KEY = "query"
ResultT = TypeVar("ResultT", bound=BaseModel)
ProcessorStats = dict[str, int]


class FormatStyleOptions(str, Enum):
    """Options for formatting the output of a class."""

    JSON = "json"
    YAML = "yaml"


def simple_formatter(obj: BaseModel, style: FormatStyleOptions) -> str:
    """Format a BaseModel instance for output.

    :param obj: The Pydantic object to format.
    :param style: The desired output format.
    :return: The formatted string.
    """
    match style:
        case FormatStyleOptions.JSON:
            return str(obj.json())
        case FormatStyleOptions.YAML:
            return yaml.dump(obj.dict(), sort_keys=False)
        case _:
            raise ValueError(f"Unrecognized format style: {style}.")


class Example(Protocol, Generic[QueryT, ResultT]):
    """Interface for any object with `input` and `output` attributes representing a processed example."""

    input: QueryT
    output: ResultT


class QueryTemplate(BaseChatPromptTemplate, Generic[QueryT, ResultT]):
    """Query template for a structured processor."""

    system_prompt: str
    format_query_fn: Callable[[QueryT], str]
    format_output_fn: Callable[[ResultT], str]
    examples: Iterable[Example[QueryT, ResultT]] | None

    def __init__(
        self,
        system_prompt: str,
        query_type: type[QueryT],
        format_query_fn: Callable[[QueryT], str],
        format_output_fn: Callable[[ResultT], str],
        examples: Iterable[Example[QueryT, ResultT]] | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            input_variables=[_QUERY_KEY],
            input_types={_QUERY_KEY: query_type},
            system_prompt=system_prompt,
            format_query_fn=format_query_fn,
            format_output_fn=format_output_fn,
        )
        self.examples = examples

    def prepare_few_shot_examples(self, examples: Iterable[Example[QueryT, ResultT]] | None) -> list[BaseMessage]:
        """Prepare few shot examples as human-assistant message pairs.

        :param examples: List of few shot examples.
        :return: List of messages (HumanMessage, AIMessage) for the chat prompt.
        """
        few_shot_messages: list[BaseMessage] = []
        if not examples:
            return few_shot_messages
        for example in examples:
            human_query = self.format_query_fn(example.input)
            ai_response = self.format_output_fn(example.output)
            few_shot_messages.append(HumanMessage(content=human_query))
            few_shot_messages.append(AIMessage(content=ai_response))
        return few_shot_messages

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format the final chat prompt messages including the system prompt, few-shot examples, and the query."""
        query: QueryT = kwargs[_QUERY_KEY]
        formatted_query = self.format_query_fn(query)

        few_shot_messages = self.prepare_few_shot_examples(
            examples=self.examples
        )  # Not yet sure how to precompute this
        return (
            [SystemMessage(content=self.system_prompt)]
            + few_shot_messages  # Note this is empty if none are provided
            + [HumanMessage(content=formatted_query)]
        )


class StructuredProcessor(BaseProcessor[QueryT, ResultT]):
    """Processor for structured queries and results."""

    NUM_FAILURES = "num_failures"
    NUM_SUCCESS = "num_success"

    def __init__(
        self,
        query_type: type[QueryT],
        result_type: type[ResultT],
        system_prompt: str | Path,
        format_query_fn: Callable[[QueryT], str],
        format_output_fn: Callable[[ResultT], str] | None = None,
        model: BaseLanguageModel[str] | BaseLanguageModel[BaseMessage] | None = None,
        log_dir: Path | None = None,
        few_shot_examples: Iterable[Example[QueryT, ResultT]] | None = None,
        validate_result_fn: Callable[[QueryT, ResultT], None] | None = None,
        use_schema_in_system_prompt: bool = True,
        output_format: FormatStyleOptions = FormatStyleOptions.JSON,
    ) -> None:
        """
        :param query_type: Type of the input query, e.g. `str`, `pd.Series`, or a Pydantic `BaseModel`.
        :param result_type: Type of the structured Pydantic output.
        :param system_prompt: The part of system message describing the desired model behaviour. This will be
            complemented by a description of the output JSON schema for `result_type`.
        :param format_query_fn: A function to format the query into a "human" chat message for the model.
        :param format_output_fn: A function to format the (expected) output into an "AI" chat message for the model.
            If not provided, the output will be formatted as JSON.
        :param model: Langchain model to use for processing.
        :param log_dir: If given, directory where to save error log files containing details of each failed query, named
            `error_{query_id}.txt`. Otherwise, errors will only be printed to stdout.
        :param few_shot_examples: Optional list of few-shot examples to be included in the prompt as human-assistant
            message pairs. Each example can be any Python object with `input` and `output` attributes of `query_type`
            and `result_type`, respectively.
        :param validate_result_fn: Optional function to validate the result of each query. It should take the query and
            the result and raise an exception if the result is invalid. For example, this may leverage Pydantic's
            validation mechanics.
        :param use_schema_in_system_prompt: If True, the system prompt will also include a description of the
            expected output schema for `result_type`, as generated by the output parser (either YAML or JSON).
        :param output_format: The format of the output. Either "json" or "yaml".
        """
        self.query_type = query_type
        self.result_type = result_type

        self.system_prompt = system_prompt if isinstance(system_prompt, str) else system_prompt.read_text()
        self.format_query_fn = format_query_fn
        self.output_format = output_format
        if format_output_fn is None:
            # Use the generic formatter based on the output format style
            self.format_output_fn: Callable[[ResultT], str] = partial(simple_formatter, style=self.output_format)
        else:
            self.format_output_fn = format_output_fn

        self.validate_result_fn = validate_result_fn

        self.parser: PydanticOutputParser[ResultT] | YamlOutputParser[ResultT]
        match self.output_format:
            case FormatStyleOptions.YAML:
                self.parser = YamlOutputParser(pydantic_object=result_type)
            case FormatStyleOptions.JSON:
                self.parser = PydanticOutputParser(pydantic_object=result_type)
            case _:
                raise ValueError(
                    f"Unrecognized output format: {self.output_format}. Should be one of {FormatStyleOptions}."
                )

        if use_schema_in_system_prompt:
            self.system_prompt += "\n\n" + self.parser.get_format_instructions()

        self.query_template = QueryTemplate(
            system_prompt=self.system_prompt,
            query_type=query_type,
            format_query_fn=self.format_query_fn,
            format_output_fn=self.format_output_fn,
            examples=few_shot_examples,
        )
        self.log_dir = log_dir
        self.chain: LLMChain | None = None
        if model is not None:
            self.set_model(model=model)

        # For logging
        self.num_failures: int = 0
        self.num_success: int = 0

    def set_model(self, model: BaseLanguageModel[str] | BaseLanguageModel[BaseMessage]) -> None:
        self.chain = LLMChain(prompt=self.query_template, llm=model, output_parser=self.parser)

    def _write_error(self, ex: Exception, query: QueryT, query_id: str) -> None:
        formatted_query = self.format_query_fn(query)
        error_message = f"{ex}\n----\nQuery {query_id=}:\n{formatted_query=}"
        if self.log_dir:
            self.log_dir.mkdir(exist_ok=True, parents=True)
            error_log_path = self.log_dir / f"error_{query_id}.txt"
            error_log_path.write_text(error_message)
            logger.info(f"Error details saved to {error_log_path}.")

    def run(self, query: QueryT, query_id: str) -> ResultT | None:
        assert self.chain, "Model not set. Call `set_model` first."
        try:
            response: ResultT = self.chain.invoke({_QUERY_KEY: query})[self.chain.output_key]
            if self.validate_result_fn:
                self.validate_result_fn(query, response)
            self.num_success += 1
            return response
        except Exception as ex:
            self._write_error(ex, query, query_id)
            self.num_failures += 1
            return None

    def get_processor_stats(self) -> ProcessorStats:
        return {
            self.NUM_FAILURES: self.num_failures,
            self.NUM_SUCCESS: self.num_success,
        }

    def aggregate_processor_stats(self, stats_per_processor: dict[str, ProcessorStats]) -> ProcessorStats:
        result: ProcessorStats = {}
        for _, stats in stats_per_processor.items():
            for key, value in stats.items():
                result[key] = result.get(key, 0) + value
        return result
