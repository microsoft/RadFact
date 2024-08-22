#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from abc import ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

QueryT = TypeVar("QueryT")
BaseResultT = TypeVar("BaseResultT")


class BaseModelWithId(BaseModel):
    """Base class for models that have an ID."""

    id: int | str | None = None


class BaseProcessor(Generic[QueryT, BaseResultT], metaclass=ABCMeta):
    """Base class for processors that interact with language models."""

    @abstractmethod
    def set_model(self, model: BaseLanguageModel[str] | BaseLanguageModel[BaseMessage]) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, query: QueryT, query_id: str) -> BaseResultT | None:
        raise NotImplementedError

    def get_processor_stats(self) -> Any:
        """Return statistics that the processor collects."""
        return None

    def aggregate_processor_stats(self, stats_per_processor: dict[str, Any]) -> Any:
        """Aggregate statistics from multiple processors.

        :param stats_per_processor: A dictionary of statistics from multiple processors. The dictionary key is the
            processor ID (usually the endpoint name), the dictionary value is the statistics returned by
            `get_processor_stats()`.
        :return: The aggregated statistics, which should be of the same type as returned by `get_processor_stats()`
        """
        return None
