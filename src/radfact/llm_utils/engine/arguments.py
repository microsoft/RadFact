#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from azureml._restclient.models.error_response import ErrorResponseException
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from radfact.llm_utils.endpoint import Endpoint, EndpointType

logger = logging.getLogger(__name__)


@dataclass
class OpenaiAPIArguments(metaclass=ABCMeta):
    """Base class for OpenAI API models."""

    endpoint: Endpoint | None = field(default=None)
    api_version: str = field(default="2023-06-01-preview")
    max_retries: int = field(default=10)
    timeout: float | None = field(default=None)

    def set_endpoint(self, endpoint: Endpoint) -> None:
        """Set the endpoint for the API."""
        self.endpoint = endpoint

    def get_params(self) -> dict[str, Any]:
        """Get LLM params as a dict. The dict keys match the expected arguments of the API. Check the OpenAI API
        documentation for more details.

        :raises ValueError: If the endpoint type is not supported.
        :return: The LLM params as a dict.
        """
        if self.endpoint is None:
            raise ValueError("Endpoint must be set before getting the params.")
        match self.endpoint.type:
            case EndpointType.AZURE_CHAT_OPENAI:
                params = dict(
                    deployment_name=self.endpoint.deployment_name,
                    azure_endpoint=self.endpoint.url,
                    openai_api_version=self.api_version,
                    max_retries=self.max_retries,
                    request_timeout=self.timeout,
                )
                try:
                    params["openai_api_key"] = self.endpoint.api_key
                except (ValueError, ErrorResponseException):
                    logger.info(
                        "Could not find API key in environment variables nor in the keyvault... Trying token provider."
                    )
                    params["azure_ad_token_provider"] = self.endpoint.token_provider
                return params
            case EndpointType.CHAT_OPENAI:
                return dict(
                    model=self.endpoint.deployment_name,
                    base_url=self.endpoint.url,
                    openai_api_key=self.endpoint.api_key,
                    max_retries=self.max_retries,
                    request_timeout=self.timeout,
                )
            case _:
                raise ValueError(f"Unsupported endpoint type {self.endpoint.type}")

    @abstractmethod
    def get_model(self) -> ChatOpenAI | AzureChatOpenAI:
        """Returns the chat model."""
        raise NotImplementedError(f"get_model() must be implemented in a subclass {self.__class__.__name__}")


@dataclass
class LLMAPIArguments(OpenaiAPIArguments):
    """Chat API for an LLM expects arguments to match ChatOpenAI or AzureChatOpenAI."""

    temperature: float = field(default=0.0)
    max_tokens: int = field(default=1024)
    top_p: float = field(default=0.95)
    frequency_penalty: float = field(default=0.0)
    presence_penalty: float = field(default=0.0)
    stop: list[str] | None = field(default=None)
    n_completions: int = field(default=1)

    def get_chat_completion_params(self) -> dict[str, Any]:
        """Get the chat completion params. Note that these params are specific to the chat completion API, the dict
        keys match the expected arguments of the API. Check the OpenAI API documentation for more details.
        https://api.python.langchain.com/en/stable/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html#langchain_openai.chat_models.azure.AzureChatOpenAI
        """
        return dict(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.n_completions,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
        )

    def get_params(self) -> dict[str, Any]:
        """Get LLM params as a dict."""
        params = super().get_params()
        params.update(self.get_chat_completion_params())
        return params

    def get_model(self) -> ChatOpenAI | AzureChatOpenAI:
        assert self.endpoint is not None  # for mypy
        match self.endpoint.type:
            case EndpointType.AZURE_CHAT_OPENAI:
                return AzureChatOpenAI(**self.get_params())
            case EndpointType.CHAT_OPENAI:
                return ChatOpenAI(**self.get_params())
            case _:
                raise ValueError(f"Unsupported endpoint type {self.endpoint.type}")


@dataclass
class LLMEngineArguments:
    """Arguments for the LLM engine wrapper around a processor.

    :param index_col: The name of the index column in the dataset.
    :param batch_size: The batch size for processing the dataset.
    :param start_index: The start index for processing the dataset.
    :param end_index: The end index for processing the dataset.
    :param output_filename: The name of the output file.
    """

    index_col: str
    batch_size: int = 100
    start_index: int = 0
    dataset_name: str | None = field(default=None)
    end_index: int | None = field(default=None)
    output_filename: str = "output.json"
