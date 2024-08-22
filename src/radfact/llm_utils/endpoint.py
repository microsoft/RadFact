#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from radfact.azure_utils.auth import get_azure_token_provider, get_from_env_or_vault
from radfact.paths import WORKSPACE_CONFIG_PATH

# The default name under which an endpoint API Key is stored in environment variables.
ENV_API_KEY = "API_KEY"


class EndpointType(Enum):
    AZURE_CHAT_OPENAI = "azure_chat_openai"
    CHAT_OPENAI = "chat_openai"


@dataclass(frozen=False)
class Endpoint:
    url: str
    deployment_name: str
    type: EndpointType = EndpointType.AZURE_CHAT_OPENAI
    speed_factor: float = 1.0
    num_parallel_processes: int = 1
    api_key_env_var_name: str = ENV_API_KEY
    keyvault_secret_name: str = ""
    # The name of the Redis cache for this endpoint. If empty, no cache is used. Make sure to update the cache
    # location if the model type changes significantly, and we expect different responses.
    redis_cache: str = ""
    _api_key: str | None = field(default=None, init=False)
    _token_provider: Callable[[], str] | None = field(default=None, init=False)

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = get_from_env_or_vault(
                env_var_name=self.api_key_env_var_name,
                secret_name=self.keyvault_secret_name,
                workspace_config_path=WORKSPACE_CONFIG_PATH,
            )
        assert self._api_key is not None  # for mypy
        return self._api_key

    @property
    def token_provider(self) -> Callable[[], str]:
        if self._token_provider is None:
            self._token_provider = get_azure_token_provider()
        assert self._token_provider is not None  # for mypy
        return self._token_provider
