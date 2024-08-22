#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# This is copied from https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/azure/identity/_bearer_token_provider.py because we are unable to build the environment
# to use the latest version of azure-identity. This is a temporary solution until we can update the environment.
# This code is licensed under the MIT License. https://github.com/Azure/azure-sdk-for-python/blob/main/LICENSE
from typing import Callable

from azure.core.credentials import TokenCredential
from azure.core.pipeline import PipelineContext, PipelineRequest
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from azure.core.rest import HttpRequest


def _make_request() -> PipelineRequest[HttpRequest]:
    return PipelineRequest(HttpRequest("CredentialWrapper", "https://fakeurl"), PipelineContext(None))


def get_bearer_token_provider(credential: TokenCredential, *scopes: str) -> Callable[[], str]:
    """Returns a callable that provides a bearer token.

    It can be used for instance to write code like:

    .. code-block:: python

        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        credential = DefaultAzureCredential()
        bearer_token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

        # Usage
        request.headers["Authorization"] = "Bearer " + bearer_token_provider()

    :param credential: The credential used to authenticate the request.
    :type credential: ~azure.core.credentials.TokenCredential
    :param str scopes: The scopes required for the bearer token.
    :rtype: callable
    :return: A callable that returns a bearer token.
    """

    policy = BearerTokenCredentialPolicy(credential, *scopes)

    def wrapper() -> str:
        request = _make_request()
        policy.on_request(request)
        return request.http_request.headers["Authorization"][len("Bearer ") :]

    return wrapper
