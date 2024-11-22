#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import re

import redis
from langchain_community.cache import RedisCache

from radfact.azure_utils.auth import extract_object_id_from_token, get_credential

logger = logging.getLogger(__name__)


def remove_endpoint_from_json_string(llm_string: str) -> str:
    """Remove the endpoint URL from the llm_string, using either the format used by AzureOpenAI or ChatOpenAI."""
    # azure_endpoint is used by AzureOpenAI, api_endpoint is used by ChatOpenAI
    pattern = r'"(azure|api)_endpoint": "https://[^"]+"'
    return re.sub(pattern, "", llm_string)


class RedisCacheWithoutEndpoint(RedisCache):
    """A RedisCache object that ignores the 'azure_endpoint' field when querying the cache."""

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string, while ignoring the 'azure_endpoint' field in the llm_string."""
        # We simply remove the 'azure_endpoint' entry in the LLM string (JSON serialized dictionary) before
        # computing the hash. The LLM string is then no longer a valid JSON, but that does not matter for cache lookup
        return RedisCache._key(prompt, remove_endpoint_from_json_string(llm_string))


def get_redis_cache(redis_cache_name: str) -> RedisCache:
    """Gets a RedisCache object that points to the given Redis cache object in Azure.
    When running in AzureML, the cache is accessed using the cluster managed identity,
    otherwise it uses the current users default Azure credentials.

    :param redis_cache_name: The name of the Redis cache in Azure, without the .redis.cache.windows.net suffix.
    :return: A RedisCache object that points to the given Redis cache object in Azure.
    """
    credential = get_credential()
    token = credential.get_token("https://redis.azure.com/.default").token
    # The Redis username is the object id of the managed identity, which can be read out from the OID field of the token
    redis_username = extract_object_id_from_token(token)
    redis_url = f"{redis_cache_name}.redis.cache.windows.net"
    logger.info(f"Connecting to Redis cache {redis_url} with AAD object id {redis_username}")
    redis_client = redis.Redis(
        host=redis_url,
        port=6380,
        password=token,
        username=redis_username,
        ssl=True,
    )
    # Set a simple test key to check authentication as early as possible
    logger.info("Testing Redis connection")
    redis_client.set("testkey", "testvalue")
    logger.info("Redis connection successful")
    return RedisCacheWithoutEndpoint(redis_client)
