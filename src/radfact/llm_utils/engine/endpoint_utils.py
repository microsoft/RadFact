#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging

from omegaconf import DictConfig

from radfact.llm_utils.endpoint import ENV_API_KEY, Endpoint, EndpointType

logger = logging.getLogger(__name__)


def get_endpoints_dict_sorted_by_speed(
    cfg: DictConfig,
    descending: bool = False,
    default_num_parallel_processes: int = 1,
) -> dict[str, Endpoint]:
    """Return a dictionary of endpoints sorted by speed factor in the order specified by the descending parameter.
    (Default: ascending order, slowest first, fastest last)

    :param cfg: The OmegaConf configuration object for the whole engine
    :param descending: If True, return the endpoints in descending speed (Fastest first, slowest last).
    :param default_num_parallel_processes: The default number of parallel processes to run for each endpoint. If the
        endpoint can handle multiple parallel processes, it will be replicated n times in the dictionary.
    :return: A dictionary of endpoint objects sorted by speed factor
    """
    endpoint_objs: list[Endpoint] = []
    for endpoint_name, endpoint in cfg.endpoints.items():
        assert isinstance(endpoint, DictConfig)
        logger.info(f"Creating Endpoint object for {endpoint_name}")
        endpoint_obj = Endpoint(
            url=endpoint.get("url"),
            type=EndpointType[endpoint.get("type")],
            api_key_env_var_name=endpoint.get("api_key_env_var_name", ENV_API_KEY),
            keyvault_secret_name=endpoint.get("keyvault_secret_name", ""),
            deployment_name=endpoint.get("deployment_name"),
            speed_factor=endpoint.get("speed_factor", 1.0),
            num_parallel_processes=endpoint.get("num_parallel_processes", default_num_parallel_processes),
        )
        endpoint_objs.append(endpoint_obj)
    endpoint_objs.sort(key=lambda x: x.speed_factor, reverse=descending)
    if not all(endpoint.deployment_name == endpoint_objs[0].deployment_name for endpoint in endpoint_objs):
        raise ValueError(
            f"All endpoints must be of the same type but got {[endpoint.deployment_name for endpoint in endpoint_objs]}"
        )
    return {endpoint.url: endpoint for endpoint in endpoint_objs}


def replicate_same_endpoint_n_times(endpoints: dict[str, Endpoint]) -> dict[str, Endpoint]:
    """Replicate each endpoint n times in the dictionary if num_parallel_processes > 1.

    :param endpoints: A dictionary of endpoint objects.
    :return: A dictionary of endpoint objects with replicated endpoints.
    """
    replicated_endpoints = {}
    for url, endpoint in endpoints.items():
        if endpoint.num_parallel_processes > 1:
            logger.info(f"Replicating endpoint {url} {endpoint.num_parallel_processes} times")
            for i in range(endpoint.num_parallel_processes):
                # this gives the illusion that it's a different endpoint for dataset sharing but endpoint.url is
                # unchanged so we can spawn/fork n processes and send parallel requests to the same endpoint
                replicated_endpoints[f"{url}_{i}"] = endpoint
        else:
            replicated_endpoints[url] = endpoint
    return replicated_endpoints
