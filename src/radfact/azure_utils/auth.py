#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from azure.identity import AzureCliCredential, DefaultAzureCredential
from azureml._restclient.models.error_response import ErrorResponseException
from health_azure import get_workspace

from radfact.azure_utils.bearer_token_provider import get_bearer_token_provider

logger = logging.getLogger(__name__)

# The default scope for the Azure Cognitive Services. Tokens are retrieve from this page, and later used instead
# of the API key.
AZURE_COGNITIVE_SERVICES = "https://cognitiveservices.azure.com"


def get_from_vault(secret_name: str, workspace_config_path: Path | None = None) -> str:
    """Reads a secret from the keyvault given the secret name.

    :param secret_name: The name of the secret in the keyvault.
    :param workspace_config_path: The path to the workspace configuration file.
    :return: Requested value.
    """
    workspace = get_workspace(workspace_config_path=workspace_config_path)
    try:
        keyvault = workspace.get_default_keyvault()
        secret_value = str(keyvault.get_secret(name=secret_name))
        return secret_value
    except ErrorResponseException:
        logger.warning("Unable to retrive secret key from keyvault.")
        raise


def get_from_env_or_vault(
    env_var_name: str = "", secret_name: str = "", workspace_config_path: Path | None = None
) -> str:
    """Reads a value from an environment variable if possible.
    Otherwise, tries to read it from the keyvault given the secret name.

    :param env_var_name: The name of the environment variable.
    :param secret_name: The name of the secret in the keyvault.
    :param workspace_config_path: The path to the workspace configuration file.
    :return: Requested value from the environment variable or the keyvault.
    """
    if not env_var_name and not secret_name:
        raise ValueError("Either env_var_name or secret_name must be provided.")
    value = os.environ.get(env_var_name, None)
    if value is not None:
        return value
    if not secret_name:
        raise ValueError("Secret name must be provided if the environment variable is not set.")
    value = get_from_vault(secret_name, workspace_config_path)
    return value


def get_credential() -> AzureCliCredential | DefaultAzureCredential:
    """Get the appropriate Azure credential based on the environment. If the Azure CLI is installed and logged in,
    the Azure CLI credential is returned. Otherwise, the default Azure credential is returned."""
    try:
        return AzureCliCredential()
    except Exception:
        logger.info("Failed to get Azure CLI credential. Trying default Azure credential.")
        return DefaultAzureCredential()


def get_azure_token_provider() -> Callable[[], str]:
    """Get a token provider for Azure Cognitive Services. The bearer token provider gets authentication tokens and
    refreshes them automatically upon expiry.
    """
    credential = get_credential()
    token = credential.get_token(AZURE_COGNITIVE_SERVICES)
    logger.info(f"Credentials: {print_token_details(token.token)}")
    return get_bearer_token_provider(credential, AZURE_COGNITIVE_SERVICES)


def token_to_json(token: str) -> Any:
    """Converts an Azure access token to its underlying JSON structure.

    :param token: The access token.
    :return: The JSON object that is stored in the token.
    """
    # This is code to dissect the token, taken from
    # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/azure/identity/_internal/decorators.py#L38
    base64_meta_data = token.split(".")[1].encode("utf-8") + b"=="
    json_bytes = base64.decodebytes(base64_meta_data)
    json_string = json_bytes.decode("utf-8")
    return json.loads(json_string)


def print_token_details(token: str) -> str:
    """Creates a human-readable string with details stored in the Azure token.

    :param token: The access token.
    :return: A string with information about the identity that was given the access token.
    """
    json_dict = token_to_json(token)
    NOT_PRESENT = "(not available)"
    oid = NOT_PRESENT
    upn = NOT_PRESENT
    name = NOT_PRESENT
    appid = NOT_PRESENT
    try:
        oid = json_dict["oid"]
    except Exception:
        pass
    try:
        upn = json_dict["upn"]
    except Exception:
        pass
    try:
        name = json_dict["name"]
    except Exception:
        pass
    try:
        appid = json_dict["appid"]
    except Exception:
        pass
    return f"EntraID object ID {oid}, user principal name (upn) {upn}, name {name}, appid {appid}"


def extract_object_id_from_token(token: str) -> str:
    """Extracts the object ID from an access token.
    The object ID is the unique identifier for the user or service principal in Azure Active Directory.

    :param token: The access token.
    :return: The object ID of the token.
    """
    json_dict = token_to_json(token)
    return json_dict["oid"]  # type: ignore
