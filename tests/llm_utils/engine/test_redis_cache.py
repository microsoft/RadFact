#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from radfact.llm_utils.engine.redis_cache import remove_endpoint_from_json_string


def test_remove_endpoint() -> None:
    assert remove_endpoint_from_json_string("foo") == "foo"
    # Full string matches
    assert remove_endpoint_from_json_string('"azure_endpoint": "https://foo"') == ""
    # Quotes missing on URL, should not match
    assert remove_endpoint_from_json_string('"azure_endpoint": https://foo') == '"azure_endpoint": https://foo'
    # Should match
    assert remove_endpoint_from_json_string('1, "azure_endpoint": "https://foo", 2') == "1, , 2"
    # Should match - api_endpoint is the format used by ChatOpenAI
    assert remove_endpoint_from_json_string('1, "api_endpoint": "https://foo", 2') == "1, , 2"
