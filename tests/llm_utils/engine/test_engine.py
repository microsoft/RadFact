#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from unittest.mock import Mock

import pandas as pd

from radfact.llm_utils.engine.engine import LLMEngine


def mock_dataset_sharding(n_data: int, speed_factors: list[float]) -> list[int]:
    class DummyEndpoint:
        def __init__(self, speed_factor: float) -> None:
            self.speed_factor = speed_factor

    mock_engine = Mock(spec=LLMEngine)
    mock_engine.dataset_df = pd.DataFrame([{"dummy": i} for i in range(n_data)])
    mock_engine.endpoints = {
        str(i): DummyEndpoint(speed_factor=speed_factor) for i, speed_factor in enumerate(speed_factors)
    }

    weighted_splits = LLMEngine.get_weighted_splits(mock_engine)
    return list(weighted_splits.values())


def test_dataset_sharding() -> None:
    assert mock_dataset_sharding(10, [1.0]) == [10]
    assert mock_dataset_sharding(10, [1.0, 1.0]) == [5, 5]
    assert mock_dataset_sharding(10, [1.0, 1.0, 1.0]) == [4, 3, 3]
    assert mock_dataset_sharding(10, [1.0, 2.0]) == [3, 7]
    assert mock_dataset_sharding(10, [1.0, 1.0, 2.0]) == [3, 2, 5]
    assert mock_dataset_sharding(10, [1.0 for _ in range(11)]) == [1 for _ in range(10)] + [0]
    assert mock_dataset_sharding(20, [1.0 for _ in range(11)]) == [2 for _ in range(9)] + [1, 1]
