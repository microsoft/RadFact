#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path

REPOSITORY_ROOT_DIR = Path(__file__).absolute().parents[2]
RADFACT_ROOT_DIR = REPOSITORY_ROOT_DIR / "src" / "radfact"
OUTPUT_DIR = REPOSITORY_ROOT_DIR / "outputs"
LLM_UTILS_DIR = RADFACT_ROOT_DIR / "llm_utils"
CONFIGS_DIR = REPOSITORY_ROOT_DIR / "configs"
EXAMPLES_DIR = REPOSITORY_ROOT_DIR / "examples"
WORKSPACE_CONFIG_PATH = REPOSITORY_ROOT_DIR / "config.json"


def get_prompts_dir(task: str) -> Path:
    return LLM_UTILS_DIR / task / "prompts"
