from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from radfact.paths import get_prompts_dir

REPORT_TO_PHRASES_PARSING_TASK = "report_to_phrases"
REPORT_TO_PHRASES_PROMPTS_DIR = get_prompts_dir(task=REPORT_TO_PHRASES_PARSING_TASK)
NEGATIVE_FILTERING_PARSING_TASK = "negative_filtering"
NEGATIVE_FILTERING_PROMPTS_DIR = get_prompts_dir(task=NEGATIVE_FILTERING_PARSING_TASK)
NLI_PARSING_TASK = "nli"
NLI_PROMPTS_DIR = get_prompts_dir(task=NLI_PARSING_TASK)


class ReportType(str, Enum):
    CXR = "cxr"
    CT = "ct"


@dataclass(frozen=True)
class PromptTask:
    name: str
    system_message_path: Path
    few_shot_examples_path: Path


class ReportToPhrasesTaskOptions(Enum):
    CXR = PromptTask(
        name=f"{ReportType.CXR.value}_report_to_phrases",
        system_message_path=REPORT_TO_PHRASES_PROMPTS_DIR / ReportType.CXR.value / "system_message.txt",
        few_shot_examples_path=REPORT_TO_PHRASES_PROMPTS_DIR / ReportType.CXR.value / "few_shot_examples.json",
    )
    CT = PromptTask(
        name=f"{ReportType.CT.value}_report_to_phrases",
        system_message_path=REPORT_TO_PHRASES_PROMPTS_DIR / ReportType.CT.value / "system_message.txt",
        few_shot_examples_path=REPORT_TO_PHRASES_PROMPTS_DIR / ReportType.CT.value / "few_shot_examples.json",
    )


class NegativeFilteringTaskOptions(Enum):
    CT = PromptTask(
        name=f"{ReportType.CT.value}_negative_filtering",
        system_message_path=NEGATIVE_FILTERING_PROMPTS_DIR / ReportType.CT.value / "system_message.txt",
        few_shot_examples_path=NEGATIVE_FILTERING_PROMPTS_DIR / ReportType.CT.value / "few_shot_examples.json",
    )


class NLITaskOptions(Enum):
    CXR = PromptTask(
        name=f"{ReportType.CXR.value}_nli",
        system_message_path=NLI_PROMPTS_DIR / ReportType.CXR.value / "system_message_ev_singlephrase.txt",
        few_shot_examples_path=NLI_PROMPTS_DIR / ReportType.CXR.value / "few_shot_examples.json",
    )
    CT = PromptTask(
        name=f"{ReportType.CT.value}_nli",
        system_message_path=NLI_PROMPTS_DIR / ReportType.CT.value / "system_message_ev_singlephrase.txt",
        few_shot_examples_path=NLI_PROMPTS_DIR / ReportType.CT.value / "few_shot_examples.json",
    )
