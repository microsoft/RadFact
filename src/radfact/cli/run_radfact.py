#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from radfact.data_utils.grounded_phrase_list import GroundedPhraseList
from radfact.llm_utils.report_to_phrases.processor import StudyIdType
from radfact.metric.bootstrapping import MetricBootstrapper
from radfact.metric.print_utils import print_bootstrap_results, print_results
from radfact.metric.radfact import InputDict, RadFactMetric
from radfact.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)


def validate_config_file(config_name: str | None) -> None:
    if config_name is not None:
        config_path = CONFIGS_DIR / f"{config_name}"
        if not config_path.exists():
            message = (
                f"Config file {config_name} does not exist. "
                "Make sure the config file is saved in the `configs` directory."
            )
            raise FileNotFoundError(message)


def get_candidates_and_references_from_csv(csv_path: Path) -> tuple[dict[StudyIdType, str], dict[StudyIdType, str]]:
    """Reads the csv file containing the samples to compute RadFact for and returns the candidates and references in
    the expected format."""
    findings_generation_samples = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(findings_generation_samples)} samples from {csv_path}")
    candidates = findings_generation_samples.set_index("example_id")["prediction"].to_dict()
    references = findings_generation_samples.set_index("example_id")["target"].to_dict()
    return candidates, references


def get_candidates_and_references_from_json(
    json_path: Path,
) -> tuple[dict[StudyIdType, GroundedPhraseList], dict[StudyIdType, GroundedPhraseList]]:
    """Reads the json file containing the samples to compute RadFact for and returns the candidates and references in
    the expected format."""
    with open(json_path, "r", encoding="utf-8") as f:
        grounded_reporting_samples = json.load(f)
    logger.info(f"Loaded {len(grounded_reporting_samples)} samples from {json_path}")
    candidates = {
        example["example_id"]: GroundedPhraseList.from_list_of_dicts(example["prediction"])
        for example in grounded_reporting_samples
    }
    references = {
        example["example_id"]: GroundedPhraseList.from_list_of_dicts(example["target"])
        for example in grounded_reporting_samples
    }
    return candidates, references


def compute_radfact_scores(
    radfact_config_name: str | None,
    phrases_config_name: str | None,
    candidates: InputDict,
    references: InputDict,
    is_narrative_text: bool,
    bootstrap_samples: int,
) -> dict[str, float]:
    radfact_metric = RadFactMetric(
        nli_config_name=radfact_config_name,
        phrase_config_name=phrases_config_name,
        is_narrative_text=is_narrative_text,
    )
    if bootstrap_samples == 0:
        _, results = radfact_metric.compute_metric_score(candidates, references)
        return results
    bootstrapper = MetricBootstrapper(metric=radfact_metric, num_samples=10, seed=42)
    results_per_sample = radfact_metric.compute_results_per_sample(candidates, references)
    return bootstrapper.compute_bootstrap_metrics(results_per_sample=results_per_sample)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Compute RadFact metric for a set of samples and saves the results to a json file."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="The path to the csv or json file containing the samples to compute RadFact for. For finding generation "
        "samples, the csv file should have columns 'example_id', 'prediction', and 'target' similar to the example in "
        "`examples/findings_generation_examples.csv`. For grounded reporting samples, provide a json file in the same "
        "format as `examples/grounded_reporting_examples.json`.",
        required=True,
    )
    parser.add_argument(
        "--is_narrative_text",
        action="store_true",
        help="Whether the input samples are narrative text or not. If true, the input samples are expected to be "
        "narrative text, otherwise they are expected to be grounded phrases.",
    )
    parser.add_argument(
        "--radfact_config_name",
        type=str,
        help="The name of the config file for RadFact processing. We use the default config file but you can provide a "
        "custom config. Make sure the config follows the same structure as `configs/radfact.yaml` and is saved in the "
        "`configs` directory. This is necessary for hydra initialization from the `configs` directory.",
        default=None,
    )
    parser.add_argument(
        "--phrases_config_name",
        type=str,
        help="The name of the config file for reports to phrases conversion. We use the default config file but you "
        "can provide a custom config. Make sure the config follows the same structure as "
        "`configs/report_to_phrases.yaml` and is saved in the `configs` directory. This is necessary for hydra "
        "initialization from the `configs` directory.",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory where the results will be saved as a json file.",
        default="outputs",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        help="Number of bootstrap samples to use for computing the confidence intervals. Set to 0 to disable "
        "bootstrapping.",
        default=500,
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    is_narrative_text = args.is_narrative_text
    radfact_config_name = args.radfact_config_name
    phrases_config_name = args.phrases_config_name
    bootstrap_samples = args.bootstrap_samples

    assert input_path.suffix in [".csv", ".json"], "Input file must be a csv or json file."
    assert input_path.suffix == ".csv" or not is_narrative_text, (
        "Input file must be a json file for grounded phrases and is_narrative_text must be False. For narrative text, "
        "input file must be a csv file and is_narrative_text must be True."
    )
    validate_config_file(radfact_config_name)
    validate_config_file(phrases_config_name)

    candidates: InputDict
    references: InputDict

    if is_narrative_text:
        candidates, references = get_candidates_and_references_from_csv(input_path)
    else:
        candidates, references = get_candidates_and_references_from_json(input_path)

    results = compute_radfact_scores(
        radfact_config_name=radfact_config_name,
        phrases_config_name=phrases_config_name,
        candidates=candidates,
        references=references,
        is_narrative_text=is_narrative_text,
        bootstrap_samples=bootstrap_samples,
    )

    print_fn = print_results if bootstrap_samples == 0 else print_bootstrap_results
    if is_narrative_text:
        print("RadFact scores for narrative text samples")
        print_fn(results=results, metrics=["logical_precision", "logical_recall", "logical_f1", "num_llm_failures"])
    else:
        print("RadFact scores for grounded phrases samples")
        print_fn(
            results=results,
            metrics=[
                "logical_precision",
                "logical_recall",
                "logical_f1",
                "spatial_precision",
                "spatial_recall",
                "spatial_f1",
                "grounding_precision",
                "grounding_recall",
                "grounding_f1",
                "num_llm_failures",
            ],
        )

    output_path = output_dir / f"radfact_scores_{input_path.stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
