#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import json
import logging

from radfact.data_utils.grounded_phrase_list import GroundedPhrase, GroundedPhraseList
from radfact.metric.bootstrapping import MetricBootstrapper
from radfact.metric.radfact import InputDict, RadFactMetric
from radfact.paths import EXAMPLES_DIR
from radfact.metric.print_utils import print_bootstrap_results

logger = logging.getLogger(__name__)


def read_examples() -> tuple[InputDict, InputDict]:
    json_path = EXAMPLES_DIR / "test_examples.json"
    with open(json_path, "r", encoding="utf-8") as f:
        examples_json = json.load(f)
    candidates = {
        example["example_id"]: GroundedPhraseList([GroundedPhrase(phrase) for phrase in example["input"]["phrases_A"]])
        for example in examples_json
    }
    references = {
        example["example_id"]: GroundedPhraseList([GroundedPhrase(phrase) for phrase in example["input"]["phrases_B"]])
        for example in examples_json
    }
    return candidates, references


def run_radfact() -> None:
    candidates, references = read_examples()
    metric = RadFactMetric()
    logger.info(f"Computing RadFact metric for {len(candidates)} examples")
    results_per_sample = metric.compute_results_per_sample(candidates, references)
    metric.is_narrative_text = True  # to avoid computing box metrics that are not relevant for this test
    bootstrapper = MetricBootstrapper(metric=metric, num_samples=500, seed=42)
    results_with_error_bars = bootstrapper.compute_bootstrap_metrics(results_per_sample=results_per_sample)

    metrics = ["logical_precision", "logical_recall", "logical_f1", "num_llm_failures"]
    print("RadFact results using Llama-3-70b-Instruct model")
    print_bootstrap_results(results_with_error_bars, metrics)

    expected_results = {
        "logical_precision/median": 0.3554,
        "logical_precision/p2.5th": 0.2745,
        "logical_precision/p97.5th": 0.4327,
        "logical_recall/median": 0.3211,
        "logical_recall/p2.5th": 0.2328,
        "logical_recall/p97.5th": 0.4093,
        "logical_f1/median": 0.3385,
        "logical_f1/p2.5th": 0.2607,
        "logical_f1/p97.5th": 0.4140,
        "num_llm_failures/median": 0.0,
        "num_llm_failures/p2.5th": 0.0,
        "num_llm_failures/p97.5th": 0.0,
    }
    print("Expected results range")
    # You should expect the results to be within the range printed here - doube check num_llm_failures if you notice
    # major discrepancies. The results may vary if you encounter many LLM failures or if you're using a different model.
    print_bootstrap_results(expected_results, metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    run_radfact()
