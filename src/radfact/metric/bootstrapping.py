#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Generator

import numpy as np
from tqdm import tqdm

from radfact.metric.radfact import InputDict, PerSampleResultType, RadFactMetric, ReturnType

logger = logging.getLogger(__name__)


def _collate_list_of_dicts(dicts: list[dict[str, float]]) -> dict[str, list[float]]:
    """Convert a list of dictionaries into a dictionary of lists, preserving order."""
    keys = dicts[0].keys()
    if not all(elem.keys() == keys for elem in dicts[1:]):
        raise ValueError("Every dict in the list should have the same keys")
    return {key: [elem[key] for elem in dicts] for key in keys}


class MetricBootstrapper:
    """Utility to report bootstrapping statistics for RadFact metric."""

    def __init__(self, metric: RadFactMetric, num_samples: int, seed: int | None = None) -> None:
        """
        :param metric: The metric for which to compute bootstrap statistics (e.g. RadFactMetric).
        :param num_samples: Number of bootstrap samples to generate, ideally in the hundreds.
        :param seed: RNG seed for reproducibility. By default (`None`), will give different results every time.
        """
        self.metric = metric
        self.num_samples = num_samples
        self.seed = seed

    def _generate_bootstrap_results(
        self,
        candidates: InputDict | None = None,
        references: InputDict | None = None,
        results_per_sample: PerSampleResultType | None = None,
    ) -> Generator[ReturnType, None, None]:
        """Compute the bootstrap results for a radfact metric by drawing `num_samples` samples with replacement,
        and re-computing the metric.


        :param candidates: The set of candidate reports.
        :param references: The set of reference reports.
        :param results_per_sample: Intermediate per-sample results. This can be used to pass in pre-computed per-sample
            results. If provided, the arguments `candidates` and `references` will be ignored for sample wise metrics.
        :yield: A generator of bootstrap results, where each result of type `MetricReturnType`.
        """
        if results_per_sample is None:
            assert candidates is not None and references is not None
            results_per_sample = self.metric.compute_results_per_sample(candidates=candidates, references=references)
        assert results_per_sample is not None  # for mypy
        num_records = len(results_per_sample)
        if num_records == 0:
            logger.warning("No samples to bootstrap. Metrics will all be NaN.")
        rng = np.random.default_rng(seed=self.seed)
        boot_indices_generator = (
            # Draw with replacement from the range of indices.
            (None, rng.choice(num_records, size=num_records, replace=True))
            for _ in tqdm(range(self.num_samples), total=self.num_samples)
        )
        for _, boot_indices in boot_indices_generator:
            boot_results_per_sample = self.metric.reindex_results_per_sample(results_per_sample, boot_indices)
            boot_results = self.metric.aggregate_results(boot_results_per_sample)
            yield boot_results

    @staticmethod
    def _compute_bootstrap_stats(values: list[float]) -> dict[str, float]:
        q025, q25, median, q75, q975 = np.nanquantile(np.asarray(values), [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
        numpy_stats = {
            "mean": np.nanmean(values, axis=0),
            "stderr": np.nanstd(values, axis=0),
            "p2.5th": q025,
            "p25th": q25,
            "median": median,
            "p75th": q75,
            "p97.5th": q975,
        }
        return {stat_name: value.tolist() for stat_name, value in numpy_stats.items()}

    def compute_bootstrap_metrics(
        self,
        candidates: InputDict | None = None,
        references: InputDict | None = None,
        results_per_sample: PerSampleResultType | None = None,
    ) -> dict[str, float]:
        """Calculate bootstrap statistics for RadFact metric that has intermediate per-sample results.

        :param candidates: The set of candidate reports to bootstrap.
        :param references: The set of reference reports to bootstrap.
        :param results_per_sample: Intermediate per-sample results. This can be used to pass in pre-computed per-sample
            results. If provided, the arguments `candidates` and `references` will be ignored for sample wise metrics.
        :return: A dictionary of bootstrap statistics, containing the mean (`mean`), standard error (`stderr`), 95%
            confidence interval (`p2.5th` and `p97.5th`), quartiles (`p25th` and `p75th`), and median (`median`) of the
            bootstrap distribution. If `metric` returns detailed submetrics, bootstrap statistics will also be included,
            e.g. `submetric/mean`, `submetric/stderr`, etc.
        """
        boot_results_generator = self._generate_bootstrap_results(
            candidates=candidates, references=references, results_per_sample=results_per_sample
        )
        boot_main_scores: list[float] = []
        boot_detailed_scores_dicts: list[dict[str, float]] = []
        for boot_results in boot_results_generator:
            assert isinstance(boot_results, tuple)
            main_score, detailed_scores_dict = boot_results
            boot_main_scores.append(main_score)
            boot_detailed_scores_dicts.append(detailed_scores_dict)

        stats_dict = self._compute_bootstrap_stats(boot_main_scores)
        collated_detailed_scores_dict = _collate_list_of_dicts(boot_detailed_scores_dicts)
        for submetric_name, values in collated_detailed_scores_dict.items():
            submetric_stats_dict = self._compute_bootstrap_stats(values)
            for stat_name, stat_value in submetric_stats_dict.items():
                stats_dict[f"{submetric_name}/{stat_name}"] = stat_value
        return stats_dict
