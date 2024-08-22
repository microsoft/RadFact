#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from dataclasses import asdict, replace
from typing import Any, Iterable, Mapping

import hydra
import numpy as np
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from radfact.data_utils.grounded_phrase_list import GroundedPhraseList, NormalizedBox
from radfact.llm_utils.nli.processor import get_report_nli_engine
from radfact.llm_utils.nli.schema import EVState, NLISample
from radfact.llm_utils.report_to_phrases.processor import FINDINGS_SECTION, StudyIdType, get_report_to_phrases_engine
from radfact.llm_utils.report_to_phrases.schema import ParsedReport
from radfact.metric.box_metrics import PRECISION, compute_box_metrics
from radfact.metric.schema import (
    GroundedPhraseEvidenced,
    OneWayNLIFractions,
    PerSampleNLIResult,
    RadFactScore,
    SpatialEntailmentStatus,
)
from radfact.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)
# To avoid HTTP logs from the LLM processor
http_logger = logging.getLogger("httpx")
http_logger.setLevel(logging.WARNING)


ReturnType = tuple[float, dict[str, float]]
InputType = GroundedPhraseList | str
InputDict = Mapping[StudyIdType, InputType]
GroundedPhraseListDict = dict[StudyIdType, GroundedPhraseList]
PerSampleResultType = list[PerSampleNLIResult]

# The YAML config file for the RadFact processor, specifying the different endpoints that it will use
RADFACT_CONFIG = "radfact.yaml"
# The YAML config file for the phrase processor in this setting.
REPORT_TO_PHRASES_CONFIG = "report_to_phrases.yaml"


def init_hydra_config(config_name: str) -> DictConfig:
    """Initialize Hydra with the given config name."""
    if GlobalHydra().is_initialized():
        # Hydra is already initalized if this code gets started from a script that has the @hydra.main decorator
        cfg = hydra.compose(config_name=config_name)
    else:
        # For use in unit tests: initialize Hydra afresh with the config directory
        with hydra.initialize_config_dir(str(CONFIGS_DIR), version_base="1.1"):
            cfg = hydra.compose(config_name=config_name)
    return cfg


def _divide_or_nan(numerator: float | int, denominator: float | int) -> float:
    """Divide the numerator by the denominator, or return NaN if the denominator is zero."""
    return numerator / denominator if denominator != 0 else np.nan


class RadFactMetric:
    def __init__(
        self,
        nli_config_name: str | None = None,
        phrase_config_name: str | None = None,
        image_size: int = 224,
        box_precision_threshold: float = 0.5,
        is_narrative_text: bool = False,
    ) -> None:
        """
        Initializes the RadFactMetric with the necessary configurations. We need to know the image size so we can
        compute box-based metrics.

        :param nli_config_name: The name of the NLI processing config file. This is the config file that specifies the
            different endpoints that the NLI processor will use. If None, the default config will be used.
        :param phrase_config_name: The name of the phrase processing config file. This is the config file that specifies
            the different endpoints that the phrase processor will use. If None, the default config will be used.
        :param image_size: The size of the images in the reports.
        :param box_precision_threshold: The threshold for precision computation for boxes.
        :param is_narrative_text: If True, we are running the metric on data narrative text data, e.g. the original
            findings section. We need to convert this to lists GroundedPhrase before conducting entailment verification.
            If False, we are running the metric on grounded reports, where the phrases are already in the correct
            format for entailment verification.
        """
        self.llm_nli_cfg = init_hydra_config(nli_config_name or RADFACT_CONFIG)
        self.llm_phrase_cfg = init_hydra_config(phrase_config_name or REPORT_TO_PHRASES_CONFIG)
        self.image_size = image_size
        self.box_precision_threshold = box_precision_threshold
        self.is_narrative_text = is_narrative_text
        self.meta_metrics: dict[str, float] = {}  # Metrics about the metric, derived from processors. Not per-sample.

    def _are_boxes_entailed(self, boxes: list[NormalizedBox] | None, evidence_boxes: list[NormalizedBox]) -> bool:
        """
        Compute whether the boxes are sufficiently covered by the evidence boxes.
        """
        if boxes is None or boxes == []:
            logger.warning("Empty boxes were passed to _are_boxes_entailed - this should not happen!")
        if len(evidence_boxes) == 0:
            return False
        assert boxes is not None  # for mypy
        box_metrics = compute_box_metrics(pred_boxes=boxes, true_boxes=evidence_boxes, mask_size=self.image_size)
        return box_metrics[PRECISION] > self.box_precision_threshold

    def _get_spatial_entailment_status(self, phrase: GroundedPhraseEvidenced) -> SpatialEntailmentStatus:
        """Get the spatial entailment status of a phrase."""
        phrase_is_entailed = phrase.status == EVState.ENTAILMENT
        if phrase.boxes is None:
            return SpatialEntailmentStatus.NO_BOXES
        elif phrase_is_entailed and self._are_boxes_entailed(phrase.boxes, phrase.get_all_evidence_boxes()):
            return SpatialEntailmentStatus.SPATIAL_ENTAILMENT
        else:
            return SpatialEntailmentStatus.NO_SPATIAL_ENTAILMENT

    def compute_radfact_score_oneway(self, phrases: list[GroundedPhraseEvidenced]) -> OneWayNLIFractions:
        """
        Compute the factual metrics for a single direction (e.g. candidate -> reference).

        We are aiming to compute the following numbers:
        - entailed_fraction: Fraction of all phrases that are logically entailed (implies denominator for
        entailed_box_fraction)
        - entailed_box_fraction: Fraction of all entailed phrases with boxes whose boxes are logically entailed
        - full_box_fraction: Fraction of all phrases with boxes whose boxes are logically entailed
        - num_phrases: Number of phrases in the report  (denominator for entailed_fraction)
        - num_phrases_with_boxes: Number of phrases in the report with boxes (denominator for full_box_fraction)

        :param phrases: List of phrases from either candidate or reference report, with logical state and evidence.
        :return: OneWayNLIFractions with the computed ratios/numbers.
        """
        num_phrases = len(phrases)
        # Add box-entailment status in-place so we can export it to JSON in the outer scope
        for i in range(num_phrases):
            spatial_entailment_status = self._get_spatial_entailment_status(phrases[i])
            phrases[i] = replace(phrases[i], spatial_entailment_status=spatial_entailment_status)

        # Phrases who are logically entailed
        entailed = [phrase for phrase in phrases if phrase.status == EVState.ENTAILMENT]

        # METRIC: Fraction of all phrases that are logically entailed
        entailed_fraction = _divide_or_nan(len(entailed), num_phrases)

        # === Now for box-based metrics ===

        # Phrases with boxes
        boxed = [phrase for phrase in phrases if phrase.spatial_entailment_status != SpatialEntailmentStatus.NO_BOXES]

        # Entailed phrases with any boxes
        entailed_boxed = [phrase for phrase in boxed if phrase.status == EVState.ENTAILMENT]

        # Entailed phrases with entailed boxes
        entailed_boxed_entailed = [
            phrase
            for phrase in phrases
            if phrase.spatial_entailment_status == SpatialEntailmentStatus.SPATIAL_ENTAILMENT
        ]

        # METRIC: Fraction of all ENTAILED phrases with boxes whose boxes are logically entailed
        entailed_box_fraction = _divide_or_nan(len(entailed_boxed_entailed), len(entailed_boxed))

        # METRIC: Fraction of ALL phrases with boxes whose boxes are logically entailed
        # Note that box entailment requires the phrase to also be logically entailed.
        full_box_fraction = _divide_or_nan(len(entailed_boxed_entailed), len(boxed))

        return OneWayNLIFractions(
            entailed_fraction=entailed_fraction,
            entailed_box_fraction=entailed_box_fraction,
            full_box_fraction=full_box_fraction,
            num_phrases=num_phrases,
            num_phrases_with_boxes=len(boxed),
        )

    def compute_radfact_score(
        self, candidate_phrases: list[GroundedPhraseEvidenced], reference_phrases: list[GroundedPhraseEvidenced]
    ) -> RadFactScore:
        """Return the factual precision and recall based on the (evidenced) list of phrases.
        These are essentially symmetrical, so the logic happens in compute_radfact_score_oneway.
        The difference is that "recall-like" metrics are computed on the reference (=ground truth) phrases,
        and "precision-like" metrics are computed on the candidate (=generated) phrases.

        :param candidate_phrases: List of phrases from the candidate report, with logical state and evidence.
        :param reference_phrases: List of phrases from the reference report, with logical state and evidence.
        :return: RadFactScore with factual precision and recall.
        """
        candidate_fractions = self.compute_radfact_score_oneway(candidate_phrases)
        reference_fractions = self.compute_radfact_score_oneway(reference_phrases)

        return RadFactScore.from_candidate_and_reference_fractions(
            candidate=candidate_fractions, reference=reference_fractions
        )

    def convert_narrative_text_to_phrases(
        self, texts: GroundedPhraseListDict, metric_prefix: str
    ) -> GroundedPhraseListDict:
        """
        Given a dictionary of study IDs and narrative texts (contained in GroundedPhraseList), convert the texts to
        phrases. Updates the meta-metrics with processor statistics.

        Returns a dictionary of processed texts.
        """
        texts_as_str = {k: v.get_all_text() for k, v in texts.items()}
        id_col = self.llm_phrase_cfg.processing.index_col
        texts_as_str_df = pd.DataFrame(
            {id_col: study_id, FINDINGS_SECTION: texts_as_str[study_id]} for study_id in texts_as_str.keys()
        )
        engine = get_report_to_phrases_engine(self.llm_phrase_cfg, texts_as_str_df)
        parsed_reports: list[ParsedReport] = engine.run()
        processed_texts = {
            parsed.id: parsed.to_grounded_phrases_list() for parsed in parsed_reports if parsed.id is not None
        }
        if engine.aggregated_processor_stats is not None:
            self.meta_metrics.update(
                {f"{metric_prefix}/{k}": float(v) for k, v in engine.aggregated_processor_stats.items()}
            )
        if set(processed_texts.keys()) != set(texts.keys()):
            logger.warning(
                f"Key mismatch between processed and input texts. #input keys: {len(set(texts.keys()))}. #processed "
                f"keys: {len(set(processed_texts.keys()))}"
            )
        return processed_texts

    def filter_candidates_and_references_to_common_keys(
        self, candidates: InputDict, references: InputDict
    ) -> tuple[InputDict, InputDict]:
        """
        Helper method that takes a list of dictionaries and filters them to only include the common keys.
        """
        common_keys = set(candidates.keys()).intersection(references.keys())
        candidates_out = {k: v for k, v in candidates.items() if k in common_keys}
        references_out = {k: v for k, v in references.items() if k in common_keys}

        num_dropped_candidates = len(candidates) - len(candidates_out)
        num_dropped_references = len(references) - len(references_out)
        if num_dropped_candidates > 0 or num_dropped_references > 0:
            logging.warning(
                f"The candidate keys (n={len(candidates)}) do not match the reference keys (n={len(references)})"
            )
            logging.warning(f"Dropped {num_dropped_candidates} candidates and {num_dropped_references} references.")
            logging.warning(f"Resulting in {len(candidates_out)} candidates and {len(references_out)} references.")
        return candidates_out, references_out

    def convert_candidates_and_references_to_phrases(
        self, candidates_mm: GroundedPhraseListDict, references_mm: GroundedPhraseListDict
    ) -> tuple[GroundedPhraseListDict, GroundedPhraseListDict]:
        """
        Convert the candidates and references to a list of phrases. This is done by converting the narrative text into
        single phrases describing one finding each. This is necessary for the entailment verification step.
        """
        report_to_phrases_prefix = "report_to_phrases"

        logger.info("CONVERTING GENERATIONS TO PHRASES...")
        candidates_mm = self.convert_narrative_text_to_phrases(
            candidates_mm, metric_prefix=f"{report_to_phrases_prefix}/generations"
        )
        # Ideally we already have GroundedPhrase for the references, but if not we also process
        logger.info("CONVERTING GROUND TRUTH TO PHRASES...")
        references_mm = self.convert_narrative_text_to_phrases(
            references_mm, metric_prefix=f"{report_to_phrases_prefix}/ground_truth"
        )
        candidates_filtered, references_filtered = self.filter_candidates_and_references_to_common_keys(
            candidates_mm, references_mm
        )
        num_dropped_candidates = len(candidates_mm) - len(candidates_filtered)
        num_dropped_references = len(references_mm) - len(references_filtered)
        self.meta_metrics[f"{report_to_phrases_prefix}/num_dropped_candidates"] = num_dropped_candidates
        self.meta_metrics[f"{report_to_phrases_prefix}/num_dropped_references"] = num_dropped_references

        return candidates_filtered, references_filtered  # type: ignore [return-value]

    def convert_input_to_multimodal(
        self, candidates: InputDict, references: InputDict
    ) -> tuple[GroundedPhraseListDict, GroundedPhraseListDict]:
        """
        Convert the input to multimodal format, where the values are GroundedPhraseList objects.
        """
        candidates_multimodal = {
            study_id: sequence if isinstance(sequence, GroundedPhraseList) else GroundedPhraseList([sequence])
            for study_id, sequence in candidates.items()
        }
        references_multimodal = {
            study_id: sequence if isinstance(sequence, GroundedPhraseList) else GroundedPhraseList([sequence])
            for study_id, sequence in references.items()
        }
        return candidates_multimodal, references_multimodal

    def compute_results_per_sample(self, candidates: InputDict, references: InputDict) -> PerSampleResultType:
        candidates_mm, references_mm = self.convert_input_to_multimodal(candidates, references)
        assert all(
            isinstance(value, GroundedPhraseList) for value in candidates_mm.values()
        ), f"Expected all values to be GroundedPhraseList, got {candidates_mm=}"
        assert all(
            isinstance(value, GroundedPhraseList) for value in references_mm.values()
        ), f"Expected all values to be GroundedPhraseList, got {references_mm=}"

        if self.is_narrative_text:
            candidates_mm, references_mm = self.convert_candidates_and_references_to_phrases(
                candidates_mm, references_mm
            )

        # Convert study IDs to strings because the NLI processor expects it
        candidates_str_ids = {str(study_id): sequence for study_id, sequence in candidates_mm.items()}
        references_str_ids = {str(study_id): sequence for study_id, sequence in references_mm.items()}

        llm_ev_engine = get_report_nli_engine(self.llm_nli_cfg, candidates_str_ids, references_str_ids)
        processed_samples: list[NLISample] = llm_ev_engine.run()
        if llm_ev_engine.aggregated_processor_stats:
            self.meta_metrics.update(llm_ev_engine.aggregated_processor_stats)
        processed_samples_by_study_id = {sample.example_id: sample for sample in processed_samples}
        # Attach the evidence and compute the scores
        results_per_sample: list[PerSampleNLIResult] = []
        for study_id_str in candidates_str_ids.keys():
            if study_id_str not in processed_samples_by_study_id:  # Invalid processed sample
                results_per_sample.append(PerSampleNLIResult(study_id=study_id_str))
                continue
            sample = processed_samples_by_study_id[study_id_str]
            candidate_grounded_phrases = candidates_str_ids[study_id_str].get_all_grounded_phrases(
                fail_if_non_grounded_phrase=True
            )
            reference_grounded_phrases = references_str_ids[study_id_str].get_all_grounded_phrases(
                fail_if_non_grounded_phrase=True
            )
            candidate_phrases_evidenced = GroundedPhraseEvidenced.attach_evidence_to_all_hypotheses(
                evidenced_phrases=sample.output.phrases_A_evidenced,
                hypothesis_grounded_phrases=candidate_grounded_phrases,
                premise_grounded_phrases=reference_grounded_phrases,
            )
            reference_phrases_evidenced = GroundedPhraseEvidenced.attach_evidence_to_all_hypotheses(
                evidenced_phrases=sample.output.phrases_B_evidenced,
                hypothesis_grounded_phrases=reference_grounded_phrases,
                premise_grounded_phrases=candidate_grounded_phrases,
            )
            # Now we can compute the score
            score = self.compute_radfact_score(candidate_phrases_evidenced, reference_phrases_evidenced)
            results_per_sample.append(
                PerSampleNLIResult(
                    study_id=study_id_str,
                    candidate_phrases=candidate_phrases_evidenced,
                    reference_phrases=reference_phrases_evidenced,
                    scores=score,
                )
            )
        assert len(results_per_sample) == len(
            candidates_str_ids
        ), f"Mismatch between input and output samples {len(results_per_sample)=} vs {len(candidates_str_ids)=}"
        return results_per_sample

    def results_per_sample_to_dataframe(self, results_per_sample: PerSampleResultType) -> pd.DataFrame:
        """Convert the results per sample to a DataFrame. This is useful for exporting the results to a CSV file."""
        score_dicts = [asdict(score) if (score := result.scores) is not None else {} for result in results_per_sample]
        return pd.DataFrame(score_dicts)  # Missing scores from invalid results will get filled with NaN

    def results_per_sample_to_dicts(self, results_per_sample: PerSampleResultType) -> list[dict[str, Any]]:
        """Convert the results per sample to a list of dictionaries. This is useful for exporting the results to JSON
        for reviewing sample wise RadFact results."""
        return [asdict(result) for result in results_per_sample]

    def aggregate_results(self, results_per_sample: PerSampleResultType) -> ReturnType:
        """Aggregate the scores from a list of samples."""
        scores = [score for result in results_per_sample if (score := result.scores) is not None]
        aggregate_score = RadFactScore.from_aggregate(scores, only_factual_scores=self.is_narrative_text)
        aggregate_dict = asdict(aggregate_score)
        aggregate_dict['logical_f1'] = aggregate_score.logical_f1
        aggregate_dict['spatial_f1'] = 0.0 if self.is_narrative_text else aggregate_score.spatial_f1
        aggregate_dict['grounding_f1'] = 0.0 if self.is_narrative_text else aggregate_score.grounding_f1
        aggregate_dict['num_samples'] = len(scores)
        # Meta-metrics such as processor stats are not part of the per-sample scores.
        aggregate_dict.update(self.meta_metrics)
        aggregate_dict["num_invalid_processed_samples"] = sum(result.scores is None for result in results_per_sample)
        return aggregate_dict['logical_f1'], aggregate_dict

    def compute_metric_score(self, candidates: InputDict, references: InputDict) -> ReturnType:
        results_per_sample = self.compute_results_per_sample(candidates=candidates, references=references)
        return self.aggregate_results(results_per_sample)

    def reindex_results_per_sample(
        self, results_per_sample: PerSampleResultType, indices: Iterable[int]
    ) -> PerSampleResultType:
        """Reindex the results per sample. This is useful for bootstrapping the sample wise results."""
        reindexed_results = [results_per_sample[i] for i in indices]
        return reindexed_results
