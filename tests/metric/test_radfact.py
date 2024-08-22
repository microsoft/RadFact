#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import shutil
from pathlib import Path

import mock
import pandas as pd
import pytest
from numpy.testing import assert_equal
from omegaconf import DictConfig

from radfact.data_utils.grounded_phrase_list import GroundedPhrase, GroundedPhraseList, NormalizedBox
from radfact.llm_utils.engine.engine import LLMEngine
from radfact.llm_utils.nli.processor import RADFACT_SUBFOLDER
from radfact.llm_utils.nli.schema import (
    BidirectionalEvidence,
    ComparisonQuery,
    EvidencedPhrase,
    EVState,
    NLIQuerySample,
    NLISample,
    NLIState,
)
from radfact.llm_utils.report_to_phrases.processor import StudyIdType
from radfact.llm_utils.report_to_phrases.schema import ParsedReport, SentenceWithRephrases
from radfact.metric.radfact import RadFactMetric
from radfact.metric.schema import GroundedPhraseEvidenced, OneWayNLIFractions, SpatialEntailmentStatus

candidate = GroundedPhraseList(
    [
        GroundedPhrase("The cat", boxes=[NormalizedBox(0.1, 0.1, 0.2, 0.2)]),
        GroundedPhrase(
            "The dog",
            boxes=[
                NormalizedBox(0.3, 0.3, 0.4, 0.4),
                NormalizedBox(0.5, 0.5, 0.6, 0.6),
            ],
        ),
        GroundedPhrase("The bird"),
        GroundedPhrase("The rabbit"),
    ]
)

reference = GroundedPhraseList(
    [
        GroundedPhrase("The cat", boxes=[NormalizedBox(0.1, 0.1, 0.2, 0.2)]),
        GroundedPhrase("The dog", boxes=[NormalizedBox(0.3, 0.3, 0.4, 0.4)]),
        GroundedPhrase("The bird", boxes=[NormalizedBox(0.5, 0.5, 0.6, 0.6)]),
        GroundedPhrase("The shark"),
    ]
)

expected_bidirectional_evidence = BidirectionalEvidence(
    phrases_A_evidenced=[
        EvidencedPhrase(phrase="The cat", status=EVState.ENTAILMENT, evidence=["The cat"]),
        EvidencedPhrase(phrase="The dog", status=EVState.ENTAILMENT, evidence=["The dog"]),
        EvidencedPhrase(phrase="The bird", status=EVState.ENTAILMENT, evidence=["The bird"]),
        EvidencedPhrase(phrase="The rabbit", status=EVState.NOT_ENTAILMENT, evidence=[]),
    ],
    phrases_B_evidenced=[
        EvidencedPhrase(phrase="The cat", status=EVState.ENTAILMENT, evidence=["The cat"]),
        EvidencedPhrase(phrase="The dog", status=EVState.ENTAILMENT, evidence=["The dog"]),
        EvidencedPhrase(phrase="The bird", status=EVState.ENTAILMENT, evidence=["The bird"]),
        EvidencedPhrase(phrase="The shark", status=EVState.NOT_ENTAILMENT, evidence=[]),
    ],
)

candidates: dict[StudyIdType, GroundedPhraseList] = {"study1": candidate}
references: dict[StudyIdType, GroundedPhraseList] = {"study1": reference}

candidates_narrative: dict[StudyIdType, str] = {"study1": "The cat The dog The bird The rabbit"}
references_narrative: dict[StudyIdType, str] = {"study1": "The cat The dog The bird The shark"}


def test_attach_evidence() -> None:
    candidate_phrases = candidate.get_all_grounded_phrases()
    reference_phrases = reference.get_all_grounded_phrases()
    evidenced_candidate = GroundedPhraseEvidenced.attach_evidence_to_all_hypotheses(
        hypothesis_grounded_phrases=candidate_phrases,
        premise_grounded_phrases=reference_phrases,
        evidenced_phrases=expected_bidirectional_evidence.phrases_A_evidenced,
    )

    def get_evidenced(i: int) -> GroundedPhraseEvidenced:
        status = expected_bidirectional_evidence.phrases_A_evidenced[i].status
        return GroundedPhraseEvidenced(
            text=candidate_phrases[i].text,
            boxes=candidate_phrases[i].boxes,
            status=status,
            evidence=[reference_phrases[i]] if status != EVState.NOT_ENTAILMENT else [],
            evidence_indices=[i] if status != EVState.NOT_ENTAILMENT else [],
        )

    expected_evidenced_candidate = [get_evidenced(i) for i in range(len(candidate_phrases))]
    assert evidenced_candidate == expected_evidenced_candidate


def test_get_spatial_entailment_status() -> None:
    metric = RadFactMetric()

    box_A = NormalizedBox(0.1, 0.1, 0.2, 0.2)
    box_B = NormalizedBox(0.3, 0.3, 0.4, 0.4)  # Does not overlap box_A
    # Phrase is entailed but has no box
    entailed_nobox = GroundedPhraseEvidenced(
        text="phrase1",
        boxes=None,
        status=EVState.ENTAILMENT,
        evidence=[GroundedPhrase("evidence_1", boxes=None)],
    )
    # Phrase is entailed and has a box (box is also entailed)
    entailed_box = GroundedPhraseEvidenced(
        text="phrase2",
        boxes=[box_A],
        status=EVState.ENTAILMENT,
        evidence=[GroundedPhrase("evidence_2", boxes=[box_A])],
    )
    # Phrase is entailed and has a box (box is not entailed)
    entailed_box_notentailed = GroundedPhraseEvidenced(
        text="phrase2.5",
        boxes=[box_A],
        status=EVState.ENTAILMENT,
        evidence=[GroundedPhrase("evidence_2.5", boxes=[box_B])],
    )
    # Phrase is not entailed and has no box
    notentailed_nobox = GroundedPhraseEvidenced(text="phrase3", boxes=None, status=EVState.NOT_ENTAILMENT, evidence=[])
    # Phrase is not entailed and does have a box
    notentailed_box = GroundedPhraseEvidenced(text="phrase4", boxes=[box_A], status=EVState.NOT_ENTAILMENT, evidence=[])

    for phrase, expected_spatial_entailment_status in [
        (entailed_nobox, SpatialEntailmentStatus.NO_BOXES),
        (entailed_box, SpatialEntailmentStatus.SPATIAL_ENTAILMENT),
        (entailed_box_notentailed, SpatialEntailmentStatus.NO_SPATIAL_ENTAILMENT),
        (notentailed_nobox, SpatialEntailmentStatus.NO_BOXES),
        (notentailed_box, SpatialEntailmentStatus.NO_SPATIAL_ENTAILMENT),
    ]:
        assert metric._get_spatial_entailment_status(phrase) == expected_spatial_entailment_status


def test_compute_radfact_score_oneway() -> None:
    metric = RadFactMetric()

    box_A = NormalizedBox(0.1, 0.1, 0.2, 0.2)
    box_B = NormalizedBox(0.3, 0.3, 0.4, 0.4)  # Does not overlap box_A
    phrases: list[GroundedPhraseEvidenced] = [
        GroundedPhraseEvidenced(
            text="entailed phrase no boxes",
            boxes=None,
            status=EVState.ENTAILMENT,
            evidence=[GroundedPhrase("evidence without box", boxes=None)],
        ),
        GroundedPhraseEvidenced(
            text="entailed phrase with correct box",
            boxes=[box_A],
            status=EVState.ENTAILMENT,
            evidence=[GroundedPhrase("evidence with box", boxes=[box_A])],
        ),
        GroundedPhraseEvidenced(
            text="entailed phrase with wrong box",
            boxes=[box_B],
            status=EVState.ENTAILMENT,
            evidence=[GroundedPhrase("evidence with box", boxes=[box_A])],
        ),
        GroundedPhraseEvidenced(
            text="not entailed phrase no boxes",
            boxes=None,
            status=EVState.NOT_ENTAILMENT,
            evidence=[],
        ),
        GroundedPhraseEvidenced(
            text="not entailed phrase with box",
            boxes=[box_A],
            status=EVState.NOT_ENTAILMENT,
            evidence=[],
        ),
    ]

    expected_values = OneWayNLIFractions(
        entailed_fraction=3 / 5,  # 3 entailed out of 5
        entailed_box_fraction=1 / 2,  # 2 entailed phrases with boxes, 1 is right
        full_box_fraction=1 / 3,  # 3 phrases with boxes, 1 is right
        num_phrases=5,
        num_phrases_with_boxes=3,
    )

    result = metric.compute_radfact_score_oneway(phrases)
    assert expected_values == result


@pytest.fixture
def mock_nli_engine() -> mock.Mock:
    study_id = "study1"
    query_sample = NLIQuerySample.from_grounded_phrases_list_pair(
        example_id=study_id, candidate=candidates[study_id], reference=references[study_id]
    )
    processed_sample = NLISample(
        example_id=query_sample.example_id, input=query_sample.input, output=expected_bidirectional_evidence
    )
    mock_llm_engine = mock.Mock()
    mock_llm_engine.run.return_value = [processed_sample]
    mock_llm_engine.aggregated_processor_stats = {
        'num_llm_failures': 0,
        'num_llm_phrase_rewrites': 0,
        'num_llm_success': 8,
    }
    return mock_llm_engine


def test_nli_processing_with_endpoint(mock_nli_engine: mock.Mock) -> None:
    """Test that the metric works end-to-end, when connecting to an actual endpoint."""
    # Subsequent runs of the metric save progress information. Delete that so that we always compute afresh
    progress_subfolder = Path(LLMEngine.OUTPUT_FILES_PREFIX) / RADFACT_SUBFOLDER
    shutil.rmtree(progress_subfolder, ignore_errors=True)
    metric = RadFactMetric()
    with mock.patch('radfact.metric.radfact.get_report_nli_engine', return_value=mock_nli_engine):
        result, details = metric.compute_metric_score(candidates, references)

    assert isinstance(result, float)
    assert result == 0.75
    assert isinstance(details, dict)
    len_candidates = len(candidates["study1"].get_all_grounded_phrases())
    len_references = len(references["study1"].get_all_grounded_phrases())
    assert details == {
        "logical_precision": 0.75,
        "logical_recall": 0.75,
        "spatial_precision": 0.5,
        "spatial_recall": 0.6666666666666666,
        "grounding_precision": 0.5,
        "grounding_recall": 0.6666666666666666,
        "num_candidate_phrases": float(len_candidates),
        "num_reference_phrases": float(len_references),
        "num_candidate_phrases_with_boxes": 2.0,
        "num_reference_phrases_with_boxes": 3.0,
        "logical_f1": 0.75,
        "spatial_f1": 0.5714285714285715,
        "grounding_f1": 0.5714285714285715,
        "num_samples": 1,
        "num_llm_failures": 0,
        "num_llm_success": len_candidates + len_references,
        "num_llm_phrase_rewrites": 0,
        "num_invalid_processed_samples": 0,
    }


def get_mock_phrase_engine(llm_phrase_cfg: DictConfig, df: pd.DataFrame) -> mock.Mock:
    mock_phrase_engine = mock.Mock()
    if df["FINDINGS"].values[0] == "The cat The dog The bird The rabbit":
        mock_phrase_engine.run.return_value = [
            ParsedReport(
                id="study1",
                sentence_list=[
                    SentenceWithRephrases(
                        orig="The cat The dog The bird The rabbit",
                        new=["The cat", "The dog", "The bird", "The rabbit"],
                    )
                ],
            )
        ]
    else:
        mock_phrase_engine.run.return_value = [
            ParsedReport(
                id="study1",
                sentence_list=[
                    SentenceWithRephrases(
                        orig="The cat The dog The bird The shark",
                        new=["The cat", "The dog", "The bird", "The shark"],
                    )
                ],
            )
        ]
    mock_phrase_engine.aggregated_processor_stats = {'num_failures': 0, 'num_success': 1}
    return mock_phrase_engine


def test_nli_processing_with_endpoint_and_report_to_phrases(mock_nli_engine: mock.Mock) -> None:
    """Test that the RadFact metric works end-to-end, with a mocked engine including report-to-phrases processing."""
    progress_subfolder = Path(LLMEngine.OUTPUT_FILES_PREFIX) / RADFACT_SUBFOLDER
    shutil.rmtree(progress_subfolder, ignore_errors=True)
    metric = RadFactMetric(is_narrative_text=True)
    with mock.patch('radfact.metric.radfact.get_report_nli_engine', return_value=mock_nli_engine):
        with mock.patch('radfact.metric.radfact.get_report_to_phrases_engine', side_effect=get_mock_phrase_engine):
            result, details = metric.compute_metric_score(candidates_narrative, references_narrative)
    assert isinstance(result, float)
    assert result == 0.75
    assert isinstance(details, dict)
    expected_details = {
        "logical_precision": 0.75,
        "logical_recall": 0.75,
        "spatial_precision": 0.0,
        "spatial_recall": 0.0,
        "grounding_precision": 0.0,
        "grounding_recall": 0.0,
        "num_candidate_phrases": 4,
        "num_reference_phrases": 4,
        "num_candidate_phrases_with_boxes": 0,
        "num_reference_phrases_with_boxes": 0,
        "logical_f1": 0.75,
        "spatial_f1": 0.0,
        "grounding_f1": 0.0,
        "num_samples": 1,
        "num_llm_failures": 0,
        "num_llm_success": 8,
        "num_llm_phrase_rewrites": 0,
        "num_invalid_processed_samples": 0,
        "report_to_phrases/generations/num_failures": 0,
        "report_to_phrases/generations/num_success": 1,
        "report_to_phrases/ground_truth/num_failures": 0,
        "report_to_phrases/ground_truth/num_success": 1,
        "report_to_phrases/num_dropped_candidates": 0,
        "report_to_phrases/num_dropped_references": 0,
    }
    assert_equal(actual=details, desired=expected_details, verbose=True)


def test_convert_input_to_multimodal() -> None:
    """
    Test that we can convert the input to a multimodal grounded sequence correctly.
    """
    metric = RadFactMetric()
    # If it's already multimodal, it should return the same thing
    converted_candidates, converted_references = metric.convert_input_to_multimodal(candidates, references)
    assert converted_candidates == candidates
    assert converted_references == references

    # If it's a narrative text, it should convert it to multimodal.
    converted_candidates_narrative, converted_references_narrative = metric.convert_input_to_multimodal(
        candidates_narrative, references_narrative
    )
    for study_id in candidates:
        assert converted_candidates_narrative[study_id].get_all_text() == candidates[study_id].get_all_text()
    for study_id in references:
        assert converted_references_narrative[study_id].get_all_text() == references[study_id].get_all_text()


def test_normalize_phrases() -> None:
    """Test if punctuation differences and case differences are handled correctly"""
    phrase_A1 = "The cat"
    phrase_A2 = "The dog."
    phrase_B1 = "the cat"
    phrase_B2 = "the dog"
    comparison_query = ComparisonQuery(phrases_A=[phrase_A1, phrase_A2], phrases_B=[phrase_B1, phrase_B2])
    evidence = BidirectionalEvidence(
        phrases_A_evidenced=[
            EvidencedPhrase(phrase=phrase_A1, evidence=["the cat"], status=NLIState.ENTAILMENT),
            EvidencedPhrase(phrase=phrase_A2, evidence=["The dog"], status=NLIState.ENTAILMENT),
        ],
        phrases_B_evidenced=[
            EvidencedPhrase(phrase=phrase_B1, evidence=[phrase_A1], status=NLIState.ENTAILMENT),
            EvidencedPhrase(phrase=phrase_B2, evidence=[phrase_A2], status=NLIState.ENTAILMENT),
        ],
    )
    # This test passes if the validators of the NliSample do not raise an exception
    NLISample(example_id="id", input=comparison_query, output=evidence)


def test_per_sample_results() -> None:
    metric = RadFactMetric()

    duplicated_candidates = candidates | {"study2": candidate}
    duplicated_references = references | {"study2": reference}
    query_samples = [
        NLIQuerySample.from_grounded_phrases_list_pair(
            example_id=study_id,  # type: ignore
            candidate=duplicated_candidates[study_id],
            reference=duplicated_references[study_id],
        )
        for study_id in duplicated_candidates
    ]
    processed_samples = [
        NLISample(
            example_id=query_samples[0].example_id,
            input=query_samples[0].input,
            output=expected_bidirectional_evidence,
        ),
    ]

    mock_llm_engine = mock.Mock()
    mock_llm_engine.run.return_value = processed_samples
    mock_llm_engine.aggregated_processor_stats = {}
    with mock.patch('radfact.metric.radfact.get_report_nli_engine', return_value=mock_llm_engine):
        results_per_sample = metric.compute_results_per_sample(duplicated_candidates, duplicated_references)

    assert len(results_per_sample) == 2
    assert all(result.study_id == study_id for result, study_id in zip(results_per_sample, duplicated_candidates))
    assert results_per_sample[0].scores is not None
    assert results_per_sample[1].scores is None

    per_sample_df = metric.results_per_sample_to_dataframe(results_per_sample)
    assert per_sample_df.shape == (2, 10)
    assert per_sample_df.iloc[0].notna().all()
    assert per_sample_df.iloc[1].isna().all()

    per_sample_dicts = metric.results_per_sample_to_dicts(results_per_sample)
    assert len(per_sample_dicts) == 2
    for result_dict in per_sample_dicts:
        assert set(result_dict.keys()) == {'study_id', 'scores', 'candidate_phrases', 'reference_phrases'}


def test_convert_narrative_text_to_phrases() -> None:
    metric = RadFactMetric(is_narrative_text=True)
    input_texts: dict[StudyIdType, GroundedPhraseList] = {
        "study1": GroundedPhraseList(
            [
                GroundedPhrase("No pneumothorax or pleural effusion."),
            ]
        ),
    }
    expected_texts = {
        "study1": GroundedPhraseList(
            [
                GroundedPhrase("No pneumothorax."),
                GroundedPhrase("No pleural effusion."),
            ]
        ),
    }
    metric_prefix = "report_to_phrases"

    mock_phrase_engine = mock.Mock()
    mock_phrase_engine.run.return_value = [
        ParsedReport(
            id="study1",
            sentence_list=[
                SentenceWithRephrases(
                    orig="No pneumothorax or pleural effusion.", new=["No pneumothorax.", "No pleural effusion."]
                )
            ],
        )
    ]
    mock_phrase_engine.aggregated_processor_stats = {'num_failures': 0, 'num_success': 1}

    with mock.patch('radfact.metric.radfact.get_report_to_phrases_engine', return_value=mock_phrase_engine):
        processed_texts = metric.convert_narrative_text_to_phrases(input_texts, metric_prefix)
    assert processed_texts == expected_texts
