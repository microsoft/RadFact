#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
import numpy.typing as npt

from radfact.metric.box_metrics import IOU, PRECISION, RECALL, compute_box_metrics, get_mask_from_boxes
from radfact.data_utils.grounded_phrase_list import NormalizedBox


def assert_score(metric_name: str, expected_score: float, actual_score: float) -> None:
    assert isinstance(actual_score, float)
    is_close = np.isclose(expected_score, actual_score, equal_nan=True)
    message = "metric: {}, expected: {}, actual: {}".format(metric_name, expected_score, actual_score)
    assert is_close, message


def _get_box_names() -> list[str]:
    return ["box1", "box2", "box3", "box4", "box5", "box6", "box7", "box8", "box9", "box10", "box10"]


def _get_box_pairs() -> list[tuple[str, str]]:
    return [
        ("box5", "box6"),  # we assume (gt_box, pred_box)
        ("box2", "box7"),
        ("box3", "box8"),
        ("box4", "box4"),
        ("box5", "box7"),
        ("box9", "box10"),
        ("box10", "box11"),
    ]


def _get_box(key: str) -> NormalizedBox:
    box1 = NormalizedBox(0.1, 0.1, 0.3, 0.3)
    box2 = NormalizedBox(0.2, 0.2, 0.4, 0.4)
    box3 = NormalizedBox(0.5, 0.5, 0.7, 0.7)
    box4 = NormalizedBox(0.6, 0.6, 0.8, 0.8)
    box5 = NormalizedBox(0.3, 0.3, 0.5, 0.5)
    box6 = NormalizedBox(0.4, 0.4, 0.6, 0.6)
    box7 = NormalizedBox(0.1, 0.1, 0.4, 0.4)
    box8 = NormalizedBox(0.7, 0.7, 0.9, 0.9)
    box9 = NormalizedBox(0.3, 0.6, 0.5, 0.8)
    box10 = NormalizedBox(0.3, 0.65, 0.5, 0.85)
    box11 = NormalizedBox(0.0, 0.0, 0.0, 0.0)
    return {
        "box1": box1,
        "box2": box2,
        "box3": box3,
        "box4": box4,
        "box5": box5,
        "box6": box6,
        "box7": box7,
        "box8": box8,
        "box9": box9,
        "box10": box10,
        "box11": box11,
    }[key]


def _get_mask_for_box(key: str) -> npt.NDArray[np.bool_]:
    mask_size = 224
    mask = np.zeros((224, 224), dtype=np.bool_)
    box = _get_box(key)
    box_coord = (box.x_min, box.y_min, box.x_max, box.y_max)
    x1, y1, x2, y2 = (np.array(box_coord) * mask_size).astype(int)
    mask[x1:x2, y1:y2] = True
    return mask


def test_get_mask_from_boxes() -> None:
    boxes: list[str] = _get_box_names()
    for box in boxes:
        expected = _get_mask_for_box(box)
        actual = get_mask_from_boxes([_get_box(box)])
    assert np.array_equal(actual, expected)


def get_iou_precision_recall_for_box(key1: str, key2: str) -> dict[str, float]:  #: key1 is pred box and key2 is gt box
    box1_mask = _get_mask_for_box(key1)
    box2_mask = _get_mask_for_box(key2)
    pred_area = box1_mask.sum()
    true_area = box2_mask.sum()
    inter_area = (box1_mask & box2_mask).sum()
    union_area = (box1_mask | box2_mask).sum()
    iou = inter_area / union_area
    if pred_area > 0:
        precision = inter_area / pred_area
    else:
        precision = np.nan
    recall = inter_area / true_area
    return {
        IOU: iou,
        PRECISION: precision,
        RECALL: recall,
    }


def test_compute_box_metrics() -> None:
    box_pairs_list: list[tuple[str, str]] = _get_box_pairs()
    # Test for cases where the pred. and gt. has one box. Tests for perfect overlap/zero overlap & intermediate cases
    for box_pairs in box_pairs_list:
        expected = get_iou_precision_recall_for_box(box_pairs[1], box_pairs[0])
        actual = compute_box_metrics([_get_box(box_pairs[1])], [_get_box(box_pairs[0])], mask_size=224)
    assert_score(IOU, expected_score=expected[IOU], actual_score=round(actual[IOU], 2))
    assert_score(PRECISION, expected_score=expected[PRECISION], actual_score=round(actual[PRECISION], 2))
    assert_score(RECALL, expected_score=expected[RECALL], actual_score=round(actual[RECALL], 2))

    # Test for cases where the pred. and gt. has two boxes and perfect overlap
    actual = compute_box_metrics(
        [_get_box(box_pairs_list[0][0]), _get_box(box_pairs_list[0][1])],
        [_get_box(box_pairs_list[0][0]), _get_box(box_pairs_list[0][1])],
        mask_size=224,
    )
    assert_score(IOU, expected_score=1.0, actual_score=round(actual[IOU], 2))
    assert_score(PRECISION, expected_score=1.0, actual_score=round(actual[PRECISION], 2))
    assert_score(RECALL, expected_score=1.0, actual_score=round(actual[RECALL], 2))

    # Test for cases where the pred. and gt. has two boxes and zero overlap
    actual = compute_box_metrics(
        [_get_box(box_pairs_list[2][0]), _get_box(box_pairs_list[2][0])],
        [_get_box(box_pairs_list[2][1]), _get_box(box_pairs_list[2][1])],
        mask_size=224,
    )
    assert_score(IOU, expected_score=0, actual_score=round(actual[IOU], 2))
    assert_score(PRECISION, expected_score=0, actual_score=round(actual[PRECISION], 2))
    assert_score(RECALL, expected_score=0, actual_score=round(actual[RECALL], 2))

    # Test for cases where the pred. and gt. has two boxes and intermediate overlap
    actual = compute_box_metrics(
        [_get_box(box_pairs_list[0][0]), _get_box(box_pairs_list[0][1])],
        [_get_box(box_pairs_list[1][0]), _get_box(box_pairs_list[1][1])],
        mask_size=224,
    )
    assert_score(IOU, expected_score=0.06, actual_score=round(actual[IOU], 2))
    assert_score(PRECISION, expected_score=0.14, actual_score=round(actual[PRECISION], 2))
    assert_score(RECALL, expected_score=0.11, actual_score=round(actual[RECALL], 2))


def test_compute_box_metrics_zero_area_box() -> None:
    # Make sure the metrics produce what we expect when there is a zero area box in both gt and prediction
    pred_has_zero_area = compute_box_metrics(
        pred_boxes=[_get_box("box11")], true_boxes=[_get_box("box1")], mask_size=224
    )
    assert_score(IOU, expected_score=0, actual_score=round(pred_has_zero_area[IOU], 2))
    assert_score(PRECISION, expected_score=np.nan, actual_score=round(pred_has_zero_area[PRECISION], 2))
    assert_score(RECALL, expected_score=0, actual_score=round(pred_has_zero_area[RECALL], 2))

    gt_has_zero_area = compute_box_metrics(pred_boxes=[_get_box("box1")], true_boxes=[_get_box("box11")], mask_size=224)
    assert_score(IOU, expected_score=0, actual_score=round(gt_has_zero_area[IOU], 2))
    assert_score(PRECISION, expected_score=0, actual_score=round(gt_has_zero_area[PRECISION], 2))
    assert_score(RECALL, expected_score=np.nan, actual_score=round(gt_has_zero_area[RECALL], 2))
