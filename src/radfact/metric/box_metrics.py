#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging

import numpy as np
import numpy.typing as npt

from radfact.data_utils.grounded_phrase_list import NormalizedBox


IOU = "iou"
PRECISION = "precision"
RECALL = "recall"

logger = logging.getLogger(__name__)


def get_mask_from_boxes(boxes: list[NormalizedBox], mask_size: int = 224) -> npt.NDArray[np.bool_]:
    """Gets a pixel mask from a list of boxes.

    It creates a numpy array of zeros with the shape of (mask_size, mask_size). It then iterates over
    the boxes and sets the corresponding pixels to True in the mask. It returns the mask as a numpy array.

    :param boxes: A list of Box objects to convert into a mask.
    :param mask_size: The image size of the mask. Defaults to 224.
    :returns: A numpy array of boolean values representing the pixel mask.
    """
    mask = np.zeros((mask_size, mask_size), dtype=np.bool_)
    for box in boxes:
        box_coord = (box.x_min, box.y_min, box.x_max, box.y_max)
        x1, y1, x2, y2 = (np.array(box_coord) * mask_size).astype(int)
        mask[x1:x2, y1:y2] = True
    return mask


def compute_box_metrics(
    pred_boxes: list[NormalizedBox], true_boxes: list[NormalizedBox], mask_size: int
) -> dict[str, float]:
    """Computes the IOU, precision, and recall scores for a pair of box lists.
    It converts the boxes into pixel masks and calculates the prediction area, ground truth area, intersection area
    and union area. It then returns a dictionary of IOU, precision, and recall scores based on these areas.

    :param pred_boxes: A list of Box objects for the prediction boxes.
    :param true_boxes: A list of Box objects for the ground truth boxes.
    :param mask_size: The image size of the masks.

    :returns: A dictionary of IOU, precision, and recall scores.
    """
    pred_mask = get_mask_from_boxes(pred_boxes, mask_size)
    true_mask = get_mask_from_boxes(true_boxes, mask_size)

    pred_area = pred_mask.sum()
    true_area = true_mask.sum()
    if true_area <= 0:
        logger.warning(f"WARNING: True area is not positive, {true_area=}. {true_boxes=}, {pred_boxes=}")
    intersection_area = (pred_mask & true_mask).sum()
    union_area = (pred_mask | true_mask).sum()
    iou = intersection_area / union_area
    if pred_area > 0:
        precision = intersection_area / pred_area
    else:
        precision = np.nan
    if true_area > 0:
        recall = intersection_area / true_area
    else:
        recall = np.nan
    return {
        IOU: iou,
        PRECISION: precision,
        RECALL: recall,
    }
