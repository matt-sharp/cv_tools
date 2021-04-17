import numpy as np
from typing import List


def np_box_area(boxes):
    """Compute the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (np.ndarray[N, 4]): boxes for which the area will be computed.

    Returns:
        area (np.ndarray[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def np_box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y2, x2, y2) format.

    Args:
        boxes1 (np.ndarray [N,4])
        boxes2 (np.ndarray [M,4])

    Returns:
        iou (np.ndarray[N, M]): the NxM matrix containing the pairwise IoU values
        for every element in boxes1 and boxes2
    """
    area1 = np_box_area(boxes1)
    area2 = np_box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def np_half_iou(boxes1, boxes2):
    """Return intersection over boxes1

    Args:
        boxes1 (np.ndarray, [N, 4])
        boxes2 (np.ndarray, [M, 4])

    Returns:
        half_iou (np.ndarray[N, M]): the NxM matrix containing
        inter_boxes (np.ndarray[N, M, 4]): 
    """
    area1 = np_box_area(boxes1)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    half_iou = inter / area1[:, None]
    return half_iou, np.concatenate((lt, rb), axis=2)


def clip_nms(boxes1, boxes2, iou_thresh=0.5):
    half_iou, inter_bboxes = np_half_iou(boxes1, boxes2)
    keep = half_iou >= iou_thresh

    return inter_bboxes[keep]


def py_cpu_nms(dets: np.ndarray, thresh: float) -> List:
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]                     # pred bbox top_x
    y1 = dets[:, 1]                     # pred bbox top_y
    x2 = dets[:, 2]                     # pred bbox bottom_x
    y2 = dets[:, 3]                     # pred bbox bottom_y
    scores = dets[:, 4]              # pred bbox cls score

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)    # pred bbox areas
    # 对pred bbox按score做降序排序，对应step-2
    order = scores.argsort()[::-1]

    keep = []    # NMS后，保留的pred bbox
    while order.size > 0:
        i = order[0]          # top-1 score bbox
        keep.append(i)        # top-1 score的话，自然就保留了
        # top-1 bbox（score最大）与order中剩余bbox计算NMS
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] -
                       inter)      # 无处不在的IoU计算~~~

        # 这个操作可以对代码断点调试理解下，结合step-3，我们希望剔除所有与当前top-1 bbox IoU >
        # thresh的冗余bbox，那么保留下来的bbox，自然就是ovr <= thresh的非冗余bbox，其inds保留下来，作进一步筛选
        inds = np.where(ovr <= thresh)[0]
        # 保留有效bbox，就是这轮NMS未被抑制掉的幸运儿，为什么 + 1？因为ind =
        # 0就是这轮NMS的top-1，剩余有效bbox在IoU计算中与top-1做的计算，inds对应回原数组，自然要做 +1
        # 的映射，接下来就是step-4的循环
        order = order[inds + 1]

    return keep    # 最终NMS结果返回
