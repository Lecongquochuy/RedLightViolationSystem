import numpy as np
def non_max_suppression_fast(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return detections

    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]

    indices = np.argsort(scores)[::-1]  # Sắp xếp theo confidence giảm dần
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        union = ((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1) +
                 (x2[indices[1:]] - x1[indices[1:]] + 1) *
                 (y2[indices[1:]] - y1[indices[1:]] + 1) - intersection)
        iou = intersection / union

        indices = indices[np.where(iou <= iou_threshold)[0] + 1]

    return detections[keep]
