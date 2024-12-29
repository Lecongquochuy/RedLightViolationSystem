import numpy as np
import cv2
def apply_nms(bboxes, scores, iou_threshold=0.7):
    bboxes = np.array(bboxes)
    scores = np.array(scores)

    if len(bboxes) == 0:
        return [], []

    indices = cv2.dnn.NMSBoxes(
        bboxes.tolist(), scores.tolist(), score_threshold=0.25, nms_threshold=iou_threshold
    )

    if indices is None or len(indices) == 0:
        return [], []

    indices = indices.flatten()  # Đảm bảo indices là một danh sách phẳng
    return [bboxes[i] for i in indices], [scores[i] for i in indices]