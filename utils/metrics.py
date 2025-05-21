import torch
from torchvision.ops import box_iou

def get_detection(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    y_true = []
    y_pred = []

    matched_gt = set()
    ious = box_iou(pred_boxes,gt_boxes) if len(pred_boxes) > 0 and len(gt_boxes) > 0 else torch.zeros((len(pred_boxes), len(gt_boxes)))
    
    for pred_idx in range(len(pred_labels)):
        if ious.shape[1] == 0:
            # No ground truth
            y_true.append(0)
            y_pred.append(1)
            continue
        max_iou, gt_idx = torch.max(ious[pred_idx], dim=0)
        if max_iou >= iou_threshold and gt_labels[gt_idx] == pred_labels[pred_idx] and gt_idx.item() not in matched_gt:
            y_true.append(1)  # True object
            y_pred.append(1)  # Correct prediction
            matched_gt.add(gt_idx.item())
        else:
            y_true.append(0)  # No matching GT
            y_pred.append(1)  # But predicted something → FP
    return y_pred,y_true


