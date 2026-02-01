import torch
import numpy as np
from typing import List, Tuple

# =============================
# CONFIG (match training)
# =============================

WINDOW_SIZE = 0.63
OVERLAP = 0.875
FRAME_SHIFT = WINDOW_SIZE * (1 - OVERLAP)  # seconds per prediction


# =============================
# Interval utilities
# =============================

def predictions_to_intervals(
    pred_bool: torch.Tensor,
) -> List[Tuple[float, float]]:
    """
    Convert boolean predictions to time intervals in seconds.
    """
    if pred_bool.numel() == 0:
        return []

    pred_bool = pred_bool.cpu()

    changes = torch.where(pred_bool[:-1] != pred_bool[1:])[0] + 1

    if pred_bool[0]:
        changes = torch.cat([torch.tensor([0]), changes])
    if pred_bool[-1]:
        changes = torch.cat([changes, torch.tensor([len(pred_bool)])])

    intervals = []
    for s, e in zip(changes[::2], changes[1::2]):
        start = s.item() * FRAME_SHIFT
        end = e.item() * FRAME_SHIFT
        intervals.append((start, end))

    return intervals


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for cur in intervals[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], cur[1]))
        else:
            merged.append(cur)
    return merged


def intersect_intervals(A, B):
    i = j = 0
    out = []
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0])
        e = min(A[i][1], B[j][1])
        if s < e:
            out.append((s, e))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def interval_length(intervals):
    return sum(e - s for s, e in intervals)


def union_length(A, B):
    return interval_length(merge_intervals(A + B))


# =============================
# Time-based metrics
# =============================

def compute_time_metrics(
    gt_intervals,
    pred_intervals,
    total_length,
):
    gt = merge_intervals(gt_intervals)
    pred = merge_intervals(pred_intervals)

    inter = intersect_intervals(gt, pred)

    TP = interval_length(inter)
    FP = interval_length(pred) - TP
    FN = interval_length(gt) - TP
    TN = total_length - union_length(gt, pred)

    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    accuracy = (TP + TN) / total_length if total_length > 0 else 0.0

    return dict(
        TP=TP, FP=FP, FN=FN, TN=TN,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
    )


def compute_time_metrics_multi(
    gt_list,
    pred_list,
    total_lengths,
):
    TP = FP = FN = TN = 0.0
    total_time = sum(total_lengths)

    for gt, pred, L in zip(gt_list, pred_list, total_lengths):
        m = compute_time_metrics(gt, pred, L)
        TP += m["TP"]
        FP += m["FP"]
        FN += m["FN"]
        TN += m["TN"]

    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    accuracy = (TP + TN) / total_time if total_time > 0 else 0.0

    return dict(
        TP=TP, FP=FP, FN=FN, TN=TN,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
    )


# =============================
# Time-based ROC / PR
# =============================

def time_based_pr_roc(
    gt_list,
    logits_list,
    total_lengths,
    thresholds=None,
):
    if thresholds is None:
        thresholds = torch.linspace(0.0, 1.0, 51)

    precisions, recalls, fprs, tprs = [], [], [], []

    for t in thresholds:
        TP = FP = FN = TN = 0.0

        for gt, logits, L in zip(gt_list, logits_list, total_lengths):
            # probs = torch.sigmoid(torch.tensor(logits))
            probs = torch.tensor(logits)
            preds = probs >= t

            pred_intervals = predictions_to_intervals(preds)
            gt_intervals = merge_intervals(gt)

            inter = intersect_intervals(gt_intervals, pred_intervals)

            tp = interval_length(inter)
            fp = interval_length(pred_intervals) - tp
            fn = interval_length(gt_intervals) - tp
            tn = L - union_length(gt_intervals, pred_intervals)

            TP += tp
            FP += fp
            FN += fn
            TN += tn

        precision = TP / (TP + FP) if TP + FP > 0 else 1.0
        recall = TP / (TP + FN) if TP + FN > 0 else 1.0
        fpr = FP / (FP + TN) if FP + TN > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        tprs.append(recall)

    # AUROC
    order = np.argsort(fprs)
    auc = np.trapz(np.array(tprs)[order], np.array(fprs)[order])

    return dict(
        precisions=np.array(precisions),
        recalls=np.array(recalls),
        fprs=np.array(fprs),
        tprs=np.array(tprs),
        AUROC=auc,
    )
