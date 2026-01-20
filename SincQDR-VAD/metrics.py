import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
# Each interval: (start_time, end_time)
# Each scored interval: (start_time, end_time, score in [0,1])

def merge_intervals(intervals):
    """Merge overlapping or adjacent intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def interval_length(intervals):
    return sum(e - s for s, e in intervals)

def intersect_intervals(A, B):
    """Return list of intersections between interval lists A and B."""
    i, j = 0, 0
    intersections = []
    while i < len(A) and j < len(B):
        a1, a2 = A[i]
        b1, b2 = B[j]
        start, end = max(a1, b1), min(a2, b2)
        if start < end:
            intersections.append((start, end))
        if a2 < b2:
            i += 1
        else:
            j += 1
    return intersections

def union_length(A, B):
    """Length of union of two (already merged) interval lists."""
    return interval_length(merge_intervals(A + B))

# ---------------- TIME-BASED (continuous) ----------------

def compute_time_metrics(gt_segments, pred_segments, total_length):
    gt = merge_intervals(gt_segments)
    pred = merge_intervals(pred_segments)
    inter = merge_intervals(intersect_intervals(gt, pred))

    TP = interval_length(inter)
    FP = interval_length(pred) - TP
    FN = interval_length(gt) - TP
    TN = total_length - union_length(gt, pred)

    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (TP + TN) / total_length if total_length > 0 else 0.0

    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)

def compute_time_metrics_multi(gt_list, pred_list, total_lengths):
    """Micro-average across files by summing durations."""
    assert len(gt_list) == len(pred_list) == len(total_lengths)
    TP = FP = FN = TN = 0.0
    T = sum(total_lengths)
    for gt, pred, L in zip(gt_list, pred_list, total_lengths):
        m = compute_time_metrics(gt, pred, L)
        TP += m["TP"]; FP += m["FP"]; FN += m["FN"]; TN += m["TN"]
    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (TP + TN) / T if T > 0 else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)

def time_based_pr_multi(gt_list, preds_with_scores_list, total_lengths):
    """
    Build time-based PR and ROC across files.
    At each global threshold, keep segments with score>=t in each file,
    then sum durations to get TP/FP/FN/TN and compute precision/recall/FPR/TPR.
    """
    assert len(gt_list) == len(preds_with_scores_list) == len(total_lengths)
    thresholds = sorted(
        set(s for preds in preds_with_scores_list for (_, _, s) in preds),
        reverse=True
    )

    recalls, precisions, fprs, tprs = [], [], [], []
    for t in thresholds:
        TP = FP = FN = TN = 0.0
        for gt, preds_scored, L in zip(gt_list, preds_with_scores_list, total_lengths):
            gt_m = merge_intervals(gt)
            pred_m = merge_intervals([(a, b) for (a, b, s) in preds_scored if s >= t])
            inter = intersect_intervals(gt_m, pred_m)

            tp = interval_length(inter)
            p_len = interval_length(pred_m)
            g_len = interval_length(gt_m)
            fp = p_len - tp
            fn = g_len - tp
            tn = L - union_length(gt_m, pred_m)

            TP += tp; FP += fp; FN += fn; TN += tn

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tpr = recall  # duration-based TPR equals recall

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        tprs.append(tpr)

    # Add endpoints for curve completeness (optional)
    if len(recalls) == 0 or recalls[-1] != 0.0:
        recalls.append(0.0); precisions.append(1.0)
    if (0.0, 0.0) not in zip(fprs, tprs):
        fprs.append(0.0); tprs.append(0.0)

    # Average Precision (area under P(R))
    order_pr = np.argsort(recalls)
    R_sorted = np.array(recalls)[order_pr]
    P_sorted = np.array(precisions)[order_pr]
    AP = float(np.trapz(P_sorted, R_sorted))  # trapezoid over recall

    # AUC-ROC
    order_roc = np.argsort(fprs)
    F_sorted = np.array(fprs)[order_roc]
    T_sorted = np.array(tprs)[order_roc]
    AUC = float(np.trapz(T_sorted, F_sorted))

    return dict(recalls=np.array(recalls),
                precisions=np.array(precisions),
                AP=AP,
                fprs=np.array(fprs),
                tprs=np.array(tprs),
                AUC=AUC)

# ---------------- EVENT-BASED (segment-wise) ----------------

def iou(seg1, seg2):
    inter = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / union if union > 0 else 0

def event_based_metrics(gt_segments, pred_segments, iou_thresh=0.5):
    """Greedy one-to-one matching within a file using IoU threshold."""
    gt = merge_intervals(gt_segments)
    used_gt = set()
    TP = FP = 0
    for p in pred_segments:
        best_iou, best_idx = -1.0, -1
        for j, g in enumerate(gt):
            ov = iou(p, g)
            if ov > best_iou:
                best_iou, best_idx = ov, j
        if best_iou >= iou_thresh and best_idx not in used_gt:
            TP += 1
            used_gt.add(best_idx)
        else:
            FP += 1
    FN = len(gt) - len(used_gt)
    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1)

def compute_event_metrics_multi(gt_list, pred_list, iou_thresh=0.5):
    """Micro-average event counts across files (sum TP/FP/FN)."""
    assert len(gt_list) == len(pred_list)
    TP = FP = FN = 0
    for gt, pred in zip(gt_list, pred_list):
        m = event_based_metrics(gt, pred, iou_thresh=iou_thresh)
        TP += int(m["TP"]); FP += int(m["FP"]); FN += int(m["FN"])
    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1)

def event_based_pr_multi(gt_list, preds_with_scores_list, iou_thresh=0.5):
    """
    Global event-based PR: sort all predictions (from all files) by score
    and reveal them one by one, matching within their own file.
    """
    # Flatten predictions with file index
    flat = []
    for f_idx, preds in enumerate(preds_with_scores_list):
        for a, b, s in preds:
            flat.append((f_idx, a, b, s))
    flat.sort(key=lambda x: x[3], reverse=True)

    gt_merged = [merge_intervals(gt) for gt in gt_list]
    matched_gt = [set() for _ in gt_list]
    total_gt_events = sum(len(g) for g in gt_merged)

    TP = FP = 0
    precisions, recalls = [], []
    for f_idx, a, b, s in flat:
        p = (a, b)
        # Match within the same file
        best_iou, best_idx = -1.0, -1
        for j, g in enumerate(gt_merged[f_idx]):
            ov = iou(p, g)
            if ov > best_iou:
                best_iou, best_idx = ov, j
        if best_iou >= iou_thresh and best_idx not in matched_gt[f_idx]:
            matched_gt[f_idx].add(best_idx)
            TP += 1
        else:
            FP += 1
        precision = TP / (TP + FP) if TP + FP > 0 else 1.0
        recall = TP / total_gt_events if total_gt_events > 0 else 1.0
        precisions.append(precision); recalls.append(recall)

    # AP as area under P(R)
    order = np.argsort(recalls)
    R_sorted = np.array(recalls)[order]
    P_sorted = np.array(precisions)[order]
    AP = float(np.trapz(P_sorted, R_sorted))
    return dict(recalls=np.array(recalls), precisions=np.array(precisions), AP=AP)

# ---------------- DEMO (3 files) ----------------
if __name__ == "__main__":
    # Ground truth per file
    data=pd.read_csv("JavadPreds26.csv")
    # gt_list = [
    #     [(5, 15), (30, 35), (45, 55)],          # file 1 (T1=60)
    #     [(0, 8), (20, 25)],                     # file 2 (T2=45)
    #     [(10, 15), (18, 22)],                   # file 3 (T3=30)
    # ]
    data["Hypotheses"] = data["Hypotheses"].fillna("[]").apply(ast.literal_eval)
    data["Reference"]  = data["Reference"].fillna("[]").apply(ast.literal_eval)
    data["Logits"]  = data["Logits"].fillna("[]").apply(ast.literal_eval)
#    data["Length"]  = data["Length"].fillna("[]").apply(ast.literal_eval)

    gt_list=list(data["Reference"])
    prepreds=list(data["Hypotheses"])
    logs=list(data["Logits"])
    total_lengths=list(data["Length"])
    preds_with_scores_list=[[(*seg, score) for seg, score in zip(segments, scores)] 
          for segments, scores in zip(prepreds,logs)]
    print(total_lengths)
    print(preds_with_scores_list)
    # Predictions with scores per file
    # preds_with_scores_list = [
    #     [(4, 14, 0.9), (16, 18, 0.6), (44, 52, 0.8)],
    #     [(0, 6, 0.7), (10, 12, 0.4), (22, 28, 0.9)],
    #     [(9, 13, 0.5), (14, 16, 0.3), (18, 21, 0.85)],
    # ]
    # total_lengths = [60.0, 45.0, 30.0]

    # Pred lists without scores
    pred_list = [[(a, b) for (a, b, _) in preds] for preds in preds_with_scores_list]

    # ---- Time-based (micro across files)
    time_stats = compute_time_metrics_multi(gt_list, pred_list, total_lengths)
    curves_t = time_based_pr_multi(gt_list, preds_with_scores_list, total_lengths)
    print("Time-based (micro):", time_stats)
    print(f"Time-based AP={curves_t['AP']:.3f}, AUC={curves_t['AUC']:.3f}")

    # ---- Event-based (micro across files)
    event_stats = compute_event_metrics_multi(gt_list, pred_list, iou_thresh=0.5)
    curves_e = event_based_pr_multi(gt_list, preds_with_scores_list, iou_thresh=0.5)
    print("Event-based (micro):", event_stats)
    print(f"Event-based AP={curves_e['AP']:.3f}")

    # ---- Plots
    # Time-based PR
    plt.figure(figsize=(6, 4))
    plt.plot(curves_t["recalls"], curves_t["precisions"], marker='o')
    plt.xlabel("Recall (time)")
    plt.ylabel("Precision (time)")
    plt.title(f"Time-based PR (AP={curves_t['AP']:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_pr_curve.png", dpi=300)
    plt.show()

    # Time-based ROC
    plt.figure(figsize=(6, 4))
    plt.plot(curves_t["fprs"], curves_t["tprs"], marker='o')
    plt.xlabel("False Positive Rate (time)")
    plt.ylabel("True Positive Rate (time)")
    plt.title(f"Time-based ROC (AUC={curves_t['AUC']:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_roc_curve.png", dpi=300)
    plt.show()

    # Event-based PR
    plt.figure(figsize=(6, 4))
    plt.plot(curves_e["recalls"], curves_e["precisions"], marker='s')
    plt.xlabel("Recall (events)")
    plt.ylabel("Precision (events)")
    plt.title(f"Event-based PR (AP={curves_e['AP']:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("event_pr_curve.png", dpi=300)
    plt.show()


