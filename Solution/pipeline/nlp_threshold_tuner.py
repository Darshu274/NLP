"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def choose_threshold(scores, labels, target_precision=0.90):
    """
    Select an optimal decision threshold based on precision constraints.

    The function evaluates multiple candidate thresholds derived from the
    score distribution. It selects the smallest threshold that achieves at
    least the specified target precision while maximizing recall. If no
    threshold satisfies the precision constraint, the threshold that
    maximizes the F1-score is returned instead.

    Parameters
    ----------
    scores : array-like
        Continuous prediction scores or confidence values.
    labels : array-like
        Ground-truth binary labels (0 or 1).
    target_precision : float, optional
        Minimum required precision for threshold selection.
        Default is 0.90.

    Returns
    -------
    tuple
        A tuple ``(threshold, precision, recall, f1)`` representing the
        selected threshold and its corresponding evaluation metrics.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    cand = np.unique(np.round(scores, 4))
    best = None
    best_f1 = -1

    for t in cand:
        pred = (scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        if p >= target_precision:
            # pick the one with max recall among those meeting precision
            if best is None or r > best[1]:
                best = (t, p, r, f1)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_tuple = (t, p, r, f1)

    return best if best is not None else best_f1_tuple
