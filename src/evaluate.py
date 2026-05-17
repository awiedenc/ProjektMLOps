from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np

def evaluate_classification(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    if y_proba is not None:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            if y_proba.ndim == 1:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            if y_proba.ndim == 1:
                raise ValueError("For multiclass, y_proba should be 2D (n_samples, n_classes)")
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba,
                    multi_class='ovr',
                    average='macro'
                )

    return metrics