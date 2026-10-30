"""Metric helpers shared by all tasks."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, average_precision_score)


def regression_metrics(y, p) -> dict:
    mse = mean_squared_error(y, p)
    return {"MAE": mean_absolute_error(y, p), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y, p)}


def classification_metrics(y, pred, proba=None) -> dict:
    m = {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
         "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}
    if proba is not None:
        m["ROC_AUC"] = roc_auc_score(y, proba)
        m["PR_AUC"] = average_precision_score(y, proba)
    return m
