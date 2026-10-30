"""Task B - IsCanceled classification: UNTUNED baselines (Milestone 1). Train (CV) + validation only."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import classification_metrics
from src.preprocessing.pipelines import build_supervised_preprocessor
from src.utils import out_dir, task_features

MODELS = {
    "dummy_majority (reference)": lambda seed: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
}


def run_classification_baseline(splits: dict, cfg: dict) -> pd.DataFrame:
    seed, d = cfg["seed"], out_dir("classification")
    feats = task_features(cfg, "classification"); tgt = cfg["tasks"]["classification"]["target"]
    tr, va = splits["train"], splits["val"]
    Xtr, ytr, Xva, yva = tr[feats], tr[tgt], va[feats], va[tgt]
    print(f"[classification] train={len(tr):,} val={len(va):,} features={len(feats)}  positive rate={ytr.mean():.3f}")
    cv = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=seed)
    rows = []
    for name, make in MODELS.items():
        pipe = Pipeline([("prep", build_supervised_preprocessor(cfg, "classification")), ("model", make(seed))])
        cvr = cross_validate(pipe, Xtr, ytr, cv=cv, n_jobs=-1, scoring=["accuracy", "f1", "roc_auc"])
        t0 = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
        pred, proba = pipe.predict(Xva), pipe.predict_proba(Xva)[:, 1]
        m = classification_metrics(yva, pred, proba)
        rows.append({"model": name, "config": "baseline (untuned)",
                     "CV_Accuracy": cvr["test_accuracy"].mean(), "CV_F1": cvr["test_f1"].mean(), "CV_ROC_AUC": cvr["test_roc_auc"].mean(),
                     **{f"VAL_{k}": v for k, v in m.items()}, "fit_time_s": fit_s})
        if name == "logistic_regression":
            cm = confusion_matrix(yva, pred); np.savetxt(d / "logreg_confusion_matrix_val.csv", cm, fmt="%d", delimiter=",")
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay(cm, display_labels=["Not canceled", "Canceled"]).plot(ax=ax, colorbar=False)
            ax.set_title("Logistic Regression - validation"); ax.grid(False)
            fig.tight_layout(); fig.savefig(d / "logreg_confusion_matrix_val.png"); plt.close(fig)
            coefs = pd.Series(pipe.named_steps["model"].coef_[0], index=pipe.named_steps["prep"].get_feature_names_out())
            coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(20).to_csv(d / "logreg_top_coefficients.csv")
    res = pd.DataFrame(rows).round(4)
    res.to_csv(d / "baseline_results.csv", index=False)
    print(res.to_string(index=False))
    return res
