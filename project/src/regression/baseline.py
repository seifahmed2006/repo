"""Task A - ADR regression: UNTUNED baselines (Milestone 1). Only train (CV) and validation are used."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.loader import task_view
from src.evaluation.metrics import regression_metrics
from src.preprocessing.pipelines import build_supervised_preprocessor
from src.utils import out_dir, task_features

MODELS = {
    "dummy_mean (reference)": lambda seed: DummyRegressor(strategy="mean"),
    "linear_regression": lambda seed: LinearRegression(),
}


def run_regression_baseline(splits: dict, cfg: dict) -> pd.DataFrame:
    seed, d = cfg["seed"], out_dir("regression")
    feats = task_features(cfg, "regression"); tgt = cfg["tasks"]["regression"]["target"]
    tr, va = task_view(splits["train"], cfg, "regression"), task_view(splits["val"], cfg, "regression")
    Xtr, ytr, Xva, yva = tr[feats], tr[tgt], va[feats], va[tgt]
    print(f"[regression] train={len(tr):,} val={len(va):,} features={len(feats)}  (ADR>0 only)")
    cv = KFold(cfg["split"]["cv_folds"], shuffle=True, random_state=seed)
    rows, best_pred = [], None
    for name, make in MODELS.items():
        pipe = Pipeline([("prep", build_supervised_preprocessor(cfg, "regression")), ("model", make(seed))])
        cvr = cross_validate(pipe, Xtr, ytr, cv=cv, n_jobs=-1,
                             scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
        t0 = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
        pred = pipe.predict(Xva); m = regression_metrics(yva, pred)
        rows.append({"model": name, "config": "baseline (untuned)",
                     "CV_MAE": -cvr["test_neg_mean_absolute_error"].mean(), "CV_MAE_std": cvr["test_neg_mean_absolute_error"].std(),
                     "CV_RMSE": -cvr["test_neg_root_mean_squared_error"].mean(), "CV_R2": cvr["test_r2"].mean(),
                     **{f"VAL_{k}": v for k, v in m.items()}, "fit_time_s": fit_s,
                     "n_features_after_prep": pipe.named_steps["prep"].get_feature_names_out().shape[0]})
        if name == "linear_regression":
            best_pred = pred   # (coefficients not exported: full one-hot + intercept is collinear, so OLS weights are not interpretable)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(d / "baseline_results.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(yva, best_pred, s=4, alpha=0.3); lim = [0, max(yva.max(), best_pred.max())]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Actual ADR"); ax[0].set_ylabel("Predicted ADR"); ax[0].set_title("Linear Regression - validation")
    ax[1].hist(yva - best_pred, bins=80, color="#4c78a8"); ax[1].set_title("Residuals (validation)")
    fig.tight_layout(); fig.savefig(d / "linear_regression_diagnostics.png"); plt.close(fig)
    print(res.to_string(index=False))
    return res
