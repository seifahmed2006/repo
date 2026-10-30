"""Reusable preprocessing pipelines. All fitted statistics (medians, means, category
vocabularies, rare-category groupings) are learned ONLY when .fit() is called on the training split."""
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _ohe(cfg):
    return OneHotEncoder(handle_unknown="infrequent_if_exist",
                         min_frequency=cfg["preprocessing"]["rare_category_min_frequency"],
                         sparse_output=False)


def _cat_pipe(cfg):
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=cfg["preprocessing"]["missing_category_label"])),
        ("onehot", _ohe(cfg)),
    ])


def build_supervised_preprocessor(cfg: dict, task: str) -> ColumnTransformer:
    """Numeric: median impute + standardise. Categorical: 'Unknown' impute + one-hot (rare grouped)."""
    t = cfg["tasks"][task]
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("num", num, t["numeric"]),
        ("cat", _cat_pipe(cfg), t["categorical"]),
    ], remainder="drop", verbose_feature_names_out=False)


def build_clustering_preprocessor(cfg: dict) -> ColumnTransformer:
    """Distance-based -> everything on a comparable scale. Skewed counts get log1p first."""
    t = cfg["tasks"]["clustering"]
    log_cols = list(t["log_numeric"])
    plain_cols = [c for c in t["numeric"] if c not in log_cols]
    log_pipe = Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                         ("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    plain_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("log", log_pipe, log_cols),
        ("num", plain_pipe, plain_cols),
        ("cat", _cat_pipe(cfg), t["categorical"]),
    ], remainder="drop", verbose_feature_names_out=False)


def prepare_clustering_frame(df, cfg):
    """Stateless step: cap extreme ADR (fixed constant from config, not data-dependent)."""
    out = df[cfg["tasks"]["clustering"]["numeric"] + cfg["tasks"]["clustering"]["categorical"]].copy()
    out["ADR"] = out["ADR"].clip(upper=cfg["tasks"]["clustering"]["adr_cap"])
    return out
