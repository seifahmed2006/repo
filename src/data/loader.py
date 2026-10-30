"""Loading, cleaning and splitting H2.csv (City Hotel).

Source labels are kept EXACTLY as in the dataset (no renaming) -> one-to-one mapping.
"""
from __future__ import annotations
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ROOT


def download_data(cfg: dict, force: bool = False) -> Path:
    path = ROOT / cfg["data"]["path"]
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading H2.csv -> {path}")
    urllib.request.urlretrieve(cfg["data"]["url"], path)
    return path


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load H2.csv, strip padding whitespace, convert 'NULL' strings to NaN, verify schema."""
    path = download_data(cfg)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    df = df.replace({"NULL": np.nan, "nan": np.nan})
    # Schema verification (spec: H2 = 79,330 rows x 31 variables)
    assert df.shape == (cfg["data"]["expected_rows"], cfg["data"]["expected_cols"]), \
        f"Unexpected shape {df.shape} - is this really H2.csv?"
    return df


def basic_clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Task-independent cleaning. Only DETERMINISTIC row filters (no fitted statistics),
    so it is safe to run before the split. Returns (clean_df, log)."""
    log = {"rows_raw": len(df)}
    d = cfg["data"]
    if d["drop_duplicates"]:
        n = len(df)
        df = df.drop_duplicates()
        log["duplicates_removed"] = n - len(df)
    if d["drop_zero_guest_rows"]:
        n = len(df)
        guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
        df = df[guests > 0]
        log["zero_guest_rows_removed"] = n - len(df)
    n = len(df)
    df = df[df["ADR"] <= d["max_valid_adr"]]
    log["adr_above_cap_removed"] = n - len(df)
    log["rows_clean"] = len(df)
    return df.reset_index(drop=True), log


def make_splits(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split (70/15/15), fixed seed.
    The test split is created here ONCE and must not be used in Milestone 1."""
    s, seed = cfg["split"], cfg["seed"]
    strat = df[s["stratify_on"]]
    train_val, test = train_test_split(df, test_size=s["test_size"], stratify=strat, random_state=seed)
    rel_val = s["val_size"] / (1 - s["test_size"])
    train, val = train_test_split(train_val, test_size=rel_val,
                                  stratify=train_val[s["stratify_on"]], random_state=seed)
    return {"train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def prepare_data(cfg: dict):
    """raw -> clean -> engineered features -> splits. Returns (splits, cleaning_log)."""
    from src.features.engineering import add_features
    raw = load_raw(cfg)
    clean, log = basic_clean(raw, cfg)
    feats = add_features(clean, cfg)
    splits = make_splits(feats, cfg)
    log.update({f"rows_{k}": len(v) for k, v in splits.items()})
    return splits, log


def task_view(split_df: pd.DataFrame, cfg: dict, task: str) -> pd.DataFrame:
    """Apply task-specific row filters (regression: ADR must be > 0)."""
    if task == "regression":
        return split_df[split_df["ADR"] > cfg["tasks"]["regression"]["min_target"]].reset_index(drop=True)
    return split_df
