"""Automated integrity checks (run: pytest -q). They encode the leakage audit so it cannot silently regress."""
import numpy as np
import pytest
from src.utils import load_config, task_features
from src.data.loader import prepare_data, task_view
from src.preprocessing.pipelines import build_supervised_preprocessor

cfg = load_config()


@pytest.fixture(scope="module")
def prepared():
    return prepare_data(cfg)


@pytest.mark.parametrize("task", ["regression", "classification", "clustering"])
def test_forbidden_columns_never_used_as_features(task):
    t = cfg["tasks"][task]
    used = set(t["numeric"]) | set(t["categorical"])
    assert used.isdisjoint(t["forbidden"]), used & set(t["forbidden"])


def test_targets_not_in_their_own_features():
    for task in ("regression", "classification"):
        assert cfg["tasks"][task]["target"] not in task_features(cfg, task)


def test_direct_leak_columns_excluded_everywhere():
    for task in ("regression", "classification", "clustering"):
        t = cfg["tasks"][task]
        used = set(t["numeric"]) | set(t["categorical"])
        assert not used & {"IsCanceled", "ReservationStatus", "ReservationStatusDate"}


def test_splits_complete_and_no_duplicate_rows_across_partitions(prepared):
    import pandas as pd
    from src.data.loader import load_raw
    splits, log = prepared
    assert sum(len(splits[k]) for k in splits) == log["rows_clean"]
    raw_cols = list(load_raw(cfg).columns)            # the 31 source variables
    allrows = pd.concat([splits[k][raw_cols] for k in ("train", "val", "test")])
    assert allrows.duplicated().sum() == 0            # dedup happened BEFORE the split -> no identical row in two partitions


def test_stratification_preserved(prepared):
    splits, _ = prepared
    rates = [splits[k]["IsCanceled"].mean() for k in splits]
    assert max(rates) - min(rates) < 0.01


def test_split_is_reproducible():
    a, _ = prepare_data(cfg); b, _ = prepare_data(cfg)
    assert a["train"].equals(b["train"]) and a["test"].equals(b["test"])


def test_no_whitespace_padding_or_null_strings(prepared):
    splits, _ = prepared
    df = splits["train"]
    for c in ("Meal", "ReservedRoomType", "DepositType", "MarketSegment"):
        assert (df[c] == df[c].str.strip()).all()
    assert not (df["Agent"] == "NULL").any()


def test_preprocessor_fits_on_train_only(prepared):
    splits, _ = prepared
    feats = task_features(cfg, "classification")
    prep = build_supervised_preprocessor(cfg, "classification")
    prep.fit(splits["train"][feats])
    mean_before = prep.named_transformers_["num"].named_steps["scale"].mean_.copy()
    prep.transform(splits["test"][feats])           # transforming test must not change fitted state
    assert np.allclose(mean_before, prep.named_transformers_["num"].named_steps["scale"].mean_)


def test_regression_view_excludes_nonpositive_adr(prepared):
    splits, _ = prepared
    assert (task_view(splits["train"], cfg, "regression")["ADR"] > 0).all()
