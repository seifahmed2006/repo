"""Stateless, deterministic feature engineering (no fitted statistics -> no leakage risk).
Every derived feature uses only fields known at booking creation."""
from __future__ import annotations
import numpy as np
import pandas as pd

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    unk = cfg["preprocessing"]["missing_category_label"]
    df["Children"] = df["Children"].fillna(0)                      # only 4 NaN rows -> 0 children
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
    df["HasCompany"] = df["Company"].notna().astype(int)           # Company is 95% NULL -> presence flag only
    df["Agent"] = df["Agent"].fillna(unk).astype(str).str.replace(r"\.0$", "", regex=True)
    df["Country"] = df["Country"].fillna(unk)
    month_num = df["ArrivalDateMonth"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    date = pd.to_datetime(dict(year=df["ArrivalDateYear"], month=month_num,
                               day=df["ArrivalDateDayOfMonth"]), errors="coerce")
    df["ArrivalWeekday"] = date.dt.weekday                         # 0 = Monday
    return df
