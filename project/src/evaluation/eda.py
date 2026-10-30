"""Exploratory data analysis: tables + figures saved to outputs/eda/. Run: python main.py analyze"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_raw, basic_clean
from src.features.engineering import add_features, MONTHS
from src.utils import out_dir

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name, d):
    fig.tight_layout()
    fig.savefig(d / name, bbox_inches="tight")
    plt.close(fig)


def run_eda(cfg: dict) -> dict:
    d = out_dir("eda")
    raw = load_raw(cfg)
    findings: dict = {}

    # ---------------- 1. structure / quality tables ----------------
    schema = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "n_missing": raw.isna().sum(),
        "pct_missing": (raw.isna().mean() * 100).round(2),
        "n_unique": raw.nunique(),
    })
    schema.to_csv(d / "schema_missing_cardinality.csv")
    num = raw.select_dtypes("number")
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc.round(3).to_csv(d / "numeric_summary.csv")

    findings["shape"] = list(raw.shape)
    findings["duplicates_exact"] = int(raw.duplicated().sum())
    findings["class_balance_IsCanceled"] = raw["IsCanceled"].value_counts(normalize=True).round(4).to_dict()
    findings["adr_zero"] = int((raw["ADR"] == 0).sum())
    findings["adr_negative"] = int((raw["ADR"] < 0).sum())
    findings["adr_max"] = float(raw["ADR"].max())
    findings["adults_zero"] = int((raw["Adults"] == 0).sum())
    findings["zero_guests"] = int(((raw["Adults"] + raw["Children"].fillna(0) + raw["Babies"]) == 0).sum())
    findings["zero_nights"] = int(((raw["StaysInWeekendNights"] + raw["StaysInWeekNights"]) == 0).sum())
    findings["high_skew_columns"] = desc["skew"][desc["skew"].abs() > 2].round(2).to_dict()
    findings["date_range_arrival"] = [f"{raw.ArrivalDateYear.min()}", f"{raw.ArrivalDateYear.max()}"]

    clean, log = basic_clean(raw, cfg)
    findings["cleaning_log"] = log
    df = add_features(clean, cfg)
    df["ArrivalMonthNum"] = df["ArrivalDateMonth"].map({m: i + 1 for i, m in enumerate(MONTHS)})

    # ---------------- 2. leakage evidence tables ----------------
    status = pd.crosstab(raw["ReservationStatus"], raw["IsCanceled"])
    status.to_csv(d / "leakage_reservationstatus_vs_iscanceled.csv")
    rows = []
    for col in ["RequiredCarParkingSpaces", "BookingChanges", "TotalOfSpecialRequests", "DaysInWaitingList"]:
        g = df.assign(bin=pd.cut(df[col], [-1, 0, 1, 2, 1000], labels=["0", "1", "2", "3+"])) \
              .groupby("bin", observed=True)["IsCanceled"].agg(["mean", "size"])
        g.insert(0, "feature", col)
        rows.append(g.reset_index())
    pd.concat(rows).round(4).to_csv(d / "leakage_cancel_rate_by_postbooking_fields.csv", index=False)
    findings["room_type_mismatch_rate"] = round(float((raw["ReservedRoomType"] != raw["AssignedRoomType"]).mean()), 4)
    findings["cancel_rate_by_room_mismatch"] = (
        df.assign(mm=df["ReservedRoomType"] != df["AssignedRoomType"]).groupby("mm")["IsCanceled"].mean().round(4)
        .rename({True: "mismatch", False: "match"}).to_dict())

    # ---------------- 3. figures ----------------
    # 3.1 target balance + missing
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vc = raw["IsCanceled"].value_counts().sort_index()
    ax[0].bar(["Not canceled (0)", "Canceled (1)"], vc.values, color=["#4c78a8", "#e45756"])
    for i, v in enumerate(vc.values):
        ax[0].text(i, v, f"{v:,}\n({v/len(raw):.1%})", ha="center", va="bottom")
    ax[0].set_title("Target balance: IsCanceled"); ax[0].set_ylim(0, vc.max() * 1.2)
    miss = (raw.isna().mean() * 100); miss = miss[miss > 0].sort_values()
    ax[1].barh(miss.index, miss.values, color="#f58518")
    for i, v in enumerate(miss.values):
        ax[1].text(v, i, f" {v:.2f}%", va="center")
    ax[1].set_title("Missing values (incl. 'NULL' strings)"); ax[1].set_xlabel("% of rows")
    _save(fig, "01_target_and_missing.png", d)

    # 3.2 ADR distribution
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(raw["ADR"].clip(upper=500), bins=80, color="#4c78a8"); ax[0].set_title("ADR (raw, clipped at 500 for display)")
    ax[1].hist(np.log1p(df.loc[df.ADR > 0, "ADR"]), bins=80, color="#54a24b"); ax[1].set_title("log1p(ADR), ADR>0 (after cleaning)")
    ax[2].boxplot(df.loc[df.ADR > 0, "ADR"], vert=False); ax[2].set_title("ADR boxplot (outliers)")
    _save(fig, "02_adr_distribution.png", d)

    # 3.3 skewed counts
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for a, c in zip(ax.ravel(), ["LeadTime", "TotalNights", "TotalGuests", "PreviousCancellations", "DaysInWaitingList", "BookingChanges"]):
        a.hist(df[c], bins=min(60, max(5, df[c].nunique())), color="#72b7b2"); a.set_title(f"{c} (skew={df[c].skew():.1f})")
        a.set_yscale("log")
    fig.suptitle("Skewed count variables (log-scale y)")
    _save(fig, "03_skewed_variables.png", d)

    # 3.4 cancel rate by category
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for a, c in zip(ax.ravel(), ["MarketSegment", "DepositType", "CustomerType", "DistributionChannel"]):
        r = df.groupby(c)["IsCanceled"].agg(["mean", "size"]).sort_values("mean")
        a.barh(r.index, r["mean"], color="#e45756")
        for i, (m, n) in enumerate(zip(r["mean"], r["size"])):
            a.text(m, i, f" {m:.0%} (n={n:,})", va="center", fontsize=8)
        a.set_title(f"Cancellation rate by {c}"); a.set_xlim(0, 1.15)
    _save(fig, "04_cancel_rate_by_category.png", d)

    # 3.5 lead time vs cancel + seasonality
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    lt = df.assign(b=pd.cut(df["LeadTime"], [-1, 7, 30, 90, 180, 365, 1000])).groupby("b", observed=True)["IsCanceled"].mean()
    ax[0].bar(lt.index.astype(str), lt.values, color="#e45756"); ax[0].set_title("Cancel rate by LeadTime bin"); ax[0].tick_params(axis="x", rotation=30)
    m = df.groupby("ArrivalMonthNum").agg(n=("ADR", "size"), adr=("ADR", "mean"))
    ax[1].bar(m.index, m["n"], color="#4c78a8"); ax[1].set_title("Bookings per arrival month"); ax[1].set_xlabel("month")
    ax[2].plot(m.index, m["adr"], marker="o", color="#54a24b"); ax[2].set_title("Mean ADR per arrival month"); ax[2].set_xlabel("month")
    _save(fig, "05_leadtime_and_seasonality.png", d)

    # 3.6 ADR by market segment / room type
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    order = df.groupby("MarketSegment")["ADR"].median().sort_values().index
    ax[0].boxplot([df.loc[(df.MarketSegment == s) & (df.ADR <= 400), "ADR"] for s in order], tick_labels=order, vert=True)
    ax[0].tick_params(axis="x", rotation=30); ax[0].set_title("ADR by MarketSegment (<=400)")
    order = sorted(df["ReservedRoomType"].unique())
    ax[1].boxplot([df.loc[(df.ReservedRoomType == s) & (df.ADR <= 400), "ADR"] for s in order], tick_labels=order)
    ax[1].set_title("ADR by ReservedRoomType (<=400)")
    _save(fig, "06_adr_by_segment_room.png", d)

    # 3.7 correlation heatmap
    cols = ["IsCanceled", "ADR", "LeadTime", "TotalNights", "TotalGuests", "ArrivalMonthNum", "IsRepeatedGuest",
            "PreviousCancellations", "PreviousBookingsNotCanceled", "BookingChanges", "DaysInWaitingList",
            "RequiredCarParkingSpaces", "TotalOfSpecialRequests"]
    corr = df[cols].corr()
    corr.round(3).to_csv(d / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=75, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.grid(False); fig.colorbar(im); ax.set_title("Correlation (numeric)")
    _save(fig, "07_correlation_heatmap.png", d)

    # 3.8 leakage evidence
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, col in zip(ax, ["RequiredCarParkingSpaces", "BookingChanges", "TotalOfSpecialRequests", "DaysInWaitingList"]):
        g = df.assign(bin=pd.cut(df[col], [-1, 0, 1, 2, 1000], labels=["0", "1", "2", "3+"])).groupby("bin", observed=True)["IsCanceled"].mean()
        a.bar(g.index.astype(str), g.values, color="#b279a2"); a.set_title(f"Cancel rate by {col}", fontsize=9); a.set_ylim(0, 1)
    fig.suptitle("Post-booking fields strongly encode the outcome -> excluded (leakage evidence)")
    _save(fig, "08_leakage_evidence.png", d)

    with open(d / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(json.dumps(findings, indent=2, default=str))
    print(f"\nEDA tables/figures saved to {d}")
    return findings
