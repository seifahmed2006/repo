"""Task C - unsupervised clustering: untuned KMeans baseline + k-sweep (Milestone 1).
Target labels (IsCanceled) are NEVER used for fitting or choosing k. Fit on TRAIN, report on train-sample and validation."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline

from src.preprocessing.pipelines import build_clustering_preprocessor, prepare_clustering_frame
from src.utils import out_dir


def _internal(X, labels, n_sample, seed):
    return {"silhouette": silhouette_score(X, labels, sample_size=min(n_sample, len(X)), random_state=seed),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels)}


def run_clustering_baseline(splits: dict, cfg: dict) -> pd.DataFrame:
    seed, d = cfg["seed"], out_dir("clustering")
    c = cfg["tasks"]["clustering"]
    tr, va = splits["train"], splits["val"]
    Ftr, Fva = prepare_clustering_frame(tr, cfg), prepare_clustering_frame(va, cfg)
    prep = build_clustering_preprocessor(cfg)
    Xtr = prep.fit_transform(Ftr); Xva = prep.transform(Fva)          # scaler/encoder fitted on TRAIN only
    names = prep.get_feature_names_out()
    print(f"[clustering] train={Xtr.shape} val={Xva.shape} features after prep={len(names)}")

    # ---- untuned baseline: sklearn-default KMeans (k=8) ----
    t0 = time.time(); base = KMeans(n_clusters=c["baseline_k"], random_state=seed, n_init=10).fit(Xtr); fit_s = time.time() - t0
    rows = [{"algorithm": "kmeans", "config": f"baseline (k={c['baseline_k']}, untuned)", "k": c["baseline_k"],
             **{f"TRAIN_{k}": v for k, v in _internal(Xtr, base.labels_, c["silhouette_sample"], seed).items()},
             **{f"VAL_{k}": v for k, v in _internal(Xva, base.predict(Xva), c["silhouette_sample"], seed).items()},
             "inertia": base.inertia_, "fit_time_s": fit_s}]
    pd.DataFrame(rows).round(4).to_csv(d / "baseline_results.csv", index=False)

    # ---- preliminary k-sweep to justify k (internal criteria only) ----
    sweep = []
    for k in range(c["k_range"][0], c["k_range"][1] + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xtr)
        sweep.append({"k": k, "inertia": km.inertia_, **_internal(Xtr, km.labels_, c["silhouette_sample"], seed)})
    sweep = pd.DataFrame(sweep); sweep.round(4).to_csv(d / "kmeans_k_sweep.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(sweep.k, sweep.inertia, "o-"); ax[0].set_title("Elbow (inertia)")
    ax[1].plot(sweep.k, sweep.silhouette, "o-g"); ax[1].set_title("Silhouette (higher = better)")
    ax[2].plot(sweep.k, sweep.davies_bouldin, "o-r"); ax[2].set_title("Davies-Bouldin (lower = better)")
    for a in ax: a.set_xlabel("k")
    fig.tight_layout(); fig.savefig(d / "kmeans_k_sweep.png"); plt.close(fig)

    # ---- PCA 2-D view + interpretation of the baseline clusters ----
    pca = PCA(n_components=2, random_state=seed).fit(Xtr)
    Z = pca.transform(Xtr); rs = np.random.RandomState(seed); idx = rs.choice(len(Z), min(8000, len(Z)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(Z[idx, 0], Z[idx, 1], c=base.labels_[idx], s=5, cmap="tab10")
    ax.set_title(f"KMeans baseline (k={c['baseline_k']}) in PCA space\n(PC1+PC2 explain {pca.explained_variance_ratio_.sum():.1%})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); fig.colorbar(sc); fig.tight_layout(); fig.savefig(d / "kmeans_baseline_pca.png"); plt.close(fig)

    prof = Ftr.assign(cluster=base.labels_).groupby("cluster").agg(
        size=("ADR", "size"), LeadTime=("LeadTime", "mean"), TotalNights=("TotalNights", "mean"), TotalGuests=("TotalGuests", "mean"),
        ADR=("ADR", "mean"), PrevCancel=("PreviousCancellations", "mean"), RepeatedGuest=("IsRepeatedGuest", "mean"),
        top_segment=("MarketSegment", lambda s: s.mode()[0]), top_deposit=("DepositType", lambda s: s.mode()[0]),
        top_customer=("CustomerType", lambda s: s.mode()[0]))
    # EXTERNAL check only (never used for fitting / choosing k): does the unsupervised grouping relate to the outcome?
    prof["cancel_rate_(external_check)"] = tr.assign(cluster=base.labels_).groupby("cluster")["IsCanceled"].mean()
    prof.round(3).to_csv(d / "kmeans_baseline_cluster_profile.csv")
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False)); print("\nk-sweep:\n", sweep.round(4).to_string(index=False)); print("\nProfile:\n", prof.round(2).to_string())
    return res
