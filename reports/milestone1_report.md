# CSE382 – Milestone 1 Progress Report
**City Hotel (H2) Booking Demand – data preparation and baseline experiments**

Repository: `<GITHUB URL>`  |  Commit SHA: `<SHA>`  |  Team: `<members>`

All numbers below are produced by `python main.py milestone1` (seed 42). Tables/figures are in `outputs/`.
Only the **training** split (cross-validation) and the **validation** split were used. The test split was created once and never read.

---
## 1. Completed work
* Repository structure, `config.yaml`, `requirements.txt`, CLI (`main.py`), README with exact commands, 11 pytest checks.
* H2.csv loaded and verified (79,330 × 31).
* EDA, data-quality audit, prediction point, leakage audit, reproducible split.
* Reusable preprocessing pipelines (fit on train only).
* One untuned baseline per task (+ a trivial reference model each).

## 2. Data understanding (outputs/eda/)

| Finding | Value | Decision |
|---|---|---|
| Padded strings (`'HB       '`) and `"NULL"` stored as text | Agent 8,131 (10.2%), **Company 75,641 (95.4%)**; Country 24, Children 4 are true NaN | Strip whitespace on load; convert `"NULL"` → missing. |
| Company 95% missing | – | Replaced by binary `HasCompany`. |
| Agent 10% missing, 224 IDs | – | Missing → category `Unknown`; rare IDs (<1% of train) grouped by the encoder. |
| Country 166 levels, Children 4 NaN | – | Country: `Unknown` + rare grouping. Children NaN → 0. |
| **Exact duplicate rows** | **25,902 (32.7%)** | Dropped **before** splitting (see §4). |
| Zero-guest bookings (Adults+Children+Babies = 0) | 167 raw (154 after dedup) | Dropped (invalid). |
| ADR = 0 | 1,208 | Kept for classification/clustering; **excluded from the regression task** (complimentary stays are not a price signal). |
| ADR max 5,400 | 1 row > 1000 | Dropped (data-entry error). |
| Zero-night stays | 331 | Kept (flagged). |
| Skewed variables (abs. skew > 2) | Babies 44.8, PreviousBookingsNotCanceled 23.1, ADR 23.2, PreviousCancellations 16.6, DaysInWaitingList 10.3, BookingChanges 7.3 … | `log1p` for clustering; trees (Milestone 2) are insensitive; linear baselines are standardised. |
| Class balance (raw) | 41.7% canceled | After de-duplication the **training** cancel rate is **30.1%** (see caveat in §4). |

Figures: target/missing (01), ADR distribution (02), skewed variables (03), cancel rate by category (04), lead time & seasonality (05), ADR by segment/room (06), correlations (07), leakage evidence (08).

## 3. Prediction point, features, preliminary leakage audit

**Prediction/analysis point for all three tasks: when the booking is first created.** A field is allowed only if its value is fixed at that moment.

### Direct leakage
| Column | Evidence | Decision |
|---|---|---|
| `ReservationStatus` | Perfect mapping: Canceled/No-Show → IsCanceled=1 (33,102 rows), Check-Out → 0 (46,228) | Excluded everywhere |
| `ReservationStatusDate` | Date the final status was set = cancellation/check-out date | Excluded everywhere |

### Indirect leakage / timing (excluded because the value is set or changes after booking)
| Column | Evidence (cancel rate by value, outputs/eda/leakage_cancel_rate_by_postbooking_fields.csv) |
|---|---|
| `RequiredCarParkingSpaces` | 31.2% cancel at 0 spaces, **0.0% at ≥1** – no canceled booking has parking |
| `TotalOfSpecialRequests` | 38.4% → 23.1% → 20.9% → 16.2% as requests rise; requests can be added after booking |
| `BookingChanges` | Accumulates until cancellation/check-in; outcome-dependent |
| `DaysInWaitingList` | Resolved after booking; outcome-correlated |
| `AssignedRoomType` | Assigned near check-in; cancel rate 33.2% when room matches vs **5.3% when it differs** (a mismatch can only exist for stays that happened) |
| `ADR` (classification only) | Final averaged rate, can be revised after booking; ablation planned for Milestone 2 |

### Task-specific points
* **Regression (ADR):** also exclude `IsCanceled` (not known at booking). All other inputs are booking-time fields. `ArrivalDateYear` is kept as a price-trend proxy – a caveat when predicting for future years.
* **Classification (IsCanceled):** inputs = lead time, arrival date fields, stay length, party size, meal, country, market segment, channel, repeat-guest history, reserved room, deposit type, agent, customer type, `HasCompany`.
* **Clustering:** only a small booking-profile set: `LeadTime, TotalNights, TotalGuests, ADR, PreviousCancellations, IsRepeatedGuest` + `MarketSegment, DepositType, CustomerType, Meal`. Excluded: outcome/post-booking fields above, and high-cardinality `Country/Agent/Company` (a 166-level one-hot would dominate Euclidean distance). ADR is kept here because it describes the booking (value segment), not an outcome; `IsCanceled` is used **only as an external check after clustering**, never to fit or choose parameters.

### Open risks to revisit in Milestone 2
1. `DepositType`: **Non Refund bookings cancel 97.2%** of the time (571 train rows) – implausible commercially, likely a recording artifact. Kept (known at booking) but will be ablated.
2. `PreviousCancellations` / `IsRepeatedGuest` are booking-time but strongly outcome-linked in some clusters (cluster 2 below: 100% cancel). Will be inspected.
3. Random split ignores time (2015–2017 bookings). A time-based hold-out is a stricter alternative.

## 4. Split strategy and preprocessing
1. Load → strip/NULL fix → drop duplicates, zero-guest, ADR>1000 (**deterministic filters only, no fitted statistics**) → 53,273 rows.
2. Stratified (on `IsCanceled`) split, seed 42: **train 37,291 / validation 7,991 / test 7,991** (70/15/15).
3. Regression additionally keeps only ADR>0: train 36,643 / val 7,836.
4. Pipeline per task: numeric → median impute → standardise; categorical → `Unknown` impute → one-hot (categories <1% of train grouped). Fitted **inside** `Pipeline`, so cross-validation refits it per fold. Clustering adds `log1p` on skewed counts and ADR capped at 500 (constants from config).
5. Feature engineering is stateless (`TotalNights`, `TotalGuests`, `HasCompany`, `ArrivalWeekday`).

**Why de-duplicate:** the file has no booking ID, so some "duplicates" are genuine separate bookings. But an identical row in train and validation/test makes the score partly a lookup. In an un-deduplicated random split we measured that ~39% of validation rows had an exact (features + label) twin in train. **Caveat:** de-duplication lowers the cancel rate (41.7% → 30.1%), so results describe *distinct* booking profiles. A sensitivity run on the non-deduplicated data is planned for Milestone 2.

## 5. Initial results (untuned baselines)

### Regression – ADR (ADR>0)
| Model | CV MAE (5-fold, train) | Val MAE | Val RMSE | Val R² |
|---|---|---|---|---|
| Mean predictor (reference) | 30.63 | 30.29 | 38.71 | −0.000 |
| **Linear Regression** | **14.91 ± 0.20** | **14.96** | **21.03** | **0.705** |

Interpretation: linear regression halves the error of the naive baseline and explains ~70% of ADR variance from booking-time fields alone (season, room type, market segment, party size). CV and validation agree, so there is no sign of overfitting. Errors are largest at the price extremes (validation, by ADR quartile): the most expensive quartile (ADR > 133) is under-predicted by ≈ 15 on average (MAE 20.2) and the cheapest quartile is over-predicted by ≈ 9 (MAE 15.1) – the usual shrinkage of a linear model toward the mean. September has the highest monthly MAE (21.0). This suggests non-linear models (Milestone 2) should help, which we will test rather than assume.

### Classification – IsCanceled
| Model | CV Acc | CV F1 | CV ROC-AUC | Val Acc | Val Prec | Val Rec | Val F1 | Val ROC-AUC | Val PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Majority class (reference) | 0.699 | 0.000 | 0.500 | 0.699 | 0.000 | 0.000 | 0.000 | 0.500 | 0.301 |
| **Logistic Regression** | 0.731 | 0.432 | 0.780 | 0.730 | 0.586 | 0.344 | 0.434 | 0.779 | 0.581 |

Interpretation: accuracy is only slightly above the majority baseline (0.730 vs 0.699), so accuracy alone is misleading. ROC-AUC 0.78 shows real ranking signal, but **recall is low (0.34)** at the default 0.5 threshold – most cancellations are missed. Planned: class weights/threshold tuning, non-linear models. The largest coefficients are `DepositType` (Non Refund ↑), specific countries (PRT ↑) and agents – consistent with the risks in §3.

### Clustering – K-Means (untuned: k = 8, sklearn default)
| Config | Train silhouette | Train Davies–Bouldin | Val silhouette | Val Davies–Bouldin |
|---|---|---|---|---|
| k = 8 | 0.162 | 1.381 | 0.165 | 1.362 |

k-sweep (train, outputs/clustering/kmeans_k_sweep.csv): among k ≥ 3 the best silhouette/DB is **k = 5** (0.216 / 1.287). k = 2 has silhouette 0.62 but only because it isolates 1,147 repeat guests (3% of rows) – a trivial split, not a useful segmentation. Structure is weak overall (silhouette ≈ 0.2), typical for mixed booking data.

Baseline cluster profile (k = 8; kmeans_baseline_cluster_profile.csv) – examples:
* Cluster 1 (6,309): last-minute bookings (≈ 5 days lead time), short stays (1.7 nights), Online TA.
* Cluster 0 (5,690): Offline TA/TO, long lead time (≈ 119 days), `Transient-Party` customers.
* Cluster 6 (9,039): long stays (≈ 5.4 nights), booked ≈ 97 days ahead.
* Clusters 4 (5,406): ≈ 3.2 guests per booking, highest ADR (≈ 170) – family/premium bookings.
* Clusters 3 & 5 (807 / 336): Corporate **repeat guests**, low ADR (≈ 67); cluster 5 also has many previous cancellations (mean 2.5).
* Cluster 7 (9,237): typical online booking, ≈ 2.3 nights, ≈ 85 days lead time.
* Cluster 2 (467): long lead time (≈ 139 days) **and previous cancellations (mean 1.0); 100% were canceled** (external check only).
Clusters differ in booking behaviour, and some align with cancellation, which is expected but must not be used to tune k.

## 6. Member contributions
*(Fill in with verifiable facts from the GitHub history – do not leave placeholders.)*

| Member | GitHub account | Work completed (commits / modules) |
|---|---|---|
| `<name>` | `<@handle>` | `<e.g. repo + EDA module>` |
| `<name>` | `<@handle>` | `<e.g. preprocessing pipelines + tests>` |
| `<name>` | `<@handle>` | `<e.g. regression/classification baselines>` |
| `<name>` | `<@handle>` | `<e.g. clustering baseline + report>` |

## 7. Plan for Milestone 2 (due 11 Dec 2026)
1. **Regression & classification:** add KNN, Decision Tree, Random Forest, XGBoost, SVM, MLP / Naive Bayes; ≥3 per task, each baseline **and** tuned (Optuna/RandomizedSearchCV, search space + metric documented, CV on train only).
2. **PCA/LDA:** explained-variance curve, ≥2 supervised models with/without PCA, original vs PCA vs LDA for one classifier.
3. **Clustering:** add Agglomerative (sampled), DBSCAN, GMM; tune with silhouette/DB; dendrogram; PCA 2-D plot; interpretation.
4. Ablations from §3 (ADR, DepositType, special requests, duplicates kept).
5. Final evaluation: test set used **once** after model selection.
6. Application (CLI/Streamlit) loading saved pipelines from `artifacts/`; final report and leakage audit.
