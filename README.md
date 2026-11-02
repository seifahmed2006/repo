# CSE382 – City Hotel (H2) Booking Demand: Regression, Classification, Clustering

**Milestone 1 submission – data preparation and baseline experiments.**

| Item | Value |
|---|---|
| GitHub repository | `https://github.com/seifahmed2006/repo` |
| Submitted commit SHA | `<PASTE `git rev-parse HEAD` OUTPUT HERE>` |
| Team members / GitHub accounts | `<name – @github_handle>` (4–5 members, each committing from their own account) |
| Dataset | Hotel Booking Demand – City Hotel **H2.csv** only (79,330 rows × 31 variables). H1 is never used. |

## 1. Environment setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Google Colab: open `notebooks/colab_milestone1.ipynb` (it clones the repo, installs requirements and calls the modules).

## 2. Exact run commands

```bash
python main.py download                 # fetch H2.csv into data/ (also happens automatically)
python main.py analyze                  # EDA tables + figures      -> outputs/eda/
python main.py split                    # cleaning log + split sizes -> outputs/cleaning_and_split_log.json
python main.py baseline --task regression
python main.py baseline --task classification
python main.py baseline --task clustering
python main.py milestone1               # everything above, in order
pytest -q                               # 11 automated leakage / reproducibility checks
```

All randomness uses `seed: 42` from `config.yaml`. Results are deterministic.

## 3. Repository layout

```
config.yaml            seed, split sizes, per-task feature lists AND forbidden (leaky) columns
main.py                CLI entry point
src/data/              download, load, clean, split
src/features/          stateless feature engineering (TotalNights, TotalGuests, ArrivalWeekday, HasCompany)
src/preprocessing/     sklearn ColumnTransformer pipelines (fit on train only)
src/regression|classification|clustering/   baseline experiments (Milestone 1)
src/evaluation/        EDA + metric helpers
src/tuning|inference/  reserved for Milestone 2
outputs/               tables and figures produced by the commands above
reports/               milestone1_report.md
tests/                 pytest leakage / reproducibility checks
artifacts/             fitted pipelines (Milestone 2; never included in the ZIP)
```

## 4. Column names
Source labels are used **unchanged** (no renaming), so the mapping to the paper is one-to-one.
Engineered columns: `TotalNights`, `TotalGuests`, `HasCompany`, `ArrivalWeekday`.

## 5. Key design decisions (details in `reports/milestone1_report.md`)
* **Prediction point:** the moment the booking is created. Only fields known then are inputs.
* **Cleaning (before the split, deterministic only):** strip padded whitespace; convert `"NULL"` strings to missing; drop 25,902 exact duplicates (prevents identical rows landing in train and test); drop 154 zero-guest rows; drop 1 ADR > 1000 row.
* **Split:** stratified 70/15/15 train/validation/test on `IsCanceled`. Test is created once and **not read** in Milestone 1.
* **Pipelines:** imputation, scaling, one-hot (rare categories grouped) are fitted on the training split only.

## 6. Milestone 1 scope
Done: data understanding/EDA, leakage audit, preprocessing pipelines, one untuned baseline per task.
Not yet (Milestone 2): ≥3 models per task, tuning, PCA/LDA, final test evaluation, saved artifacts, final application.
