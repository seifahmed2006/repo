# CSE382 – City Hotel (H2) Booking Demand: Regression, Classification, Clustering

**Milestone 1 submission – data preparation and baseline experiments.**

| Item | Value |
|---|---|
| GitHub repository | `https://github.com/seifahmed2006/repo` |
| Submitted commit SHA | (to be filled on submission day) |
| Team members / GitHub accounts | Cherif - @chichochago2109-lgtm; Seif Ahmed - @seifahmed2006; Peter - @handle; Hamza - @handle; Youssef - @handle || Dataset | Hotel Booking Demand – City Hotel **H2.csv** only (79,330 rows × 31 variables). H1 is never used. |

## 1. Environment setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/seifahmed2006/repo
cd repo
python -m venv .venv
```

Activate it:

- **Windows PowerShell:**
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Tested on Python 3.13. Internet access is needed on the first run to download H2.csv.

Google Colab: open `notebooks/colab_milestone1.ipynb` (it clones the repo, installs requirements and calls the modules).

## 2. Exact run commands

Run these from the repository root with the virtual environment activated:

```bash
python main.py download                 # fetch H2.csv into data/ (also happens
automatically)
python main.py analyze                  # EDA tables + figures      -> outputs/eda/
python main.py split                    # cleaning log + split sizes -> outputs/cleaning_and_split_log.json
python main.py baseline --task regression
python main.py baseline --task classification
python main.py baseline --task clustering
python main.py milestone1               # everything above, in order
python -m pytest -q                     # 11 automated leakage / reproducibility checks
```

All randomness uses `seed: 42` from `config.yaml`. Metrics are deterministic; only the `fit_time_s` timing columns vary between machines.

## 3. Repository layout

```
config.yaml            seed, split sizes, per-task feature lists AND forbidden (leaky) columns
main.py                CLI entry point
requirements.txt       Python dependencies
pytest.ini             pytest configuration
data/                  H2.csv, downloaded automatically on first run (git-ignored)
src/data/              download, load, clean, split
src/features/          stateless feature engineering (TotalNights, TotalGuests, ArrivalWeekday, HasCompany)
src/preprocessing/     sklearn ColumnTransformer pipelines (fit on train only)
src/regression|classification|clustering/   baseline experiments (Milestone 1)
src/evaluation/        EDA + metric helpers
src/tuning|inference/  reserved for Milestone 2
outputs/               tables and figures produced by the commands above
reports/               milestone1_report.md
tests/                 pytest leakage / reproducibility checks
notebooks/             optional Colab runner (supplementary only)
artifacts/             fitted pipelines (created in Milestone 2; git-ignored, never included in the ZIP)
```
## 4. Column names
Source labels are used **unchanged** (no renaming), so the mapping to the paper is one-to-one.
Engineered columns: `TotalNights`, `TotalGuests`, `HasCompany`, `ArrivalWeekday`.

## 5. Key design decisions (details in `reports/milestone1_report.md`)
* **Prediction point:** the moment the booking is created. Only fields known then are inputs.
* **Leakage control:** outcome and post-booking columns are excluded per task via the `forbidden` lists in `config.yaml` (`ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList`, `RequiredCarParkingSpaces`, `TotalOfSpecialRequests`; `ADR` for classification). `tests/` enforces this.
* **Cleaning (before the split, deterministic only):** strip padded whitespace; convert `"NULL"` strings to missing; drop 25,902 exact duplicates (prevents identical rows landing in train and test); drop 154 zero-guest rows; drop 1 ADR > 1000 row.
* **Fixed rules (no fitted statistics):** `Children` NaN -> 0; `Company` -> `HasCompany` flag; missing `Agent`/`Country` -> `Unknown`. Regression keeps only rows with ADR > 0.
* **Split:** stratified 70/15/15 train/validation/test on `IsCanceled`, seed 42. The test split is created once and is not used for any modelling, metric or tuning in Milestone 1 (only the integrity tests touch it).
* **Pipelines:** imputation, scaling, one-hot (rare categories grouped) are fitted on the training split only; cross-validation refits them per fold.
* **Outputs:** the tables and figures in `outputs/` are committed. Re-running regenerates identical metrics; only `fit_time_s` varies by machine.

## 6. Milestone 1 scope
Done: data understanding/EDA, leakage audit, preprocessing pipelines, one untuned baseline per task (linear regression, logistic regression, K-Means k=8), each with a trivial reference model where applicable.
Not yet (Milestone 2): >=3 models per task, tuning, PCA/LDA, final test evaluation, ablations (ADR, DepositType, duplicates kept), saved artifacts, final application.