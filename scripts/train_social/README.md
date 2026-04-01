# ⚠️ Training Pipeline Override

> **Only use these scripts if you intend to retrain the model on new data.**

This directory contains alternative scripts that replace specific steps in the standard pipeline when running a full training run. Swap out the files listed below before executing the pipeline.

---

## Script Replacements

### Step 03 — Market Volatility Target

| | Path |
|---|---|
| **Replace** | `scripts/03_fetch_market_volatility_target.py` |
| **With** | `scripts/train_social/032_fetch_market_volatility_compound.py` |

---

### Step 04 — Social Target

| | Path |
|---|---|
| **Replace** | `scripts/04_fetch_social_target.py` |
| **With** | `scripts/train_social/042_fetch_social_train.py` |

---

### Step 11 — VADER Execution

| | Path |
|---|---|
| **Replace** | `scripts/11_execute_vader.py` |
| **With** | `scripts/train_social/112_execute_vader_train.py` |

---

### Step 21 — Social Data Cleaning

| | Path |
|---|---|
| **Replace** | `scripts/21_clean_social_data.py` |
| **With** | `scripts/train_social/212_clean_social_data_train.py` |