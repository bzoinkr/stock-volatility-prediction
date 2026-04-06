# ⚠️ Training Pipeline Override

> **Only use these scripts if you intend to retrain the model on new data.**

This directory contains alternative scripts that replace specific steps in the standard pipeline when running a full training run. Swap out the files listed below before executing the pipeline.

---

## Script Replacements

### Step 03 — Market Volatility Target

| | Path |
|---|---|
| **Replace** | `scripts/03_fetch_market_volatility_target.py` |
| **With** | `scripts/train_social/033_fetch_market_volatility_target.py` |

---

### Step 11 — VADER Execution

| | Path |
|---|---|
| **Replace** | `scripts/12_execute_finbert.py` |
| **With** | `scripts/train_social/113_execute_finbert_train.py` |

---

### Step 21 — Social Data Cleaning

| | Path |
|---|---|
| **Replace** | `scripts/22_clean_news_data.py` |
| **With** | `scripts/train_social/213_clean_news_data_train.py` |