# Stock Volatility Prediction

This project predicts future stock market volatility using multiple independent data sources:

- **VADER** → social media sentiment (e.g., Reddit)
- **FinBERT** → financial news sentiment

---

## Project Structure

```text
stock-volatility-prediction
├─ data/               # datasets (usually gitignored)
│  ├─ raw/             # original downloads
│  ├─ interim/         # cleaned / cached
│  └─ processed/       # model-ready tables
│
├─ artifacts/          # outputs (models, predictions, reports)
│
├─ src/                # all reusable code
│  ├─ common/          # paths, io, configs, metrics, utils
│  ├─ apis/            # data provider wrappers
│  ├─ pipelines/       # feature & label builders
│  ├─ models/          # vader/, finbert/, ensemble/
│  └─ evaluation/      # backtests, plots, diagnostics
│
├─ scripts/            # runnable entrypoints
├─ configs/            # YAML configuration files
├─ requirements.txt
└─ README.md
```

---

### Folder Responsibilities

`data/`

Holds datasets at different processing stages.

```text
raw/        -> downloaded data
interim/    -> cleaned / cached
processed/ -> final features & labels
```

Nothing inside `data/` is treated as source code.

---

`artifacts/`

Model outputs.

```text
artifacts/
├─ models/        # saved model objects
├─ predictions/   # prediction files
└─ reports/       # metrics, plots
```

---

`src/common/`

Shared infrastructure used everywhere.

- `paths.py` → resolves project root & key directories
- `io.py` → resolves project root & key directories
- `config.py` → resolves project root & key directories
- `metrics.py`, `utils.py` → resolves project root & key directories

This is the backbone of routing.

---

`src/apis/`

Wrappers around external data providers.

Examples:

- market data
- news data
- social media data
- ollama keywords

Keeps API logic out of pipelines.

---

`src/pipelines/`

End-to-end dataset construction.

- `make_labels.py`
- `vader_pipeline.py`
- `finbert_pipeline.py`
- `merge_features.py`
- `keywords_pipeline.py`

Each pipeline reads raw/interim data and writes a processed table.

---

`src/models/`

```text
models/
├─ base.py          # common interface
├─ vader/
├─ finbert/
└─ ensemble/
```

Each model folder contains:

- `model.py` → training + inference logic
- `features.py` → model-specific feature logic
- `README.md` → notes for that model

Each model is owned by one contributor.

---

`scripts/`


Scripts are the only files you execute.

Examples:

```text
04_build_keywords.py
10_make_labels.py
11_build_vader_features.py
12_build_finbert_features.py
20_train_vader.py
21_train_finbert.py
30_predict.py
40_evaluate.py
```

Scripts are thin:

- load config
- call pipeline/model functions
- save outputs

No heavy logic inside scripts.

---

### Configuration System

All settings live in `configs/`:

- `run.yaml` → tickers, dates, seeds, keyword_count
- `labels.yaml` → volatility target definition
- `vader.yaml` → VADER pipeline settings
- `finbert.yaml` → FinBERT pipeline settings
- `ensemble.yaml` → merge & ensemble settings

Accessed via:

```python
from common.config import load_config
cfg = load_config("run.yaml")
```

---

### Universal Join Contract

All processed feature tables must contain:

```text
ticker
date
```

Labels use the same keys.

This guarantees painless merging and ensembling.

---

## Local LLM Keyword Generation (Ollama)

This project can optionally use a locally running LLM (via Ollama) to generate
single-word finance-related keywords per ticker for text filtering and feature engineering.

Environment variables:

```
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2:1b
```

Example `run.yaml` additions:

```yaml
universe:
  tickers: ["AAPL", "MSFT", "NVDA", "TSLA"]
keyword_count: 15
```

Run once:

```bash
python scripts/04_build_keywords.py
```

Creates:

- `data/interim/keywords.json` — cached keywords per ticker

---

## Typical Workflow

```bash
python scripts/01_fetch_social.py
python scripts/02_fetch_news.py
python scripts/03_fetch_market.py

python scripts/04_build_keywords.py

python scripts/10_make_labels.py
python scripts/11_build_vader_features.py
python scripts/12_build_finbert_features.py

python scripts/20_train_vader.py
python scripts/21_train_finbert.py

python scripts/30_predict.py
python scripts/40_evaluate.py
```

---

## Routing Test

To verify imports and paths:

```bash
python scripts/99_test_routing.py
```

Expected:

```vbnet
IMPORTS OK
PIPELINE ROUTES OK
```
