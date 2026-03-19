from _bootstrap import *

import json
import yaml

from common.config import load_config
from pipelines.prediction.eval_pipeline import run_eval_pipeline


# ── Paths — fill these in ─────────────────────────────────────────────────────

SENTIMENT_PATH  = "data/processed/social/social_data_train.json"   # path to sentiment JSON file
VOLATILITY_PATH = "data/processed/market/volatility.json"   # path to volatility JSON file
MODEL_DIR       = "artifacts/models"   # directory where vol_model.pkl will be saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run             = load_config("run.yaml")
    date            = run["universe"]["end_date"]
    day_weights     = run["day_weights"]
    feature_weights = run["feature_weights"]

    print(f"Evaluating model on date        : {date}")

    results = run_eval_pipeline(
        sentiment_path  = SENTIMENT_PATH,
        volatility_path = VOLATILITY_PATH,
        model_dir       = MODEL_DIR,
        target_date     = str(date),
        day_weights     = day_weights,
        feature_weights = feature_weights,
    )

    output_path = "artifacts/evaluations/social_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()