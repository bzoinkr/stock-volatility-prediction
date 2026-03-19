from _bootstrap import *

import json

from common.config import load_config
from pipelines.prediction.ridgeCreateModel_pipeline import run_pipeline


# ── Paths — fill these in ─────────────────────────────────────────────────────

SENTIMENT_PATH  = "data/processed/social/social_data_train.json"   # path to sentiment JSON file
VOLATILITY_PATH = "data/processed/market/volatility.json"   # path to volatility JSON file
MODEL_DIR       = "artifacts/models"   # directory where vol_model.pkl will be saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run             = load_config("run.yaml")
    date            = run["universe"]["end_datey"]
    day_weights     = run["day_weights"]
    feature_weights = run["feature_weights"]

    print(f"Predicting volatility for date  : {date}")

    predictions = run_pipeline(
        sentiment_path  = SENTIMENT_PATH,
        volatility_path = VOLATILITY_PATH,
        model_dir       = MODEL_DIR,
        target_date     = str(date),
        day_weights     = day_weights,
        feature_weights = feature_weights,
    )

    output_path = "data/predictions/social_predictions.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()