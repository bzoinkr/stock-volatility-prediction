from _bootstrap import *

from common.config import load_config
from pipelines.prediction.ridgeRegression.ridgeCreateModel_social_pipeline import run_pipeline


# ── Paths ─────────────────────────────────────────────────────────────────────

SENTIMENT_PATH  = "data/processed/social/social_data.json"
VOLATILITY_PATH = "data/processed/market/volatility.json"
MODEL_DIR       = "artifacts/models"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run         = load_config("run.yaml")
    day_weights = run["day_weights"]

    run_pipeline(
        sentiment_path  = SENTIMENT_PATH,
        volatility_path = VOLATILITY_PATH,
        model_dir       = MODEL_DIR,
        day_weights     = day_weights,
    )


if __name__ == "__main__":
    main()