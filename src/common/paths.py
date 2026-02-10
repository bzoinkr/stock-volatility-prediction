from pathlib import Path

# Project root = stock-vol-pred/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
MARKET_RAW_DATA_DIR = RAW_DATA_DIR / "market"
NEWS_RAW_DATA_DIR = RAW_DATA_DIR / "news"
SOCIAL_RAW_DATA_DIR = RAW_DATA_DIR / "social"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
MARKET_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "market"
NEWS_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "news"
SOCIAL_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "social"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

CONFIG_DIR = PROJECT_ROOT / "configs"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_dirs():
    """Create core folders if missing."""
    for d in [
        RAW_DATA_DIR,
        MARKET_RAW_DATA_DIR,
        NEWS_RAW_DATA_DIR,
        SOCIAL_RAW_DATA_DIR,

        INTERIM_DATA_DIR,

        PROCESSED_DATA_DIR,
        MARKET_PROCESSED_DATA_DIR,
        NEWS_PROCESSED_DATA_DIR,
        SOCIAL_PROCESSED_DATA_DIR,

        MODELS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
