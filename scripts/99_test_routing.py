from _bootstrap import *

from common.paths import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    ARTIFACTS_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR,
    CONFIG_DIR, SRC_DIR, SCRIPTS_DIR, ensure_dirs
)

from pipelines.make_labels import build_labels
from pipelines.vader_pipeline import build_vader_features
from pipelines.finbert_pipeline import build_finbert_features


def main():
    print("=== IMPORTS OK ===")

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("SRC_DIR:", SRC_DIR)
    print("SCRIPTS_DIR:", SCRIPTS_DIR)
    print("CONFIG_DIR:", CONFIG_DIR)

    print("DATA_DIR:", DATA_DIR)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("INTERIM_DATA_DIR:", INTERIM_DATA_DIR)
    print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)

    print("ARTIFACTS_DIR:", ARTIFACTS_DIR)
    print("MODELS_DIR:", MODELS_DIR)
    print("PREDICTIONS_DIR:", PREDICTIONS_DIR)
    print("REPORTS_DIR:", REPORTS_DIR)

    ensure_dirs()
    print("=== DIRS ENSURED ===")

    build_labels()
    build_vader_features()
    build_finbert_features()
    print("=== PIPELINE ROUTES OK ===")


if __name__ == "__main__":
    main()
