import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts._bootstrap import *
from pipelines.prediction.sentimentModels.finbert_pipeline import run_finbert_on_yahoo_news


def main() -> None:
    result = run_finbert_on_yahoo_news()
    print("FINBERT completed.")
    print(f"  Input : {result['input']}")
    print(f"  Output: {result['output']}")
    print(f"  Rows  : {result['rows_written']}")


if __name__ == "__main__":
    main()