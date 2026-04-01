from _bootstrap import *
from pipelines.prediction.sentimentModels.vader_pipeline import run_vader_on_reddit_posts


def main() -> None:

    INPUT_PATH = Path("data/raw/social/reddit_posts_train.json")
    OUTPUT_PATH = Path("data/processed/social/reddit_posts_vader_scored_train.json")

    result = run_vader_on_reddit_posts(INPUT_PATH, OUTPUT_PATH)

    print("VADER scoring completed.")
    print(f"Input : {result['input']}")
    print(f"Output: {result['output']}")
    print(f"Rows  : {result['rows_written']}")


if __name__ == "__main__":
    main()
