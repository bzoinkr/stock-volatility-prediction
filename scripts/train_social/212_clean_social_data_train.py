import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts._bootstrap import *

from pathlib import Path

from pipelines.social.sentiment_stats import build_sentiment_stats


def main():
    INPUT_PATH  = Path("data/processed/social/reddit_posts_vader_scored_train.json")
    OUTPUT_PATH = Path("data/processed/social/social_data.json")

    build_sentiment_stats(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()