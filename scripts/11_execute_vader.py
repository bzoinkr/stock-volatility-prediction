from _bootstrap import *
from pipelines.vader_pipeline import run_vader_on_reddit_posts


def main() -> None:
    result = run_vader_on_reddit_posts()

    print("VADER scoring completed.")
    print(f"Input : {result['input']}")
    print(f"Output: {result['output']}")
    print(f"Rows  : {result['rows_written']}")


if __name__ == "__main__":
    main()
