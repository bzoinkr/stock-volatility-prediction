from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from _bootstrap import *
from pipelines.reddit_social_pipeline import run_reddit_social_pipeline


def load_run_config(path: str = "configs/run.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find run config at: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    run_cfg = load_run_config("configs/run.yaml")

    run_reddit_social_pipeline(
        run_cfg,
        keywords_path="data/interim/keywords.json",
        raw_dir="data/raw/social",
        verbose_print_urls=True,
    )


if __name__ == "__main__":
    main()
