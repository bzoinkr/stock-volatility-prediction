from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from models.vader.model import score_vader


# ---------- FIXED PATHS ----------
INPUT_PATH = Path("data/raw/social/reddit_posts.jsonl")
OUTPUT_PATH = Path("data/processed/social/reddit_posts_vader_scored.jsonl")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def run_vader_on_reddit_posts() -> Dict[str, Any]:

    def scored_rows() -> Iterator[Dict[str, Any]]:
        for row in _iter_jsonl(INPUT_PATH):
            text = row.get("text", "")
            row_id = row.get("id")
            match_term = row.get("matched_term")
            date = row.get("date_utc")
            permalink = row.get("permalink")

            scores = score_vader(text)

            yield {
                "id": row_id,
                "match_term": match_term,
                "date": date,
                "neg": scores["neg"],
                "neu": scores["neu"],
                "pos": scores["pos"],
                "compound": scores["compound"],
                "permalink": permalink,
            }

    n_written = _write_jsonl(OUTPUT_PATH, scored_rows())

    return {
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_PATH),
        "rows_written": n_written,
    }