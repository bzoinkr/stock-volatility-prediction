from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from tqdm import tqdm

from models.vader.model import score_vader


def _count_lines(path: Path) -> int:
    """Fast line count without loading file into memory."""
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


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


def run_vader_on_reddit_posts(INPUT_PATH, OUTPUT_PATH) -> Dict[str, Any]:

    total = _count_lines(INPUT_PATH)
    print(f"Scoring {total:,} posts with VADER...")

    def scored_rows() -> Iterator[Dict[str, Any]]:
        bar = tqdm(
            _iter_jsonl(INPUT_PATH),
            total=total,
            unit="post",
            dynamic_ncols=True,
            desc="VADER",
            smoothing=0.05,          # smoother ETA on large files
        )
        tickers_seen: set[str] = set()
        for row in bar:
            text        = row.get("text", "")
            row_id      = row.get("source_id")
            ticker      = row.get("ticker")
            match_term  = row.get("matched_term")
            date        = row.get("created_utc")
            permalink   = row.get("permalink")
            impressions = row.get("num_comments")

            scores = score_vader(text)

            # keep postfix showing unique tickers processed so far
            if ticker:
                tickers_seen.add(ticker)
                bar.set_postfix(tickers=len(tickers_seen), ticker=ticker, refresh=False)

            yield {
                "id":          row_id,
                "ticker":      ticker,
                "match_term":  match_term,
                "date":        date,
                "neg":         scores["neg"],
                "neu":         scores["neu"],
                "pos":         scores["pos"],
                "compound":    scores["compound"],
                "impressions": impressions,
            }

    n_written = _write_jsonl(OUTPUT_PATH, scored_rows())

    return {
        "input":        str(INPUT_PATH),
        "output":       str(OUTPUT_PATH),
        "rows_written": n_written,
    }