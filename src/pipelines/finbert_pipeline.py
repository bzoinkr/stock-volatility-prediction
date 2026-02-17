from __future__ import annotations

import json
import hashlib

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from models.finbert.model import score_finbert

NEWS_RAW_DIR = Path("data/raw/news")
OUTPUT_PATH = Path("data/processed/news/yahoo_news_finbert_scored.jsonl")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _latest_yahoo_jsonl(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("yahoo_news_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No yahoo_news_*.jsonl found in {raw_dir.resolve()}")
    return files[-1]


def _get_permalink(row: Dict[str, Any]) -> str:
    return row.get("permalink") or row.get("link") or row.get("url") or row.get("article_url") or ""


def make_yahoo_row_id(row: Dict[str, Any], ticker: str) -> str:
    permalink = _get_permalink(row)

    # Use permalink as the stable unique base (best)
    base = permalink

    # Fallback if permalink missing
    if not base:
        title = (row.get("title") or "").strip()
        summary = (row.get("description") or row.get("summary") or "").strip()
        base = title + "|" + summary

    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]  # 8–12 chars is fine
    return f"yahoo:{h}:{ticker}"


def run_finbert_on_yahoo_news(input_path: Optional[Path] = None) -> Dict[str, Any]:
    input_path = input_path or _latest_yahoo_jsonl(NEWS_RAW_DIR)

    def scored_rows() -> Iterator[Dict[str, Any]]:
        for row in _iter_jsonl(input_path):
            ticker = (row.get("symbol") or row.get("ticker") or "").upper()
            date_val = row.get("date")
            permalink = _get_permalink(row)
            row_id = make_yahoo_row_id(row, ticker)

            title = (row.get("title") or "").strip()
            desc = (row.get("description") or row.get("summary") or "").strip()
            body = (row.get("text") or "").strip()
            merged = " ".join([t for t in [title, desc, body] if t])

            scores = score_finbert(merged)

            yield {
                "id": row_id,
                "ticker": ticker,
                "date": date_val,
                "neg": scores["neg"],
                "neu": scores["neu"],
                "pos": scores["pos"],
                "compound": scores["compound"],
                "permalink": permalink,
            }

    n_written = _write_jsonl(OUTPUT_PATH, scored_rows())
    return {"input": str(input_path), "output": str(OUTPUT_PATH), "rows_written": n_written}
