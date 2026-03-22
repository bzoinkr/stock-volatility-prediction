import json
import statistics
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
import sys
from urllib.parse import urlparse


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _domain_from_permalink(url: str | None) -> str:
    if not url:
        return "unknown"
    try:
        return urlparse(url).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def build_sentiment_stats(
    input_path: Path,
    output_path: Path,
    start_date: str | date,
    end_date: str | date,
) -> None:
    """
      - match_term = permalink domain (e.g., finance.yahoo.com)
      - date       = already "YYYY-MM-DD" in input
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)

    groups: dict[tuple, list[dict]] = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_num} — {e}", file=sys.stderr)
                continue

            if "date" not in record:
                continue
            try:
                record_date = _parse_date(record["date"])
            except Exception:
                continue

            if not (start <= record_date <= end):
                continue

            ticker = str(record.get("ticker", "")).upper()
            if not ticker:
                continue

            compound = record.get("compound", record.get("sentiment_score"))
            if compound is None:
                continue

            record["compound"] = float(compound)
            record["pos"] = float(record.get("pos", 0.0))
            record["neg"] = float(record.get("neg", 0.0))
            record["neu"] = float(record.get("neu", 0.0))
            record["match_term"] = _domain_from_permalink(record.get("permalink"))

            date_str = record_date.strftime("%Y-%m-%d")
            groups[(ticker, date_str)].append(record)

    results = []

    for (ticker, date_str), records in sorted(groups.items()):
        compounds = [r["compound"] for r in records]
        n = len(records)

        mean_compound = statistics.mean(compounds)
        variance_compound = statistics.variance(compounds) if n > 1 else None
        std_compound = statistics.stdev(compounds) if n > 1 else None

        total_pos = sum(r["pos"] for r in records)
        total_neg = sum(r["neg"] for r in records)
        total_neu = sum(r["neu"] for r in records)
        mean_pos = total_pos / n
        mean_neg = total_neg / n
        mean_neu = total_neu / n

        pos_neg_ratio = total_pos / total_neg if total_neg > 0 else None
        sentiment_balance = (total_pos - total_neg) / n

        bullish_posts = sum(1 for r in records if r["compound"] > 0.05)
        bearish_posts = sum(1 for r in records if r["compound"] < -0.05)
        neutral_posts = n - bullish_posts - bearish_posts
        bullish_ratio = bullish_posts / n
        bearish_ratio = bearish_posts / n
        neutral_ratio = neutral_posts / n

        unique_terms = set(r["match_term"] for r in records)
        term_diversity = len(unique_terms)

        term_counts: dict[str, int] = defaultdict(int)
        for r in records:
            term_counts[r["match_term"]] += 1
        top_term_ratio = max(term_counts.values()) / n

        results.append({
            "ticker": ticker,
            "date": date_str,
            "num_posts": n,
            "mean_compound": round(mean_compound, 6),
            "variance_compound": round(variance_compound, 6) if variance_compound is not None else None,
            "std_compound": round(std_compound, 6) if std_compound is not None else None,
            "mean_pos": round(mean_pos, 6),
            "mean_neg": round(mean_neg, 6),
            "mean_neu": round(mean_neu, 6),
            "total_pos": round(total_pos, 6),
            "total_neg": round(total_neg, 6),
            "total_neu": round(total_neu, 6),
            "pos_neg_ratio": round(pos_neg_ratio, 6) if pos_neg_ratio is not None else None,
            "sentiment_balance": round(sentiment_balance, 6),
            "bullish_ratio": round(bullish_ratio, 6),
            "bearish_ratio": round(bearish_ratio, 6),
            "neutral_ratio": round(neutral_ratio, 6),
            "term_diversity": term_diversity,
            "top_term_ratio": round(top_term_ratio, 6),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done — {len(results)} ticker-date groups written to '{output_path}'")