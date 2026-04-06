import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import sys


def _parse_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).date()

    if isinstance(value, str):
        value = value.strip()

        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            pass

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            pass

    raise ValueError(f"Unsupported date format: {value!r}")


def _parse_range_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _empty_day() -> dict:
    return {
        "num_posts": 0,
        "mean_compound": None,
        "variance_compound": None,
        "std_compound": None,
        "mean_pos": None,
        "mean_neg": None,
        "mean_neu": None,
        "total_pos": None,
        "total_neg": None,
        "total_neu": None,
        "pos_neg_ratio": None,
        "sentiment_balance": None,
        "bullish_ratio": None,
        "bearish_ratio": None,
        "neutral_ratio": None,
    }


def build_sentiment_stats(
    input_path: Path,
    output_path: Path,
    start_date: str | date,
    end_date: str | date,
) -> None:
    start = _parse_range_date(start_date)
    end = _parse_range_date(end_date)

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

            try:
                record_date = _parse_date(record["date"])
            except Exception as e:
                print(
                    f"Warning: skipping line {line_num} due to bad date {record.get('date')!r} — {e}",
                    file=sys.stderr,
                )
                continue

            if not (start <= record_date <= end):
                continue

            date_str = record_date.strftime("%Y-%m-%d")
            groups[(record["ticker"], date_str)].append(record)

    # Load existing output file if present, so we can merge into it
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing: dict[str, dict[str, dict]] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: existing output file is malformed, starting fresh — {e}", file=sys.stderr)
                existing = {}
    else:
        existing = {}

    all_tickers = sorted({ticker for (ticker, _) in groups})

    output: dict[str, dict[str, dict]] = existing

    for ticker in all_tickers:

        if ticker not in output:
            output[ticker] = {}

        for d in _date_range(start, end):
            date_str = d.strftime("%Y-%m-%d")
            records = groups.get((ticker, date_str), [])

            if not records:
                output[ticker][date_str] = _empty_day()
                continue

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

            output[ticker][date_str] = {
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
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    total_days = sum(len(dates) for dates in output.values())
    date_span = len(list(_date_range(start, end)))
    print(
        f"Done — merged {len(all_tickers)} tickers × {date_span} days into '{output_path}' "
        f"(file now contains {len(output)} tickers, {total_days} ticker-date entries total)"
    )