from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt

import shutil


def _clean_directory(path: Path, verbose: bool = True) -> None:
    """
    Deletes all contents inside the given directory.
    Does NOT delete the directory itself.
    """
    if not path.exists():
        return

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    if verbose:
        print(f"[sentiment-plots] Cleaned existing contents in: {path}")


def read_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _mkdir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def daily_stats(df: pd.DataFrame, *, date_col: str, score_col: str, id_col: str = "id") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, score_col]).sort_values(date_col)

    daily = (
        df.groupby(df[date_col].dt.date)
          .agg(
              n_posts=(id_col, "count"),
              score_mean=(score_col, "mean"),
              score_std=(score_col, "std"),
              score_var=(score_col, "var"),
              score_median=(score_col, "median"),
          )
          .reset_index()
          .rename(columns={date_col: "day"})
    )
    daily["day"] = pd.to_datetime(daily["day"])
    return daily.sort_values("day")


def plot_daily_post_volume(daily: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    plt.figure()
    plt.plot(daily["day"], daily["n_posts"])
    plt.title(f"{title_prefix} — Daily Article/Post Volume")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "daily_volume.png", dpi=200)
    plt.close()


def plot_daily_std(daily: pd.DataFrame, out_dir: Path, title_prefix: str, rolling_window_days: int) -> None:
    plt.figure()
    plt.plot(daily["day"], daily["score_std"], label="Daily std(compound)")
    plt.plot(daily["day"], daily["score_std_roll"], linestyle="--", label=f"{rolling_window_days}D rolling std")
    plt.title(f"{title_prefix} — Sentiment Volatility (Std of Compound)")
    plt.xlabel("Date")
    plt.ylabel("Std Dev")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "daily_compound_std.png", dpi=200)
    plt.close()


def plot_mean_with_std_band(daily: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    std = daily["score_std"].fillna(0.0)
    upper = daily["score_mean"] + std
    lower = daily["score_mean"] - std

    plt.figure()
    plt.plot(daily["day"], daily["score_mean"], label="Daily mean(compound)")
    plt.fill_between(daily["day"], lower, upper, alpha=0.25, label="± 1 std")
    plt.title(f"{title_prefix} — Mean Compound ± 1 Std")
    plt.xlabel("Date")
    plt.ylabel("Compound")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "daily_mean_band.png", dpi=200)
    plt.close()


def plot_boxplot_recent_days(
    df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    *,
    date_col: str,
    score_col: str,
    n_days: int = 14,
) -> None:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, score_col])
    df["day"] = df[date_col].dt.normalize()

    day_order = sorted(df["day"].unique())[-n_days:]
    recent = df[df["day"].isin(day_order)]

    data = [recent.loc[recent["day"] == d, score_col].values for d in day_order]
    labels = [pd.to_datetime(d).strftime("%m-%d") for d in day_order]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"{title_prefix} — Compound Distribution (Recent Days)")
    plt.xlabel("Day (MM-DD)")
    plt.ylabel("Compound")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "compound_boxplot_recent.png", dpi=200)
    plt.close()


def make_sentiment_report(
    input_jsonl: str | Path,
    out_dir: str | Path,
    *,
    date_col: str = "date",
    score_col: str = "compound",
    group_col: Optional[str] = None,
    id_col: str = "id",
    rolling_window_days: int = 5,
    recent_days_for_boxplot: int = 14,
    clean_output: bool = True,   # NEW
    verbose: bool = True,
) -> None:
    out_dir = Path(out_dir)

    if clean_output:
        if verbose:
            print(f"\n[sentiment-plots] Cleaning output directory: {out_dir}")
        _clean_directory(out_dir, verbose=verbose)

    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[sentiment-plots] Input: {input_jsonl}")
        print(f"[sentiment-plots] Output dir: {out_dir}")
        print(f"[sentiment-plots] group_col={group_col}")
        print(f"[sentiment-plots] rolling_window_days={rolling_window_days}\n")

    df = read_jsonl(input_jsonl)
    if df.empty:
        print("[sentiment-plots] No rows. Done.")
        return

    # -------------------------
    # 1) Combined report
    # -------------------------
    if verbose:
        print(f"[sentiment-plots] Building COMBINED report ({len(df):,} rows)")

    combined_daily = daily_stats(df, date_col=date_col, score_col=score_col, id_col=id_col)
    combined_daily["score_std_roll"] = combined_daily["score_std"].rolling(
        rolling_window_days, min_periods=2
    ).mean()

    combined_daily.to_csv(out_dir / "daily_sentiment_stats.csv", index=False)

    plot_daily_post_volume(combined_daily, out_dir, "Overall")
    plot_daily_std(combined_daily, out_dir, "Overall", rolling_window_days)
    plot_mean_with_std_band(combined_daily, out_dir, "Overall")
    plot_boxplot_recent_days(
        df, out_dir, "Overall", date_col=date_col, score_col=score_col, n_days=recent_days_for_boxplot
    )

    # -------------------------
    # 2) Grouped reports
    # -------------------------
    if group_col and group_col in df.columns:
        groups = sorted(df[group_col].dropna().unique())

        if verbose:
            print(f"\n[sentiment-plots] Building GROUPED reports by '{group_col}' ({len(groups)} groups)")

        for g in groups:
            g_df = df[df[group_col] == g].copy()
            g_out = out_dir / str(g)
            g_out.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"[sentiment-plots] -> {group_col}={g} ({len(g_df):,} rows)")

            daily = daily_stats(g_df, date_col=date_col, score_col=score_col, id_col=id_col)
            daily["score_std_roll"] = daily["score_std"].rolling(
                rolling_window_days, min_periods=2
            ).mean()

            daily.to_csv(g_out / "daily_sentiment_stats.csv", index=False)

            plot_daily_post_volume(daily, g_out, f"{group_col}={g}")
            plot_daily_std(daily, g_out, f"{group_col}={g}", rolling_window_days)
            plot_mean_with_std_band(daily, g_out, f"{group_col}={g}")
            plot_boxplot_recent_days(
                g_df,
                g_out,
                f"{group_col}={g}",
                date_col=date_col,
                score_col=score_col,
                n_days=recent_days_for_boxplot,
            )

    if verbose:
        print("\n[sentiment-plots] Report generation complete.\n")

