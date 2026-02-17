from __future__ import annotations

from typing import Any, Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_vader(text: str) -> Dict[str, float]:
    """
    Score a single text string using VADER.

    Returns a dict with keys: neg, neu, pos, compound
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    scores: Dict[str, float] = _analyzer.polarity_scores(text)
    return scores
