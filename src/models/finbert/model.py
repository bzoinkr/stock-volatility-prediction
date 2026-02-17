from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
MAX_LEN = 256

_tokenizer = None
_model = None
_device = None
_id2label = None


def _load() -> None:
    global _tokenizer, _model, _device, _id2label
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(_device)
    _model.eval()
    _id2label = {int(k): str(v).lower() for k, v in _model.config.id2label.items()}


@torch.no_grad()
def score_finbert(text: str) -> Dict[str, float]:
    """
    Returns keys: neg, neu, pos, compound
    compound = pos - neg (like a VADER-style signed score)
    """
    _load()

    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    inputs = _tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    logits = _model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    probs_by_label = {_id2label[i]: float(p) for i, p in enumerate(probs)}
    pos = probs_by_label.get("positive", 0.0)
    neu = probs_by_label.get("neutral", 0.0)
    neg = probs_by_label.get("negative", 0.0)

    return {"neg": neg, "neu": neu, "pos": pos, "compound": pos - neg}
