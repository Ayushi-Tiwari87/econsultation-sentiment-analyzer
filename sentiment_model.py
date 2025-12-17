"""
Sentiment and summarization utilities.

This module provides both simple placeholder functions (for quick demos)
and production-ready functions powered by Hugging Face Transformers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Any


# ------------------------
# Placeholder functions
# ------------------------
def analyze_sentiments(comments: List[str]) -> Dict[str, object]:
    """
    Simple keyword-based analysis kept for backward compatibility.
    Returns a dict with counts and per-item labels among
    {'positive','negative','neutral'} (lowercase).
    """
    positive_markers = {"positive", "beneficial", "needed", "clear", "better"}
    negative_markers = {"burden", "confusing", "difficult", "unnecessary"}

    labels: List[str] = []
    for c in comments:
        cl = c.lower()
        if any(w in cl for w in positive_markers):
            labels.append("positive")
        elif any(w in cl for w in negative_markers):
            labels.append("negative")
        else:
            labels.append("neutral")

    counts = {
        "positive": sum(1 for l in labels if l == "positive"),
        "negative": sum(1 for l in labels if l == "negative"),
        "neutral": sum(1 for l in labels if l == "neutral"),
    }

    return {"counts": counts, "labels": labels}


def summarize_comments(comments: List[str]) -> str:
    """
    Placeholder summary: returns the first and last comments concatenated.
    """
    if not comments:
        return "No comments to summarize."
    if len(comments) == 1:
        return f"Single comment: {comments[0]}"
    return (
        "High-level takeaway (placeholder): "
        + comments[0].strip()
        + " ... "
        + comments[-1].strip()
    )


# ------------------------
# Hugging Face powered functions
# ------------------------
@lru_cache(maxsize=1)
def _get_sentiment_pipeline():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as e:
        raise ImportError(
            "transformers is required for analyze_sentiment. Install with: pip install transformers torch"
        ) from e

    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
    )


def analyze_sentiment(comments: List[str]) -> List[str]:
    """
    Analyze sentiment using CardiffNLP RoBERTa model.

    Args:
        comments: list of input strings.

    Returns:
        List of labels, each in {"Positive", "Negative", "Neutral"}.
    """
    if not comments:
        return []

    nlp = _get_sentiment_pipeline()
    results = nlp(comments)

    def _normalize(label: str) -> str:
        lbl = label.lower()
        if "pos" in lbl:
            return "Positive"
        if "neg" in lbl:
            return "Negative"
        if "neu" in lbl:
            return "Neutral"
        # Fallback for generic labels like LABEL_0/1/2 using known Cardiff mapping
        mapping = {
            "label_0": "Negative",
            "label_1": "Neutral",
            "label_2": "Positive",
        }
        return mapping.get(lbl, label.capitalize())

    return [_normalize(r.get("label", "Neutral")) for r in results]


def analyze_sentiment_with_scores(comments: List[str]) -> List[Dict[str, Any]]:
    """
    Hugging Face sentiment returning a list of {'label': 'NEGATIVE|NEUTRAL|POSITIVE', 'score': float}.
    Labels are normalized to UPPERCASE.
    """
    if not comments:
        return []

    nlp = _get_sentiment_pipeline()
    results = nlp(comments)

    def _normalize_upper(label: str) -> str:
        lbl = label.lower()
        if "pos" in lbl:
            return "POSITIVE"
        if "neg" in lbl:
            return "NEGATIVE"
        if "neu" in lbl:
            return "NEUTRAL"
        mapping = {
            "label_0": "NEGATIVE",
            "label_1": "NEUTRAL",
            "label_2": "POSITIVE",
        }
        return mapping.get(lbl, label.upper())

    out: List[Dict[str, Any]] = []
    for r in results:
        out.append({
            "label": _normalize_upper(r.get("label", "NEUTRAL")),
            "score": float(r.get("score", 0.0)),
        })
    return out


@lru_cache(maxsize=1)
def _get_summarization_pipeline():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    except ImportError as e:
        raise ImportError(
            "transformers is required for summarize_text. Install with: pip install transformers torch"
        ) from e

    model_id = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
    )


def summarize_text(text: str) -> str:
    """
    Summarize input text into 1–2 sentences using BART-Large-CNN.

    For very long inputs, the text will be truncated to the model's max tokens.
    """
    if not text or not text.strip():
        return ""

    pipe = _get_summarization_pipeline()
    # Tune to encourage 1–2 sentence summaries
    summary = pipe(
        text,
        min_length=30,
        max_length=120,
        do_sample=False,
        num_beams=4,
        truncation=True,
    )
    return summary[0]["summary_text"].strip()

