"""
Utility functions for visualization and preprocessing.
"""
from __future__ import annotations

from typing import List, Optional, Dict
import re
from collections import Counter


def generate_wordcloud(comments: List[str]):
    """
    Generate a word cloud as a matplotlib.figure.Figure from a list of comments.

    1. Join all comments into one string.
    2. Use wordcloud.WordCloud to generate the word cloud.
    3. Render with matplotlib and return the Figure.

    Returns None gracefully if dependencies are missing or input is empty.
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not comments:
        return None
    text = " ".join(c for c in comments if isinstance(c, str)).strip()
    if not text:
        return None

    # Use transparent background so it blends with page gradient; words are black
    wc = WordCloud(width=800, height=400, background_color=None, mode="RGBA", colormap=None)
    wc_img = wc.generate(text)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.imshow(wc_img.recolor(color_func=lambda *args, **kwargs: (0, 0, 0)), interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def get_wordcloud_data(comments: List[str], max_words: int = 200) -> List[Dict[str, float]]:
    """
    Compute word frequencies for the given comments to back the word cloud.

    Returns a list of dicts: [{"word": str, "count": int, "weight": float}],
    sorted by descending count. Weight is normalized to [0,1] by max count.

    If `wordcloud` is available, uses its STOPWORDS and preprocessing; otherwise
    falls back to a simple tokenizer and a small stopword set.
    """
    if not comments:
        return []

    text = " ".join(c for c in comments if isinstance(c, str))
    if not text.strip():
        return []

    try:
        from wordcloud import WordCloud, STOPWORDS
        wc = WordCloud(stopwords=STOPWORDS)
        # process_text returns a frequency dict (word -> count)
        freq = wc.process_text(text)
    except Exception:
        # Fallback: simple tokenization and stopword removal
        fallback_stopwords = {
            'the','and','to','of','in','a','is','for','on','with','that','as','are','be','it','this',
            'by','or','an','from','at','we','will','can','may','should','would','could','our','their','your'
        }
        tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
        tokens = [t for t in tokens if t not in fallback_stopwords and len(t) > 1]
        freq = dict(Counter(tokens))

    if not freq:
        return []

    # Sort and limit
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_words]
    max_count = float(items[0][1]) if items else 1.0
    data = [
        {"word": w, "count": int(c), "weight": (float(c) / max_count if max_count else 0.0)}
        for w, c in items
    ]
    return data
