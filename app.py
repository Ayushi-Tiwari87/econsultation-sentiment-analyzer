"""
eConsultation Sentiment Analyzer - Streamlit App

Features:
- Upload CSV/Excel with a `comment` column (case-insensitive).
- Preview first 5 rows.
- Per-row Sentiment (Placeholder or Hugging Face) and Summary (HF) columns.
- Sentiment distribution chart.
- Word cloud of all comments.
- Download enriched CSV.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    # Allow importing this module without Streamlit installed (e.g., for tests)
    class _Stub:
        def __getattr__(self, name):
            def _fn(*args, **kwargs):
                return None
            return _fn

    st = _Stub()  # type: ignore

from sentiment_model import (
    analyze_sentiments,
    summarize_comments,
    analyze_sentiment,  # HF labels only
    analyze_sentiment_with_scores,  # HF labels + scores
    summarize_text,     # HF summarization
)
from utils import generate_wordcloud, get_wordcloud_data


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if "comment" not in df.columns:
        raise ValueError("Expected a 'comment' column in the CSV.")
    return df


def _find_comment_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() == "comment":
            return c
    return None


def _read_uploaded(uploaded) -> pd.DataFrame | None:
    name = getattr(uploaded, "name", "").lower()
    try:
        if name.endswith(".csv") or name == "":
            return pd.read_csv(uploaded)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            try:
                return pd.read_excel(uploaded)
            except Exception as e:
                st.error("Failed to read Excel. Try installing 'openpyxl' or upload CSV. Error: " + str(e))
                return None
        # Fallback attempt as CSV
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return None


def main():
    st.set_page_config(page_title="eConsultation Sentiment Analyzer", layout="wide")
    # Global gradient sea green theme and professional layout
    st.markdown(
                """
                <style>
                    :root {
                        --sg-start: #FFFFFF; /* light sea green */
                        --sg-end:   #FFFFFF; /* dark sea green  */
                        --text: #000000;     /* black text */
                        --button-hover: #1F8B7A; /* darker hover */
                        --card-bg: #ADADC9; /* uniform card/box color */
                        --shadow: 0 6px 18px rgba(0,0,0,0.18);
                    }

                    /* App background gradient */
                    .stApp {
                        background: linear-gradient(135deg, var(--sg-start) 0%, var(--sg-end) 100%) !important;
                        color: var(--text) !important;
                    }

                    /* Make most text black for readability */
                    html, body, [class^="css"], .stMarkdown, .stText, .stCaption, .stRadio, .stCheckbox,
                    .stDataFrame, .stTable, .stMetric, .st-emotion-cache, [data-testid="stMarkdownContainer"],
                    [data-testid="stForm"] label, [data-testid="stSidebar"], [data-testid="column"] {
                        color: var(--text) !important;
                    }

                    /* Buttons: gradient background, black text, hover effect */
                    .stButton > button {
                        background: linear-gradient(135deg, var(--sg-start), var(--sg-end)) !important;
                        color: var(--text) !important;
                        border: none !important;
                        border-radius: 8px !important;
                        padding: 0.5rem 1rem !important;
                        box-shadow: var(--shadow) !important;
                    }
                    .stButton > button:hover {
                        background: var(--button-hover) !important;
                        filter: brightness(0.95);
                    }

                    /* Cards with subtle shadow/padding/rounded corners */
                    .card {
                        background: var(--card-bg) !important;
                        border-radius: 12px;
                        padding: 1rem 1.25rem;
                        margin: 0.75rem 0;
                        box-shadow: var(--shadow);
                    }

                    /* Ensure inner data containers also reflect the card color */
                    .card [data-testid="stDataFrame"],
                    .card [data-testid="stTable"],
                    .card [role="grid"],
                    .card .stMarkdown,
                    .card .stText {
                        background: var(--card-bg) !important;
                    }

                    /* DataFrames & charts inside cards should align nicely */
                    .card .element-container, .card [data-testid="stDataFrame"], .card canvas, .card svg {
                        filter: none;
                    }

                    /* Sidebar readability */
                    [data-testid="stSidebar"] * { color: var(--text) !important; }

                    /* Make Streamlit default containers transparent so gradient shows (except Sidebar) */
                    [data-testid="stHeader"], [data-testid="stToolbar"] {
                        background: transparent !important;
                    }

                    /* Sidebar background color */
                    [data-testid="stSidebar"] {
                        background: #ADADC9 !important;
                    }
                    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                        background: #ADADC9 !important;
                    }

                    /* Make file uploader text white (main area and sidebar) */
                    [data-testid="stFileUploader"] *,
                    [data-testid="stFileUploaderDropzone"] * {
                        color: #FFFFFF !important;
                    }
                    /* Button inside file uploader */
                    [data-testid="stFileUploader"] button {
                        color: #FFFFFF !important;
                    }
                    /* Sidebar-specific override to ensure white text despite sidebar global text color */
                    [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
                    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
                        color: #FFFFFF !important;
                    }

                    /* Make selectbox (dropdown) text and icon white */
                    [data-testid="stSelectbox"] *,
                    div[data-baseweb="select"] *,
                    div[data-baseweb="select"] {
                        color: #FFFFFF !important;
                    }
                    [data-testid="stSelectbox"] svg,
                    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
                        fill: #FFFFFF !important;
                        color: #FFFFFF !important;
                    }
                    /* Sidebar-specific override for selectbox */
                    [data-testid="stSidebar"] [data-testid="stSelectbox"] *,
                    [data-testid="stSidebar"] div[data-baseweb="select"] *,
                    [data-testid="stSidebar"] div[data-baseweb="select"] {
                        color: #FFFFFF !important;
                    }

                    /* Download buttons: gray background with white text (main + sidebar) */
                    [data-testid="stDownloadButton"] button,
                    [data-testid="stDownloadButton"] a,
                    [data-testid="stDownloadButton"] > div > button,
                    [data-testid="stDownloadButton"] > div > a,
                    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button,
                    [data-testid="stSidebar"] [data-testid="stDownloadButton"] a {
                        background-color: #6C757D !important; /* gray */
                        color: #FFFFFF !important;            /* white text */
                        border: none !important;
                        border-radius: 8px !important;
                        box-shadow: var(--shadow) !important;
                        outline: none !important;
                    }
                    [data-testid="stDownloadButton"] button:hover,
                    [data-testid="stDownloadButton"] a:hover,
                    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover,
                    [data-testid="stSidebar"] [data-testid="stDownloadButton"] a:hover {
                        background-color: #5A6268 !important; /* darker gray on hover */
                        color: #FFFFFF !important;
                    }
                    /* Disabled state styling for download buttons */
                    [data-testid="stDownloadButton"] button:disabled,
                    [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:disabled,
                    [data-testid="stDownloadButton"] a[aria-disabled="true"],
                    [data-testid="stSidebar"] [data-testid="stDownloadButton"] a[aria-disabled="true"] {
                        background-color: #6C757D !important; /* keep gray when disabled */
                        color: #FFFFFF !important;            /* keep text white */
                        opacity: 1 !important;                /* avoid dimming */
                        cursor: not-allowed !important;
                    }

                    /* Plotly fullscreen/chart visibility: force white background & black text */
                    .js-plotly-plot, .plotly, .plot-container, .plotly .svg-container, .plotly .main-svg,
                    .plotly-container, body.fullscreen, .plotly .modebar, .plotly .hoverlayer {
                        background: #FFFFFF !important;
                        color: #000000 !important;
                    }
                    /* Plotly internal background layers */
                    .plotly .bglayer rect, .plotly rect.bg, .plotly rect.draglayer, .plotly .layer-above rect,
                    .plotly g.bglayer, .plotly g.layer-below, .plotly g.layer-above {
                        fill: #FFFFFF !important;
                    }
                    /* Additional layers & subplots to guarantee white in fullscreen */
                    .plotly .cartesianlayer, .plotly .subplot, .plotly .layer-subplot, .plotly g.cartesianlayer g,
                    .plotly .gridlayer, .plotly .gridlayer rect, .plotly .zoomlayer, .plotly .hoverlayer path {
                        background: #FFFFFF !important;
                        fill: #FFFFFF !important;
                    }
                    /* Force transparent plot area overlays to not darken */
                    .plotly rect.hoverlayer, .plotly path.bg { fill: #FFFFFF !important; }
                    /* Axis labels, tick labels, title */
                    .plotly .titletext, .plotly .xtick text, .plotly .ytick text,
                    .plotly .xaxis-title, .plotly .yaxis-title, .plotly .legendtext {
                        fill: #000000 !important;
                        color: #000000 !important;
                    }
                    /* Narrow chart container to avoid overly wide visuals */
                    .chart-narrow {
                        max-width: 760px;
                        margin-left: auto !important;
                        margin-right: auto !important;
                    }
                </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='text-align:center; margin-bottom: 0.5rem;'>eConsultation Sentiment Analyzer</h1>",
        unsafe_allow_html=True,
    )

    default_csv = Path(__file__).parent / "sample.csv"
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader(
        "Upload data file (CSV or Excel) with a 'comment' column",
        type=["csv", "xlsx", "xls"],
    )
    st.sidebar.header("Engine")
    engine = st.sidebar.radio(
        "Sentiment engine",
        options=["Hugging Face"],
        index=0,
        help="Using Hugging Face models for sentiment and summarization.",
    )
    st.sidebar.header("Charts")
    chart_type = st.sidebar.selectbox("Chart type", ["Bar", "Pie", "Line"], index=0)
    st.sidebar.header("Terms")
    min_count = st.sidebar.slider("Min word frequency", min_value=1, max_value=20, value=3, step=1,
                                  help="Only show and export words that appear at least this many times.")

    if uploaded is not None:
        df = _read_uploaded(uploaded)
        if df is None:
            return
    else:
        try:
            df = load_data(default_csv)
        except Exception as e:
            st.error(str(e))
            return

    # Ensure required column exists (case-insensitive)
    comment_col = _find_comment_column(df)
    if not comment_col:
        st.error("File must contain a 'comment' column (case-insensitive). Columns found: " + ", ".join(df.columns.astype(str)))
        return

    # Normalize comment column
    df[comment_col] = df[comment_col].astype(str).fillna("")

    st.subheader("Preview (first 5 rows)")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Analysis")
    with st.spinner("Analyzing sentiments and generating summaries..."):
        if engine == "Hugging Face":
            results = analyze_sentiment_with_scores(df[comment_col].tolist())
            df["Sentiment"] = [r["label"].capitalize() for r in results]
            df["score"] = [float(r.get("score", 0.0)) for r in results]
            # Ensure numeric scores rounded to 2 decimals
            df["score"] = pd.to_numeric(df["score"], errors="coerce").round(2)
        else:
            import numpy as np
            placeholder = analyze_sentiments(df[comment_col].tolist())
            df["Sentiment"] = [str(l).capitalize() for l in placeholder.get("labels", [])]
            df["score"] = np.nan

        # Detect 'Suggestion' category and override sentiment label if applicable
        import re
        def _is_suggestion(text: str) -> bool:
            if not text:
                return False
            t = str(text).lower()
            # Common suggestion/feature-request indicators
            patterns = [
                r"\bshould\b", r"\bcould\b", r"\bwould\b", r"\bsuggest\w*\b", r"\brecommend\w*\b",
                r"\bplease\b", r"\bcan you\b", r"\bcould you\b", r"\bwould you\b",
                r"it would be (great|helpful|better)", r"\bfeature request\b", r"\bi wish\b",
                r"\bconsider\b", r"\badd\b", r"\bimprov\w*\b", r"\brequest\b",
            ]
            return any(re.search(p, t) for p in patterns)

        sugg_mask = df[comment_col].apply(_is_suggestion)
        if sugg_mask.any():
            df.loc[sugg_mask, "Sentiment"] = "Suggestion"
            # Provide a heuristic "suggestion confidence" score (0-1)
            def _suggestion_score(text: str) -> float:
                t = (str(text) or "").lower()
                strong = [r"feature request", r"\bi wish\b", r"it would be (great|helpful|better)"]
                base = 0.9 if any(re.search(p, t) for p in strong) else 0.75
                pats = [
                    r"\bshould\b", r"\bcould\b", r"\bwould\b", r"\bsuggest\w*\b", r"\brecommend\w*\b",
                    r"\bplease\b", r"\bcan you\b", r"\bcould you\b", r"\bwould you\b",
                    r"\bconsider\b", r"\badd\b", r"\bimprov\w*\b", r"\brequest\b",
                ]
                matches = sum(1 for p in pats if re.search(p, t))
                score = min(0.99, base + 0.04 * max(0, matches - 1))
                return round(score, 2)

            try:
                df.loc[sugg_mask, "score"] = df.loc[sugg_mask, comment_col].apply(_suggestion_score)
            except Exception:
                pass
        # Ensure numeric scores after any overrides
        try:
            df["score"] = pd.to_numeric(df["score"], errors="coerce").round(2)
        except Exception:
            pass

        def _row_summary(t: str) -> str:
            t = (t or "").strip()
            if not t:
                return ""
            if len(t.split()) < 25:
                return t
            try:
                return summarize_text(t)
            except Exception:
                return t

        df["Summary"] = [_row_summary(t) for t in df[comment_col].tolist()]

    # Sentiment counts for charting
    counts = {
        "Positive": int((df["Sentiment"] == "Positive").sum()) if "Sentiment" in df.columns else 0,
        "Negative": int((df["Sentiment"] == "Negative").sum()) if "Sentiment" in df.columns else 0,
        "Neutral": int((df["Sentiment"] == "Neutral").sum()) if "Sentiment" in df.columns else 0,
        "Suggestion": int((df["Sentiment"] == "Suggestion").sum()) if "Sentiment" in df.columns else 0,
    }
    st.write("Sentiment counts:", counts)

    # Chart (bar, pie, or line) with a vibrant multi-color palette
    try:
        chart_df = pd.DataFrame(
            {"sentiment": list(counts.keys()), "count": list(counts.values())}
        )
        # Transparent plot background and black text for visibility over gradient
        plot_font = {"color": "#000000"}
        # Multi-color, high-contrast palette per sentiment
        sentiment_colors = {
            "Positive": "#2E7D32",   # deep green
            "Negative": "#C62828",   # deep red
            "Neutral":  "#546E7A",   # blue-gray
            "Suggestion": "#FF8F00", # amber (dark)
        }
        mpl_colors = [sentiment_colors.get(s, "#607D8B") for s in chart_df["sentiment"].tolist()]
        st.markdown("<div class='card chart-narrow'>", unsafe_allow_html=True)
        if chart_type == "Bar":
            # Prefer Plotly for custom colors; fallback to Altair; last resort Matplotlib
            try:
                import importlib
                px = importlib.import_module("plotly.express")
                fig = px.bar(
                    chart_df,
                    x="sentiment",
                    y="count",
                    color="sentiment",
                    color_discrete_map=sentiment_colors,
                    title="Sentiment Distribution",
                    text="count",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    autosize=True, height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title="Sentiment", yaxis_title="Count",
                    template="plotly_white",
                    font=plot_font,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    import altair as alt
                    domain = list(sentiment_colors.keys())
                    range_ = list(sentiment_colors.values())
                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X('sentiment:N', title='Sentiment'),
                            y=alt.Y('count:Q', title='Count'),
                            color=alt.Color('sentiment:N', scale=alt.Scale(domain=domain, range=range_), legend=None),
                        )
                        .properties(height=240, title='Sentiment Distribution', background='transparent')
                    )
                    labels = (
                        alt.Chart(chart_df)
                        .mark_text(dy=-5, fontSize=12, color='#000000')
                        .encode(x='sentiment:N', y='count:Q', text='count:Q')
                    )
                    st.altair_chart(
                        (chart + labels).configure_axis(labelColor="#000000", titleColor="#000000"),
                        use_container_width=True,
                    )
                except Exception:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5.0, 2.6))
                    fig.patch.set_facecolor('#FFFFFF')
                    ax.set_facecolor('#FFFFFF')
                    ax.bar(chart_df["sentiment"], chart_df["count"], color=mpl_colors)
                    ax.set_title("Sentiment Distribution")
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
                    for i, v in enumerate(chart_df["count"].tolist()):
                        ax.text(i, v + (max(chart_df["count"]) * 0.02 if max(chart_df["count"]) > 0 else 0.1), str(v),
                                ha='center', va='bottom', fontsize=9)
                    st.pyplot(fig, use_container_width=True)
        elif chart_type == "Pie":
            try:
                import importlib
                px = importlib.import_module("plotly.express")
                fig = px.pie(
                    chart_df, names="sentiment", values="count", title="Sentiment Distribution",
                    color="sentiment", color_discrete_map=sentiment_colors
                )
                # Reduce plot size and avoid stretching across container
                fig.update_layout(autosize=True, height=240, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white", font=plot_font, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # Fallback to Matplotlib pie if Plotly isn't available
                import matplotlib.pyplot as plt
                # Smaller figure size for a more compact pie chart
                fig, ax = plt.subplots(figsize=(3.0, 3.0))
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')
                ax.pie(chart_df["count"], labels=chart_df["sentiment"], autopct='%1.1f%%', startangle=90, colors=mpl_colors)
                ax.axis('equal')
                st.pyplot(fig, use_container_width=True)
        else:  # Line
            try:
                import importlib
                px = importlib.import_module("plotly.express")
                fig = px.line(
                    chart_df, x="sentiment", y="count", title="Sentiment Distribution (Line)", markers=True,
                    color_discrete_sequence=["#3F51B5"],
                )
                # Enforce absolutely white background (including fullscreen) & black font
                fig.update_layout(
                    autosize=True,
                    height=240,
                    margin=dict(l=10, r=10, t=40, b=10),
                    template=None,  # remove any inherited theme that could darken fullscreen
                    font=plot_font,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                )
                # Add a white rectangle shape behind everything as final safeguard
                try:
                    fig.add_shape(
                        type="rect",
                        xref="paper", yref="paper",
                        x0=0, y0=0, x1=1, y1=1,
                        fillcolor="#FFFFFF",
                        line=dict(width=0),
                        layer="below"
                    )
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    import altair as alt
                    accent = "#3F51B5"
                    chart = (
                        alt.Chart(chart_df)
                        .mark_line(point=True, strokeWidth=3, color=accent)
                        .encode(
                            x=alt.X('sentiment:N', title='Sentiment'),
                            y=alt.Y('count:Q', title='Count'),
                        )
                        .properties(height=240, title='Sentiment Distribution (Line)', background='#FFFFFF')
                    )
                    st.altair_chart(chart.configure_axis(labelColor="#000000", titleColor="#000000"), use_container_width=True)
                except Exception:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(4.6, 2.4))
                    fig.patch.set_facecolor('#FFFFFF')
                    ax.set_facecolor('#FFFFFF')
                    ax.plot(chart_df["sentiment"], chart_df["count"], marker='o', color="#3F51B5", markerfacecolor="#3F51B5")
                    ax.set_title("Sentiment Distribution (Line)")
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                    st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        pass

    # Show enriched table (first 5 rows with new columns)
    st.subheader("Results (first 5 rows)")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Build display table explicitly to ensure score visibility
    sentiment_series = df["Sentiment"].str.upper() if "Sentiment" in df.columns else pd.Series([None]*len(df))
    score_series = df["score"] if "score" in df.columns else pd.Series([None]*len(df))
    summary_series = df["Summary"] if "Summary" in df.columns else pd.Series([None]*len(df))
    display_df = pd.DataFrame({
        "comment": df[comment_col].astype(str),
        "sentiment": sentiment_series,
        "score": score_series,
        "summary": summary_series,
    })
    st.dataframe(display_df.head())
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Most Frequent Words")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    wc_data = get_wordcloud_data(df[comment_col].tolist())
    if wc_data:
        wc_df = pd.DataFrame(wc_data)
        # Apply user-selected minimum frequency threshold
        wc_df = wc_df[wc_df["count"] >= min_count].sort_values("count", ascending=False).reset_index(drop=True)
        if wc_df.empty:
            st.caption(f"No terms with frequency >= {min_count}.")
        else:
            st.dataframe(wc_df.head(50))
        # Downloads
        if not wc_df.empty:
            st.download_button(
                label="Download Word Frequencies (CSV)",
                data=wc_df.to_csv(index=False).encode("utf-8"),
                file_name="word_frequencies.csv",
                mime="text/csv",
            )
        try:
            import io
            excel_buf = io.BytesIO()
            if not wc_df.empty:
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    wc_df.to_excel(writer, index=False, sheet_name="frequencies")
                st.download_button(
                    label="Download Word Frequencies (Excel)",
                    data=excel_buf.getvalue(),
                    file_name="word_frequencies.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception:
            st.caption("Install 'openpyxl' to enable Excel download for word frequency data.")
    else:
        st.caption("No terms available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Download enriched results
    st.subheader("Download Results")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV (with Sentiment + Summary)",
        data=csv_bytes,
        file_name="econsultation_sentiment_results.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Overall 1–2 line summary at the very end
    st.subheader("Overall Summary")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    def _counts_fallback_summary(cnt: dict[str, int]) -> str:
        total = max(1, sum(cnt.values()))
        dominant = max(cnt, key=cnt.get) if total > 0 else "Neutral"
        ordered = ["Positive", "Negative", "Neutral", "Suggestion"]
        parts = []
        for k in ordered:
            v = cnt.get(k, 0)
            if v > 0:
                pct = int(round(v * 100 / total))
                parts.append(f"{k}: {pct}%")
        dist = ", ".join(parts) if parts else "No data"
        return f"Overall tone: {dominant.upper()}. Distribution — {dist}."

    try:
        texts = [str(t).strip() for t in df[comment_col].astype(str).tolist() if str(t).strip()]
        if texts:
            # Join comments and truncate to a reasonable size to keep inference fast
            combined = ". ".join(texts)
            if len(combined) > 6000:
                combined = combined[:6000]
            with st.spinner("Generating overall 1–2 line summary..."):
                overall_summary = summarize_text(combined)
            final_summary = overall_summary or _counts_fallback_summary(counts)
        else:
            final_summary = "No comments to summarize."
    except Exception:
        final_summary = _counts_fallback_summary(counts)

    st.write(final_summary)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    # If running as a script (not via `streamlit run`), just print a note.
    # Typical usage: `streamlit run app.py`
    try:
        main()
    except Exception as exc:
        print(f"App failed to start: {exc}")
