# eConsultation Sentiment Analyzer

A Streamlit web application for analyzing sentiment in public consultation comments using Hugging Face transformers. 

## Features
- Upload CSV/Excel files with comments
- Sentiment analysis (Positive, Negative, Neutral, Suggestion)
- AI-powered comment summarization
- Interactive charts (Bar, Pie, Line)
- Word frequency analysis and word cloud
- Export results with sentiment scores

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Files
- `app.py` - Main Streamlit application
- `sentiment_model.py` - Sentiment analysis and summarization logic
- `utils.py` - Visualization utilities
- `sample.csv` - Sample dataset

## Models Used
- Sentiment:  `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Summarization: `facebook/bart-large-cnn`
