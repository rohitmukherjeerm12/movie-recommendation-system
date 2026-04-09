# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Python and Machine Learning.

## What it does
Type any movie name and get 5 similar movie recommendations instantly.

## How it works
- Extracts features like genres, cast, crew, and keywords from 5000 movies
- Converts text data into numerical vectors using CountVectorizer
- Calculates similarity between movies using Cosine Similarity
- Displays results through an interactive Streamlit web app

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (CountVectorizer, Cosine Similarity)
- Streamlit (Web UI)
- Dataset: TMDB 5000 Movies Dataset

## How to run
```bash
pip install pandas scikit-learn streamlit numpy
python app.py
python -m streamlit run streamlit_app.py
```
