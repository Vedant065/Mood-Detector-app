# Mood Detector

A simple Streamlit app that analyzes text sentiment using VADER (NLTK) and logs mood entries.

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run mood_detector.py
   ```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and push these files (`mood_detector.py`, `requirements.txt`, `README.md`).
2. Go to https://streamlit.io/cloud, connect your GitHub account, and create a new app using this repo and file path `mood_detector.py`.