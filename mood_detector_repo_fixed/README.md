# Mood Detector (fixed)

This version auto-downloads the NLTK vader_lexicon if missing.

Run locally:
1. Create & activate venv
2. pip install -r requirements.txt
3. streamlit run mood_detector.py

If automatic download fails, run:
python -c "import nltk; nltk.download('vader_lexicon')"