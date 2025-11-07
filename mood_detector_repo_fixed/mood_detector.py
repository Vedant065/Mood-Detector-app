# mood_detector.py
import streamlit as st
from datetime import datetime
import pandas as pd
import io
import os

st.set_page_config(page_title="Mood Detector", page_icon="😊", layout="centered")
st.title("🧠 Mood Detector — Sentiment Analyzer")
st.markdown("Type how you're feeling (or paste a sentence) and I'll detect your mood.")

@st.cache_resource
def get_vader():
    import nltk
    # prepare local nltk_data folder next to this script
    local_nltk_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
    if not os.path.exists(local_nltk_dir):
        try:
            os.makedirs(local_nltk_dir, exist_ok=True)
        except Exception:
            pass
    if local_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_dir)
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        try:
            nltk.download('vader_lexicon', download_dir=local_nltk_dir, quiet=True)
            if local_nltk_dir not in nltk.data.path:
                nltk.data.path.insert(0, local_nltk_dir)
        except Exception as e:
            raise RuntimeError(
                "NLTK vader_lexicon is missing and automatic download failed.\n"
                "If this persists, download the resource locally and add the 'nltk_data' folder to the repo.\n"
                "Locally you can run:\n"
                "  python -c \"import nltk; nltk.download('vader_lexicon', download_dir='nltk_data')\"\n"
            ) from e
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

sia = get_vader()

def analyze_text(text):
    scores = sia.polarity_scores(text)
    c = scores["compound"]
    if c >= 0.5:
        mood = "Very Positive"
        emoji = "😁"
    elif c >= 0.05:
        mood = "Positive"
        emoji = "🙂"
    elif c > -0.05:
        mood = "Neutral"
        emoji = "😐"
    elif c > -0.5:
        mood = "Negative"
        emoji = "☹️"
    else:
        mood = "Very Negative"
        emoji = "😢"
    return {"mood": mood, "emoji": emoji, "scores": scores, "compound": c}

if "log" not in st.session_state:
    st.session_state.log = []

with st.form("mood_form"):
    text = st.text_area("How are you feeling?", placeholder="I am feeling great today!", height=120)
    submitted = st.form_submit_button("Analyze")

if submitted and text.strip():
    result = analyze_text(text)
    st.markdown(f"### Result: {result['emoji']} **{result['mood']}**")
    st.write("Compound score:", round(result["compound"], 3))
    st.write("Detailed scores:", result["scores"])

    suggestions = {
        "Very Positive": "Keep it up! Share your positivity with someone 😊",
        "Positive": "Nice — keep the momentum going!",
        "Neutral": "A neutral day — maybe try a quick walk or a break?",
        "Negative": "Sorry you feel that way. Try a short breathing exercise.",
        "Very Negative": "If you’re struggling, consider talking to a friend or seeking support."
    }
    st.info(suggestions[result["mood"]])

    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "mood": result["mood"],
        "compound": result["compound"],
        "neg": result["scores"]["neg"],
        "neu": result["scores"]["neu"],
        "pos": result["scores"]["pos"]
    }
    st.session_state.log.append(entry)

st.markdown("---")
st.subheader("Mood Log & Trend")

if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df[["timestamp", "text", "mood", "compound"]]
                 .sort_values(by="timestamp", ascending=False).reset_index(drop=True))
    chart_df = df[["timestamp", "compound"]].copy()
    chart_df["ts"] = pd.to_datetime(chart_df["timestamp"])
    chart_df = chart_df.sort_values("ts")
    st.line_chart(chart_df.set_index("ts")["compound"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download log as CSV", data=csv, file_name="mood_log.csv", mime="text/csv")
    if st.button("Clear log"):
        st.session_state.log = []
        st.experimental_rerun()
else:
    st.info("No mood entries yet. Type something and press Analyze to start logging.")

st.markdown("---")
st.markdown("**About:** This app uses the VADER sentiment analyzer (NLTK) to detect emotional tone in short text. It's a lightweight prototype — not a medical or clinical tool.")
