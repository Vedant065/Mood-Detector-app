# mood_detector_text_fix.py
import streamlit as st
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Mood Detector (Text Fix)", page_icon="😊", layout="centered")
st.title("🧠 Mood Detector — Text (Safe Mode)")
st.markdown("This is a safe minimal version. The text box will always appear even if other libs fail.")

def safe_analyze_text(text):
    try:
        from textblob import TextBlob
    except Exception:
        # fallback simple polarity: count positive/negative words (very basic)
        pos_words = {"good","great","happy","awesome","fantastic","love","nice","amazing"}
        neg_words = {"bad","sad","angry","hate","terrible","awful","upset","mad"}
        words = {w.lower().strip(".,!?") for w in text.split()}
        score = (len(words & pos_words) - len(words & neg_words)) / max(1, len(words))
        # map to same buckets used before
        if score >= 0.3: mood, emoji = "Very Positive", "😁"
        elif score >= 0.05: mood, emoji = "Positive", "🙂"
        elif score > -0.05: mood, emoji = "Neutral", "😐"
        elif score > -0.5: mood, emoji = "Negative", "☹️"
        else: mood, emoji = "Very Negative", "😢"
        return {"mood": mood, "emoji": emoji, "score": score}
    # if TextBlob available, use it
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.5: mood, emoji = "Very Positive", "😁"
    elif polarity >= 0.05: mood, emoji = "Positive", "🙂"
    elif polarity > -0.05: mood, emoji = "Neutral", "😐"
    elif polarity > -0.5: mood, emoji = "Negative", "☹️"
    else: mood, emoji = "Very Negative", "😢"
    return {"mood": mood, "emoji": emoji, "score": polarity}

if "log" not in st.session_state:
    st.session_state.log = []

with st.form("text_form", clear_on_submit=False):
    text_input = st.text_area("How are you feeling?", placeholder="I am feeling great today!", height=140)
    submitted = st.form_submit_button("Analyze")
if submitted and text_input.strip():
    res = safe_analyze_text(text_input)
    st.markdown(f"### {res['emoji']} {res['mood']}")
    st.write("Score:", round(res["score"], 3))
    st.session_state.log.append({"timestamp": datetime.now().isoformat(), "source": "text", "text": text_input, "mood": res["mood"], "score": float(res["score"])})

st.markdown("---")
st.subheader("Log")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log).sort_values(by="timestamp", ascending=False)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "mood_log.csv")
else:
    st.info("No entries yet.")
