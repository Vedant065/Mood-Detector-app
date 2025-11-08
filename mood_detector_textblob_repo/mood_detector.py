# mood_detector.py
import streamlit as st
from datetime import datetime
import pandas as pd
from textblob import TextBlob
import io
import numpy as np

st.set_page_config(page_title="Mood Detector", page_icon="😊", layout="centered")
st.title("🧠 Mood Detector — Sentiment Analyzer (Text + Image)")
st.markdown("Type how you're feeling (or paste a sentence) and I'll detect your mood. You can also upload a photo to estimate mood from facial expression and get suggestions to improve mood.")

# -------------------- Text-based analyzer (TextBlob) --------------------
def analyze_text(text):
    polarity = TextBlob(text).sentiment.polarity
    c = polarity
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
    scores = {"polarity": polarity}
    return {"mood": mood, "emoji": emoji, "scores": scores, "compound": c}

# -------------------- Image-based analyzer (OpenCV Haar cascades) --------------------
def analyze_image_for_mood(image_bytes):
    """Return inferred mood and raw details using Haar cascades for face and smile detection.
    image_bytes: raw bytes of uploaded image."""
    try:
        import cv2
    except Exception as e:
        return {
            "error": "OpenCV not installed. Install with `pip install opencv-python` to enable image mood detection."
        }

    # Read image bytes into numpy array
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Unable to decode image."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade classifiers bundled with opencv
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    except Exception:
        return {"error": "Haar cascade data not found in OpenCV installation."}

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return {"mood": "Unknown", "emoji": "❓", "faces": 0, "smiles": 0, "note": "No face detected."}

    # pick largest face by area
    faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces_sorted[0]
    face_roi_gray = gray[y:y+h, x:x+w]
    # detect smiles in the face ROI
    smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(15,15))

    # heuristics: if smiles detected -> Positive; else Neutral/Negative
    if len(smiles) > 0:
        mood = "Positive"
        emoji = "🙂"
    else:
        # no smile — could be neutral or negative; we label Neutral for caution
        mood = "Neutral"
        emoji = "😐"

    details = {
        "mood": mood,
        "emoji": emoji,
        "faces": len(faces),
        "smiles": len(smiles),
        "face_box": (int(x), int(y), int(w), int(h))
    }
    return details

# -------------------- Recommendations --------------------
RECOMMENDATIONS = {
    "Very Positive": ["Share your joy with a friend", "Keep doing what you're doing", "Write down 3 things you're grateful for"],
    "Positive": ["Go for a short walk", "Listen to uplifting music", "Call a friend and share good news"],
    "Neutral": ["Try a 10-minute guided breathing session", "Do a short stretching routine", "Watch a light-hearted short video"],
    "Negative": ["Try a short breathing exercise", "Do 10 minutes of light exercise or yoga", "Call a close friend and talk about it"],
    "Very Negative": ["If you're struggling, consider reaching out to someone you trust", "Try deep breathing + a short walk", "If needed, seek professional support"],
    "Unknown": ["Try to get a clearer photo or type how you feel", "If possible, speak with a friend or take a short break"]
}

def get_recommendations(mood, limit=3):
    opts = RECOMMENDATIONS.get(mood, RECOMMENDATIONS["Neutral"])
    # return up to limit items
    return opts[:limit]

# -------------------- Session state for logs --------------------
if "log" not in st.session_state:
    st.session_state.log = []

# -------------------- UI: Text input --------------------
st.header("Text-based mood detection")
with st.form("mood_form"):
    text = st.text_area("How are you feeling?", placeholder="I am feeling great today!", height=120)
    submitted = st.form_submit_button("Analyze text")

if submitted and text.strip():
    result = analyze_text(text)
    st.markdown(f"### Result: {result['emoji']} **{result['mood']}** (Text)")
    st.write("Polarity score:", round(result["compound"], 3))
    st.write("Detailed scores:", result["scores"])
    st.info(get_recommendations(result["mood"]))
    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": "text",
        "text": text,
        "mood": result["mood"],
        "score": float(result["compound"])
    }
    st.session_state.log.append(entry)

st.markdown("---")

# -------------------- UI: Image upload --------------------
st.header("Image-based mood estimation (upload a clear photo of a single person)")
st.markdown("Upload a photo (front-facing) and the app will try to detect a face and smile to estimate mood. This is a simple heuristic — not clinical.")

img_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
if img_file is not None:
    image_bytes = img_file.read()
    st.image(image_bytes, caption="Uploaded image", use_column_width=True)
    with st.spinner("Analyzing image for face & smile..."):
        img_result = analyze_image_for_mood(image_bytes)
    if "error" in img_result:
        st.error(img_result["error"])
    else:
        mood = img_result.get("mood", "Unknown")
        emoji = img_result.get("emoji", "❓")
        faces = img_result.get("faces", 0)
        smiles = img_result.get("smiles", 0)
        st.markdown(f"### Image result: {emoji} **{mood}**")
        st.write(f"Faces detected: {faces} — Smiles detected (in primary face): {smiles}")
        recs = get_recommendations(mood)
        st.subheader("Recommendations to uplift mood")
        for r in recs:
            st.write("- " + r)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": "image",
            "filename": getattr(img_file, "name", "uploaded_image"),
            "mood": mood,
            "faces": int(faces),
            "smiles": int(smiles)
        }
        st.session_state.log.append(entry)

st.markdown("---")

# -------------------- Mood Log and Trend --------------------
st.subheader("Mood Log & Trend")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df.sort_values(by="timestamp", ascending=False).reset_index(drop=True))
    # show trend from text scores only (images have no numeric score)
    text_scores = df[df["source"] == "text"][["timestamp", "score"]].copy()
    if not text_scores.empty:
        text_scores["ts"] = pd.to_datetime(text_scores["timestamp"])
        text_scores = text_scores.sort_values("ts")
        st.line_chart(text_scores.set_index("ts")["score"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download log as CSV", data=csv, file_name="mood_log.csv", mime="text/csv")
    if st.button("Clear log"):
        st.session_state.log = []
        st.experimental_rerun()
else:
    st.info("No mood entries yet. Type or upload an image to start logging.")

st.markdown("---")
st.markdown("**About:** Image-based mood estimation uses a simple smile-detection heuristic with OpenCV. Results are approximate. For production use, train a dedicated facial-emotion model and obtain informed consent before analyzing people's photos.")
