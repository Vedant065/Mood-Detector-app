# mood_detector.py
import streamlit as st
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Mood Detector", page_icon="😊", layout="centered")
st.title("🧠 Mood Detector — Text + Image Mood Estimator (Emotion)")
st.markdown(
    "Type how you're feeling (text) or upload a front-facing photo to estimate emotion (happy, sad, angry, surprise, neutral, fear, disgust, contempt)."
)

def analyze_text(text):
    try:
        from textblob import TextBlob
    except Exception:
        return {"error": "TextBlob not installed. Add 'textblob' to requirements.txt."}
    polarity = TextBlob(text).sentiment.polarity
    c = polarity
    if c >= 0.5:
        mood = "happy"; emoji = "😁"
    elif c >= 0.05:
        mood = "positive"; emoji = "🙂"
    elif c > -0.05:
        mood = "neutral"; emoji = "😐"
    elif c > -0.5:
        mood = "sad"; emoji = "☹️"
    else:
        mood = "very sad"; emoji = "😢"
    return {"mood": mood, "emoji": emoji, "compound": c}

def analyze_image_for_mood(image_bytes):
    # Try DeepFace first (best). If not available, try FER. Otherwise fallback to OpenCV smile heuristic.
    # Returns: {"emotion": <label>, "emoji": <emoji>, ...} or {"error": "..."}.
    # Convert image bytes to suitable object for libraries when needed.
    import io
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Unable to decode image. Upload a valid JPG/PNG file."}

    # 1) DeepFace (preferred)
    try:
        from deepface import DeepFace
        import numpy as np
        # DeepFace.analyze accepts file path or np.ndarray (RGB)
        img_arr = np.array(img)
        # analyse may download models first time — that's okay if environment allows
        result = DeepFace.analyze(img_arr, actions=["emotion"], enforce_detection=True)
        dominant = result.get("dominant_emotion")
        # map to common labels and emojis
        label = dominant.lower() if dominant else "unknown"
        emoji_map = {
            "happy":"😁","sad":"☹️","angry":"😠","surprise":"😲","surprised":"😲",
            "neutral":"😐","fear":"😨","disgust":"🤢","contempt":"😒"
        }
        return {"emotion": label, "emoji": emoji_map.get(label,"❓"), "raw": result}
    except Exception as e:
        deepface_err = str(e)

    # 2) FER library
    try:
        from fer import FER
        import numpy as np
        img_arr = np.array(img)
        detector = FER(mtcnn=True)
        # detector.top_emotion returns (label, score)
        top = detector.top_emotion(img_arr)
        if top is None:
            # detector may return None if no face
            return {"emotion": "unknown", "emoji": "❓", "faces": 0, "note": "No face detected"}
        label, score = top
        label = label.lower()
        emoji_map = {"happy":"😁","sad":"☹️","angry":"😠","surprise":"😲","neutral":"😐","fear":"😨","disgust":"🤢","contempt":"😒"}
        return {"emotion": label, "emoji": emoji_map.get(label,"❓"), "score": float(score)}
    except Exception as e:
        fer_err = str(e)

    # 3) Fallback: OpenCV smile heuristic (fast, low-accuracy)
    try:
        import numpy as np
        import cv2
    except Exception:
        return {
            "error": "No supported image-emotion library available. To enable full emotion prediction, add 'deepface' (and its deps) or 'fer' to requirements.txt. "
                     "Fallback OpenCV/NumPy not available either."
        }
    # Use OpenCV cascades
    img_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return {"emotion":"unknown", "emoji":"❓", "faces":0}
    faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces_sorted[0]
    face_roi = gray[y:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20, minSize=(15,15))
    if len(smiles)>0:
        return {"emotion":"happy","emoji":"😁","faces":len(faces),"smiles":len(smiles)}
    else:
        return {"emotion":"neutral","emoji":"😐","faces":len(faces),"smiles":0, "note":"no smile detected (could be neutral or other emotion)"}

# Recommendations for fine-grained emotions
RECOMMENDATIONS = {
    "happy": ["Keep it up! Share your happiness with someone", "Do a short celebration - dance or play music"],
    "positive": ["Go for a short walk", "Listen to uplifting music"],
    "neutral": ["Try a 10-minute breathing session", "Do light stretching or yoga", "Watch a short comedy clip"],
    "sad": ["Try a brief walk outside", "Do 5 minutes of deep breathing", "Call a close friend"],
    "very sad": ["If you're struggling, consider talking to someone you trust", "Try grounding exercises or seek support"],
    "angry": ["Try 5 minutes of box-breathing (inhale 4 — hold 4 — exhale 4)", "Go for a brisk walk to release energy"],
    "surprise": ["Take a moment to breathe and process the surprise", "Share the surprise with someone"],
    "fear": ["Try grounding and breathing exercises", "If persistent, consider reaching out to a professional"],
    "disgust": ["Step away and take deep breaths", "Try a calming activity like listening to gentle music"],
    "contempt": ["Reflect on what triggered this feeling and consider a calming break"],
    "unknown": ["Try a clearer front-facing photo or describe how you feel in text"]
}

def get_recommendations_for_emotion(emotion, limit=3):
    return RECOMMENDATIONS.get(emotion, RECOMMENDATIONS["neutral"])[:limit]

if "log" not in st.session_state:
    st.session_state.log = []

st.header("Text-based mood detection")
with st.form("text_form"):
    text_input = st.text_area("How are you feeling?", placeholder="I am feeling great today!", height=120)
    submitted_text = st.form_submit_button("Analyze text")
if submitted_text and text_input.strip():
    text_res = analyze_text(text_input)
    if "error" in text_res:
        st.error(text_res["error"])
    else:
        st.markdown(f"### Text result: {text_res['emoji']} **{text_res['mood']}**")
        st.write("Score:", round(text_res["compound"], 3))
        st.info(get_recommendations_for_emotion(text_res["mood"]))
        st.session_state.log.append({"timestamp": datetime.now().isoformat(), "source":"text", "text": text_input, "mood": text_res["mood"], "score": float(text_res["compound"])})

st.markdown("---")

st.header("Image-based emotion detection")
st.markdown("Upload a **front-facing** photo of one person for best results.")
img_file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
if img_file is not None:
    image_bytes = img_file.read()
    st.image(image_bytes, caption="Uploaded image", use_column_width=True)
    with st.spinner("Analyzing image for emotion..."):
        img_res = analyze_image_for_mood(image_bytes)
    if "error" in img_res:
        st.error(img_res["error"])
        st.info("To enable better emotion detection, add 'deepface' to requirements.txt (heavier) or 'fer' + 'mtcnn'. For server use opencv-python-headless.")
    else:
        emo = img_res.get("emotion","unknown")
        emoji = img_res.get("emoji","❓")
        st.markdown(f"### Image result: {emoji} **{emo}**")
        st.write({k: v for k, v in img_res.items() if k not in ("emotion","emoji")})
        st.subheader("Recommendations")
        for r in get_recommendations_for_emotion(emo):
            st.write("- " + r)
        entry = {"timestamp": datetime.now().isoformat(), "source":"image", "mood": emo}
        st.session_state.log.append(entry)

st.markdown("---")
st.subheader("Mood Log & Trend")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df.sort_values(by="timestamp", ascending=False).reset_index(drop=True))
    # show text trend if present
    if "score" in df.columns and not df[df["source"]=="text"].empty:
        ts_df = df[df["source"]=="text"][["timestamp","score"]].copy()
        ts_df["ts"] = pd.to_datetime(ts_df["timestamp"])
        ts_df = ts_df.sort_values("ts")
        st.line_chart(ts_df.set_index("ts")["score"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download log as CSV", data=csv, file_name="mood_log.csv", mime="text/csv")
    if st.button("Clear log"):
        st.session_state.log = []
        st.experimental_rerun()
else:
    st.info("No mood entries yet. Type or upload an image to start logging.")

st.markdown("---")
st.markdown("**About:** Image emotion detection tries DeepFace (best) then FER then OpenCV-smile fallback. DeepFace/FER require heavier dependencies; add them to requirements.txt if you want accurate results.")
