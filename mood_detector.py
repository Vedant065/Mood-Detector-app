# mood_detector.py
import streamlit as st
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Mood Detector", page_icon="😊", layout="centered")
st.title("🧠 Mood Detector — Emotion from Text + Image")
st.markdown("Upload a clear front-facing image (one person) to detect emotion (happy, sad, angry, surprise, neutral, fear, disgust, contempt).")

def analyze_text_blob(text):
    try:
        from textblob import TextBlob
    except Exception:
        return {"error": "TextBlob missing. Add 'textblob' to requirements.txt."}
    p = TextBlob(text).sentiment.polarity
    if p >= 0.5: m, e = "happy", "😁"
    elif p >= 0.05: m, e = "positive", "🙂"
    elif p > -0.05: m, e = "neutral", "😐"
    elif p > -0.5: m, e = "sad", "☹️"
    else: m, e = "very sad", "😢"
    return {"mood": m, "emoji": e, "score": p}

def use_deepface(img_rgb):
    """Attempt DeepFace analysis. Returns dict or raises."""
    from deepface import DeepFace
    import numpy as np
    img_arr = np.array(img_rgb)  # DeepFace expects RGB ndarray
    # enforce_detection may throw if no face found
    result = DeepFace.analyze(img_arr, actions=["emotion"], enforce_detection=True)
    # result often includes 'dominant_emotion' and 'emotion' dict
    dominant = result.get("dominant_emotion")
    emotions = result.get("emotion") or {}
    # normalize keys to lowercase
    emotions = {k.lower(): float(v) for k, v in emotions.items()}
    return {"engine": "deepface", "dominant": (dominant or "").lower(), "scores": emotions, "raw": result}

def use_fer(img_rgb):
    """Attempt FER analysis. Returns dict or raises."""
    from fer import FER
    import numpy as np
    img_arr = np.array(img_rgb)
    detector = FER(mtcnn=True)  # mtcnn improves face detection
    # detect_emotions returns list of dicts
    detections = detector.detect_emotions(img_arr)
    if not detections:
        return {"engine": "fer", "dominant": "unknown", "scores": {}, "faces": 0}
    # take largest face by box area
    best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    emotions = best.get("emotions", {})
    # emotions keys already lowercase, values between 0..1
    emotions = {k: float(v) for k, v in emotions.items()}
    dom = max(emotions.items(), key=lambda kv: kv[1])[0] if emotions else "unknown"
    return {"engine": "fer", "dominant": dom, "scores": emotions, "faces": len(detections)}

def use_opencv_fallback(img_rgb):
    """OpenCV smile/face heuristic fallback (fast, low-accuracy)."""
    try:
        import numpy as np, cv2
    except Exception:
        raise
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return {"engine": "opencv", "dominant": "unknown", "scores": {}, "faces": 0}
    faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces_sorted[0]
    roi = gray[y:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=20, minSize=(15,15))
    if len(smiles) > 0:
        return {"engine": "opencv", "dominant": "happy", "scores": {"happy": 1.0}, "faces": len(faces), "smiles": len(smiles)}
    else:
        return {"engine": "opencv", "dominant": "neutral", "scores": {"neutral": 1.0}, "faces": len(faces), "smiles": 0}

# mapping/emoji
EMOJI = {"happy":"😁","sad":"☹️","angry":"😠","surprise":"😲","neutral":"😐","fear":"😨","disgust":"🤢","contempt":"😒","positive":"🙂","very sad":"😢","unknown":"❓"}

def aggregate_emotion(result):
    """Given engine result dict, return (label, emoji, detail). Applies thresholds to avoid false-neutral for anger etc."""
    if not result:
        return "unknown", EMOJI.get("unknown"), {}
    engine = result.get("engine", "unknown")
    dom = (result.get("dominant") or "unknown").lower()
    scores = result.get("scores") or {}
    # Some engines use 0..100 (DeepFace) or 0..1 (FER). Normalize to 0..1
    norm = {}
    for k, v in scores.items():
        val = float(v)
        if val > 1.5:  # probably DeepFace percent-style
            val = val / 100.0
        norm[k.lower()] = val
    # If dominant is available, consider it, but allow scores to override
    label = dom
    # If dominant is neutral but anger score is significant, prefer angry
    anger_score = norm.get("angry", norm.get("anger", 0.0))
    if label in ("neutral","unknown") and anger_score >= 0.25:
        label = "angry"
    # If dominant is neutral but sadness or fear is higher than threshold, pick them
    for emo in ("sad","fear","disgust","surprise","happy","angry","contempt"):
        if norm.get(emo, 0.0) >= 0.4 and (label == "neutral" or label == "unknown"):
            label = emo
            break
    # final fallback: if no strong scores but dominant exists, use it
    if label in ("unknown","neutral") and dom not in ("unknown","neutral"):
        label = dom
    emoji = EMOJI.get(label, "❓")
    return label, emoji, {"engine": engine, "raw_scores": norm}

def analyze_image(image_bytes):
    """Try DeepFace -> FER -> OpenCV fallback. Return final emotion label, emoji, detail."""
    from PIL import Image
    import io
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file."}
    # Try DeepFace
    try:
        res = use_deepface(img)
        label, emoji, detail = aggregate_emotion(res)
        detail.update({"engine": res.get("engine")})
        return {"emotion": label, "emoji": emoji, "detail": detail}
    except Exception as e:
        deep_err = str(e)
    # Try FER
    try:
        res = use_fer(img)
        label, emoji, detail = aggregate_emotion(res)
        detail.update({"engine": res.get("engine")})
        return {"emotion": label, "emoji": emoji, "detail": detail}
    except Exception as e:
        fer_err = str(e)
    # Try OpenCV fallback
    try:
        res = use_opencv_fallback(img)
        label, emoji, detail = aggregate_emotion(res)
        detail.update({"engine": res.get("engine")})
        return {"emotion": label, "emoji": emoji, "detail": detail}
    except Exception as e:
        return {"error": "No supported image-emotion library available. To enable full emotion prediction, add 'deepface' or 'fer' to requirements.txt. Fallback OpenCV/NumPy not available either."}

RECOMMENDATIONS = {
    "happy":["Keep it up! Share your happiness","Dance or listen to music"],
    "positive":["Go for a short walk","Listen to uplifting music"],
    "neutral":["Try a 10-minute breathing session","Do light stretching or yoga"],
    "sad":["Try a brief walk outside","Call a friend","Try 5 minutes of breathing"],
    "very sad":["Reach out to someone you trust or seek support","Try grounding exercises"],
    "angry":["Try box-breathing (4-4-4), take a brisk walk","Count to 10 and breathe deeply"],
    "surprise":["Take a moment to process it and breathe","Share with someone"],
    "fear":["Grounding + breathing; seek support if needed"],
    "disgust":["Step away, take deep breaths, calm down"],
    "contempt":["Take a calming break, reflect"],
    "unknown":["Try a clearer photo or type how you feel"]
}

def get_recs(emotion):
    return RECOMMENDATIONS.get(emotion, RECOMMENDATIONS["unknown"])[:3]

if "log" not in st.session_state:
    st.session_state.log = []

st.header("Image Emotion Detection")
st.markdown("Upload a front-facing photo of one person.")
img_file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
if img_file:
    image_bytes = img_file.read()
    st.image(image_bytes, caption="Uploaded", use_column_width=True)
    with st.spinner("Analyzing..."):
        out = analyze_image(image_bytes)
    if "error" in out:
        st.error(out["error"])
        st.info("To enable better detection add 'deepface' or 'fer' to requirements.txt (and use opencv-python-headless for server).")
    else:
        emo = out["emotion"]
        emoji = out["emoji"]
        st.markdown(f"### Detected emotion: {emoji} **{emo}**")
        st.write("Details:", out.get("detail", {}))
        st.subheader("Recommendations")
        for r in get_recs(emo):
            st.write("- " + r)
        st.session_state.log.append({"timestamp": datetime.now().isoformat(), "source":"image", "filename": getattr(img_file, "name", ""), "emotion": emo})

st.markdown("---")
st.subheader("Log")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log).sort_values(by="timestamp", ascending=False)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="mood_log.csv")
    if st.button("Clear log"):
        st.session_state.log = []
        st.experimental_rerun()
else:
    st.info("No logs yet.")
