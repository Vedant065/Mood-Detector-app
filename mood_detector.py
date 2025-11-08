import streamlit as st
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Mood Detector", page_icon="😊", layout="centered")
st.title("🧠 Mood Detector — Text + Image Mood Estimator")
st.markdown("Type a sentence to detect mood from text, or upload a clear photo of a single person to estimate mood from facial expression.")

def analyze_text_blob(text):
    from textblob import TextBlob
    polarity = TextBlob(text).sentiment.polarity
    c = polarity
    if c >= 0.5:
        mood = "Very Positive"; emoji = "😁"
    elif c >= 0.05:
        mood = "Positive"; emoji = "🙂"
    elif c > -0.05:
        mood = "Neutral"; emoji = "😐"
    elif c > -0.5:
        mood = "Negative"; emoji = "☹️"
    else:
        mood = "Very Negative"; emoji = "😢"
    scores = {"polarity": polarity}
    return {"mood": mood, "emoji": emoji, "scores": scores, "compound": c}

def analyze_image_for_mood(image_bytes):
    import cv2, numpy as np
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Unable to decode image."}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return {"mood":"Unknown","emoji":"❓","faces":0,"smiles":0}
    faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces_sorted[0]
    face_roi_gray = gray[y:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(15,15))
    if len(smiles) > 0:
        mood = "Positive"; emoji = "🙂"
    else:
        mood = "Neutral"; emoji = "😐"
    return {"mood":mood,"emoji":emoji,"faces":int(len(faces)),"smiles":int(len(smiles))}

RECOMMENDATIONS = {
    "Very Positive":["Share your joy with a friend","Keep doing what you're doing","Write down 3 things you're grateful for"],
    "Positive":["Go for a short walk","Listen to uplifting music","Call a friend and share good news"],
    "Neutral":["Try a 10-minute guided breathing session","Do a short stretching routine or light yoga","Watch a light-hearted short video"],
    "Negative":["Try a short breathing exercise","Do 10 minutes of light exercise or yoga","Call a close friend and talk about how you feel"],
    "Very Negative":["Consider reaching out to someone you trust","Try deep breathing and a short walk","If needed, seek professional support"],
    "Unknown":["Try a clearer photo or type how you feel","Take a short break and do deep breathing","Reach out to a friend if possible"]
}
def get_recommendations(mood, limit=3):
    return RECOMMENDATIONS.get(mood, RECOMMENDATIONS["Neutral"])[:limit]

if "log" not in st.session_state:
    st.session_state.log = []

st.header("Text-based mood detection")
with st.form("text_form"):
    text_input = st.text_area("How are you feeling?", placeholder="I am feeling great today!", height=120)
    submitted_text = st.form_submit_button("Analyze text")
if submitted_text and text_input.strip():
    text_result = analyze_text_blob(text_input)
    st.markdown(f"### Result: {text_result['emoji']} **{text_result['mood']}** (Text)")
    st.write("Polarity score:", round(text_result["compound"],3))
    st.write("Detailed scores:", text_result["scores"])
    st.info(get_recommendations(text_result["mood"]))
    st.session_state.log.append({"timestamp":datetime.now().isoformat(),"source":"text","text":text_input,"mood":text_result["mood"],"score":float(text_result["compound"])})

st.markdown("---")

st.header("Image-based mood estimation")
st.markdown("Upload a clear front-facing photo of one person.")
img_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
if img_file is not None:
    image_bytes = img_file.read()
    st.image(image_bytes, caption="Uploaded image", use_column_width=True)
    img_result = analyze_image_for_mood(image_bytes)
    if "error" in img_result:
        st.error(img_result["error"])
    else:
        mood = img_result.get("mood","Unknown"); emoji = img_result.get("emoji","❓")
        st.markdown(f"### Image result: {emoji} **{mood}**")
        st.write(f"Faces detected: {img_result.get('faces',0)} — Smiles detected: {img_result.get('smiles',0)}")
        st.subheader("Recommendations to uplift mood")
        for r in get_recommendations(mood):
            st.write("- " + r)
        st.session_state.log.append({"timestamp":datetime.now().isoformat(),"source":"image","filename":getattr(img_file,"name","uploaded_image"),"mood":mood,"faces":img_result.get("faces",0),"smiles":img_result.get("smiles",0)})

st.markdown("---")

st.subheader("Mood Log & Trend")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df.sort_values(by="timestamp",ascending=False).reset_index(drop=True))
    text_scores = df[df["source"]=="text"][["timestamp","score"]].copy()
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
st.markdown("Notes: Image-based mood estimation is heuristic (face & smile detection). For production, use a trained model and obtain consent.")
