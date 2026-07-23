<div align="center">

# 😊 AI Mood Detector

### *An Intelligent Emotion Detection System using Text Sentiment Analysis & Facial Expression Recognition*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/DeepFace-Facial%20Recognition-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TextBlob-Sentiment%20Analysis-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

*A Machine Learning-powered application that detects a user's emotional state through **Text Sentiment Analysis** and **Facial Emotion Recognition**, providing personalized wellness recommendations via an intuitive Streamlit interface.*

</div>

---

# 📖 About the Project

AI Mood Detector is a Python-based intelligent emotion recognition system designed to analyze human emotions using two independent approaches:

- 📝 **Text Sentiment Analysis**
- 😀 **Facial Emotion Detection**

The application predicts a user's emotional state from either written text or facial expressions and provides meaningful suggestions based on the detected emotion.

This project combines **Machine Learning**, **Natural Language Processing**, and **Computer Vision** into a single user-friendly application.

---

# ✨ Features

- ✅ Text-based Mood Detection
- ✅ Facial Emotion Recognition
- ✅ Real-time Emotion Prediction
- ✅ Interactive Streamlit Interface
- ✅ Image Upload Support
- ✅ Personalized Mood Recommendations
- ✅ Fast and Lightweight
- ✅ Easy Local Deployment

---

# 🛠 Tech Stack

## Programming Language
- Python

## Frontend
- Streamlit

## Machine Learning
- DeepFace
- FER

## Computer Vision
- OpenCV

## NLP
- TextBlob

## Libraries
- NumPy
- Pandas
- Pillow

---

# 📂 Project Structure

```text
Mood-Detector-App/
│
├── app.py
├── mood_detector.py
├── recommendation.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── home.png
│   ├── text_detection.png
│   ├── facial_detection.png
│
└── images/
```

---

# ⚙ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Vedant065/Mood-Detector-app.git
```

## 2️⃣ Navigate to Project Directory

```bash
cd Mood-Detector-app
```

## 3️⃣ Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

# 🚀 Usage

### 📝 Text Mood Detection

1. Open the application.
2. Navigate to **Text Analysis**.
3. Enter any sentence or paragraph.
4. Click **Analyze**.
5. View the detected mood and recommendations.

---

### 😀 Facial Emotion Detection

1. Navigate to **Facial Detection**.
2. Upload an image or use webcam (if supported).
3. Click **Detect Emotion**.
4. View detected facial emotion.
5. Receive personalized suggestions.

---

# 🧠 How It Works

## Text Sentiment Analysis

```
User Text
      │
      ▼
TextBlob
      │
      ▼
Sentiment Analysis
      │
      ▼
Mood Classification
      │
      ▼
Recommendations
```

---

## Facial Emotion Detection

```
Input Image
      │
      ▼
OpenCV
      │
      ▼
Face Detection
      │
      ▼
DeepFace / FER
      │
      ▼
Emotion Prediction
      │
      ▼
Suggestions
```

---

# 🔄 Complete Workflow

```text
                User
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   Text Input         Image Upload
        │                   │
        ▼                   ▼
 TextBlob NLP          OpenCV
        │                   │
        ▼                   ▼
 Sentiment          Face Detection
 Analysis                 │
        │                 ▼
        │          DeepFace / FER
        │                 │
        └──────────┬──────┘
                   ▼
          Emotion Classification
                   │
                   ▼
      Mood Recommendation System
                   │
                   ▼
             Final Result
```

---

# 📷 Screenshots

> Replace these images with your own screenshots.

## 🏠 Home Page

```
assets/home.png
```

---

## 📝 Text Analysis

```
assets/text_detection.png
```

---

## 😀 Facial Emotion Detection

```
assets/facial_detection.png
```

---

# 📊 Supported Emotions

| Emotion | Description |
|----------|-------------|
| 😀 Happy | Positive emotional state |
| 😢 Sad | Low emotional state |
| 😠 Angry | High frustration |
| 😨 Fear | Anxiety or fear |
| 😲 Surprise | Unexpected reaction |
| 😐 Neutral | Balanced emotional state |
| 🤢 Disgust | Negative response |

---

# 📈 Future Enhancements

- 🎤 Voice Emotion Recognition
- 🌐 Multi-language Support
- 📊 Mood History Dashboard
- 🔐 User Authentication
- ☁ Cloud Deployment
- 📱 Mobile Responsive UI
- 🤖 AI Chat-based Mental Wellness Assistant
- 🎵 Music Recommendation System
- 📅 Daily Mood Tracking
- 📈 Emotion Analytics

---

# 💡 Applications

- Mental Wellness Monitoring
- Educational Demonstrations
- Human-Computer Interaction
- Emotion-aware Applications
- AI Research
- Healthcare Assistance
- Student Well-being Analysis

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

## Vedant Dhamele

🎓 Final Year Engineering Student

💻 Passionate about Artificial Intelligence, Machine Learning, Computer Vision, Full Stack Development, and Open Source.

### Connect with Me

- **GitHub:** https://github.com/Vedant065
- **LinkedIn:** https://www.linkedin.com/in/vedant-dhamele

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a Star!

**Made with ❤️ using Python, Streamlit, Machine Learning, NLP & Computer Vision**

</div>
