# 🏏 CricketSensei

**CricketSensei** is an intelligent cricket shot analysis system that uses pose estimation, shot-based keyframe extraction, and machine learning to classify cricket shots and provide detailed feedback on posture and technique.

## 🔍 Overview

Upload a video of your cricket shot and CricketSensei will:
- Predict the **type of shot played**
- Calculate a **match percentage** comparing your form with ideal standards
- Provide **frame-wise and phase-wise posture feedback**
- Suggest personalized tips for improvement

---

## 🚀 Run the Web App

> Run the full system with a user-friendly interface via Flask.

### 🔧 Requirements

- Python 3.8+
- Required libraries (install with `pip install -r requirements.txt`):
  - Flask
  - OpenCV
  - scikit-learn
  - joblib
  - numpy
  - mediapipe

### ▶️ Start the App

```bash
python app.py
```

> The app will start at `http://127.0.0.1:5000/`

---

## 📊 Machine Learning Model

- **Model**: RandomForestClassifier (Scikit-learn)
- **Features**: Mean joint angles from video frames
- **Shot Types**: e.g., Cover Drive, Pull Shot
- **Frame Analysis**: Each frame gets form prediction + quality score

---

## 📋 Features

- ✅ **Shot Prediction**
- 📈 **Match Percentage Score**
- 🧍 **Pose-Based Frame Analysis**
- 🎯 **Generalized Phase Feedback** (Start, Middle, End)
- 💬 **Personalized Tips for Form Improvement**
- 🧠 **JSON Output for Advanced Review**

---

## 📚 Reference Paper (Literature Base)

**"Shot-Based Keyframe Extraction using Edge LBP Approach"**

- **Method Used**: Edge detection + Local Binary Patterns to extract keyframes
- **Relevance**: Inspired our own keyframe extraction for critical posture frames
- **Advantages**: Lightweight, shot-aware
- **Disadvantages**: LBP lacks semantic motion awareness (solved via pose)

---

## 🌱 Future Enhancements

- Add more shot types (square cut, straight drive etc.)
- Integrate 3D pose estimation
- Real time feedback
- Export performance reports

---

## 🤝 Contributors

- Rohan Raghavan
- Nandana A Dev
- Rohn Raphael
- Shobin Shino Job

---

> ⚡ CricketSensei — your AI cricket coach in the cloud! 🏏🤖
