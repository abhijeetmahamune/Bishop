# Bishop-Brain_Checker_Post_Counselling_Bot
# 🧠 Brain Checker – AI Post-Counseling Guidance System

An AI-powered web system that helps parents and students understand their psychometric counseling report and guides them through career planning, college selection, action planning, and parent guidance.

> Built for **Brain Checker®** — India's Largest Career Counseling Company.

---

## ✨ Features

| Mode | Description |
|------|-------------|
| 📄 Understand Report | Upload PDF report, ask questions in simple language |
| 🗺 Career Roadmap | Timeline from 10th → 12th → Degree → Career |
| 🎓 Find Best College | Guided Q&A → personalised college suggestions |
| 📅 Action Plan | 30-60-90 day plan for student + parent |
| 👨‍👩‍👧 Parent Guide | How to support, motivate, communicate with child |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/brain-checker-ai.git
cd brain-checker-ai
```

### 2. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and add your OPENROUTER_API_KEY:
```
OPENROUTER_API_KEY=your_real_key_here
```
Get your key from: https://aistudio.google.com/app/apikey

### 3. Install Python dependencies
```bash
cd backend
pip install -r backend/requirements.txt
```

### 4. Run the backend
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 5. Open the frontend
Simply open `index.html` in your browser, or serve it:
```bash
cd ..
python -m http.server 5500
```
Then visit: `http://localhost:5500`

---

## 📁 Project Structure

```
brain-checker-ai/
├── index.html              ← Frontend (single file)
├── .env                    ← Your API key (DO NOT commit)
├── .env.example            ← Safe placeholder for GitHub
├── .gitignore              ← Ignores .env and other sensitive files
├── README.md               ← This file
└── backend/
    ├── main.py             ← FastAPI server
    ├── pdf_utils.py        ← PDF text extraction
    └── requirements.txt    ← Python dependencies
```

---

## 🔐 API Key Safety

- ✅ API key is stored in `.env` (never committed to GitHub)
- ✅ `.gitignore` excludes `.env` automatically
- ✅ Backend handles all API calls — key never exposed in frontend
- ✅ `.env.example` shows structure without real values

---

## 🛠 Tech Stack

- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **Backend:** Python, FastAPI
- **AI:** OPENROUTER_API_KEY - nvidia/nemotron-3-super-120b-a12b:free
- **PDF Parsing:** PyMuPDF / pypdf
- **Fonts:** Poppins, DM Sans

---

## ⚠️ Disclaimer

This system is a **post-counseling guidance tool** only. It does not replace professional psychological counseling. Always refer to the original counselor's report and recommendations.

---

## 📞 Contact

**Brain Checker®**  
107, Business Court, 100 Feet Ring Road, Nashik-422006  
📱 +91-99700-57774  
🌐 [www.brainchecker.in](https://brainchecker.in)
