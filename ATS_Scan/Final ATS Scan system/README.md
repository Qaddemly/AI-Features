# ATS Scan System

ATS Scan System is a resume parsing and scoring API built using FastAPI. It analyzes and scores job applicants based on the similarity of their resumes to a job description using a variety of NLP techniques, including TF-IDF and Sentence-BERT (SBERT).

## 🔧 Features

- Extracts and cleans text from uploaded resumes (PDFs)
- Identifies and parses common resume sections like Skills, Education, Experience, etc.
- Calculates multiple similarity scores:
  - Exact keyword match
  - TF-IDF cosine similarity
  - SBERT semantic similarity
- Ranks candidates based on scoring
- FastAPI-based REST endpoint for easy integration
- CORS-enabled for front-end use

## 🗂 Project Structure

ATSScan/
│
├── model.py # All NLP processing, scoring, and PDF extraction
├── main.py # FastAPI app for scoring endpoint
└── input.json # Example input format for testing the API

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repo-url>
cd ATSScan
```

### 2. Create & Activate Conda Environment

```bash
conda create -n ATSScan python=3.8
conda activate ATSScan
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.Run the Server

```bash
uvicorn main:app --reload
```

The API will be running at http://127.0.0.1:8000.