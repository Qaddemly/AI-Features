#  Qaddemly AI Assistant Bot

Qaddemly AI Assistant is a multi-agent chatbot system that intelligently handles user questions in a job platform. It utilizes classification, task analysis, RAG (Retrieval-Augmented Generation), and LLM reasoning to provide helpful answers based on system features, user data, or documentation.

---

##  Features

- 🔍 **Intent Classification** (General vs Specific)
- 🧽 **Feature Task Detection** (e.g., Recommendation, Resume Builder)
- 🧠 **RAG System** for general queries using LangChain + FAISS
- 📡 **User-Aware Answering** using personalized profile data
- 🧵 **Multi-step Reasoning** via [CrewAI](https://www.crewai.com/)
- ⚙️ **LLM-Powered Agents** using `Groq` API + `llama3`

---

## 💠 Technologies Used

| Stack         | Description                      |
| ------------- | -------------------------------- |
| **FastAPI**   | API backend                      |
| **CrewAI**    | Agent orchestration              |
| **LangChain** | RAG system with FAISS embeddings |
| **FAISS**     | Vector search database           |
| **Groq**      | LLM provider (LLaMA3)            |
| **Pydantic**  | Data validation                  |
| **dotenv**    | Environment variable loader      |

---

## 🧩 System Architecture

```mermaid
flowchart TD
    A[User Question via API] --> B[Classifier Agent (GENERAL / SPECIFIC)]
    B -->|GENERAL| C[RAG System → Answer from JSON (FAISS)]
    B -->|SPECIFIC| D[Task Classifier Agent]
    D -->|Feature Matched| E[Fixed Feature Response]
    D -->|OTHER| F[Query Agent → Check Needed Data]
    F -->|Data Not Needed| G[Final Answer Agent (No Data)]
    F -->|Data Needed| H[Pass user_data]
    H --> I[Final Answer Agent (Personalized Answer)]
    E --> Z[Response to User]
    G --> Z
    I --> Z
    C --> Z
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/qaddemly-bot.git
cd qaddemly-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

```env
GROQ_API_KEY=your_groq_api_key
AGENTOPS_API_KEY=your_agentops_key  # Optional
```

### 4. Run the FastAPI server

```bash
uvicorn main:app --reload
```

### 5. (Optional) Expose using ngrok

```bash
ngrok http 8000
```

---

## 📦 API Usage

### Endpoint

`POST /qaddemly-bot`

### Request Body

```json
{
  "question": "What job roles am I best suited for based on my profile?",
  "user_type": "candidate",
  "user_data": {
    "first_name": "Abdo",
    "skills": ["Node.js", "Spring boot", "Java"],
    "experiences": [...],
    ...
  }
}
```

### Response

```json
{
  "classification": "SPECIFIC",
  "task_type": "OTHER",
  "needed_data": "USER_PROFILE",
  "answer": "Based on your profile, here are roles that suit you best..."
}
```

---

## 📁 Folder Structure

```
qaddemly-bot/
├── main.py                  # FastAPI endpoint
├── bot_runner.py            # Core logic to run multi-agent workflow
├── rag_system.py            # Retrieval-Augmented Generation system
├── agents.py                # CrewAI agents definition
├── tasks.py                 # Tasks per agent
├── data/
│   └── QA.json              # Static FAQ data for RAG
├── QA_faiss_index/          # Saved FAISS index
├── .env                     # Secrets (not committed)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 📌 Notes

- The `QA.json` file is used for **general queries**. If missing or modified, the FAISS index is automatically rebuilt.
- Add new question-answer pairs in `QA.json` to enhance general responses.
- CrewAI allows for sequential task execution between agents.

---
