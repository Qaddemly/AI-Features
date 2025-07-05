## 🚀 Qaddemly AI Chatbot — Fetching Data from Node.js to FastAPI

This project demonstrates communication between two backend servers:

- A **Node.js server** (hosted by your teammate)
- A **FastAPI server** (your backend)

---

## ⚙️ Step 1: Run Node.js Server

1. On your classmate’s machine, navigate to the Node.js project folder.

2. Run the server:

   ```bash
   node server.js
   ```

3. Start an `ngrok` tunnel to expose the server:

   ```bash
   ngrok http 3000
   ```

4. Copy the forwarded `ngrok` URL (e.g.):

   ```bash
   https://67a9-156-203-147-147.ngrok-free.app -> http://localhost:3000
   ```

5. Share this `ngrok` URL with your FastAPI backend team.

---

## ⚙️ Step 2: Run FastAPI Server

1. Open `main.py` in your FastAPI project.

2. Replace the placeholder `nodejs_url` with the real one from the Node.js server:

   ```python
   nodejs_url = "https://67a9-156-203-147-147.ngrok-free.app/api/fetch-data"
   ```

3. Run the FastAPI server (will auto-tunnel using `pyngrok`):

   ```bash
   uvicorn main:app --reload
   ```

4. Access the Swagger UI to test endpoints:

   ```
   http://127.0.0.1:8000/docs
   ```

---

## 📡 API Endpoints

### 🔹 `/qaddemly-bot` (POST)

Send user question to the Qaddemly AI Bot.

#### 📅 Request body:

```json
{
  "question": "What is the Matching Score?",
  "user_type": "candidate",
  "user_id": "12345"
}
```

---

### 🔹 `/fetch-data-from-node` (POST)

Fetch dynamic data from the Node.js database when needed.

#### 📅 Request body (used internally by the chatbot system):

```json
{
  "needed_data": ["USER_PROFILE", "USER_RESUME"],
  "user_type": "candidate",
  "user_question": "What job roles am I best suited for?",
  "user_id": "12345"
}
```

---

## 🔄 How It Works (Behind the Scenes)

1. User sends a question to `/qaddemly-bot`.
2. The bot:
   - Classifies the question.
   - If the question is `SPECIFIC`, checks if user data is needed.
   - If needed, sends a request to `/fetch-data-from-node`.
   - That request is forwarded to the Node.js backend to fetch real data (e.g., resume, profile).
   - Final answer is generated using retrieved data + LLM.

---

## 🛠️ Notes

- `HuggingFaceEmbeddings` in `rag_system.py` is deprecated in LangChain ≥ 0.2.2. Update with:

  ```bash
  pip install -U langchain-huggingface
  ```

  Replace import:

  ```python
  from langchain_huggingface import HuggingFaceEmbeddings
  ```

- Make sure `QA.json` and `QA_faiss_index/` exist before running the bot. You can run:

  ```python
  from rag_system import build_vectorstore
  build_vectorstore()
  ```

---

## ✅ Example Workflow

1. Node.js backend: Starts `ngrok` on port 3000.
2. FastAPI: Connects to the Node.js ngrok URL to fetch user-specific data.
3. Chatbot processes the question, uses data (if needed), and responds using LLM (LLaMA 3).

