from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import List, Dict
from pyngrok import ngrok
from bot_runner import run_qaddemly_bot

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"FastAPI ngrok public URL: {public_url}")

app = FastAPI()


class BotRequest(BaseModel):
    question: str
    user_type: str
    user_id: str


# This endpoint receives the question and user type, then runs the Qaddemly bot logic
# It returns the answer of question
@app.post("/qaddemly-bot")
def ask_bot(request: BotRequest):
    return run_qaddemly_bot(
        question=request.question, user_type=request.user_type, user_id=request.user_id
    )


# Data model
class DataRequest(BaseModel):
    needed_data: List[str]
    user_type: str
    user_question: str
    user_id: str


# This endpoint sends the query to Node.js and waits for response
@app.post("/fetch-data-from-node")
def fetch_data_from_node(request: DataRequest):
    payload = {
        "needed_data": request.needed_data,
        "user_type": request.user_type,
        "user_question": request.user_question,
        "user_id": request.user_id,
    }

    try:
        # Replace this with the actual ngrok URL from your classmate’s machine
        nodejs_url = "https://29f8-156-203-147-147.ngrok-free.app/api/fetch-data"

        response = requests.post(nodejs_url, json=payload)
        response.raise_for_status()

        return {"status": "success", "retrieved_data": response.json()}

    except Exception as e:
        return {"status": "error", "message": str(e)}
