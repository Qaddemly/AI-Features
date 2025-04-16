from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from generator import generate_cover_letter_parts

app = FastAPI()


class UserJobRequest(BaseModel):
    user: Dict
    job: Dict


@app.post("/generate-cover-letter")
def generate_cover_letter(request: UserJobRequest):
    try:
        # Generate the cover letter parts
        result = generate_cover_letter_parts(request.dict())
        
        # Return the merged cover letter as a single string
        return {
            result["merged_cover_letter"]
        }
    except Exception as e:
        # In case of errors, return the exception details
        raise HTTPException(status_code=500, detail=str(e))
