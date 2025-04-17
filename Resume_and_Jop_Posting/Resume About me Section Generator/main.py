from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from generator import generate_about_me_section
import re

app = FastAPI()

class UserData(BaseModel):
    user: Dict[str, Any]

def clean_about_me_text(text: str) -> str:
    """
    Remove any introductory lines from the generated About Me text.
    """
    # Split into lines and remove empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Remove common introductory patterns
    if len(lines) > 1:
        first_line = lines[0].lower()
        if ("here's a" in first_line or 
            "here is a" in first_line or 
            "about me" in first_line or
            "professional summary" in first_line):
            return "\n".join(lines[1:]).strip()
    
    return text.strip()



@app.post("/generate-about-me/")
async def get_about_me(data: UserData):
    try:
        raw_content = generate_about_me_section(data.dict())
        
        cleaned_content = clean_about_me_text(raw_content)
        
        cleaned_content = cleaned_content.strip('"').strip()
        
        return {"about_me": cleaned_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))