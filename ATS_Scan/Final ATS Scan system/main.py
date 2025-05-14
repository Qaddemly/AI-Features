from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from model import (
    clean_text,
    exact_match_score,
    calculate_cosine_similarity,
    calculate_embeddingSBERT_cosine_similarity,
    extract_text,
    extract_resume_sections_from_pdf,
    extract_skills_sections,
)

from urllib.parse import unquote
import re

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Job(BaseModel):
    id: int
    title: str
    description: str
    country: str
    city: str
    location_type: str
    status: str
    skills: List[str]
    salary: int
    employee_type: str
    keywords: List[str]
    experience: int
    business_id: int
    created_at: str
    updated_at: str


class JobApplicationResume(BaseModel):
    id: int
    url: str
    name: str
    size: int


class JobApplicationSubmit(BaseModel):
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    birth_date: Optional[str] = None
    skills: List[str]
    languages: Optional[List[str]] = None
    created_at: str
    updated_at: str
    job_application_resume: List[JobApplicationResume]
    job: Job


class InputData(BaseModel):
    success: bool
    jobApplicationSubmit: List[JobApplicationSubmit]


class UserScore(BaseModel):
    id: int
    # score: float
    normalized_score: float


class ScoringResponse(BaseModel):
    max_score: float
    sorted_users: List[UserScore]
    qualified_users: List[int]


@app.post("/score-applicants/", response_model=ScoringResponse)
async def score_applicants(data: InputData):
    try:
        max_score = 0
        sorted_users = []

        if not data.success or not data.jobApplicationSubmit:
            raise HTTPException(status_code=400, detail="Invalid input data")

        # Get job data from the first application (assuming all applications are for the same job)
        job = data.jobApplicationSubmit[0].job
        job_skills = job.skills
        job_keywords = job.keywords
        job_description = job.description

        for user in data.jobApplicationSubmit:
            user_id = user.id
            user_skills = user.skills

            if not user.job_application_resume:
                continue  # Skip users without resumes

            user_resume_path = user.job_application_resume[0].url
            user_resume_path = user_resume_path.replace("file:///", "")
            user_resume_path = unquote(user_resume_path)

            try:
                user_resume_text = extract_text(user_resume_path)
                user_resume_sections = extract_resume_sections_from_pdf(
                    user_resume_text
                )
                user_skills2 = extract_skills_sections(user_resume_sections)
                user_skills2 = clean_text(user_skills2)
            except Exception as e:
                print(f"Error processing resume for user {user_id}: {str(e)}")
                continue

            # Step 1: intersection of user skills and job skills
            matched_skills, exact_score = exact_match_score(
                user_skills, job_skills + job_keywords
            )

            # Step 2: intersection of user skills from resume and job skills & keywords using tf-idf
            resume_skills_text = " ".join(user_skills2.split()) if user_skills2 else ""
            job_skills_text = " ".join(job_skills + job_keywords)
            skills_similarity = calculate_cosine_similarity(
                resume_skills_text, job_skills_text
            )

            # Step 3: similarity between resume and job description using SBERT
            job_text = (
                job.title
                + " "
                + job.description
                + " "
                + job.country
                + " "
                + job.city
                + " "
                + job.location_type
                + " "
                + job.status
                + " "
                + " ".join(job.skills)
                + " "
                + str(job.salary)
                + " "
                + job.employee_type
                + " "
                + " ".join(job.keywords)
                + " "
                + str(job.experience)
                + " "
                + str(job.business_id)
            )
            job_text = clean_text(job_text)
            job_similarity = calculate_embeddingSBERT_cosine_similarity(
                user_resume_text, job_text
            )

            # Calculate the overall score
            overall_score = (
                0.6 * (0.65 * exact_score + 0.35 * skills_similarity)
                + 0.4 * job_similarity
            )

            max_score = max(max_score, overall_score)
            sorted_users.append({"id": user_id, "score": round(overall_score, 5)})

        # Sort users by score in descending order
        sorted_users = sorted(sorted_users, key=lambda x: x["score"], reverse=True)

        # Create response with normalized scores
        response_users = []
        for user in sorted_users:
            normalized_score = (user["score"] / max_score if max_score > 0 else 0) * 100
            response_users.append(
                {
                    "id": user["id"],
                    # "score": user["score"],
                    "normalized_score": round(normalized_score, 4),
                }
            )

        # Get IDs of qualified users (score > 70% of max)
        qualified_users = [
            user["id"]
            for user in sorted_users
            if max_score > 0 and (user["score"] / max_score) > 0.7
        ]

        return {
            "max_score": round(max_score, 2),
            "sorted_users": response_users,
            "qualified_users": qualified_users,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
