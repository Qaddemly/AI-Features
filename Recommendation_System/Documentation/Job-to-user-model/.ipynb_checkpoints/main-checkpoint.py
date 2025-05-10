from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union, List, Optional
from datetime import date
import pandas as pd
from recommendation import RecommendForUser

app = FastAPI()


# Pydantic Models
class Skill(BaseModel):
    name: str


class Education(BaseModel):
    university: str
    field_of_study: str
    gpa: Optional[float] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class Experience(BaseModel):
    job_title: str
    employment_type: str
    company_name: str
    location: str
    location_type: str
    still_working: Optional[bool] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class User(BaseModel):
    id: int
    country: str
    city: str
    about_me: str
    subtitle: str
    skills: List[Skill] = Field(default_factory=list)
    educations: List[Education] = Field(default_factory=list)
    experiences: List[Experience] = Field(default_factory=list)


class JobItem(BaseModel):
    id: int
    title: Optional[str] = None
    description: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    location_type: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    salary: Optional[Union[int, float]] = None
    employee_type: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    experience: Optional[int] = None


class RecommendationRequest(BaseModel):
    user: User
    jobs: List[JobItem]


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    recommender = RecommendForUser()

    user_data = request.user.dict()
    user_data["skills"] = [skill["name"] for skill in user_data.get("skills", [])]
    user_data["educations"] = [
        f"{edu.university} {edu.field_of_study}" for edu in request.user.educations
    ]
    user_data["experiences"] = [
        f"{exp.job_title} {exp.company_name} {exp.location}"
        for exp in request.user.experiences
    ]

    jobs_data = [job.dict() for job in request.jobs]

    user_df = pd.DataFrame([user_data])
    jobs_df = pd.DataFrame(jobs_data)

    return recommender.recommend_users(user_df, jobs_df)