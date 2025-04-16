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
    job: JobItem
    users: List[User]


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    print("Received request:", request.dict())  # Log the incoming request
    recommender = RecommendForUser()

    users_data = [user.dict() for user in request.users]
    users_df = pd.DataFrame(users_data)
    print("Users DataFrame:", users_df)  # Log the users DataFrame

    job_data = request.job.dict()
    job_df = pd.DataFrame([job_data])
    print("Job DataFrame:", job_df)  # Log the job DataFrame

    return recommender.recommend_users(users_df, job_df)
