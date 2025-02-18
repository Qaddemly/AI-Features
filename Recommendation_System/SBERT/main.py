from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from SBERTmodel import JobRecommendationSystem  # Import your class from SBERTmodel.py

app = FastAPI()

# Global variable to cache the list of jobs
cached_jobs = None


# Define Pydantic models for request validation
class Resume(BaseModel):
    url: str
    name: str
    size: int
    id: int
    accountId: int


class Job(BaseModel):
    title: str
    description: str
    location: Dict[str, str]
    location_type: str
    skills: List[str]
    salary: int
    employee_type: str
    keywords: List[str]
    experience: str
    business: Dict[str, Any]
    id: int
    status: str
    created_at: str
    updated_at: str


class JobsRequest(BaseModel):
    success: bool
    message: str
    jobs: List[Job]


class ResumeRequest(BaseModel):
    success: bool
    message: str
    resume: Resume


class RecommendedJobResponse(BaseModel):
    resume_id: int
    recommended_jobs: List[Job]  # List of recommended job objects


@app.post("/cache-jobs/")
async def cache_jobs(jobs_request: JobsRequest):
    """
    Cache the list of jobs received from the backend.
    """
    global cached_jobs
    if not jobs_request.success:
        raise HTTPException(
            status_code=400, detail="Failed to cache jobs: Invalid request"
        )

    cached_jobs = jobs_request.jobs
    return {"message": "Jobs cached successfully"}


@app.post("/recommend-jobs/", response_model=RecommendedJobResponse)
async def recommend_jobs(resume_request: ResumeRequest):
    """
    Recommend jobs for a specific resume using the cached list of jobs.
    """
    global cached_jobs

    # Check if jobs are cached
    if cached_jobs is None:
        raise HTTPException(
            status_code=400, detail="No jobs cached. Please cache jobs first."
        )

    try:
        # Extract resume data
        resume_json = resume_request.dict()

        # Initialize the JobRecommendationSystem with the cached list of jobs
        recommendation_system = JobRecommendationSystem(
            [job.dict() for job in cached_jobs]
        )

        # Get recommendations (IDs of recommended jobs)
        result = recommendation_system.recommend_jobs(resume_json)

        # Check if there was an error in processing
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Extract the recommended job objects using their IDs
        recommended_jobs = []
        for job_id in result["recommended_jobs"]:
            job = next((job for job in cached_jobs if job.id == job_id), None)

            if job:
                recommended_jobs.append(job)

        # Prepare the response
        response = RecommendedJobResponse(
            resume_id=resume_json["resume"]["id"], recommended_jobs=recommended_jobs
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
