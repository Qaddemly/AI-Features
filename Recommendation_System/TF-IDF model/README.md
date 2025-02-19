# 1) Job Recommendation API

  This project provides a Job Recommendation API built with FastAPI. It uses a hybrid recommendation approach combining TF-IDF, cosine similarity, Jaccard similarity, and Euclidean distance to suggest jobs based on a candidate's profile.

# 2)  Project Structure

   
  . recommendation.py: Contains the core logic for    processing resumes and jobs, including the  recommendation algorithm.
  . main.py: FastAPI application that defines the API   endpoints and data models.
  . requirements.txt: List of required Python packages.
  . README.md: This file.
  . inputJson.json : example of input data

# 3) Setup

    1. **Create and activate a virtual environment:**

      python -m venv venv
      # On Windows:
      venv\Scripts\activate
      # On Linux/macOS:
      source venv/bin/activate


# 4) Install Dependencis
  pip install -r requirements.txt


# 5) Run the FastAPI application:
  uvicorn main:app --reload


# 6) Access the API documentation:
  Open your browser and go to http://127.0.0.1:8000/docs to interact with the API.
