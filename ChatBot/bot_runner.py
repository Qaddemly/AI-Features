from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from rag_system import get_answer
import os
from dotenv import load_dotenv
import json
import requests
from tasks import (
    build_classifier_task,
    build_task_classifier_task,
    build_query_task,
    build_final_answer_task,
)

# Load environment variables from .env
load_dotenv()

# Access the variables
groq_key = os.getenv("GROQ_API_KEY")
agentops_key = os.getenv("AGENTOPS_API_KEY")

# Setup LLM from Groq (using llama-3.3-70b-versatile)
groq_llm = ChatGroq(
    api_key=groq_key,
    # use llama3-8b-8192 model
    model="llama3-70b-8192",
    temperature=0,
)


FIXED_FEATURE_RESPONSES = {
    "RECOMMENDATION": (
        "**Job Recommendations**: Qaddemly automatically suggests jobs tailored to your skills, experience, and preferences. "
        "Just visit your homepage to explore opportunities that fit you best."
    ),
    "MATCHING_SCORE": (
        "**Matching Score**: For every job you view, Qaddemly shows a percentage score that reflects how well your profile matches the job. "
        "It considers your resume, skills, and preferences."
    ),
    "COVER_LETTER": (
        "**Cover Letter Assistant**: You can generate a personalized, ATS-friendly cover letter or enhance an existing one using our AI-powered tool, directly from the job details page."
    ),
    "RESUME_BUILDER": (
        "**Resume Builder**: Create or improve your resume with our smart builder. It's saved in your account, fully customizable, and optimized to pass ATS filters."
    ),
    "JOB_SEARCH": (
        "**Job Search Tools**: Use advanced filters (location, type, skills, etc.) to search, view, and save job postings that interest you."
    ),
    "COMPANY_SEARCH": (
        "**Company Search & Reviews**: Explore companies, follow them for updates, and read/write reviews to learn more before applying."
    ),
    "APPLICATION_TRACKING": (
        " **Track Your Applications**: Easily monitor your applied jobs in one place using your personalized dashboard."
    ),
    "PROFILE": (
        " **Your Profile (Portfolio)**: Build a rich profile including your skills, education, experience, projects, and more — this helps with matching, recommendations, and job applications."
    ),
    "NOTIFICATIONS": (
        " **Real-Time Notifications**: Get alerts when new matching jobs are posted, applications are updated, or companies message you."
    ),
    "MESSAGING": (
        " **Messaging System**: Use our secure chat to communicate directly with employers or HR managers about job opportunities."
    ),
}


def run_qaddemly_bot(question: str, user_type: str, user_id: str):
    result = {}
    classifier_task = build_classifier_task(question, user_type)
    classification_crew = Crew(
        agents=[classifier_task.agent],
        tasks=[classifier_task],
        process=Process.sequential,
        verbose=True,
    )

    classification_result = classification_crew.kickoff().strip().upper()
    result["classification"] = classification_result

    if classification_result == "GENERAL":
        result["answer"] = get_answer(question)
        return result

    task_classification = build_task_classifier_task(question, user_type)
    task_crew = Crew(
        agents=[task_classification.agent],
        tasks=[task_classification],
        process=Process.sequential,
        verbose=True,
    )
    task_type_result = task_crew.kickoff().strip().upper()
    result["task_type"] = task_type_result

    if task_type_result != "OTHER":
        result["answer"] = FIXED_FEATURE_RESPONSES.get(task_type_result)
        return result

    query_task = build_query_task(question, user_type)
    query_crew = Crew(
        agents=[query_task.agent],
        tasks=[query_task],
        process=Process.sequential,
        verbose=True,
    )
    data_needed = query_crew.kickoff().strip().upper()
    result["needed_data"] = data_needed

    backend_data = {}
    if data_needed != "NOTNEEDED_DATA":
        fastapi_url = "http://localhost:8000/fetch-data-from-node"
        payload = {
            "needed_data": [item.strip() for item in data_needed.split(",")],
            "user_type": user_type,
            "user_question": question,
            "user_id": user_id,
        }

        try:
            response = requests.post(fastapi_url, json=payload)
            response.raise_for_status()
            result_json = response.json()
            backend_data = result_json.get("retrieved_data", {})
        except Exception as e:
            backend_data = {}
            result["error"] = str(e)

    final_task = build_final_answer_task(question, user_type, backend_data)
    final_crew = Crew(
        agents=[final_task.agent],
        tasks=[final_task],
        process=Process.sequential,
        verbose=True,
    )
    result["answer"] = final_crew.kickoff()

    return result
