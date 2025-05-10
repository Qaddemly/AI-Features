import string
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import json


def process_column(value):
    """Flattens nested structures and converts everything to a string."""
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value) if pd.notna(value) else ""


def normalize(value, min_val, max_val):
    """Normalize numerical values to a [0,1] range."""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0


class JobRecommendationSystem:
    def __init__(self, jobs_csv):
        """Initialize the system and load job data from a CSV file."""
        self.jobs_df = pd.read_csv(jobs_csv)

        # Prepare job descriptions by combining relevant columns.
        job_columns = ["position", "job_role_and_duties", "requisite_skill"]
        if "offer_details" in self.jobs_df.columns:
            job_columns.append("offer_details")

        self.jobs_df["job_text"] = self.jobs_df[job_columns].apply(
            lambda row: " ".join(process_column(val) for val in row), axis=1
        )

    def clean_text(self, text):
        """Clean text by converting to lowercase, removing punctuation, and extra spaces."""
        return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

    def recommend_jobs(self, resume_text, top_n=20):
        """Recommend jobs based on a resume using TF-IDF and hybrid similarity measures."""
        resume_text = self.clean_text(resume_text)
        all_text = [resume_text] + self.jobs_df["job_text"].tolist()

        # Compute TF-IDF vectors.
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(all_text)

        # Extract cosine similarities.
        cosine_similarities = cosine_similarity(
            tfidf_matrix[0], tfidf_matrix[1:]
        ).flatten()

        # Euclidean Distance for numerical values (Salary Normalization)
        user_salary = 0  # Assume 0 if not provided
        job_salary = self.jobs_df["salary"].fillna(0).astype(float).values

        min_salary, max_salary = job_salary.min(), job_salary.max()
        user_salary_norm = normalize(user_salary, min_salary, max_salary)
        job_salary_norm = np.array(
            [normalize(sal, min_salary, max_salary) for sal in job_salary]
        )

        euclidean_distances = np.array(
            [
                euclidean([user_salary_norm], [job_salary_norm[i]])
                for i in range(len(self.jobs_df))
            ]
        )
        max_euclidean = max(euclidean_distances) if len(euclidean_distances) > 0 else 1
        euclidean_similarities = 1 - (euclidean_distances / max_euclidean)

        # Final Hybrid Score (Weighted Combination)
        weight_cosine, weight_euclidean = 0.8, 0.2
        final_scores = (
            weight_cosine * cosine_similarities
            + weight_euclidean * euclidean_similarities
        )

        # Get Top N job recommendations
        top_indices = final_scores.argsort()[::-1][:top_n]
        recommended_jobs = self.jobs_df.iloc[top_indices].copy()
        recommended_jobs["similarity_score"] = final_scores[top_indices]

        return {"recommended_jobs": recommended_jobs.to_dict(orient="records")}
