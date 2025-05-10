import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
import numpy as np


class JobRecommendationSystem:
    def __init__(self, job_data_path: str = "JobsFE.csv"):
        """
        Initialize the system and load job data from a CSV file.
        """
        # Load job data
        self.jobs_df = pd.read_csv(job_data_path)

        # Combine relevant job description fields into a single text column
        self.jobs_df["job_text"] = (
            self.jobs_df[
                ["position", "job_role_and_duties", "requisite_skill", "offer_details"]
            ]
            .astype(str)
            .agg(" ".join, axis=1)
        )

        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # Fit the vectorizer on job descriptions
        self.job_vectors = self.vectorizer.fit_transform(self.jobs_df["job_text"])

        # Convert job vectors to a NumPy array and normalize
        self.job_vectors_np = normalize(self.job_vectors.toarray(), axis=1)

        # Build BallTree (using Euclidean distance on normalized vectors)
        self.tree = BallTree(self.job_vectors_np, metric="euclidean")

    def recommend_jobs(self, resume_text: str, top_n: int = 20):
        """
        Recommend jobs based on a single resume text.
        """
        # Transform the resume text into a TF-IDF vector
        resume_vector = self.vectorizer.transform([resume_text])

        # Convert and normalize the resume vector
        resume_vector_np = normalize(resume_vector.toarray(), axis=1)

        # Find the top N nearest jobs using BallTree
        distances, indices = self.tree.query(resume_vector_np, k=top_n)

        # Retrieve the recommended jobs
        recommended_jobs = self.jobs_df.iloc[indices[0]].to_dict(orient="records")

        # Return the result in the same format as the SBERT model
        return {"recommended_jobs": recommended_jobs}
