import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


class JobRecommender:
    def __init__(self, weight_cosine=0.5, weight_jaccard=0.3, weight_euclidean=0.2):
        self.weight_cosine = weight_cosine
        self.weight_jaccard = weight_jaccard
        self.weight_euclidean = weight_euclidean

    def process_column(self, value):
        """Flattens nested structures and converts to string, handling nulls."""
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join([self.process_column(item) for item in value])
        elif isinstance(value, dict):
            return " ".join([self.process_column(v) for v in value.values()])
        elif hasattr(value, "dict"):  # Handle Pydantic models
            return self.process_column(value.dict())
        return str(value)

    def jaccard_similarity(self, set1, set2):
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set(set1) & set(set2))
        union = len(set(set1) | set(set2))
        return intersection / union if union > 0 else 0.0

    def normalize(self, value, min_val, max_val):
        """Normalize numerical values to a [0,1] range."""
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0

    def recommend_jobs(self, user_df, jobs_df, top_n=20):
        """Recommends jobs based on hybrid similarity: Cosine, Jaccard, and Euclidean."""
        # Process user text data for cosine similarity
        user_text = user_df.apply(
            lambda row: " ".join(
                [
                    self.process_column(row["about_me"]),
                    self.process_column(row["subtitle"]),
                    self.process_column(row["skills"]),
                    self.process_column(row["educations"]),
                    self.process_column(row["experiences"]),
                    self.process_column(row["country"]),
                    self.process_column(row["city"]),
                ]
            ),
            axis=1,
        ).values[0]

        # Process job text data
        jobs_text = jobs_df.apply(
            lambda row: " ".join(
                [
                    self.process_column(row["title"]),
                    self.process_column(row["description"]),
                    self.process_column(row["country"]),
                    self.process_column(row["city"]),
                    self.process_column(row["skills"]),
                    self.process_column(row["keywords"]),
                    self.process_column(row["employee_type"]),
                ]
            ),
            axis=1,
        ).values

        # TF-IDF Cosine Similarity
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform([user_text] + list(jobs_text))
        cosine_similarities = cosine_similarity(
            tfidf_matrix[0], tfidf_matrix[1:]
        ).flatten()

        # Jaccard Similarity for skills
        user_skills = set(user_df["skills"].iloc[0])
        jaccard_similarities = np.array(
            [
                self.jaccard_similarity(user_skills, set(job_skills))
                for job_skills in jobs_df["skills"]
            ]
        )

        # Euclidean Distance for numerical values (Salary, Experience)
        user_experience = (
            user_df["experiences"]
            .apply(
                lambda x: (
                    len(" ".join(x).split())
                    if isinstance(x, list)
                    else len(str(x).split())
                )
            )
            .iloc[0]
        )
        user_salary = user_df.get("salary", pd.Series([0])).iloc[0]

        job_experience = jobs_df["experience"].fillna(0).values
        job_salary = jobs_df["salary"].fillna(0).values

        min_exp, max_exp = job_experience.min(), job_experience.max()
        min_salary, max_salary = job_salary.min(), job_salary.max()

        user_exp_norm = self.normalize(user_experience, min_exp, max_exp)
        user_salary_norm = self.normalize(user_salary, min_salary, max_salary)

        job_exp_norm = np.array(
            [self.normalize(exp, min_exp, max_exp) for exp in job_experience]
        )
        job_salary_norm = np.array(
            [self.normalize(sal, min_salary, max_salary) for sal in job_salary]
        )

        euclidean_distances = np.array(
            [
                euclidean(
                    [user_exp_norm, user_salary_norm],
                    [job_exp_norm[i], job_salary_norm[i]],
                )
                for i in range(len(jobs_df))
            ]
        )

        max_euclidean = max(euclidean_distances) if len(euclidean_distances) > 0 else 1
        euclidean_similarities = 1 - (euclidean_distances / max_euclidean)

        # Final Hybrid Score
        final_scores = (
            self.weight_cosine * cosine_similarities
            + self.weight_jaccard * jaccard_similarities
            + self.weight_euclidean * euclidean_similarities
        )

        # Get Top N job recommendations
        top_indices = final_scores.argsort()[::-1][:top_n]
        recommended_jobs = jobs_df.iloc[top_indices].copy()
        recommended_jobs["similarity_score"] = final_scores[top_indices]

        return recommended_jobs.to_dict(orient="records")
