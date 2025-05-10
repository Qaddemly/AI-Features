import string
import numpy as np
import faiss
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# Load the model using "paraphrase-MiniLM-L6-v2"
MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")
MODEL = torch.quantization.quantize_dynamic(
    MODEL, {torch.nn.Linear}, dtype=torch.qint8
)  # Apply dynamic quantization to Linear layers


class JobRecommendationSystem:
    def __init__(self, jobs_csv):
        """Initialize the system and load job data from a CSV file"""
        self.jobs_df = pd.read_csv(jobs_csv)
        print("Loaded job data with columns:", self.jobs_df.columns)

        # Prepare job descriptions from CSV by combining selected text columns
        self.jobs_df["job_text"] = (
            self.jobs_df["workplace"].astype(str)
            + " "
            + self.jobs_df["working_mode"].astype(str)
            + " "
            + self.jobs_df["position"].astype(str)
            + " "
            + self.jobs_df["job_role_and_duties"].astype(str)
            + " "
            + self.jobs_df["requisite_skill"].astype(str)
        )

        self.jobs_texts = self.jobs_df["job_text"].tolist()

        # Store the entire job DataFrame (all columns) for later retrieval.
        self.job_info = self.jobs_df.copy()

        # Encode job descriptions using the INT8 quantized model
        self.job_embeddings = MODEL.encode(
            self.jobs_texts, convert_to_numpy=True
        ).astype(np.float16)

        # Build FAISS IndexFlatIP
        self.dim = self.job_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(
            self.job_embeddings.astype(np.float16)
        )  # Run FAISS on float16 embeddings

    def clean_text(self, text):
        """Clean text by removing extra spaces, converting to lowercase, and stripping punctuation"""
        return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

    def filter_top_jobs(self, resume_text, top_n=100):
        """Reduce the number of jobs using TF-IDF to select the top 100 relevant ones"""
        vectorizer = TfidfVectorizer()
        job_vectors = vectorizer.fit_transform(self.jobs_texts)
        resume_vector = vectorizer.transform([resume_text])
        similarity_scores = (job_vectors @ resume_vector.T).toarray().flatten()

        # Get the top jobs based on highest similarity scores
        top_indices = np.argsort(similarity_scores)[-top_n:]

        # Extract required data: all columns from job_info
        return (
            [self.jobs_texts[i] for i in top_indices],  # Filtered job descriptions
            self.job_info.iloc[top_indices].reset_index(drop=True),  # Full job details
            self.job_embeddings[
                top_indices
            ],  # Precomputed embeddings for selected jobs
        )

    def recommend_jobs(self, resume_text, top_n=20):
        """Perform job recommendation using FAISS after filtering jobs"""
        resume_text = self.clean_text(resume_text)

        # Filter top 100 jobs using TF-IDF
        filtered_jobs_texts, filtered_jobs_df, filtered_embeddings = (
            self.filter_top_jobs(resume_text)
        )

        # Generate resume embedding
        resume_embedding = MODEL.encode([resume_text], convert_to_numpy=True).astype(
            np.float16
        )

        # Build a new FAISS index for filtered jobs
        index = faiss.IndexFlatIP(self.dim)
        index.add(filtered_embeddings.astype(np.float16))

        # Find the closest jobs
        distances, indices = index.search(resume_embedding.astype(np.float16), top_n)

        # Retrieve full job details based on indices
        recommended_jobs = filtered_jobs_df.iloc[indices[0]].to_dict(orient="records")

        return {"recommended_jobs": recommended_jobs}
