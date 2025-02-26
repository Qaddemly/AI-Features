import re
import string
import time
import json
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

# Ensure necessary NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize the WordNet Lemmatizer
lm = WordNetLemmatizer()


def preprocess_text(text):
    """Preprocess the text: lowercase, remove numbers and punctuation, tokenize, remove stopwords, and lemmatize."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize words
    tokens = [
        word for word in tokens if word not in stopwords.words("english")
    ]  # Remove stopwords
    tokens = [lm.lemmatize(word) for word in tokens]  # Lemmatizing
    return tokens


class JobRecommendationSystem:
    def __init__(self, jobs_csv):
        """Initialize the system and load job data from a CSV file using BM25."""
        self.jobs_df = pd.read_csv(jobs_csv)
        print("Loaded job data with columns:", self.jobs_df.columns)

        # Prepare job descriptions by combining selected text columns
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

        # Store the entire job DataFrame (all columns) for later retrieval.
        self.job_info = self.jobs_df.copy()

        # Preprocess job_text for BM25 (create tokenized representation)
        self.jobs_df["bm25_tokens"] = self.jobs_df["job_text"].apply(preprocess_text)

        # Initialize BM25 with tokenized job texts
        self.bm25 = BM25Okapi(self.jobs_df["bm25_tokens"].tolist())

    def clean_text(self, text):
        """Clean text by removing extra spaces and converting to lowercase."""
        return text.lower().strip()

    def recommend_jobs(self, resume_text, top_n=20):
        """Perform job recommendation using the BM25 model."""
        # Clean and preprocess the resume text
        resume_text_clean = self.clean_text(resume_text)
        resume_tokens = preprocess_text(resume_text_clean)

        # Get BM25 similarity scores for each job
        bm25_scores = self.bm25.get_scores(resume_tokens)

        # Add BM25 scores to the full job DataFrame
        self.job_info["bm25_score"] = bm25_scores

        # Sort by BM25 score (higher score means more relevant) and get top_n jobs
        recommended_df = self.job_info.sort_values(
            by="bm25_score", ascending=False
        ).head(top_n)

        # Convert the recommendations to a list of dictionaries
        recommended_jobs = recommended_df.to_dict(orient="records")
        return {"recommended_jobs": recommended_jobs}
