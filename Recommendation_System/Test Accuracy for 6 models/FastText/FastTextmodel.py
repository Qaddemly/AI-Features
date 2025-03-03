import os
import pandas as pd
import numpy as np
import fasttext
import fasttext.util
import faiss  # For efficient similarity search
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from typing import List
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources (only once)
nltk.download("stopwords")
nltk.download("wordnet")


class JobRecommendationSystem:
    def __init__(self, job_data_path: str = "JobsFE.csv"):
        # Load job data
        self.jobs = pd.read_csv(job_data_path)
        # print("job columns is ", self.jobs.columns)

        # Precompute Stopwords & Stemmer
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        # Prepare and preprocess job text
        self.jobs["job_text"] = (
            self.jobs[
                [
                    "workplace",
                    "working_mode",
                    "position",
                    "job_role_and_duties",
                    "requisite_skill",
                ]
            ]
            .astype(str)
            .agg(" ".join, axis=1)
            .apply(self.clean_and_preprocess_text)
        )

        # Efficient FastText Loading
        if not os.path.exists("cc.en.300.bin"):
            fasttext.util.download_model("en", if_exists="ignore")
        self.ft_model = fasttext.load_model("cc.en.300.bin")

        # Fine-tune FastText (with reduced training time)
        self.fine_tune_fasttext(self.jobs["job_text"].tolist())

        # Compute job embeddings efficiently
        self.job_texts = self.jobs[
            "job_text"
        ].to_numpy()  # Avoid redundant Pandas overhead
        self.job_embeddings = np.array(
            [self.get_sentence_vector(text) for text in self.job_texts],
            dtype=np.float32,
        )

        # Reduce dimensionality with PCA
        self.pca = PCA(n_components=100, random_state=42)
        self.job_embeddings = self.pca.fit_transform(self.job_embeddings)
        self.job_embeddings = normalize(self.job_embeddings.astype(np.float32))

        # Use FAISS for fast nearest neighbor search
        self.index = faiss.IndexFlatIP(
            100
        )  # Inner Product (cosine similarity with normalized vectors)
        self.index.add(self.job_embeddings)

    def clean_and_preprocess_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s]", "", text.lower())  # Remove punctuation
        words = [
            self.stemmer.stem(word)
            for word in text.split()
            if word not in self.stop_words
        ]
        return " ".join(words)

    def fine_tune_fasttext(self, texts: List[str]):
        with open("job_texts.txt", "w", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)
        self.ft_model = fasttext.train_unsupervised(
            "job_texts.txt",
            model="cbow",
            dim=300,
            epoch=10,
            lr=0.05,
            thread=4,
            minCount=10,
            wordNgrams=2,
        )

    def get_sentence_vector(self, text: str):
        vectors = np.array(
            [
                self.ft_model.get_word_vector(word)
                for word in text.split()
                if word in self.ft_model.words
            ],
            dtype=np.float32,
        )
        return (
            np.mean(vectors, axis=0)
            if vectors.size
            else np.zeros(300, dtype=np.float32)
        )

    def recommend_jobs(self, resume: str, top_n: int = 20):
        # Clean and preprocess resume text
        resume = self.clean_and_preprocess_text(resume)

        # Generate resume embedding
        resume_embedding = self.get_sentence_vector(resume)
        resume_embedding = normalize(
            self.pca.transform(resume_embedding.reshape(1, -1)).astype(np.float32)
        )

        # Use FAISS to find the top N similar jobs
        similarity_scores, top_n_idx = self.index.search(resume_embedding, top_n)

        # Retrieve the top N jobs as a list of dictionaries
        recommended_jobs = []
        for idx, score in zip(top_n_idx[0], similarity_scores[0]):
            job = self.jobs.iloc[idx].to_dict()  # Get all job details
            job["similarity_score"] = float(score)  # Add similarity score
            # print(recommended_jobs)
            recommended_jobs.append(job)

        # Return the result in the desired format
        return {"recommended_jobs": recommended_jobs}
