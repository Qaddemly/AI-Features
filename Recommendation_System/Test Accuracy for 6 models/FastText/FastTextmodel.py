import pandas as pd
import numpy as np
import fasttext
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommendationSystem:
    def __init__(self, job_data_path: str = "JobsFE.csv"):
        # Load job data
        self.jobs = pd.read_csv(job_data_path)

        # Prepare job text
        self.jobs["job_text"] = (
            self.jobs["workplace"].astype(str)
            + " "
            + self.jobs["working_mode"].astype(str)
            + " "
            + self.jobs["position"].astype(str)
            + " "
            + self.jobs["job_role_and_duties"].astype(str)
            + " "
            + self.jobs["requisite_skill"].astype(str)
        )

        # Load FastText model
        fasttext.util.download_model("en", if_exists="ignore")
        self.ft_model = fasttext.load_model("cc.en.300.bin")

        # Compute job embeddings
        self.job_embeddings = np.array(
            [self.get_sentence_vector(text) for text in self.jobs["job_text"]]
        )

    def get_sentence_vector(self, text: str):
        words = text.split()
        vectors = [
            self.ft_model.get_word_vector(word)
            for word in words
            if word in self.ft_model.words
        ]
        return (
            np.mean(vectors, axis=0)
            if vectors
            else np.zeros(self.ft_model.get_dimension())
        )

    def recommend_jobs(self, resume: str, top_n: int = 20):
        # Generate resume embedding
        resume_embedding = self.get_sentence_vector(resume).reshape(1, -1)

        # Compute cosine similarity between resume and job embeddings
        similarity = cosine_similarity(resume_embedding, self.job_embeddings)

        # Get indices of top N jobs
        top_n_idx = similarity.argsort()[0][-top_n:][::-1]

        # Retrieve the top N jobs as a list of dictionaries
        recommended_jobs = self.jobs.iloc[top_n_idx].to_dict(orient="records")

        # Return the result in the same format as the SBERT model
        return {"recommended_jobs": recommended_jobs}
