import pandas as pd
import numpy as np
import nltk
import re
import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")


class JobRecommendationSystem:
    def __init__(self, job_data_path: str = "JobsFE.csv"):
        """
        Initialize the system and load job data from a CSV file.
        """
        # Load job data
        self.job_data = pd.read_csv(job_data_path)

        # Print columns for debugging
        print(self.job_data.columns)

        # Combine relevant job description fields into a single text column
        self.job_data["job_text"] = (
            self.job_data[
                ["position", "job_role_and_duties", "requisite_skill", "offer_details"]
            ]
            .astype(str)
            .agg(" ".join, axis=1)
        )

        # Initialize stopwords and lemmatizer
        self.en_stopwords = stopwords.words("english")
        self.en_stopwords.remove("not")
        self.lm = WordNetLemmatizer()

        # Preprocess job descriptions
        self.job_data["processed_text"] = self.job_data["job_text"].apply(
            self.preprocess_text
        )

        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(self.job_data["processed_text"])
        self.corpus = [
            self.dictionary.doc2bow(text) for text in self.job_data["processed_text"]
        ]

        # Train LDA model
        self.lda_model = gensim.models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=10,  # Adjust based on performance
            passes=15,
            random_state=42,
        )

        # Get topic distribution for each job
        self.job_data["topic_distribution"] = self.job_data["job_text"].apply(
            lambda x: self.get_topic_distribution(x)
        )

    def preprocess_text(self, text):
        """
        Preprocess the text by lowercasing, removing numbers and punctuation, tokenizing, removing stopwords, and lemmatizing.
        """
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenize words
        tokens = [
            word for word in tokens if word not in self.en_stopwords
        ]  # Remove stopwords
        tokens = [self.lm.lemmatize(word) for word in tokens]  # Lemmatizing
        return tokens

    def get_topic_distribution(self, text):
        """
        Get the topic distribution for a given text.
        """
        bow_vector = self.dictionary.doc2bow(
            self.preprocess_text(text)
        )  # Convert text to bag-of-words
        return self.lda_model.get_document_topics(
            bow_vector, minimum_probability=0.0
        )  # Get topic distribution

    def get_resume_vector(self, resume_text):
        """
        Convert resume text to a vector in LDA topic space.
        """
        bow_vector = self.dictionary.doc2bow(self.preprocess_text(resume_text))
        topic_vector = self.lda_model.get_document_topics(
            bow_vector, minimum_probability=0.0
        )
        return np.array([prob for _, prob in topic_vector]).reshape(1, -1)

    def recommend_jobs(self, resume_text: str, top_n: int = 20):
        """
        Recommend jobs based on a single resume text.
        """
        # Convert resume to LDA topic space
        user_vector = self.get_resume_vector(resume_text)

        # Convert all job postings to a NumPy matrix
        job_topic_matrix = np.array(
            [
                [prob for _, prob in topic]
                for topic in self.job_data["topic_distribution"]
            ]
        )

        # Compute similarity
        similarities = cosine_similarity(user_vector, job_topic_matrix)

        # Get top N recommended jobs
        self.job_data["similarity_score"] = similarities[0]
        recommended_jobs = self.job_data.sort_values(
            by="similarity_score", ascending=False
        ).head(top_n)

        # Retrieve the recommended jobs
        recommended_jobs_dict = recommended_jobs[
            ["Job Id", "position", "job_role_and_duties", "similarity_score"]
        ].to_dict(orient="records")

        # Return the result in the same format as the SBERT model
        return {"recommended_jobs": recommended_jobs_dict}
