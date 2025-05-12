#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Job recommendation module for user-job matching.

This module provides functionality to recommend jobs to users based on
various similarity metrics including skill matching, title matching,
job description similarity, and employment type.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommender:
    """
    A class for recommending jobs to users.

    Provides methods to calculate various similarity metrics between user profiles
    and job listings, and combines these metrics to generate recommendations.

    Attributes
    ----------
    title_weight : float
        Weight for title matching in the final score
    cosine_weight : float
        Weight for description cosine similarity in the final score
    jaccard_weight : float
        Weight for skill similarity in the final score
    emp_type_weight : float
        Weight for employment type matching in the final score
    loc_type_weight : float
        Weight for location type matching in the final score
    top_n : int
        Number of recommendations to return
    """

    def __init__(self, title_w=0.2, cosine_w=0.2, jaccard_w=0.2,
                 emp_type_w=0.2, loc_type_w=0.2, top_n=20):
        """
        Initialize the JobRecommender with scoring weights.

        Parameters
        ----------
        title_w : float, optional
            Weight for title matching, by default 0.2
        cosine_w : float, optional
            Weight for description similarity, by default 0.2
        jaccard_w : float, optional
            Weight for skill similarity, by default 0.2
        emp_type_w : float, optional
            Weight for employment type matching, by default 0.2
        loc_type_w : float, optional
            Weight for location type matching, by default 0.2
        top_n : int, optional
            Number of recommendations to return, by default 20
        """
        self.title_weight = title_w
        self.cosine_weight = cosine_w
        self.jaccard_weight = jaccard_w
        self.emp_type_weight = emp_type_w
        self.loc_type_weight = loc_type_w
        self.top_n = top_n

    def compute_similarity(self, user_emb, job_emb):
        """
        Compute cosine similarity between user and job skill embeddings.

        Parameters
        ----------
        user_emb : numpy.ndarray
            User's skill embedding, shape (n_features,) or (1, n_features)
        job_emb : list or pandas.Series
            List or Series of job embeddings, each shape (n_features,)

        Returns
        -------
        numpy.ndarray
            Array of similarity scores for each job
        """
        if user_emb.ndim == 1:
            user_emb = user_emb.reshape(1, -1)

        job_emb = np.vstack(job_emb)

        if user_emb.shape[0] == 0 or job_emb.shape[0] == 0:
            return np.array([0] * len(job_emb))

        sim_matrix = cosine_similarity(user_emb, job_emb)
        return sim_matrix[0]

    def compute_common_skills(self, user_skills, job_skills, threshold=0.6):
        """
        Count job skills that have at least one user skill with similarity > threshold.

        Parameters
        ----------
        user_skills : list
            List of user skills (strings)
        job_skills : list
            List of job skills (strings)
        threshold : float, optional
            Cosine similarity threshold for a match, by default 0.6

        Returns
        -------
        int
            Number of common skills
        """
        if not user_skills or not job_skills:
            return 0

        with open('skill_embeddings.pkl', 'rb') as f:
            skill_to_embedding = pickle.load(f)

        user_embeddings = np.array([skill_to_embedding.get(skill, np.zeros(384)) for skill in user_skills])
        job_embeddings = np.array([skill_to_embedding.get(skill, np.zeros(384)) for skill in job_skills])

        sim_matrix = cosine_similarity(user_embeddings, job_embeddings)

        common_skills = sum(any(sim_matrix[i, j] > threshold for i in range(len(user_skills)))
                            for j in range(len(job_skills)))
        return common_skills

    def recommend(self, user_df, jobs_df):
        """
        Generate job recommendations for a user.

        Parameters
        ----------
        user_df : pandas.DataFrame
            User data dataframe (single user)
        jobs_df : pandas.DataFrame
            Jobs data dataframe

        Returns
        -------
        list
            List of recommended job dictionaries
        """
        user_df = user_df.iloc[0]
        jobs_df = jobs_df.copy()

        # Calculate basic equality metrics
        jobs_df['Title Equal'] = jobs_df['title'].apply(lambda x: 1 if x == user_df['subtitle'] else 0)
        jobs_df['Employee Equal'] = jobs_df['employee_type'].apply(
            lambda x: 1 if x == user_df['employment_type'] else 0)
        jobs_df['Location Equal'] = jobs_df['location_type'].apply(lambda x: 1 if x == user_df['location_type'] else 0)

        # Calculate description similarity using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        job_vectors = vectorizer.fit_transform(jobs_df['description'])
        user_vector = vectorizer.transform([user_df['about_me']])

        raw_similarities = cosine_similarity(user_vector, job_vectors).flatten()
        max_score = raw_similarities.max()
        normalized_similarities = raw_similarities / max_score if max_score > 0 else raw_similarities
        jobs_df['description_similarity'] = normalized_similarities

        user_skills = set(user_df["skills"])
        jobs_df['common_skills'] = jobs_df['skills'].apply(
            lambda job_skills: self.compute_common_skills(user_skills, job_skills, 0.5)
        )

        # Normalize skill similarity
        max_common = jobs_df['common_skills'].max()
        if max_common > 0:
            jobs_df['skill_similarity'] = jobs_df['common_skills'] / max_common
        else:
            jobs_df['skill_similarity'] = 0.0

        # Calculate final score
        final_scores = (
                self.title_weight * jobs_df['Title Equal'] +
                self.cosine_weight * jobs_df['description_similarity'] +
                self.emp_type_weight * jobs_df['Employee Equal'] +
                self.loc_type_weight * jobs_df['Location Equal'] +
                self.jaccard_weight * jobs_df['skill_similarity']
        )

        top_indices = final_scores.argsort()[::-1][:self.top_n]
        recommendations = [
            {
                "id": int(jobs_df.iloc[i]['id']),
                "similarity_score": float(final_scores.iloc[i])
            }
            for i in top_indices
        ]

        return recommendations

def recommend_for_user(user_df, jobs_df, title_w=0.2, cosine_w=0.2, jaccard_w=0.2,
                       emp_type_w=0.2, loc_type_w=0.2, top_n=20):
    """
    Generate job recommendations for a user.

    Parameters
    ----------
    user_df : pandas.DataFrame
        User data dataframe
    jobs_df : pandas.DataFrame
        Jobs data dataframe
    title_w : float, optional
        Weight for title matching, by default 0.2
    cosine_w : float, optional
        Weight for description similarity, by default 0.2
    jaccard_w : float, optional
        Weight for skill similarity, by default 0.2
    emp_type_w : float, optional
        Weight for employment type matching, by default 0.2
    loc_type_w : float, optional
        Weight for location type matching, by default 0.2
    top_n : int, optional
        Number of recommendations to return, by default 20

    Returns
    -------
    list
        List of recommended job dictionaries
    """
    recommender = JobRecommender(
        title_w=title_w,
        cosine_w=cosine_w,
        jaccard_w=jaccard_w,
        emp_type_w=emp_type_w,
        loc_type_w=loc_type_w,
        top_n=top_n
    )
    return recommender.recommend(user_df, jobs_df)