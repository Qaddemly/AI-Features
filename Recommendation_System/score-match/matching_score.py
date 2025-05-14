import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import math

class MatchingScore:
    def __init__(self, model, skill_embeddings, title_w=0.2, cosine_w=0.2, jaccard_w=0.2,
                 emp_type_w=0.2, loc_type_w=0.2, top_n=20, k=10):
        self.model = model  # Preloaded SentenceTransformer
        self.skill_embeddings = skill_embeddings  # Preloaded skill embeddings
        self.title_weight = title_w
        self.cosine_weight = cosine_w
        self.jaccard_weight = jaccard_w
        self.emp_type_weight = emp_type_w
        self.loc_type_weight = loc_type_w
        self.top_n = top_n
        self.k = k  # Scaling factor for logarithmic transformation

    def logarithmic_transform(self, similarity):
        """Apply logarithmic transformation to similarity score."""
        return math.log(1 + self.k * similarity) / math.log(1 + self.k)

    def compute_title_similarity(self, user_subtitle, job_title):
        """Compute embedding-based similarity for titles with logarithmic scaling."""
        user_emb = self.model.encode([user_subtitle])[0]
        job_emb = self.model.encode([job_title])[0]
        similarity = cosine_similarity([user_emb], [job_emb])[0][0]
        return self.logarithmic_transform(similarity)

    def compute_description_similarity(self, user_about_me, job_description):
        """Compute embedding-based similarity for descriptions with logarithmic scaling."""
        user_emb = self.model.encode([user_about_me])[0]
        job_emb = self.model.encode([job_description])[0]
        similarity = cosine_similarity([user_emb], [job_emb])[0][0]
        return self.logarithmic_transform(similarity)

    def compute_skill_similarity(self, user_skills, job_skills):
        """Compute skill similarity based on common skills with threshold."""
        if not user_skills or not job_skills:
            return 0.0
        user_embeddings = np.array([self.skill_embeddings.get(skill, np.zeros(384)) for skill in user_skills])
        job_embeddings = np.array([self.skill_embeddings.get(skill, np.zeros(384)) for skill in job_skills])
        sim_matrix = cosine_similarity(user_embeddings, job_embeddings)
        common_skills = sum(any(sim_matrix[i, j] > 0.5 for i in range(len(user_skills)))
                            for j in range(len(job_skills)))
        return common_skills / len(job_skills) if len(job_skills) > 0 else 0.0

    def find_score(self, user, job):
        """Compute similarity score between a user and a job."""
        # Title similarity (embedding-based with logarithmic scaling)
        title_similarity = self.compute_title_similarity(user['subtitle'], job['title'])

        # Description similarity (embedding-based with logarithmic scaling)
        description_similarity = self.compute_description_similarity(user['about_me'], job['description'])

        # Skill similarity
        skill_similarity = self.compute_skill_similarity(user['skills'], job['skills'])

        # Employment and location type matching (binary)
        emp_equal = 1 if user['employment_type'] == job['employee_type'] else 0
        loc_equal = 1 if user['location_type'] == job['location_type'] else 0

        # Final weighted score
        final_score = (
            self.title_weight * title_similarity +
            self.cosine_weight * description_similarity +
            self.jaccard_weight * skill_similarity +
            self.emp_type_weight * emp_equal +
            self.loc_type_weight * loc_equal
        )
        return final_score

# Example usage:
# preprocessor = DataPreprocessor()
# model = preprocessor.model
# with open('skill_embeddings.pkl', 'rb') as f:
#     skill_embeddings = pickle.load(f)
# matcher = MatchingScore(model, skill_embeddings)
# score = matcher.recommend(user_dict, job_dict)
# print("Similarity Score:", score)