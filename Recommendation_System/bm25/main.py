import pandas as pd
import numpy as np
import nltk
import re
import ast
import gensim
import joblib
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.tokenize import word_tokenize
from gensim import corpora
from rank_bm25 import BM25Okapi
#
# en_stopwords = stopwords.words('english')
# en_stopwords.remove("not")
# lm = WordNetLemmatizer()
#
# def preprocess_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     tokens = word_tokenize(text)  # Tokenize words
#     tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
#     tokens = [lm.lemmatize(word) for word in tokens]  # Lemmatizing
#     return tokens
#
# job_data = pd.read_csv('job_data_sahy.csv')
# start = time.time()
# bm25 = joblib.load('bm25_model.joblib')
# user_resume = "Software engineer with Python, machine learning, and cloud computing experience."
# user_resume_tokens = preprocess_text(user_resume)
#
# bm25_scores = bm25.get_scores(user_resume_tokens)
#
# job_data["bm25_score"] = bm25_scores
#
# recommended_jobs_bm25 = job_data.sort_values(by="bm25_score", ascending=False).head(5)
#
# print(recommended_jobs_bm25[['Job Title', 'Job Description']].head(n=5))
#
# end = time.time()
#
# print("Total time:", end-start)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import re

bm25 = joblib.load('bm25_model.joblib')
job_data = pd.read_csv('job_data_sahy.csv')


class UserInfo(BaseModel):
    Skills: str

app = FastAPI()

def preprocess_text(text):
    # lm = WordNetLemmatizer()
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize words
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    # tokens = [lm.lemmatize(word) for word in tokens]  # Lemmatizing
    return tokens

def recommend_jobs(user_skills, top_n=5):
    user_skills_tokens = preprocess_text(user_skills)
    bm25_scores = bm25.get_scores(user_skills_tokens)
    job_data["bm25_score"] = bm25_scores
    recommended_jobs = job_data.sort_values(by="bm25_score", ascending=False).head(top_n)
    return [recommended_jobs["Job Id"].tolist(), recommended_jobs["Job Title"].tolist()]

@app.post("/")
async def get_recommendations(user_info: UserInfo):
    try:
        user_skills = user_info.Skills
        recommended_job_ids, job_titles = recommend_jobs(user_skills, top_n=5)
        return {"recommended_job_ids": recommended_job_ids, "job_title": job_titles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)