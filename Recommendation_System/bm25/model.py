import pandas as pd
import numpy as np
import nltk
import re
import ast
import gensim
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.tokenize import word_tokenize
from gensim import corpora
from rank_bm25 import BM25Okapi

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class LDA_Model:
    def __init__(self, data_path):
        self.data_path = data_path
        self.job_data = self.get_data()
        self.model = self.create_model()

    def get_data(self):
        job_data = pd.read_csv(self.data_path)

        job_data['Long Description'] = ""
        des_cols = ['Job Description', 'Job Title', 'Role', 'skills', 'Responsibilities']

        for col in des_cols:
            job_data['Long Description'] = job_data['Long Description'] + " " + job_data[col]
        job_data['Long Description'] = job_data['Long Description'].str.strip()

        for col in job_data.columns:
            if job_data[col].dtype == 'object':
                job_data[col] = job_data[col].str.lower().str.strip()

        en_stopwords = stopwords.words('english')
        en_stopwords.remove("not")
        lm = WordNetLemmatizer()
        def preprocess_text(text):
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'\d+', '', text)  # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            tokens = word_tokenize(text)  # Tokenize words
            tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
            tokens = [lm.lemmatize(word) for word in tokens]  # Lemmatizing
            return tokens

        job_data['processed_text'] = job_data['Long Description'].apply(preprocess_text)
        return job_data

    def create_model(self):
        dictionary = corpora.Dictionary(job_data["processed_text"])

        corpus = [dictionary.doc2bow(text) for text in job_data["processed_text"]]

        dictionary.save("lda_dictionary.dict")
        corpora.MmCorpus.serialize("lda_corpus.mm", corpus)

        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=10,
            passes=15,
            random_state=42
        )
        def get_topic_distribution(text):
            bow_vector = dictionary.doc2bow(preprocess_text(text))
            return lda_model.get_document_topics(bow_vector, minimum_probability=0.0)

        self.job_data["topic_distribution"] = self.job_data["Long Description"].apply(lambda x: get_topic_distribution(x))
        return lda_model

    def save_model(self):
        joblib.dump(self.model, 'lda_model.joblib')


class BM25_Model:
    def __init__(self, data_path):
        self.data_path = data_path
        self.job_data = self.get_data()
        self.model = self.create_model()

    def get_data(self):
        job_data = pd.read_csv(self.data_path)

        job_data['Long Description'] = ""
        des_cols = ['Job Description', 'Job Title', 'Role', 'skills', 'Responsibilities']

        for col in des_cols:
            job_data['Long Description'] = job_data['Long Description'] + " " + job_data[col]
        job_data['Long Description'] = job_data['Long Description'].str.strip()

        for col in job_data.columns:
            if job_data[col].dtype == 'object':
                job_data[col] = job_data[col].str.lower().str.strip()

        en_stopwords = stopwords.words('english')
        en_stopwords.remove("not")
        lm = WordNetLemmatizer()
        def preprocess_text(text):
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'\d+', '', text)  # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            tokens = word_tokenize(text)  # Tokenize words
            tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
            tokens = [lm.lemmatize(word) for word in tokens]  # Lemmatizing
            return tokens

        job_data['processed_text'] = job_data['Long Description'].apply(preprocess_text)
        job_data.to_csv('job_data_sahy.csv')
        return job_data

    def create_model(self):
        bm25 = BM25Okapi(self.job_data["processed_text"].tolist())
        return bm25

    def save_model(self):
        joblib.dump(self.model, 'bm25_model.joblib')

# lda_model = LDA_Model('job_data_sahy.csv')
# lda_model.save_model()
bm25 = BM25_Model('job_data_sahy.csv')
bm25.save_model()
print("Models Saved!")