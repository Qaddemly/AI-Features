import json
import string
import requests
import numpy as np
import faiss
import pdfplumber
import os
import time
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# تحميل نموذج التضمين مع int8 quantization
MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")


class JobRecommendationSystem:
    def __init__(self, jobs_json):
        """تهيئة النظام وإعداد الوظائف وتضميناتها"""
        self.jobs_json = jobs_json
        self.jobs_texts, self.job_ids = self.prepare_jobs_text()
        self.job_embeddings = MODEL.encode(
            self.jobs_texts, convert_to_numpy=True
        ).astype(np.int8)

        # بناء FAISS IndexFlatIP للبحث الأسرع
        self.dim = self.job_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.job_embeddings.astype(np.float32))  # FAISS يحتاج float32

    def clean_text(self, text):
        """تنظيف النصوص بحذف الفراغات الزائدة وتحويل الأحرف إلى صغيرة وإزالة علامات الترقيم"""
        return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

    def extract_resume_text(self, resume_source):
        """استخراج النص من ملف PDF باستخدام pdfplumber"""
        text = ""
        if os.path.exists(resume_source):
            with pdfplumber.open(resume_source) as pdf:
                text = " ".join([page.extract_text() or "" for page in pdf.pages])
        else:
            response = requests.get(resume_source)
            if response.status_code == 200:
                temp_pdf = "downloaded_resume.pdf"
                with open(temp_pdf, "wb") as f:
                    f.write(response.content)
                with pdfplumber.open(temp_pdf) as pdf:
                    text = " ".join([page.extract_text() or "" for page in pdf.pages])
        return self.clean_text(text) if text else ""

    def prepare_jobs_text(self):
        """تحضير نصوص الوظائف لاستخراج التضمينات"""
        jobs_texts = []
        job_ids = []
        for job in self.jobs_json:
            job_text = (
                f"{job['title']} {job['description']} {job['location']['city']} {job['location']['country']} "
                f"{job['location_type']} {' '.join(job['skills'])} {job['employee_type']} {' '.join(job['keywords'])} "
                f"experience: {job['experience']}"
            )
            jobs_texts.append(self.clean_text(job_text))
            job_ids.append(job["id"])
        return jobs_texts, job_ids

    def filter_top_jobs(self, resume_text, top_n=100):
        """تقليل عدد الوظائف باستخدام TF-IDF لاختيار أفضل 100 وظيفة أولية"""
        vectorizer = TfidfVectorizer()
        job_vectors = vectorizer.fit_transform(self.jobs_texts)
        resume_vector = vectorizer.transform([resume_text])
        similarity_scores = (job_vectors @ resume_vector.T).toarray().flatten()

        # الحصول على أفضل الوظائف بناءً على أعلى درجات تشابه
        top_indices = np.argsort(similarity_scores)[-top_n:]
        return [self.jobs_texts[i] for i in top_indices], [
            self.job_ids[i] for i in top_indices
        ]

    def recommend_jobs(self, resume_json, top_n=20):
        """تنفيذ عملية التوصية باستخدام FAISS بعد تصفية الوظائف"""
        resume_text = self.extract_resume_text(resume_json["resume"]["url"])
        if not resume_text:
            return {"error": "Failed to extract resume text"}

        # تصفية أفضل 100 وظيفة باستخدام TF-IDF قبل البحث في FAISS
        filtered_jobs_texts, filtered_job_ids = self.filter_top_jobs(resume_text)

        # توليد تضمين السيرة الذاتية باستخدام int8 quantization
        resume_embedding = MODEL.encode([resume_text], convert_to_numpy=True).astype(
            np.int8
        )

        # بناء FAISS Index جديد للوظائف المصفاة فقط
        filtered_embeddings = MODEL.encode(
            filtered_jobs_texts, convert_to_numpy=True
        ).astype(np.int8)
        index = faiss.IndexFlatIP(self.dim)
        index.add(filtered_embeddings.astype(np.float32))  # FAISS يتطلب float32

        # البحث عن أقرب الوظائف
        distances, indices = index.search(resume_embedding.astype(np.float32), top_n)
        recommended_job_ids = [filtered_job_ids[i] for i in indices[0]]

        return {"recommended_jobs": recommended_job_ids}
