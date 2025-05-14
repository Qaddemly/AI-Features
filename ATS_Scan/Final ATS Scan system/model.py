from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import re
from collections import defaultdict
import fitz


# clean text and convert to lowercase and erase special characters and extra spaces and new lines
def clean_text(text):
    """Clean text by removing special characters and extra spaces."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text.strip().lower()  # Trim and lowercase


# extract text from pdf file
def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text


# extract text from pdf file and extract resume sections
headers_list = [
    # Personal Information
    "Profile",
    "Personal Details",
    "Personal Information",
    "Contact Information",
    "About Me",
    "Summary",
    "Objective",
    "Career Objective",
    # Education
    "Education",
    "Educational Background",
    "Academic Background",
    "Academic Qualifications",
    "Qualifications",
    # Experience
    "Experience",
    "Work Experience",
    "Professional Experience",
    "Employment History",
    "Career History",
    "Relevant Experience",
    "Internships",
    # Skills
    "Skills",
    "Technical Skills",
    "Key Skills",
    "Core Competencies",
    "Soft Skills",
    "Hard Skills",
    "Computer Skills",
    # Certifications
    "Certificates",
    "Certifications",
    "Licenses",
    "Courses and Certifications",
    # Projects
    "Projects",
    "Key Projects",
    "Academic Projects",
    "Personal Projects",
    "Major Projects",
    # Languages
    "Languages",
    "Language Proficiency",
    "Spoken Languages",
    # Interests / Hobbies
    "Interests",
    "Hobbies",
    "Activities",
    # Awards & Honors
    "Awards",
    "Honors",
    "Achievements",
    "Recognitions",
    "Accomplishments",
    # Volunteer / Extra Activities
    "Volunteer Work",
    "Volunteering",
    "Community Service",
    "Extracurricular Activities",
    "Leadership Experience",
    "Social Activities",
    # Organizations
    "Organisations",
    "Organizations",
    "Professional Affiliations",
    "Memberships",
    # Publications
    "Publications",
    "Research",
    "Research Papers",
    "Articles",
    # References
    "References",
    "Referees",
    "Recommendation",
    # Declaration
    "Declaration",
    "Statement",
    # Courses
    "Courses",
    "Relevant Courses",
    "Training",
    "Workshops",
    # Additional Sections
    "Achievements",
    "Professional Summary",
    "Career Summary",
    "Technical Summary",
    "Professional Highlights",
    "Key Responsibilities",
    "Strengths",
    "Portfolio",
    "GitHub",
    "LinkedIn",
    "Contact",
    "Cover Letter",
    "Workshops",
]


def extract_resume_sections_from_pdf(text, headers=headers_list):
    # Step 1: Extract text from PDF
    # text = extract_text(pdf_path)
    # print(text)
    # print("---"*100)

    # Step 2: Clean and preprocess the text of resume
    text = re.sub(r"\n+", "\n", text)  # Collapse multiple newlines
    text = re.sub(r" +", " ", text)  # Collapse multiple spaces
    text = text.strip().lower()  # Trim and lowercase

    lines = text.split("\n")

    # Step 3: Prepare lowercase headers for matching
    headers_lower = [h.lower() for h in headers]
    pattern_to_header = {}
    for h in headers_lower:
        pattern = re.compile(rf"^\s*{re.escape(h)}\s*$", re.IGNORECASE)
        pattern_to_header[pattern] = h

    # print(pattern_to_header)

    # Step 4: Extract sections
    result = defaultdict(str)
    current_header = None

    for line in lines:
        line_clean = line.strip().lower()
        matched = False

        for pattern, header in pattern_to_header.items():
            if pattern.match(line_clean):
                current_header = header
                matched = True
                break

        if not matched and current_header:
            result[current_header] += line + "\n"

    return dict(result)


# extract skills from resume sections
def extract_skills_sections(result_dict):
    skills_headers = [
        "skills",
        "technical skills",
        "key skills",
        "core competencies",
        "soft skills",
        "hard skills",
        "computer skills",
    ]
    skills = ""
    for key, value in result_dict.items():
        if key in skills_headers:
            skills += value.strip() + "\n"
    return skills


def exact_match_score(resume_skills, job_skills):
    """Calculate exact match score."""
    resume_skills_set = set(resume_skills)
    job_skills_set = set(job_skills)
    matched = resume_skills_set.intersection(job_skills_set)
    score = len(matched) / len(job_skills_set) * 100  # Percentage match
    return matched, score


def calculate_cosine_similarity(text1, text2):
    """Compute cosine similarity between two text inputs using TF-IDF."""
    if not text1 or not text2:
        return 0.0
    corpus = [text1, text2]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


from sentence_transformers import SentenceTransformer, util

# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_embeddingSBERT_cosine_similarity(text1, text2):
    """Compute cosine similarity between two text inputs using SBERT."""

    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return round(cosine_scores.item() * 100, 2)
