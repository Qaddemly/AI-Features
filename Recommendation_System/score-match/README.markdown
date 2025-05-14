# Scoring Match Between a Single Job and a Single User

## Overview

The Scoring Match Between a Single Job and a Single User feature computes a similarity score to evaluate the compatibility between a specific user and a specific job. This content-based scoring system uses NLP and similarity metrics to assess alignment across job titles, descriptions, skills, employment types, and location preferences. The feature is designed for scenarios requiring precise compatibility assessment, such as targeted candidate evaluation or job fit analysis, and employs logarithmic scaling for balanced similarity contributions.

## Components

The system is implemented in a single Python module, with supporting preprocessing from other modules:

- **`matching_score.py`**: Defines the `MatchingScore` class, which computes a weighted similarity score based on embedding-based title and description similarities, skill overlap, and binary employment/location type matching. Uses a SentenceTransformer model and precomputed skill embeddings.
- **Supporting Modules** (assumed available from the job recommendation system):
  - **`data_preprocessing.py`**: Provides the SentenceTransformer model and skill embeddings required by `matching_score.py`.

## Setup

### Prerequisites

- **Python**: Version 3.8 or higher.
- **Dependencies**: Install required libraries using pip:
  ```bash
  pip install numpy sentence-transformers scikit-learn
  ```
- **Supporting Modules**: Ensure `data_preprocessing.py` is available to provide the SentenceTransformer model and skill embeddings.

### Input Data

The feature expects a user dictionary and a job dictionary, typically provided programmatically or via JSON. Example structure:

```json
{
  "user": {
    "id": 1,
    "about_me": "Passionate software engineer with expertise in AI",
    "subtitle": "Software Engineer",
    "skills": ["Python", "Machine Learning"],
    "employment_type": "Full-time",
    "location_type": "Remote"
  },
  "job": {
    "id": 101,
    "title": "Machine Learning Engineer",
    "description": "Develop advanced AI models",
    "skills": ["Python", "TensorFlow"],
    "employee_type": "Full-time",
    "location_type": "Remote"
  }
}
```

Skill embeddings must be precomputed using `data_preprocessing.py` or provided as a dictionary mapping skills to embeddings.

## Usage

### Programmatic Usage

To compute a similarity score:
1. Ensure `matching_score.py` and `data_preprocessing.py` are in the same directory.
2. Use the following example code:
   ```python
   from data_preprocessing import DataPreprocessor
   from matching_score import MatchingScore
   import pickle

   # Initialize preprocessor and load model
   preprocessor = DataPreprocessor()
   model = preprocessor.model

   # Load or generate skill embeddings (example assumes precomputed)
   with open('skill_embeddings.pkl', 'rb') as f:
       skill_embeddings = pickle.load(f)

   # Define user and job data
   user = {
       "id": 1,
       "about_me": "Passionate software engineer with expertise in AI",
       "subtitle": "Software Engineer",
       "skills": ["Python", "Machine Learning"],
       "employment_type": "Full-time",
       "location_type": "Remote"
   }
   job = {
       "id": 101,
       "title": "Machine Learning Engineer",
       "description": "Develop advanced AI models",
       "skills": ["Python", "TensorFlow"],
       "employee_type": "Full-time",
       "location_type": "Remote"
   }

   # Compute score
   matcher = MatchingScore(model, skill_embeddings)
   score = matcher.find_score(user, job)
   print(f"Similarity Score: {score}")
   ```

### Integration

To integrate with a larger system:
1. Use `data_preprocessing.py` to preprocess data and generate embeddings.
2. Pass the `model` and `skill_embeddings` to `MatchingScore` for scoring.
3. Incorporate the score into application logic (e.g., candidate evaluation).

## Technical Details

The scoring process involves the following steps:

1. **Initialization** (`matching_score.py`):
   - Accepts a SentenceTransformer model and precomputed skill embeddings.
   - Configures weights for title, description, skills, employment type, and location type (default: 0.2 each).
   - Sets a logarithmic scaling factor (`k=10`) for title and description similarities.

2. **Scoring** (`matching_score.py`):
   - **Title Similarity**: Computes cosine similarity between user subtitle and job title embeddings, scaled logarithmically.
   - **Description Similarity**: Computes cosine similarity between user’s “about me” and job description embeddings, scaled logarithmically.
   - **Skill Similarity**: Counts job skills with at least one user skill above a cosine similarity threshold (0.5), normalized by the number of job skills.
   - **Employment/Location Matching**: Assigns binary scores (1 for match, 0 otherwise).
   - Combines weighted scores into a final similarity score.

3. **Dependencies**:
   - Relies on `data_preprocessing.py` for the `all-MiniLM-L6-v2` SentenceTransformer model and skill embeddings.

## Troubleshooting

- **Missing Model**: Ensure `data_preprocessing.py` provides a valid SentenceTransformer model.
- **Embedding Issues**: Verify that `skill_embeddings` contains embeddings for all relevant skills.
- **Input Errors**: Confirm that user and job dictionaries include required fields (`subtitle`, `about_me`, `skills`, `employment_type`, `location_type` for user; `title`, `description`, `skills`, `employee_type`, `location_type` for job).
- **Memory Usage**: Embedding computations may require significant RAM; ensure sufficient resources.

## Potential Enhancements

- Add support for additional scoring criteria, such as experience duration or salary expectations.
- Optimize embedding computations for real-time scoring.
- Integrate with a database for scalable user and job data management.
- Provide a CLI or API wrapper for easier standalone usage.

## License

This feature is licensed under the MIT License, permitting use, modification, and distribution with attribution.