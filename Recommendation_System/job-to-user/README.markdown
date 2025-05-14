# Job Recommendation System for Users

## Overview

The Job Recommendation System for Users is a content-based recommendation engine designed to assist job seekers in discovering relevant job opportunities. By analyzing a user’s skills, professional background, and preferences against a collection of job listings, the system generates a ranked list of job recommendations tailored to the user’s profile. The system employs natural language processing (NLP) and similarity metrics to evaluate job titles, descriptions, skills, employment types, and location preferences, facilitating an exploratory approach to job discovery.

## Components

The system comprises four Python modules, each contributing to the recommendation pipeline:

- **`data_preprocessing.py`**: Preprocesses user and job data by cleaning text, parsing skills, extracting recent experiences, and generating skill embeddings using a SentenceTransformer model. Outputs structured Pandas DataFrames and embeddings.
- **`recommendation_to_user.py`**: Implements the `JobRecommender` class to compute similarity scores based on title matching, description similarity (via TF-IDF), skill overlap, and employment/location type alignment. Returns a ranked list of job recommendations.
- **`app.py`**: Provides a FastAPI-based REST API to accept user and job data via HTTP requests and return recommendations, enabling integration with external applications.
- **`main.py`**: Serves as the command-line entry point, orchestrating preprocessing and recommendation for quick testing with a JSON input file.

## Setup

### Prerequisites

- **Python**: Version 3.8 or higher.
- **Dependencies**: Install required libraries using pip:
  ```bash
  pip install numpy pandas nltk sentence-transformers scikit-learn fastapi uvicorn pydantic
  ```
- **NLTK Data**: Download necessary NLTK resources:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt')
  nltk.download('wordnet')
  ```

### Input Data

The system expects input in JSON format, containing a user profile and a list of job listings. Save the input as `test.json` in the project directory. Example structure:

```json
{
  "user": {
    "id": 1,
    "about_me": "Passionate software engineer with expertise in AI",
    "subtitle": "Software Engineer",
    "skills": [{"name": "Python"}, {"name": "Machine Learning"}],
    "experiences": [
      {
        "job_title": "Developer",
        "employment_type": "Full-time",
        "location_type": "Remote",
        "start_date": "2023-06-01"
      }
    ],
    "educations": []
  },
  "jobs": [
    {
      "id": 101,
      "title": "Machine Learning Engineer",
      "description": "Develop advanced AI models",
      "skills": ["Python", "TensorFlow"],
      "employee_type": "Full-time",
      "location_type": "Remote"
    },
    {
      "id": 102,
      "title": "Content Writer",
      "description": "Create engaging content",
      "skills": ["Writing", "SEO"],
      "employee_type": "Part-time",
      "location_type": "On-site"
    }
  ]
}
```

## Usage

### Command-Line Interface

To generate recommendations via the command line:
1. Ensure all Python files and `test.json` are in the same directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The system will output a list of recommended jobs with their IDs and similarity scores.

### API Interface

To use the API:
1. Start the FastAPI server:
   ```bash
   python app.py
   ```
2. The server will run on `http://localhost:8001`.
3. Send a POST request to the `/recommend` endpoint with the JSON input:
   ```bash
   curl -X POST "http://localhost:8001/recommend" -H "Content-Type: application/json" -d @test.json
   ```
4. The response will include a list of recommendations and a success message.

## Technical Details

The recommendation process involves the following steps:

1. **Preprocessing** (`data_preprocessing.py`):
   - Cleans text by removing special characters, lowercasing, and lemmatizing.
   - Parses user skills and job skills, extracting the most recent user experience.
   - Generates skill embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model.
   - Produces Pandas DataFrames and embeddings for further processing.

2. **Recommendation** (`recommendation_to_user.py`):
   - Computes similarities:
     - **Title**: Binary match between user subtitle and job title.
     - **Description**: TF-IDF-based cosine similarity between user’s “about me” and job description.
     - **Skills**: Counts common skills using embedding-based cosine similarity (threshold: 0.5).
     - **Employment/Location**: Binary matching for employment and location types.
   - Combines similarities using weights (default: 0.2 each) to compute a final score.
   - Returns the top `top_n` jobs ranked by score.

3. **API and CLI** (`app.py`, `main.py`):
   - `app.py` exposes a `/recommend` endpoint to handle JSON input and return recommendations.
   - `main.py` provides a command-line interface for quick testing.

## Troubleshooting

- **NLTK Errors**: Ensure `stopwords`, `punkt`, and `wordnet` are downloaded.
- **JSON Format Issues**: Verify that `test.json` follows the expected structure with `user` and `jobs` fields.
- **API Errors**: Check that port `8001` is available and the JSON payload matches the `InputData` model in `app.py`.
- **Memory Constraints**: Skill embedding generation may require significant RAM; ensure adequate resources.

## Potential Enhancements

- Support recommendations for multiple users simultaneously.
- Incorporate additional criteria, such as salary or experience level.
- Cache skill embeddings to improve API performance.
- Develop a web-based interface for user interaction.

## License

This feature is licensed under the MIT License, permitting use, modification, and distribution with attribution.