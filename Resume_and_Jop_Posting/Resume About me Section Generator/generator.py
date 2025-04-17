import json
import re
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file!")

client = Groq(api_key=GROQ_API_KEY)


def generate_about_me_section(user_json: dict) -> str:
    """
    Generate an "About Me" section for a Resume based on user information.
    
    Args:
        user_json (dict): A dictionary containing user information.
        
    Returns:
        str: The generated "About Me" section.
    """
    user = user_json["user"]
    #print(user)

    prompt = f"""
        You are a professional resume writer. Based on the user’s background, write a concise, professional “About Me” section for a resume that:

        - Does not exceed 7 lines.

        - Focuses solely on the user's skills, knowledge, and areas of expertise (not job experience, job titles, or future goals).

        - Uses a confident and formal tone.

        - Follows best practices from professional resume-writing guidelines (e.g., highlights strengths, uses descriptive and action words, avoids generic or vague statements).

        - Avoids mentioning company names, years of experience, or any job-specific references.


      **User information:**
      {user}
    """

    # Request to Groq API to generate the "About Me" section
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=600,
    )

    content = response.choices[0].message.content.strip()
    
    # Remove the first line if it's an intro sentence
    #lines = content.splitlines()
    #if len(lines) > 1 and "about me" in lines[0].lower():
     #   content = "\n".join(lines[1:]).strip()

    return content


# example usage
if __name__ == "__main__":
    # Path to the JSON file
    test_file_path = "data/test.json"

    # Load user data from JSON
    with open(test_file_path, "r", encoding="utf-8") as file:
        user_data = json.load(file)

    # Generate About Me section
    about_me = generate_about_me_section(user_data)

    # Print the result
    print(about_me)



"""
output:

Here's a concise and professional "About Me" section based on the user's background:

"Dedicated full-stack developer with expertise in designing and developing scalable web applications. Skilled in JavaScript, React, and Python, with a strong proficiency in Django and AWS. Proficient in containerization with Docker and orchestration with Kubernetes. Bilingual in English and Spanish, with a strong foundation in Computer Science from Stanford University and AWS Certified Solutions Architect certification."



"""

"""
"Results-driven software engineer with expertise in full-stack development, scalable architecture, and cloud computing. Skilled in JavaScript, React, Python, Django, and AWS, with a proven track record in building high-performance applications. Proficient in Docker, Kubernetes, and containerization. Fluently conversant in English and Spanish."

"""