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


def generate_cover_letter_parts(user_job_json: dict) -> dict:
    user = user_job_json["user"]
    job = user_job_json["job"]

    prompt = f"""
      You are an expert cover letter writer. Based on the following structured user and job information in JSON format, generate three professional and engaging paragraphs of a cover letter: Opening, Body, and Closing.

      1. **Opening Paragraph**:
        In the opening paragraph tell how you learned about the position. You may, for example, know of a job through: Qaddemly website

      2. **Body Paragraph**:
          This paragraph gives a summary of your background and critical skills (hard skills) that make you qualified for the position.
          This paragraph can be used to demonstrate your persuasive skills (soft skills).

      3. **Closing Paragraph**:
        At the end of the letter talk about your availability for the job, where you can be contacted, and when you are going to contact the hiring person for an appointment to discuss your application. If you have no contact name you may simply want to indicate your anticipation for a response in this part of the letter. Thank the person to whom you are writing for his/her time and consideration of your application.

      **User information:**
      {user}

      **Job information:**
      {job}
    """

    # Request to Groq API to generate the cover letter
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=600,
    )

    content = response.choices[0].message.content.strip()

    # Remove the "Opening Paragraph", "Body Paragraph", and "Closing Paragraph" labels
    content = re.sub(r"\*\*.*?Paragraph:\*\*", "", content).strip()

    # Split the content into paragraphs and clean them
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # Ensure we have exactly 3 paragraphs (opening, body, closing)
    if len(paragraphs) < 3:
        raise ValueError("The generated cover letter doesn't contain all required sections.")

    # Clean up each paragraph (strip whitespace)
    opening = paragraphs[1].strip()
    body = paragraphs[2].strip()
    closing = paragraphs[3].strip()

    # Return the result in a structure that the API expects
    return {
        "merged_cover_letter": f"{opening}\n\n{body}\n\n{closing}"
    }


"""
Result for sample Input:

  "I am writing to express my interest in the Senior Full-Stack Developer position at TechInnovate, which I came across on the Qaddemly website. As a seasoned software engineer with a passion for building scalable web applications, I am excited about the opportunity to join your engineering team and contribute to the company's mission of providing innovative productivity tools for remote teams.
  
  With over 5 years of experience in full-stack development, I possess a unique combination of technical skills and soft skills that make me an ideal fit for this role. In my current position as a Senior Software Engineer at Airbnb, I have honed my expertise in designing and implementing features across the entire stack, from database to UI, using modern JavaScript frameworks such as React, Node.js, and TypeScript. I have also developed strong skills in database design and API development, which I believe will serve me well in this position. My experience in mentoring junior developers has also given me a unique perspective on how to effectively communicate technical ideas and collaborate with cross-functional teams. I am confident that my skills, experience, and passion for clean code and user experience make me a strong candidate for this position.
  
  Thank you for considering my application for the Senior Full-Stack Developer position at TechInnovate. I am excited about the opportunity to discuss my qualifications further and learn more about your team's work. I can be reached at emma.wilson@example.com or +1 415 555 0199. I would appreciate the opportunity to schedule an appointment to meet and discuss my application. I look forward to hearing from you soon and learning about the next steps in the process. Thank you again for your time and consideration."

"""