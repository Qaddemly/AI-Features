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
      You are an expert cover letter writer. Based on the following structured user and job information in JSON format, generate three professional and engaging paragraphs of a cover letter.

      Using the provided resume and job post, craft a concise freelance proposal. The proposal should be structured in 4-5 paragraphs and utilize bullet points for clarity. Write it as code so that I can copy it. Don't use bold words. The proposal should be maximum 1500 characters. Don't write number, email, or title at the end. Write subject separately. Start with "Dear Hiring Manager". Ensure the proposal includes:

        **General Tips:**
        
        - Be Concise: Keep language clear and to the point. Use bullet points for readability.
        - Professional Tone: Maintain a courteous and professional tone.
        - Customization: Personalize the proposal for the specific client and project. Reference specific details from the job post.
        - Visual Appeal: Use a clean format with no spelling or grammatical errors.
        - Value Proposition: Highlight the value you bring and provide evidence of past successes.
        - Clear Call to Action: Clearly state the next steps for the client.
        1. **Personalized Introduction:**
            - Address the client by name.
            - Express enthusiasm for the project.
            - Briefly mention how you found the job post.
        2. **Understanding Client Needs:**
            - Reference specific needs and goals from the job post.
            - Demonstrate that the proposal is tailored to the client's requirements.
        3. **Proposed Approach:**
            - Outline your approach to completing the project.
            - Describe the key steps you will take.
        4. **Showcase Expertise and Value:**
            - Highlight relevant experience from the resume.
            - Mention specific achievements or past projects.
            - Include brief testimonials or recommendations, if available.
        5. **Call to Action:**
            - Encourage the client to take the next steps (e.g., scheduling a meeting, signing a contract).
      **Input:**
      **User information:**
      {user}

      **Job information:**
      {job}
      
      ## Style guideline:

        Avoid overused buzzwords (like ‘leverage,’ ‘harness,’ ‘elevate,’ ‘ignite,’ ‘empower,’ ‘cutting-edge,’ ‘unleash,’ ‘revolutionize,’ ‘innovate,’ ‘dynamic,’ ‘transformative power’), filler phrases (such as ‘in conclusion,’ ‘it’s important to note,’ ‘as previously mentioned,’ ‘ultimately,’ ‘to summarize,’ ‘what’s more,’ ‘now,’ ‘until recently’), clichés (like ‘game changer,’ ‘push the boundaries,’ ‘the possibilities are endless,’ ‘only time will tell,’ ‘mind-boggling figure,’ ‘breaking barriers,’ ‘unlock the potential,’ ‘remarkable breakthrough’), and flowery language (including ‘tapestry,’ ‘whispering,’ ‘labyrinth,’ ‘oasis,’ ‘metamorphosis,’ ‘enigma,’ ‘gossamer,’ ‘treasure trove,’ ‘labyrinthine’). Also, limit the use of redundant connectives and fillers like ‘moreover,’ ‘furthermore,’ ‘additionally,’ ‘however,’ ‘therefore,’ ‘consequently,’ ‘importantly,’ ‘notably,’ ‘as well as,’ ‘despite,’ ‘essentially,’ and avoid starting sentences with phrases like ‘Firstly,’ ‘Moreover,’ ‘In today’s digital era,’ ‘In the world of’. Focus on delivering the information in a concise and natural tone without unnecessary embellishments, jargon, or redundant phrases.
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