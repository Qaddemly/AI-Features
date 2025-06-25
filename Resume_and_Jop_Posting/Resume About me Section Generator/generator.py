import os
from typing import Optional, Literal, Tuple
from dotenv import load_dotenv
from groq import Groq


class ResumeSectionGenerator:
    """Base class for generating and enhancing resume sections using Groq API."""

    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        self.client = Groq(api_key=self.api_key)

        # Define max tokens for each section
        self.max_tokens = {
            "about_me": 200,
            "skills": 100
        }

    def _validate_inputs(
            self,
            user_json: dict,
            section_type: Literal["about_me", "skills"],
            mode: Literal["generate", "enhance"],
            existing_section: Optional[str]
    ) -> None:
        """Validate input parameters."""
        if not user_json.get("user"):
            raise ValueError("Input JSON must contain a 'user' key")
        if mode == "enhance" and not existing_section:
            raise ValueError("Existing section text is required for enhancement mode")
        if section_type not in self.max_tokens:
            raise ValueError(f"Invalid section_type '{section_type}'. Must be 'about_me' or 'skills'")
        if mode not in ["generate", "enhance"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'generate' or 'enhance'")

    def _get_prompts(self, section_type: str, mode: str) -> Tuple[str, str]:
        """Return system and user prompts for the section and mode."""
        prompts = {
            "about_me": {
                "generate": {
                    "system": "You are an expert resume writer with a knack for crafting natural, engaging, and ATS-optimized summaries that resonate with recruiters. Your role is to write concise, professional narratives that highlight a candidate’s unique strengths in a warm, human tone.",
                    "user": """
                        Write a professional 'About Me' section for a resume as a single paragraph, strictly adhering to these requirements:
                        - Limit to 7 lines or fewer (approximately 100 words).
                        - Highlight the user’s key skills, certifications, areas of expertise, and relevant educational background.
                        - Exclude job titles, years of experience, company names, or future career goals.
                        - Use a confident, natural tone with strong action verbs (e.g., 'specialize,' 'excel,' 'adept').
                        - Incorporate industry-standard keywords and, if provided, job-specific keywords for ATS compatibility.
                        - Avoid vague terms (e.g., 'good,' 'experienced,' 'passionate').
                        - Return only the paragraph, with no additional text, headings, or explanations.

                        **User Information:**
                        {user}

                        **Target Job Keywords (optional):**
                        {job_keywords}
                    """
                },
                "enhance": {
                    "system": "You are an expert resume editor skilled at refining ATS-optimized summaries to sound natural and engaging. Your role is to enhance existing content with a warm, professional tone, improving clarity and impact while preserving the candidate’s unique strengths.",
                    "user": """
                        Enhance the provided 'About Me' section for a resume as a single paragraph, strictly adhering to these requirements:
                        - Maintain 7 lines or fewer (approximately 100 words).
                        - Retain the user’s key skills, certifications, and areas of expertise, improving clarity and impact.
                        - Exclude job titles, years of experience, company names, or future career goals.
                        - Use a confident, natural tone with strong action verbs (e.g., 'specialize,' 'excel,' 'adept').
                        - Incorporate industry-standard and job-specific keywords (if provided) for ATS compatibility.
                        - Replace vague terms (e.g., 'good,' 'experienced,' 'passionate') with precise language.
                        - Address weaknesses in the original (e.g., lack of keywords, weak phrasing).
                        - Return only the paragraph, with no additional text, headings, or explanations.

                        **User Information:**
                        {user}

                        **Target Job Keywords (optional):**
                        {job_keywords}

                        **Existing About Me Section to Enhance:**
                        {existing_section}
                    """
                }
            },
            "skills": {
                "generate": {
                    "system": "You are an expert resume writer specializing in crafting ATS-optimized skill lists that feel professional and human-crafted. Your role is to create concise, balanced skill sets that align with job requirements and showcase a candidate’s strengths naturally.",
                    "user": """
                        Write a 'Skills' section for a resume as a single comma-separated list, strictly adhering to these requirements:
                        - Include 8–12 skills, with approximately 60% hard skills (technical, job-specific) and 40% soft skills (interpersonal, e.g., communication, adaptability).
                        - Prioritize skills from the user’s profile, supplementing with relevant skills inferred from their education, certifications, or interests.
                        - Incorporate job-specific keywords (if provided) to align with the target job.
                        - Use clear, standard terms without special characters for ATS compatibility.
                        - Exclude skills unrelated to the user’s profile or job requirements.
                        - Return only the comma-separated list (e.g., Python, Django, Teamwork), with no additional text, headings, or explanations.

                        **User Information:**
                        {user}

                        **Target Job Keywords (optional):**
                        {job_keywords}
                    """
                },
                "enhance": {
                    "system": "You are an expert resume editor skilled at refining ATS-optimized skill lists to be concise and professional. Your role is to enhance existing skill sets with a natural, human-crafted feel, improving relevance and alignment with job requirements.",
                    "user": """
                        Enhance the provided 'Skills' section for a resume as a single comma-separated list, strictly adhering to these requirements:
                        - Include 8–12 skills, maintaining approximately 60% hard skills (technical, job-specific) and 40% soft skills (interpersonal, e.g., communication, adaptability).
                        - Retain relevant skills from the original section, removing vague or irrelevant ones (e.g., 'expert,' 'general coding').
                        - Incorporate skills from the user’s profile and job-specific keywords (if provided) to align with the target job.
                        - Use clear, standard terms without special characters for ATS compatibility.
                        - Return only the comma-separated list (e.g., Python, Django, Teamwork), with no additional text, headings, or explanations.

                        **User Information:**
                        {user}

                        **Target Job Keywords (optional):**
                        {job_keywords}

                        **Existing Skills Section to Enhance:**
                        {existing_section}
                    """
                }
            }
        }
        section_prompts = prompts.get(section_type, {}).get(mode, {})
        if not section_prompts:
            raise ValueError(f"Invalid section_type '{section_type}' or mode '{mode}'")
        return section_prompts["system"], section_prompts["user"]

    def generate_section(
            self,
            user_json: dict,
            section_type: Literal["about_me", "skills"],
            mode: Literal["generate", "enhance"] = "generate",
            existing_section: Optional[str] = None,
            job_keywords: Optional[list] = None
    ) -> str:
        """Generate or enhance a resume section (to be implemented by subclasses)."""
        raise NotImplementedError


class AboutMeGenerator(ResumeSectionGenerator):
    """Class for generating and enhancing 'About Me' resume sections."""

    def generate_section(
            self,
            user_json: dict,
            section_type: Literal["about_me"] = "about_me",
            mode: Literal["generate", "enhance"] = "generate",
            existing_section: Optional[str] = None,
            job_keywords: Optional[list] = None
    ) -> str:
        """Generate or enhance an 'About Me' section."""
        self._validate_inputs(user_json, section_type, mode, existing_section)
        user = user_json["user"]
        job_keywords_str = ", ".join(job_keywords) if job_keywords else "None"

        system_prompt, user_prompt_template = self._get_prompts(section_type, mode)
        user_prompt = user_prompt_template.format(
            user=user,
            job_keywords=job_keywords_str,
            existing_section=existing_section or "None"
        )

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Slightly higher for natural tone
                max_tokens=self.max_tokens[section_type],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"Groq API error: {str(e)}")


class SkillsGenerator(ResumeSectionGenerator):
    """Class for generating and enhancing 'Skills' resume sections."""

    def generate_section(
            self,
            user_json: dict,
            section_type: Literal["skills"] = "skills",
            mode: Literal["generate", "enhance"] = "generate",
            existing_section: Optional[str] = None,
            job_keywords: Optional[list] = None
    ) -> str:
        """Generate or enhance a 'Skills' section."""
        self._validate_inputs(user_json, section_type, mode, existing_section)
        user = user_json["user"]
        job_keywords_str = ", ".join(job_keywords) if job_keywords else "None"

        system_prompt, user_prompt_template = self._get_prompts(section_type, mode)
        user_prompt = user_prompt_template.format(
            user=user,
            job_keywords=job_keywords_str,
            existing_section=existing_section or "None"
        )

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Slightly higher for natural tone
                max_tokens=self.max_tokens[section_type],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"Groq API error: {str(e)}")


# Example usage
if __name__ == "__main__":
    import json

    # Sample user data
    test_file_path = "data/test.json"
    with open(test_file_path, "r", encoding="utf-8") as file:
        user_data = json.load(file)

    # Sample job keywords (optional, from DataPreprocessor)
    job_keywords = ["Python", "Django", "AWS", "Communication"]

    # Sample existing sections for enhancement
    existing_about_me = "Experienced developer good at coding and team player."
    existing_skills = "coding, teamwork, expert in tech"

    # Initialize generators
    about_me_generator = AboutMeGenerator()
    skills_generator = SkillsGenerator()

    # Generate About Me
    about_me = about_me_generator.generate_section(
        user_json=user_data,
        section_type="about_me",
        mode="generate",
        job_keywords=job_keywords
    )
    print("Generated About Me:", about_me)

    # Enhance About Me
    enhanced_about_me = about_me_generator.generate_section(
        user_json=user_data,
        section_type="about_me",
        mode="enhance",
        existing_section=existing_about_me,
        job_keywords=job_keywords
    )
    print("Enhanced About Me:", enhanced_about_me)

    # Generate Skills
    skills = skills_generator.generate_section(
        user_json=user_data,
        section_type="skills",
        mode="generate",
        job_keywords=job_keywords
    )
    print("Generated Skills:", skills)

    # Enhance Skills
    enhanced_skills = skills_generator.generate_section(
        user_json=user_data,
        section_type="skills",
        mode="enhance",
        existing_section=existing_skills,
        job_keywords=job_keywords
    )
    print("Enhanced Skills:", enhanced_skills)