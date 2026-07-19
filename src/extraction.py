import json
from src.llm import ask_llm

EXTRACTION_PROMPT_TEMPLATE = """You are a resume data extraction assistant.

Extract the following information from the resume text below, and
return it as a single valid JSON object with EXACTLY this structure:

{{
  "skills": ["list", "of", "technical", "and", "soft", "skills"],
  "education": [
    {{"degree": "degree name", "institution": "school name", "year": "graduation year"}}
  ],
  "years_of_experience": <number, estimated total years of professional experience>
}}

Rules:
- Return ONLY the JSON object. No explanations, no markdown code fences, no extra text.
- If a field cannot be determined from the text, use an empty list [] or 0 for numbers.
- Do not invent information not present in the resume text.

Resume text:
{resume_text}
"""


def extract_structured_data(resume_text: str) -> dict:
    """
    Extract skills, education, and years of experience from resume text
    as structured data.

    Args:
        resume_text: The full or partial resume text to analyze.

    Returns:
        A dictionary with keys: 'skills', 'education', 'years_of_experience'.
        Returns a default empty structure if parsing fails.
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(resume_text=resume_text)
    raw_response = ask_llm(prompt, temperature=0.0)

    cleaned_response = raw_response.strip()
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response.strip("`")
        cleaned_response = cleaned_response.replace("json\n", "", 1)

    try:
        data = json.loads(cleaned_response)
    except json.JSONDecodeError:
        data = {
            "skills": [],
            "education": [],
            "years_of_experience": 0,
        }

    return data
