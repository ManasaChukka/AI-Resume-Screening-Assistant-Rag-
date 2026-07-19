from src.llm import ask_llm

INTERVIEW_QUESTIONS_PROMPT_TEMPLATE = """Based on the candidate excerpts below,
generate exactly 5 tailored interview questions that probe deeper into their
specific claimed skills and experience.

Candidate excerpts:
{resume_excerpts}

Rules:
- Reference specific things mentioned in the excerpts where possible.
- Mix technical and behavioral questions.
- Return the questions as a numbered list, one per line, no other text.
"""


def generate_interview_questions(resume_excerpts: str) -> list[str]:
    """
    Generate tailored interview questions based on resume content.

    Args:
        resume_excerpts: The candidate's relevant resume text.

    Returns:
        A list of question strings.
    """
    prompt = INTERVIEW_QUESTIONS_PROMPT_TEMPLATE.format(resume_excerpts=resume_excerpts)
    raw_response = ask_llm(prompt, temperature=0.4)

    lines = raw_response.strip().split("\n")
    questions = []
    for line in lines:
        cleaned = line.strip()
        if cleaned and cleaned[0].isdigit():
            cleaned = cleaned.split(".", 1)[-1].strip()
            questions.append(cleaned)

    return questions
