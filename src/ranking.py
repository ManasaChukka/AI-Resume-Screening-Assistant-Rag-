import json
from langchain_community.vectorstores import FAISS
from src.retriever import retrieve_relevant_chunks
from src.llm import ask_llm

RANKING_PROMPT_TEMPLATE = """You are an expert technical recruiter evaluating a
candidate against a job description.

Job Description:
{job_description}

Candidate's relevant resume excerpts:
{candidate_excerpts}

Evaluate how well this candidate matches the job description. Return ONLY a
valid JSON object with EXACTLY this structure:

{{
  "match_score": <integer from 0 to 100>,
  "strengths": ["list of specific matching qualifications"],
  "gaps": ["list of specific missing or unclear qualifications"],
  "reasoning": "one or two sentence explanation of the score"
}}

Rules:
- Base the score ONLY on evidence in the excerpts above.
- Return ONLY the JSON object, no extra text or markdown fences.
"""

SUMMARY_PROMPT_TEMPLATE = """Summarize this candidate's background in 2-3
concise sentences, suitable for a busy HR reviewer skimming quickly.

Resume excerpts:
{resume_excerpts}

Return only the summary text, no headers or extra formatting.
"""


def _get_candidate_excerpts(vector_store: FAISS, source_filename: str, query: str, top_k: int = 5) -> str:
    """
    Retrieve and format the most relevant excerpts for one specific
    candidate (filtered by source filename) against a query.

    Args:
        vector_store: The FAISS vector store containing all candidates.
        source_filename: The resume filename to filter results to.
        query: The query to rank relevance against (e.g., the job description).
        top_k: Max number of excerpts to retrieve for this candidate.

    Returns:
        Formatted string of that candidate's relevant excerpts.
    """
    all_results = retrieve_relevant_chunks(vector_store, query, top_k=top_k * 3)

    candidate_results = [
        (doc, score) for doc, score in all_results
        if doc.metadata.get("source") == source_filename
    ][:top_k]

    if not candidate_results:
        return "No relevant excerpts found for this candidate."

    return "\n\n".join(doc.page_content for doc, _ in candidate_results)


def rank_candidate(vector_store: FAISS, source_filename: str, job_description: str) -> dict:
    """
    Score a single candidate against a job description.

    Args:
        vector_store: The FAISS vector store containing all candidates.
        source_filename: The resume filename identifying this candidate.
        job_description: The job description text to match against.

    Returns:
        A dict with keys: match_score, strengths, gaps, reasoning.
    """
    excerpts = _get_candidate_excerpts(vector_store, source_filename, job_description)

    prompt = RANKING_PROMPT_TEMPLATE.format(
        job_description=job_description,
        candidate_excerpts=excerpts,
    )
    raw_response = ask_llm(prompt, temperature=0.0)

    try:
        cleaned = raw_response.strip().strip("`").replace("json\n", "", 1)
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "match_score": 0,
            "strengths": [],
            "gaps": ["Could not evaluate this candidate."],
            "reasoning": "Evaluation failed due to a parsing error.",
        }

    result["source"] = source_filename
    return result


def rank_all_candidates(vector_store: FAISS, source_filenames: list[str], job_description: str) -> list[dict]:
    """
    Score and rank multiple candidates against a job description.

    Args:
        vector_store: The FAISS vector store containing all candidates.
        source_filenames: List of unique resume filenames to evaluate.
        job_description: The job description text to match against.

    Returns:
        List of candidate result dicts, sorted by match_score descending.
    """
    results = [
        rank_candidate(vector_store, filename, job_description)
        for filename in source_filenames
    ]
    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results


def generate_candidate_summary(vector_store: FAISS, source_filename: str) -> str:
    """
    Generate a short summary of a candidate's background.

    Args:
        vector_store: The FAISS vector store containing all candidates.
        source_filename: The resume filename identifying this candidate.

    Returns:
        A short summary string.
    """
    excerpts = _get_candidate_excerpts(
        vector_store, source_filename, query="skills experience education", top_k=6
    )
    prompt = SUMMARY_PROMPT_TEMPLATE.format(resume_excerpts=excerpts)
    return ask_llm(prompt, temperature=0.2)
