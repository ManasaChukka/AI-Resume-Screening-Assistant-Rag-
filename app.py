from pathlib import Path

import streamlit as st

from src.pdf_processor import extract_text_from_pdf
from src.text_splitter import chunk_resume_text
from src.vector_store import create_vector_store, save_vector_store
from src.rag_pipeline import run_rag_query
from src.extraction import extract_structured_data
from src.ranking import rank_all_candidates, generate_candidate_summary
from src.interview_questions import generate_interview_questions
from src.utils import setup_logger, compute_file_hash, load_cache, save_cache

logger = setup_logger()

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "resume_texts" not in st.session_state:
    # Maps filename -> full extracted resume text (used for
    # extraction/summary/interview-question features that want full text)
    st.session_state.resume_texts = {}

if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = []


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------
def process_uploaded_resumes(uploaded_files) -> None:
    """
    Process uploaded PDF files into a searchable FAISS vector store,
    using a content-hash cache to skip redundant embedding work on
    resumes that have already been processed.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
    """
    all_chunks = []
    all_metadatas = []
    cache = load_cache()

    progress_bar = st.progress(0, text="Starting...")
    save_dir = Path("data/resumes")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(
            (i + 1) / len(uploaded_files),
            text=f"Processing {uploaded_file.name}..."
        )

        file_bytes = uploaded_file.getbuffer()
        file_hash = compute_file_hash(bytes(file_bytes))

        file_path = save_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        try:
            if file_hash in cache:
                logger.info(f"Cache hit for {uploaded_file.name}, skipping re-extraction.")
                text = cache[file_hash]["text"]
                chunks = cache[file_hash]["chunks"]
            else:
                text = extract_text_from_pdf(str(file_path))
                chunks = chunk_resume_text(text)
                cache[file_hash] = {"text": text, "chunks": chunks}
                logger.info(f"Processed {uploaded_file.name}: {len(chunks)} chunks.")

            metadatas = [
                {"source": uploaded_file.name, "chunk_id": j}
                for j in range(len(chunks))
            ]
            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)

            st.session_state.resume_texts[uploaded_file.name] = text
            if uploaded_file.name not in st.session_state.processed_filenames:
                st.session_state.processed_filenames.append(uploaded_file.name)

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            st.error(f"Error processing {uploaded_file.name}: {e}")

    save_cache(cache)

    if all_chunks:
        vector_store = create_vector_store(all_chunks, all_metadatas)
        save_vector_store(vector_store)
        st.session_state.vector_store = vector_store
        progress_bar.progress(1.0, text="Done!")
        st.success(f"Processed {len(uploaded_files)} resume(s) into {len(all_chunks)} chunks.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Process Resumes", disabled=not uploaded_files):
        process_uploaded_resumes(uploaded_files)

    if st.session_state.processed_filenames:
        st.divider()
        st.caption("Processed resumes:")
        for name in st.session_state.processed_filenames:
            st.text(f"• {name}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("AI Resume Screening Assistant")

if st.session_state.vector_store is None:
    st.info("Upload and process resumes in the sidebar to get started.")
else:
    tab_chat, tab_rank, tab_extract, tab_interview = st.tabs(
        ["💬 Chat", "🏆 Rank Candidates", "🔍 Extract Info", "🎤 Interview Questions"]
    )

    # -- Chat tab ----------------------------------------------------------
    with tab_chat:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("View sources"):
                        for src in message["sources"]:
                            st.markdown(
                                f"**{src['source']}** (relevance: {src['relevance_score']})"
                            )
                            st.text(src["text"])

        user_query = st.chat_input("Ask about your candidates...")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = run_rag_query(st.session_state.vector_store, user_query, top_k=5)
                    st.write(result["answer"])
                    with st.expander("View sources"):
                        for src in result["sources"]:
                            st.markdown(
                                f"**{src['source']}** (relevance: {src['relevance_score']})"
                            )
                            st.text(src["text"])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })

    # -- Ranking tab ---------------------------------------------------------
    with tab_rank:
        st.subheader("Rank candidates against a job description")
        job_description = st.text_area("Paste the job description here", height=150)

        if st.button("Rank Candidates", disabled=not job_description):
            with st.spinner("Scoring candidates..."):
                ranked = rank_all_candidates(
                    st.session_state.vector_store,
                    st.session_state.processed_filenames,
                    job_description,
                )
            for candidate in ranked:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{candidate['source']}**")
                        st.caption(candidate.get("reasoning", ""))
                    with col2:
                        st.metric("Match Score", f"{candidate['match_score']}/100")
                    strengths_col, gaps_col = st.columns(2)
                    with strengths_col:
                        st.markdown("✅ **Strengths**")
                        for s in candidate.get("strengths", []):
                            st.write(f"- {s}")
                    with gaps_col:
                        st.markdown("⚠️ **Gaps**")
                        for g in candidate.get("gaps", []):
                            st.write(f"- {g}")

        st.divider()
        st.subheader("Candidate summary")
        summary_choice = st.selectbox(
            "Choose a candidate", st.session_state.processed_filenames, key="summary_select"
        )
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = generate_candidate_summary(st.session_state.vector_store, summary_choice)
            st.write(summary)

    # -- Extraction tab --------------------------------------------------
    with tab_extract:
        st.subheader("Extract structured resume data")
        extract_choice = st.selectbox(
            "Choose a candidate", st.session_state.processed_filenames, key="extract_select"
        )
        if st.button("Extract Info"):
            with st.spinner("Extracting..."):
                resume_text = st.session_state.resume_texts.get(extract_choice, "")
                data = extract_structured_data(resume_text)

            st.markdown("**Skills**")
            st.write(", ".join(data.get("skills", [])) or "None found")

            st.markdown("**Education**")
            for edu in data.get("education", []):
                st.write(f"- {edu.get('degree', '')} — {edu.get('institution', '')} ({edu.get('year', '')})")

            st.markdown("**Years of Experience**")
            st.write(data.get("years_of_experience", "Unknown"))

    # -- Interview questions tab -----------------------------------------
    with tab_interview:
        st.subheader("Generate tailored interview questions")
        interview_choice = st.selectbox(
            "Choose a candidate", st.session_state.processed_filenames, key="interview_select"
        )
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                resume_text = st.session_state.resume_texts.get(interview_choice, "")
                questions = generate_interview_questions(resume_text)

            for i, q in enumerate(questions, start=1):
                st.write(f"{i}. {q}")
