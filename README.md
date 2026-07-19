# AI Resume Screening System (RAG)

An AI-powered resume screening assistant that lets HR users upload resumes,
search and chat with them using natural language, rank candidates against
job descriptions, and generate interview questions — all grounded in
Retrieval-Augmented Generation (RAG) to minimize hallucination.

## Live Demo
`Add your Streamlit Community Cloud URL here after deploying`

## Features
- Upload and process multiple PDF resumes
- Natural language chat interface with source citations and relevance scores
- Resume ranking against a job description with match scores, strengths, and gaps
- Skill, education, and experience extraction (structured JSON output)
- Auto-generated candidate summaries
- Auto-generated, resume-tailored interview questions
- Persistent chat history across a session
- Content-hash based caching to avoid re-processing duplicate resumes
- File + console logging for debugging deployed behavior

## Tech Stack
| Component         | Technology                   |
|--------------------|-------------------------------|
| LLM                | Google Gemini                |
| Framework          | LangChain                    |
| Vector Database    | FAISS                        |
| Embeddings         | Google Gemini Embeddings     |
| PDF Processing     | PyMuPDF                      |
| Frontend           | Streamlit                    |
| Deployment         | Streamlit Community Cloud    |

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index
                                                              │
User Query → Query Embedding → Similarity Search ────────────┘
                                        │
                                        ▼
                          Augmented Prompt → Gemini LLM → Grounded Answer
```

**Retrieval-Augmented Generation (RAG)** pipeline:
1. **Retrieval** — the user's question is embedded and matched against
   stored resume-chunk embeddings in FAISS to find the most relevant excerpts.
2. **Augmentation** — those excerpts are inserted into a carefully engineered
   prompt, along with strict grounding instructions.
3. **Generation** — Gemini answers using only the retrieved excerpts,
   citing sources and refusing to answer when information isn't present.

## Project Structure

```
resume-rag-system/
├── src/
│   ├── pdf_processor.py       # PDF text extraction + cleaning
│   ├── nlp_basics.py          # NLP fundamentals (reference/teaching module)
│   ├── text_splitter.py       # Chunking logic
│   ├── embeddings.py          # Gemini embedding generation
│   ├── vector_store.py        # FAISS create/save/load
│   ├── retriever.py           # Top-K similarity search
│   ├── llm.py                 # Gemini chat model wrapper
│   ├── rag_pipeline.py        # Full retrieval + augmentation + generation
│   ├── extraction.py          # Structured skills/education/experience extraction
│   ├── ranking.py             # Job-description candidate ranking + summaries
│   ├── interview_questions.py # Tailored interview question generation
│   └── utils.py                # Logging + caching helpers
├── app.py                     # Streamlit UI entry point
├── data/
│   ├── resumes/                # Uploaded PDFs (gitignored)
│   ├── cache/                  # Processed-file cache (gitignored)
│   └── faiss_index/             # Saved vector index (gitignored)
├── logs/                       # Application logs (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Running Locally

1. Clone this repo:
   ```bash
   git clone <your-repo-url>
   cd resume-rag-system
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your free Gemini API key
   (get one at [aistudio.google.com](https://aistudio.google.com)):
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (`.env` and `data/*` are gitignored automatically).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GOOGLE_API_KEY = "your_actual_key_here"
   ```
5. Click **Deploy**.

## Key Design Decisions

- **RAG over fine-tuning** — keeps the system cheap, instantly updatable
  as new resumes are uploaded, and strictly grounded in real resume content
  rather than a static, retrained model.
- **Flat FAISS index (`IndexFlatL2`)** — appropriate for this project's
  scale (hundreds to low-thousands of resumes); a clear upgrade path exists
  to approximate indexes (IVF/HNSW) for larger datasets.
- **Structured JSON outputs with defensive parsing** — every extraction/
  ranking feature asks the LLM for a strict schema and gracefully falls
  back to a safe default if the model returns malformed output, since LLM
  output is never 100% guaranteed to match the requested format.
- **Content-hash caching** — avoids redundant, costly embedding/LLM calls
  when the same resume file is uploaded more than once.
- **Per-candidate ranking** — the LLM scores one candidate at a time against
  the job description (avoiding context window limits), with Python doing
  the deterministic sorting/aggregation afterward.

## Known Limitations / Future Improvements

- FAISS runs in-memory/single-instance — a managed vector DB would be
  needed to scale to many concurrent users or very large resume sets.
- Ranking makes one LLM call per candidate sequentially; this could be
  parallelized for faster ranking at scale.
- Sentence splitting in `nlp_basics.py` is intentionally naive (regex-based)
  for teaching purposes — a production NLP pipeline would use spaCy or NLTK.

## Screenshots

*(Add 2-3 screenshots of the app in action here)*

## License

MIT
