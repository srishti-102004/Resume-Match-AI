# ResumeMatch AI — Intelligent Resume + JD Matcher

## Architecture

```
resume_matcher/
├── app.py              ← FastAPI server (orchestrator)
├── rag_engine.py       ← RAG: chunking · TF-IDF embeddings · retrieval
├── llm_analyzer.py     ← Anthropic Claude API for deep AI analysis
├── resume_parser.py    ← PDF / DOCX / TXT text extraction
├── templates/
│   └── index.html      ← Full-stack UI (dark, animated)
├── requirements.txt
└── README.md
```

## How It Works

### 1. Text Extraction (ResumeParser)
- Supports **PDF** (pypdf), **DOCX** (python-docx), and plain **TXT**
- Cleans whitespace and normalizes line breaks

### 2. RAG Engine (Retrieval-Augmented Generation)
- **Chunking**: Splits resume and JD into overlapping 150-word windows
- **Embedding**: Builds TF-IDF vectors for each chunk — no GPU or API calls needed
- **Retrieval**: Cosine similarity between JD chunks (query) and resume chunks (corpus)
- **Semantic Score**: Average max-similarity across JD chunks, scaled to 0–100
- Returns top 5 most relevant resume sections + overall semantic match score

### 3. LLM Analysis (Anthropic Claude)
- Sends: full resume + full JD + top RAG sections + semantic score
- Structured JSON output with:
  - Overall match score (0–100)
  - Verdict label
  - Executive summary
  - Skills: matched / missing / bonus
  - Experience & Education analysis
  - Strengths & Gaps
  - Actionable recommendations (High / Medium / Low priority)
  - ATS optimization score
  - Keyword density analysis
  - Interview probability

### 4. FastAPI Server
- Single `POST /api/analyze` endpoint
- Accepts multipart form: resume text or file + job description
- Orchestrates RAG → LLM pipeline
- Serves the React-free single-file UI

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Run
python app.py
# → http://localhost:8000
```

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `anthropic` | Claude LLM API |
| `numpy` | Vector math for RAG |
| `pypdf` | PDF parsing |
| `python-docx` | DOCX parsing |
| `python-multipart` | File upload support |

## Production Upgrade Path

| Component | Current (zero-dependency) | Production |
|---|---|---|
| Embeddings | TF-IDF (numpy) | `sentence-transformers` or OpenAI `text-embedding-3-small` |
| Vector Store | In-memory numpy | Pinecone / Weaviate / pgvector |
| File Storage | Local disk | AWS S3 |
| Auth | None | JWT + OAuth2 |
| Cache | None | Redis |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
