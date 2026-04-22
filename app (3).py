from dotenv import load_dotenv
load_dotenv()
"""
Resume + Job Description Matcher
AI-powered matching system using RAG + LLM analysis via Anthropic API
"""

import os
import json
import re
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rag_engine import RAGEngine
from llm_analyzer import LLMAnalyzer
from resume_parser import ResumeParser

app = FastAPI(title="Resume Matcher API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

rag = RAGEngine()
llm = LLMAnalyzer()
parser = ResumeParser()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze(
    resume_text: Optional[str] = Form(None),
    job_description: str = Form(...),
    resume_file: Optional[UploadFile] = File(None),
):
    # Extract resume text
    if resume_file and resume_file.filename:
        content = await resume_file.read()
        ext = Path(resume_file.filename).suffix.lower()
        resume_text = parser.extract_text(content, ext)
    
    if not resume_text or len(resume_text.strip()) < 50:
        raise HTTPException(400, "Please provide a resume (text or file).")

    if len(job_description.strip()) < 50:
        raise HTTPException(400, "Job description is too short.")

    # RAG: chunk and embed resume, retrieve top relevant sections vs JD
    resume_chunks = rag.chunk_text(resume_text)
    jd_chunks = rag.chunk_text(job_description)

    resume_embeddings = rag.embed_chunks(resume_chunks)
    jd_embeddings = rag.embed_chunks(jd_chunks)

    relevant_sections = rag.retrieve_relevant(
        query_embeddings=jd_embeddings,
        corpus_chunks=resume_chunks,
        corpus_embeddings=resume_embeddings,
        top_k=5
    )

    semantic_score = rag.compute_semantic_score(resume_embeddings, jd_embeddings)

    # LLM deep analysis
    analysis = llm.analyze(
        resume_text=resume_text,
        job_description=job_description,
        relevant_sections=relevant_sections,
        semantic_score=semantic_score
    )

    return JSONResponse(analysis)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
