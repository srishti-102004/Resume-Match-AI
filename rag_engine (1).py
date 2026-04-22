"""
RAG Engine — Retrieval-Augmented Generation
Handles text chunking, TF-IDF based embeddings, and semantic retrieval
for matching resume sections against job description requirements.
"""

import re
import math
import numpy as np
from typing import List, Tuple
from collections import Counter


class RAGEngine:
    """
    Lightweight RAG engine using TF-IDF vectors (no heavy ML dependencies).
    For production, swap embed_chunks() with OpenAI/Cohere/sentence-transformers.
    """

    def __init__(self, chunk_size: int = 150, overlap: int = 30):
        self.chunk_size = chunk_size  # words per chunk
        self.overlap = overlap
        self._idf_cache = {}

    # ─── Chunking ────────────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word windows."""
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, max(1, len(words) - self.overlap), step):
            chunk = ' '.join(words[i: i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    # ─── Embedding (TF-IDF) ──────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove stopwords
        stopwords = {
            'the','a','an','and','or','but','in','on','at','to','for',
            'of','with','by','from','is','are','was','were','be','been',
            'has','have','had','do','does','did','will','would','could',
            'should','may','might','this','that','these','those','i','we',
            'you','he','she','it','they','my','our','your','its','their'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _tf(self, tokens: List[str]) -> dict:
        count = Counter(tokens)
        total = len(tokens) or 1
        return {t: c / total for t, c in count.items()}

    def _idf(self, term: str, corpus_tokens: List[List[str]]) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        n_docs = len(corpus_tokens)
        df = sum(1 for doc in corpus_tokens if term in doc)
        idf = math.log((n_docs + 1) / (df + 1)) + 1
        self._idf_cache[term] = idf
        return idf

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Build TF-IDF vectors for each chunk.
        Returns shape (n_chunks, vocab_size).
        """
        tokenized = [self._tokenize(c) for c in chunks]
        vocab = sorted({t for doc in tokenized for t in doc})
        vocab_idx = {t: i for i, t in enumerate(vocab)}

        if not vocab:
            return np.zeros((len(chunks), 1))

        self._idf_cache = {}
        matrix = np.zeros((len(chunks), len(vocab)))
        for i, tokens in enumerate(tokenized):
            tf = self._tf(tokens)
            for term, tf_val in tf.items():
                if term in vocab_idx:
                    idf = self._idf(term, tokenized)
                    matrix[i, vocab_idx[term]] = tf_val * idf

        # L2 normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    # ─── Retrieval ───────────────────────────────────────────────────────────

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve_relevant(
        self,
        query_embeddings: np.ndarray,
        corpus_chunks: List[str],
        corpus_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[dict]:
        """
        For each JD chunk (query), find the most similar resume chunks.
        Returns top_k unique resume sections ranked by relevance.
        """
        if corpus_embeddings.shape[1] != query_embeddings.shape[1]:
            # Dimension mismatch — pad the smaller one
            target_dim = max(corpus_embeddings.shape[1], query_embeddings.shape[1])
            def pad(arr, dim):
                if arr.shape[1] < dim:
                    return np.hstack([arr, np.zeros((arr.shape[0], dim - arr.shape[1]))])
                return arr
            corpus_embeddings = pad(corpus_embeddings, target_dim)
            query_embeddings = pad(query_embeddings, target_dim)

        scores = {}
        for q_vec in query_embeddings:
            for idx, c_vec in enumerate(corpus_embeddings):
                sim = self._cosine_sim(q_vec, c_vec)
                scores[idx] = max(scores.get(idx, 0), sim)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"text": corpus_chunks[i], "score": round(float(s), 4)}
            for i, s in ranked
        ]

    def compute_semantic_score(
        self,
        resume_embeddings: np.ndarray,
        jd_embeddings: np.ndarray
    ) -> float:
        """
        Overall semantic similarity: avg of max-sim per JD chunk.
        """
        if resume_embeddings.shape[1] != jd_embeddings.shape[1]:
            target_dim = max(resume_embeddings.shape[1], jd_embeddings.shape[1])
            def pad(arr, dim):
                if arr.shape[1] < dim:
                    return np.hstack([arr, np.zeros((arr.shape[0], dim - arr.shape[1]))])
                return arr
            resume_embeddings = pad(resume_embeddings, target_dim)
            jd_embeddings = pad(jd_embeddings, target_dim)

        max_sims = []
        for j_vec in jd_embeddings:
            sims = [self._cosine_sim(j_vec, r_vec) for r_vec in resume_embeddings]
            max_sims.append(max(sims) if sims else 0)

        raw = float(np.mean(max_sims)) if max_sims else 0
        # Scale to 0-100 and clamp
        scaled = min(100, max(0, raw * 180))
        return round(scaled, 1)
