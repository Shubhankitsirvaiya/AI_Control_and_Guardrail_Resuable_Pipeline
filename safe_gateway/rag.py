import re
import math
import os

from dotenv import load_dotenv
from openai import OpenAI


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if token not in STOPWORDS]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def relevance_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


def retrieve_keyword(query: str, document_text: str, top_k: int = 3) -> list[str]:
    chunks = chunk_text(document_text)
    scored = sorted(
        ((relevance_score(query, chunk), chunk) for chunk in chunks),
        key=lambda item: item[0],
        reverse=True,
    )
    return [chunk for score, chunk in scored[:top_k] if score > 0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class OpenAIEmbeddingRetriever:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self._cache: dict[str, list[float]] = {}

    def enabled(self) -> bool:
        return self.client is not None

    def embed(self, text: str) -> list[float]:
        cache_key = f"{self.model}:{text}"
        if cache_key not in self._cache:
            response = self.client.embeddings.create(model=self.model, input=text)
            self._cache[cache_key] = response.data[0].embedding
        return self._cache[cache_key]

    def retrieve(self, query: str, document_text: str, top_k: int = 3) -> list[str]:
        chunks = chunk_text(document_text)
        if not chunks:
            return []

        query_embedding = self.embed(query)
        scored = sorted(
            (
                (cosine_similarity(query_embedding, self.embed(chunk)), chunk)
                for chunk in chunks
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [chunk for score, chunk in scored[:top_k] if score > 0]


def retrieve(query: str, document_text: str, top_k: int = 3, embedding_retriever: OpenAIEmbeddingRetriever | None = None) -> list[str]:
    """Retrieve with OpenAI embeddings as the primary path and keyword search as fallback."""
    if embedding_retriever and embedding_retriever.enabled():
        try:
            return embedding_retriever.retrieve(query, document_text, top_k)
        except Exception:
            return retrieve_keyword(query, document_text, top_k)
    return retrieve_keyword(query, document_text, top_k)


def generate_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "I could not find enough relevant information in the selected document to answer safely."

    query_terms = set(tokenize(query))
    sentences: list[str] = []
    for context in contexts:
        sentences.extend(re.split(r"(?<=[.!?])\s+", context))

    ranked = sorted(
        sentences,
        key=lambda sentence: len(query_terms & set(tokenize(sentence))),
        reverse=True,
    )
    selected = [sentence.strip() for sentence in ranked[:4] if sentence.strip()]

    if not selected:
        selected = [contexts[0][:700].strip()]

    return (
        "Based on the selected document, here is the safest grounded answer:\n\n"
        + " ".join(selected)
        + "\n\nThis response is limited to the uploaded document context."
    )

