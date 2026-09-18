"""
Destination Knowledge Tool (RAG)

Ingests the 4 destination guide .txt files, chunks each into its 5
labeled sections, and exposes search_destination_guide() which returns
the most relevant chunk(s) for a natural-language query using
TF-IDF + cosine similarity (no external embedding API needed).
"""

import os
import re
import logging
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DATA_DIR

logger = logging.getLogger("tripmate.rag_tool")

# The 5 section headers used consistently across every destination file.
SECTION_HEADERS = [
    "VISA & ENTRY",
    "BEST TIME TO VISIT",
    "LOCAL CUSTOMS",
    "PACKING TIPS",
    "SAFETY & HEALTH",
]


class DestinationKnowledgeBase:
    """Loads destination .txt files, chunks them by section, and
    supports similarity search over those chunks using TF-IDF."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.chunks: List[str] = []       # raw chunk text
        self.metadata: List[dict] = []    # {"city": ..., "section": ...}
        self.known_cities: List[str] = []  # cities actually present in the data pack
        self.vectorizer = None
        self.chunk_vectors = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Destination data directory not found: {self.data_dir}"
            )

        txt_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        if not txt_files:
            raise FileNotFoundError(
                f"No destination .txt files found in {self.data_dir}"
            )

        for filename in txt_files:
            city = os.path.splitext(filename)[0].replace("_", " ").title()
            self.known_cities.append(city)
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            self._chunk_by_section(city, text)

        if not self.chunks:
            raise ValueError("No chunks were produced from the destination data.")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)

        logger.info(
            "Indexed %d chunks from %d destination files.",
            len(self.chunks), len(txt_files)
        )

    def _chunk_by_section(self, city: str, text: str):
        """Split a destination document into one chunk per section header."""
        # Build a regex that finds any of the known section headers.
        pattern = "|".join(re.escape(h) for h in SECTION_HEADERS)
        matches = list(re.finditer(pattern, text))

        for i, match in enumerate(matches):
            section_name = match.group()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            chunk_text = f"{city} — {section_name}: {body}"
            self.chunks.append(chunk_text)
            self.metadata.append({"city": city, "section": section_name})

    def _match_known_cities(self, query: str) -> List[str]:
        """Return which of our known (supported) cities are explicitly named in the query."""
        query_lower = query.lower()
        return [city for city in self.known_cities if city.lower() in query_lower]

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        # Guard against false-positive matches for unsupported destinations.
        # TF-IDF similarity alone can't tell "visa for Paris" apart from
        # "visa for Barcelona" if it only matches on the surrounding words
        # ("visa", "requirements", "for") rather than the city name itself
        # (since an unknown city name won't be in the vocabulary at all).
        # So we explicitly require the query to name one of our supported
        # cities before returning any chunks.
        matched_cities = self._match_known_cities(query)

        if not matched_cities:
            logger.warning(
                "No supported destination named in query: %r. "
                "Supported cities: %s", query, self.known_cities
            )
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.chunk_vectors)[0]

        # Restrict candidates to chunks belonging to the matched city/cities,
        # so a query naming Barcelona can never surface Bangkok content.
        candidate_indices = [
            i for i, meta in enumerate(self.metadata)
            if meta["city"] in matched_cities
        ]

        ranked = sorted(candidate_indices, key=lambda i: similarities[i], reverse=True)[:top_k]

        results = []
        for idx in ranked:
            score = similarities[idx]
            if score <= 0:
                continue  # no meaningful overlap with this chunk
            results.append(self.chunks[idx])
            logger.debug(
                "Matched chunk: city=%s section=%s score=%.3f",
                self.metadata[idx]["city"], self.metadata[idx]["section"], score
            )

        if not results:
            logger.warning(
                "Query named a supported city (%s) but no relevant chunk "
                "was found for query: %r", matched_cities, query
            )

        return results


# Singleton instance, built once at import time and reused across calls.
_kb = None


def _get_kb() -> DestinationKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = DestinationKnowledgeBase()
    return _kb


def search_destination_guide(query: str) -> List[str]:
    """
    Tool function exposed to the LLM agent.

    Args:
        query: natural-language question or topic, e.g.
               "What should I pack for Tokyo in winter?"

    Returns:
        A list of the most relevant destination guide text chunks.
        Returns an empty list if nothing relevant is found.
    """
    logger.info("search_destination_guide called with query=%r", query)
    try:
        kb = _get_kb()
        return kb.search(query, top_k=3)
    except Exception as e:
        logger.error("search_destination_guide failed: %s", e)
        raise