import os
import re
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PolicyChunk:
    doc_name: str
    section_title: str
    content: str

    @property
    def source_reference(self) -> str:
        return f"{self.doc_name} > {self.section_title}"


class PolicyRetriever:
    """
    Lightweight runtime policy loader, text chunker, and TF-IDF cosine retriever.
    Automatically loads all markdown/text policy files from the specified directory.
    """

    def __init__(self, policy_dir: str):
        self.policy_dir = policy_dir
        self.chunks: List[PolicyChunk] = []
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self._load_and_chunk_docs()

    def _load_and_chunk_docs(self):
        """Scans policy_dir, reads documents, and chunks them by markdown headers."""
        if not os.path.exists(self.policy_dir):
            raise FileNotFoundError(f"Policy directory not found: {self.policy_dir}")

        for filename in sorted(os.listdir(self.policy_dir)):
            if filename.endswith(".md") or filename.endswith(".txt"):
                filepath = os.path.join(self.policy_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                self._chunk_document(filename, text)

        if not self.chunks:
            raise ValueError(f"No policy documents found in {self.policy_dir}")

        # Build TF-IDF matrix over chunk contents
        corpus = [f"{c.section_title}\n{c.content}" for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _chunk_document(self, filename: str, text: str):
        """Splits document text by markdown section headers (## or #)."""
        doc_title = filename.replace(".md", "").replace(".txt", "").replace("_", " ").title()
        
        # Split by level 1 or 2 headers (e.g. ## Section Name)
        sections = re.split(r'\n(?=#{1,3}\s+)', text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            lines = section.split('\n')
            first_line = lines[0].strip()
            
            if first_line.startswith('#'):
                section_title = first_line.lstrip('#').strip()
                content = '\n'.join(lines[1:]).strip()
            else:
                section_title = "General Information"
                content = section
                
            if content:
                self.chunks.append(PolicyChunk(
                    doc_name=doc_title,
                    section_title=section_title,
                    content=content
                ))

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.1) -> List[Tuple[PolicyChunk, float]]:
        """
        Retrieves top_k matching chunks for a given query string.
        Returns a list of (PolicyChunk, similarity_score) tuples.
        """
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score >= score_threshold:
                results.append((self.chunks[idx], score))

        return results
