"""
Reference-conditional paper profiles (embedding geometry vs cited works).

This package is intentionally separate from recommend / RAG. It does not
write into chroma_corpus or chroma_rag. Scores are offline / on-demand.

Honest framing: these numbers measure geometry relative to cited references
in a fixed embedding space: not scientific novelty in the idea sense.
"""

from .score import score_paper, score_papers

__all__ = ["score_paper", "score_papers"]
