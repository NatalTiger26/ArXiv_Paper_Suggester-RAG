"""
In-library citation graph.

Scan papers in your corpus (or RAG full-text set), collect each paper's
reference list (Semantic Scholar → OpenAlex), and count how many *library*
papers cite each referenced work. Highly cited works that are not yet in
the library are natural ingest candidates.
"""

__version__ = "1.0.0"
