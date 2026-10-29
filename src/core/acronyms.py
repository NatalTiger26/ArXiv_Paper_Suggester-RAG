"""Domain acronym dictionary and query expansion for academic literature search."""

from __future__ import annotations

import re

# Curated high-frequency academic machine learning acronyms
ACRONYM_MAP: dict[str, str] = {
    "NTK": "Neural Tangent Kernel infinite-width limit",
    "LORA": "Low-Rank Adaptation parameter-efficient fine-tuning",
    "MRL": "Matryoshka Representation Learning adaptive dimension",
    "SVD": "Singular Value Decomposition low-rank approximation",
    "BERT": "Bidirectional Encoder Representations from Transformers",
    "CLIP": "Contrastive Language-Image Pre-Training",
    "RAG": "Retrieval-Augmented Generation",
    "RLHF": "Reinforcement Learning from Human Feedback",
    "DPO": "Direct Preference Optimization",
    "MCTS": "Monte Carlo Tree Search",
    "VAE": "Variational Autoencoder",
    "GAN": "Generative Adversarial Network",
    "SGD": "Stochastic Gradient Descent",
    "PPO": "Proximal Policy Optimization",
    "MOE": "Mixture of Experts routing",
}


def expand_query_acronyms(query: str) -> str:
    """Expand recognized technical acronyms into canonical phrases to aid dense matching."""
    if not query or len(query.strip()) < 3:
        return query

    tokens = re.findall(r"\b[A-Za-z0-9_\-]+\b", query)
    expanded = query
    for token in tokens:
        upper = token.upper()
        if upper in ACRONYM_MAP:
            canonical = ACRONYM_MAP[upper]
            # Replace whole word occurrence if not already followed by definition
            pattern = re.compile(rf"\b{re.escape(token)}\b(?!\s*\([^)]*{re.escape(upper)}[^)]*\))")
            if pattern.search(expanded):
                expanded = pattern.sub(f"{token} ({canonical})", expanded, count=1)
    return expanded
