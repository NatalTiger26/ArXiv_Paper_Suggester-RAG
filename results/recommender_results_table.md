# Recommender evaluation Evaluation Results

**Query:** mechanistic interpretability circuits random matrix theory neural networks loss landscapes statistical mechanics representation learning world models

**Ground truth:** `/Users/nataltiger/Documents/Personal Projects/ArXiv_Paper_Suggester/rankings/personal_importance.csv`

**Ground-truth papers:** 48

**Evaluation k:** 40

**Retrieval pool:** 120

**Recency half-life:** 4.0

**LLM enabled:** no

## Metrics

| Method | Spearman ρ | NDCG@10 | NDCG@20 | Intersection |
|---|---:|---:|---:|---:|
| `citation_baseline` | — | 0.0000 | 0.0000 | 0 |
| `embedding_only` | — | 0.0000 | 0.0000 | 0 |
| `full` | — | 0.0000 | 0.0000 | 0 |
| `ablate_embedding` | — | 0.0000 | 0.0000 | 0 |
| `ablate_recency` | — | 0.0000 | 0.0000 | 0 |
| `ablate_citation` | — | 0.0000 | 0.0000 | 0 |
| `ablate_author` | — | 0.0000 | 0.0000 | 0 |
| `ablate_pagerank` | — | 0.0000 | 0.0000 | 0 |

## Weights

| Signal | Default weight |
|---|---:|
| `embedding` | 0.5000 |
| `recency` | 0.2000 |
| `citation` | 0.2500 |
| `author` | 0.0500 |

## Conclusion

The multi-signal ranker and citation baseline were evaluated, but Spearman ρ could not be computed for one or both methods.
