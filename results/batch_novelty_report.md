# Batch Novelty Subspace Analysis Report

**Generated:** 2026-10-27T05:59:20.414995+00:00  
**Papers Evaluated:** 2350  
**Success Rate:** 2152 / 2350 (91.6%)

## 1. Aggregate Geometric Metrics

| Metric | Definition | Cohort Mean | Interpretation |
| :--- | :--- | :--- | :--- |
| **m** | Cosine to closest parent | `0.8735` | Baseline anchor strength to closest single prior work |
| **Δ (Delta)** | Multi-parent cone synergy | `0.0849` | Energy gain from non-negative combination of multiple parents |
| **R²₊** | Cone reconstruction energy | `0.8492` | Proportion of embedding explained by the reference subspace |
| **√(1 - R²₊)** | Novelty residual gap | `0.3831` | Orthogonal departure distance from reference cone to unit circle |

## 2. Discovery Mode Distribution

| Discovery Classification | Paper Count | Percentage |
| :--- | :---: | :---: |
| **High Synergy Cone Blend** | 1930 | 89.7% |
| **Moderate Synthesis** | 208 | 9.7% |
| **Linear / Incremental Extension** | 12 | 0.6% |
| **Paradigm Shift / Subspace Outlier** | 2 | 0.1% |

## 3. Top Synergistic Syntheses (Highest Multi-Parent Synergy Δ)
Papers that achieve dramatic innovation by non-linearly fusing multiple distinct research threads:

| arXiv ID | Title | Year | Citations | Δ (Synergy) | m (Anchor) | R²₊ | Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| [2603.27331](https://arxiv.org/abs/2603.27331) | SACRED: A Faithful Annotated Multimedia Multi... | 2026 | 0 | **0.1624** | 0.829 | 0.849 | High Synergy Cone Blend |
| [2608.23593](https://arxiv.org/abs/2608.23593) | Fidelity Preference, Not Demographic Preferen... | 2026 | 0 | **0.1592** | 0.802 | 0.802 | High Synergy Cone Blend |
| [1512.01289](https://arxiv.org/abs/1512.01289) | Untitled... | — | 2 | **0.1558** | 0.850 | 0.879 | High Synergy Cone Blend |
| [2608.03711](https://arxiv.org/abs/2608.03711) | Attention is Case-Sensitive... | 2026 | 0 | **0.1558** | 0.825 | 0.837 | High Synergy Cone Blend |
| [2609.28565](https://arxiv.org/abs/2609.28565) | When Explanations Cannot Be Read: Measuring a... | 2026 | 0 | **0.1554** | 0.819 | 0.827 | High Synergy Cone Blend |
| [2608.06300](https://arxiv.org/abs/2608.06300) | Bias Analysis of L2 Speaking Assessment Syste... | 2026 | 0 | **0.1546** | 0.790 | 0.779 | High Synergy Cone Blend |
| [2607.24730](https://arxiv.org/abs/2607.24730) | KANEx: Translating Kolmogorov-Arnold Networks... | 2026 | 0 | **0.1540** | 0.841 | 0.861 | High Synergy Cone Blend |
| [2505.21513](https://arxiv.org/abs/2505.21513) | Enhancing Vision Transformer Explainability U... | 2025 | 0 | **0.1539** | 0.857 | 0.888 | High Synergy Cone Blend |
| [2603.18353](https://arxiv.org/abs/2603.18353) | Interpretability without actionability: mecha... | — | 0 | **0.1538** | 0.810 | 0.810 | High Synergy Cone Blend |
| [2507.05810](https://arxiv.org/abs/2507.05810) | Concept-Based Mechanistic Interpretability Us... | 2025 | 0 | **0.1533** | 0.817 | 0.820 | High Synergy Cone Blend |

## 4. Top Subspace Outliers / Paradigm Shifts (Highest Residual √(1 - R²₊))
Papers with large orthogonal departures that cannot be reconstructed from their references:

| arXiv ID | Title | Year | Citations | Novelty Residual | R²₊ | m (Anchor) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [1801.06562](https://arxiv.org/abs/1801.06562) | Uncertainty quantification of tabulated super... | 2018 | 5 | **0.6930** | 0.520 | 0.694 |
| [2505.16376](https://arxiv.org/abs/2505.16376) | DeCafNet: Delegate and Conquer for Efficient ... | 2025 | 0 | **0.6796** | 0.538 | 0.688 |
| [2509.14830](https://arxiv.org/abs/2509.14830) | ProtoMedX: Towards Explainable Multi-Modal Pr... | 2025 | 0 | **0.6591** | 0.566 | 0.680 |
| [1811.04544](https://arxiv.org/abs/1811.04544) | Visual Saliency Maps Can Apply to Facial Expr... | 2018 | 16 | **0.6443** | 0.585 | 0.693 |
| [1806.04398](https://arxiv.org/abs/1806.04398) | Attentive cross-modal paratope prediction... | 2018 | 67 | **0.6392** | 0.591 | 0.703 |
| [1904.03367](https://arxiv.org/abs/1904.03367) | Untitled... | — | 67 | **0.6382** | 0.593 | 0.688 |
| [1806.04838](https://arxiv.org/abs/1806.04838) | Partial AUC Maximization via Nonlinear Scorin... | 2018 | 10 | **0.6279** | 0.606 | 0.707 |
| [2602.05710](https://arxiv.org/abs/2602.05710) | Ethology of Latent Spaces... | 2026 | 0 | **0.6231** | 0.612 | 0.727 |
| [2609.32190](https://arxiv.org/abs/2609.32190) | Evaluating Single and Multi-Omics Based Expla... | 2026 | 0 | **0.6012** | 0.639 | 0.730 |
| [2608.22150](https://arxiv.org/abs/2608.22150) | Gender Attribution in Causal Beliefs... | 2026 | 0 | **0.5813** | 0.662 | 0.758 |

## 5. Top Foundationally Rooted Works (Highest Single Parent m)
Works that build directly and deeply on a primary ancestor paper:

| arXiv ID | Title | Year | Citations | m (Anchor) | Δ (Synergy) | Primary Parent Attribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| [1404.1530](https://arxiv.org/abs/1404.1530) | Provable Deterministic Leverage Score Sa... | 2014 | 88 | **1.0000** | 0.000 | `openalex:W2109818719` (λ=1.00) |
| [2504.00435](https://arxiv.org/abs/2504.00435) | User authentication on earable devices v... | 2025 | 0 | **0.9885** | 0.002 | `openalex:W4283219085` (λ=0.85) |
| [2505.08517](https://arxiv.org/abs/2505.08517) | A Deep Learning-Driven Inhalation Injury... | 2025 | 0 | **0.9595** | 0.014 | `s2:ad141b5b4594383bd6da7269da0454ad6e07cd8a` (λ=0.69) |
| [2606.14686](https://arxiv.org/abs/2606.14686) | CottonLeafVision: An Explainable and Rob... | 2026 | 0 | **0.9582** | 0.034 | `s2:58e83b5b2580cb99becb8b4424bd0a56c9d81f62` (λ=0.26) |
| [2507.10013](https://arxiv.org/abs/2507.10013) | Cross-modal Associations in Vision and L... | 2025 | 0 | **0.9575** | 0.016 | `2407.17974` (λ=0.59) |
| [2512.13434](https://arxiv.org/abs/2512.13434) | Self-Supervised Ultrasound Representatio... | 2025 | 0 | **0.9567** | 0.021 | `s2:5da5c29b4dc098e124c47c5e2d2ae01c4a07b4fb` (λ=0.59) |
| [1009.0623](https://arxiv.org/abs/1009.0623) | Untitled... | — | 2 | **0.9564** | 0.017 | `s2:466f1c8a01d5ee8308884722afa1030f3a22918e` (λ=0.69) |
| [1801.02480](https://arxiv.org/abs/1801.02480) | Facial Attributes: Accuracy and Adversar... | 2018 | 67 | **0.9552** | 0.006 | `s2:e87a38f70420b2d4a11c0bbab55677ae139bcc1b` (λ=0.82) |
| [2607.12293](https://arxiv.org/abs/2607.12293) | Adaptive Cross-Modal Fusion with Sparse ... | 2026 | 0 | **0.9545** | 0.041 | `2511.20020` (λ=0.27) |
| [2505.20027](https://arxiv.org/abs/2505.20027) | Multi-modal brain encoding models for mu... | 2025 | 0 | **0.9543** | 0.026 | `s2:32d0ad59162b07bf469b6889930ef1d10e66f7f8` (λ=0.47) |

---
### Methodology & Theoretical Foundations
Every paper embedding $\mathbf{x}$ is projected onto the non-negative convex cone of its cited references $\mathbf{Y} = [\mathbf{y}_1, \dots, \mathbf{y}_k]$ via Non-Negative Least Squares (NNLS):

$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \ge 0} \|\mathbf{x} - \mathbf{Y}\boldsymbol{\lambda}\|^2, \quad \hat{\mathbf{x}} = \mathbf{Y}\boldsymbol{\lambda}^*$$

The reconstructed energy is $R^2_+ = \|\hat{\mathbf{x}}\|^2$, the foundational anchor is $m = \max_j (\mathbf{x}^T \mathbf{y}_j)$, and multi-parent synergy is defined as $\Delta = R^2_+ - \max(m, 0)^2$.