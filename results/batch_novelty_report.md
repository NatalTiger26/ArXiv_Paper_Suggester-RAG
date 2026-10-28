# Batch Novelty Subspace Analysis Report

**Generated:** 2026-10-28T01:10:17.353924+00:00  
**Papers Evaluated:** 5752  
**Success Rate:** 5707 / 5752 (99.2%)

## 1. Aggregate Geometric Metrics

| Metric | Definition | Cohort Mean | Interpretation |
| :--- | :--- | :--- | :--- |
| **m** | Cosine to closest parent | `0.8691` | Baseline anchor strength to closest single prior work |
| **Δ (Delta)** | Multi-parent cone synergy | `0.0877` | Energy gain from non-negative combination of multiple parents |
| **R²₊** | Cone reconstruction energy | `0.8446` | Proportion of embedding explained by the reference subspace |
| **√(1 - R²₊)** | Novelty residual gap | `0.3879` | Orthogonal departure distance from reference cone to unit circle |

## 2. Discovery Mode Distribution

| Discovery Classification | Paper Count | Percentage |
| :--- | :---: | :---: |
| **High Synergy Cone Blend** | 5176 | 90.7% |
| **Moderate Synthesis** | 489 | 8.6% |
| **Linear / Incremental Extension** | 34 | 0.6% |
| **Paradigm Shift / Subspace Outlier** | 8 | 0.1% |

## 3. Top Synergistic Syntheses (Highest Multi-Parent Synergy Δ)
Papers that achieve dramatic innovation by non-linearly fusing multiple distinct research threads:

| arXiv ID | Title | Year | Citations | Δ (Synergy) | m (Anchor) | R²₊ | Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| [2108.07258](https://arxiv.org/abs/2108.07258) | Untitled... | — | 0 | **0.2117** | 0.724 | 0.736 | High Synergy Cone Blend |
| [1810.03993](https://arxiv.org/abs/1810.03993) | Untitled... | — | 0 | **0.1752** | 0.734 | 0.713 | High Synergy Cone Blend |
| [2204.03772](https://arxiv.org/abs/2204.03772) | Multi-objective optimization determines when,... | 2022 | 0 | **0.1730** | 0.694 | 0.655 | High Synergy Cone Blend |
| [2008.10138](https://arxiv.org/abs/2008.10138) | PermuteAttack: Counterfactual Explanation of ... | 2020 | 0 | **0.1719** | 0.683 | 0.639 | High Synergy Cone Blend |
| [2304.03442](https://arxiv.org/abs/2304.03442) | Untitled... | — | 0 | **0.1708** | 0.844 | 0.883 | High Synergy Cone Blend |
| [2007.00251](https://arxiv.org/abs/2007.00251) | Unifying Model Explainability and Robustness ... | 2020 | 0 | **0.1686** | 0.838 | 0.870 | High Synergy Cone Blend |
| [2310.10631](https://arxiv.org/abs/2310.10631) | Untitled... | — | 0 | **0.1669** | 0.829 | 0.854 | High Synergy Cone Blend |
| [2211.07982](https://arxiv.org/abs/2211.07982) | Evaluating the Faithfulness of Saliency-based... | 2022 | 0 | **0.1652** | 0.844 | 0.878 | High Synergy Cone Blend |
| [2603.27331](https://arxiv.org/abs/2603.27331) | SACRED: A Faithful Annotated Multimedia Multi... | 2026 | 0 | **0.1624** | 0.829 | 0.849 | High Synergy Cone Blend |
| [2302.14383](https://arxiv.org/abs/2302.14383) | Linear Spaces of Meanings: Compositional Stru... | 2023 | 0 | **0.1624** | 0.833 | 0.856 | High Synergy Cone Blend |

## 4. Top Subspace Outliers / Paradigm Shifts (Highest Residual √(1 - R²₊))
Papers with large orthogonal departures that cannot be reconstructed from their references:

| arXiv ID | Title | Year | Citations | Novelty Residual | R²₊ | m (Anchor) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [1609.02907](https://arxiv.org/abs/1609.02907) | Untitled... | — | 0 | **0.7018** | 0.507 | 0.648 |
| [2501.10292](https://arxiv.org/abs/2501.10292) | Untitled... | — | 0 | **0.6994** | 0.511 | 0.681 |
| [1801.06562](https://arxiv.org/abs/1801.06562) | Uncertainty quantification of tabulated super... | 2018 | 5 | **0.6930** | 0.520 | 0.694 |
| [2210.15678](https://arxiv.org/abs/2210.15678) | Untitled... | — | 0 | **0.6830** | 0.533 | 0.666 |
| [2505.16376](https://arxiv.org/abs/2505.16376) | DeCafNet: Delegate and Conquer for Efficient ... | 2025 | 0 | **0.6796** | 0.538 | 0.688 |
| [2501.02355](https://arxiv.org/abs/2501.02355) | CorrFill: Enhancing Faithfulness in Reference... | 2025 | 0 | **0.6775** | 0.541 | 0.662 |
| [1911.11351](https://arxiv.org/abs/1911.11351) | Untitled... | — | 0 | **0.6742** | 0.545 | 0.694 |
| [1001.3540](https://arxiv.org/abs/1001.3540) | Untitled... | — | 0 | **0.6722** | 0.548 | 0.706 |
| [2207.05374](https://arxiv.org/abs/2207.05374) | Rethinking gradient weights' influence over s... | 2022 | 0 | **0.6686** | 0.553 | 0.677 |
| [2302.08670](https://arxiv.org/abs/2302.08670) | Cascaded information enhancement and cross-mo... | 2023 | 0 | **0.6670** | 0.555 | 0.677 |

## 5. Top Foundationally Rooted Works (Highest Single Parent m)
Works that build directly and deeply on a primary ancestor paper:

| arXiv ID | Title | Year | Citations | m (Anchor) | Δ (Synergy) | Primary Parent Attribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| [1404.1530](https://arxiv.org/abs/1404.1530) | Provable Deterministic Leverage Score Sa... | 2014 | 88 | **1.0000** | 0.000 | `openalex:W2109818719` (λ=1.00) |
| [2209.06481](https://arxiv.org/abs/2209.06481) | Targeting interventions for displacement... | 2022 | 0 | **1.0000** | 0.000 | `openalex:W4315488832` (λ=1.00) |
| [2201.12828](https://arxiv.org/abs/2201.12828) | Comprehensive Saliency Fusion for Object... | 2022 | 0 | **1.0000** | 0.000 | `openalex:W4205623571` (λ=1.00) |
| [1607.00133](https://arxiv.org/abs/1607.00133) | Untitled... | — | 0 | **1.0000** | 0.000 | `openalex:W2473418344` (λ=1.00) |
| [2402.17242](https://arxiv.org/abs/2402.17242) | Scalable Community Search with Accuracy ... | 2024 | 0 | **0.9991** | 0.000 | `openalex:W4400909681` (λ=0.95) |
| [2504.00435](https://arxiv.org/abs/2504.00435) | User authentication on earable devices v... | 2025 | 0 | **0.9885** | 0.002 | `openalex:W4283219085` (λ=0.85) |
| [2109.05201](https://arxiv.org/abs/2109.05201) | Untitled... | — | 0 | **0.9843** | 0.001 | `openalex:W3097305019` (λ=0.95) |
| [2101.03164](https://arxiv.org/abs/2101.03164) | Untitled... | — | 0 | **0.9808** | 0.012 | `s2:3dc12dadcf6a2015fb3c732e35dfa56ba604ed88` (λ=0.66) |
| [1910.12828](https://arxiv.org/abs/1910.12828) | Blind Robust 3-D Mesh Watermarking based... | 2019 | 0 | **0.9798** | 0.004 | `openalex:W2911342611` (λ=0.78) |
| [2408.12398](https://arxiv.org/abs/2408.12398) | A Comparative Analysis of Faithfulness M... | 2024 | 0 | **0.9792** | 0.001 | `2406.15264` (λ=0.93) |

---
### Methodology & Theoretical Foundations
Every paper embedding $\mathbf{x}$ is projected onto the non-negative convex cone of its cited references $\mathbf{Y} = [\mathbf{y}_1, \dots, \mathbf{y}_k]$ via Non-Negative Least Squares (NNLS):

$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \ge 0} \|\mathbf{x} - \mathbf{Y}\boldsymbol{\lambda}\|^2, \quad \hat{\mathbf{x}} = \mathbf{Y}\boldsymbol{\lambda}^*$$

The reconstructed energy is $R^2_+ = \|\hat{\mathbf{x}}\|^2$, the foundational anchor is $m = \max_j (\mathbf{x}^T \mathbf{y}_j)$, and multi-parent synergy is defined as $\Delta = R^2_+ - \max(m, 0)^2$.