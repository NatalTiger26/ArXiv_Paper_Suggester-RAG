# Batch Novelty Subspace Analysis Report

**Generated:** 2026-10-26T22:19:51.291561+00:00  
**Papers Evaluated:** 50  
**Success Rate:** 20 / 50 (40.0%)

## 1. Aggregate Geometric Metrics

| Metric | Definition | Cohort Mean | Interpretation |
| :--- | :--- | :--- | :--- |
| **m** | Cosine to closest parent | `0.8903` | Baseline anchor strength to closest single prior work |
| **Δ (Delta)** | Multi-parent cone synergy | `0.0825` | Energy gain from non-negative combination of multiple parents |
| **R²₊** | Cone reconstruction energy | `0.8761` | Proportion of embedding explained by the reference subspace |
| **√(1 - R²₊)** | Novelty residual gap | `0.3478` | Orthogonal departure distance from reference cone to unit circle |

## 2. Discovery Mode Distribution

| Discovery Classification | Paper Count | Percentage |
| :--- | :---: | :---: |
| **High Synergy Cone Blend** | 17 | 85.0% |
| **Moderate Synthesis** | 3 | 15.0% |

## 3. Top Synergistic Syntheses (Highest Multi-Parent Synergy Δ)
Papers that achieve dramatic innovation by non-linearly fusing multiple distinct research threads:

| arXiv ID | Title | Year | Citations | Δ (Synergy) | m (Anchor) | R²₊ | Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| [1901.09392](https://arxiv.org/abs/1901.09392) | On the (In)fidelity and Sensitivity for Expla... | 2019 | 651 | **0.1334** | 0.839 | 0.838 | High Synergy Cone Blend |
| [1807.03903](https://arxiv.org/abs/1807.03903) | Deep Imbalanced Attribute Classification usin... | 2018 | 245 | **0.1128** | 0.885 | 0.896 | High Synergy Cone Blend |
| [1904.02323](https://arxiv.org/abs/1904.02323) | Untitled... | — | 242 | **0.1074** | 0.857 | 0.842 | High Synergy Cone Blend |
| [1512.00570](https://arxiv.org/abs/1512.00570) | Attribute2Image: Conditional Image Generation... | 2015 | 810 | **0.1057** | 0.843 | 0.816 | High Synergy Cone Blend |
| [1806.02724](https://arxiv.org/abs/1806.02724) | Untitled... | — | 679 | **0.1034** | 0.879 | 0.876 | High Synergy Cone Blend |
| [1704.05858](https://arxiv.org/abs/1704.05858) | The Hyper Suprime-Cam SSP Survey: Overview an... | 2017 | 679 | **0.0987** | 0.843 | 0.810 | High Synergy Cone Blend |
| [1711.08998](https://arxiv.org/abs/1711.08998) | Visual Feature Attribution using Wasserstein ... | 2017 | 160 | **0.0959** | 0.891 | 0.891 | High Synergy Cone Blend |
| [1806.06176](https://arxiv.org/abs/1806.06176) | Untitled... | — | 578 | **0.0915** | 0.895 | 0.893 | High Synergy Cone Blend |
| [1803.08024](https://arxiv.org/abs/1803.08024) | Stacked Cross Attention for Image-Text Matchi... | 2018 | 1,441 | **0.0860** | 0.873 | 0.848 | High Synergy Cone Blend |
| [1510.02927](https://arxiv.org/abs/1510.02927) | DeepFix: A Fully Convolutional Neural Network... | 2015 | 516 | **0.0858** | 0.907 | 0.908 | High Synergy Cone Blend |

## 4. Top Subspace Outliers / Paradigm Shifts (Highest Residual √(1 - R²₊))
Papers with large orthogonal departures that cannot be reconstructed from their references:

| arXiv ID | Title | Year | Citations | Novelty Residual | R²₊ | m (Anchor) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [1704.05858](https://arxiv.org/abs/1704.05858) | The Hyper Suprime-Cam SSP Survey: Overview an... | 2017 | 679 | **0.4362** | 0.810 | 0.843 |
| [1512.00570](https://arxiv.org/abs/1512.00570) | Attribute2Image: Conditional Image Generation... | 2015 | 810 | **0.4294** | 0.816 | 0.843 |
| [1901.09392](https://arxiv.org/abs/1901.09392) | On the (In)fidelity and Sensitivity for Expla... | 2019 | 651 | **0.4024** | 0.838 | 0.839 |
| [1904.02323](https://arxiv.org/abs/1904.02323) | Untitled... | — | 242 | **0.3975** | 0.842 | 0.857 |
| [1607.06215](https://arxiv.org/abs/1607.06215) | A Comprehensive Survey on Cross-modal Retriev... | 2016 | 336 | **0.3941** | 0.845 | 0.873 |
| [1803.08024](https://arxiv.org/abs/1803.08024) | Stacked Cross Attention for Image-Text Matchi... | 2018 | 1,441 | **0.3902** | 0.848 | 0.873 |
| [1812.11788](https://arxiv.org/abs/1812.11788) | Untitled... | — | 1,111 | **0.3859** | 0.851 | 0.877 |
| [1904.11223](https://arxiv.org/abs/1904.11223) | Untitled... | — | 146 | **0.3790** | 0.856 | 0.892 |
| [1802.00682](https://arxiv.org/abs/1802.00682) | Untitled... | — | 268 | **0.3655** | 0.866 | 0.888 |
| [1609.02077](https://arxiv.org/abs/1609.02077) | Untitled... | — | 367 | **0.3531** | 0.875 | 0.919 |

## 5. Top Foundationally Rooted Works (Highest Single Parent m)
Works that build directly and deeply on a primary ancestor paper:

| arXiv ID | Title | Year | Citations | m (Anchor) | Δ (Synergy) | Primary Parent Attribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| [1709.06316](https://arxiv.org/abs/1709.06316) | Predicting Video Saliency with Object-to... | 2017 | 125 | **0.9400** | 0.044 | `s2:25c65c87968b94631d3dcca253478d9f50806be3` (λ=0.27) |
| [1710.11063](https://arxiv.org/abs/1710.11063) | Grad-CAM++: Improved Visual Explanations... | 2017 | 3,549 | **0.9380** | 0.047 | `s2:5582bebed97947a41e3ddd9bd1f284b73f1648c2` (λ=0.50) |
| [1611.09571](https://arxiv.org/abs/1611.09571) | Predicting Human Eye Fixations via an LS... | 2016 | 613 | **0.9320** | 0.069 | `1510.02927` (λ=0.31) |
| [1701.01299](https://arxiv.org/abs/1701.01299) | Untitled... | — | 244 | **0.9281** | 0.070 | `1610.03107` (λ=0.27) |
| [1609.02077](https://arxiv.org/abs/1609.02077) | Untitled... | — | 367 | **0.9187** | 0.031 | `1503.08663` (λ=0.51) |
| [1510.02927](https://arxiv.org/abs/1510.02927) | DeepFix: A Fully Convolutional Neural Ne... | 2015 | 516 | **0.9070** | 0.086 | `1603.00845` (λ=0.28) |
| [1603.00845](https://arxiv.org/abs/1603.00845) | Shallow and Deep Convolutional Networks ... | 2016 | 499 | **0.9070** | 0.064 | `1510.02927` (λ=0.39) |
| [1806.06176](https://arxiv.org/abs/1806.06176) | Untitled... | — | 578 | **0.8954** | 0.091 | `2301.04856` (λ=0.25) |
| [1904.11223](https://arxiv.org/abs/1904.11223) | Untitled... | — | 146 | **0.8920** | 0.061 | `1212.0504` (λ=0.19) |
| [1711.08998](https://arxiv.org/abs/1711.08998) | Visual Feature Attribution using Wassers... | 2017 | 160 | **0.8914** | 0.096 | `1511.06434` (λ=0.13) |

---
### Methodology & Theoretical Foundations
Every paper embedding $\mathbf{x}$ is projected onto the non-negative convex cone of its cited references $\mathbf{Y} = [\mathbf{y}_1, \dots, \mathbf{y}_k]$ via Non-Negative Least Squares (NNLS):

$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \ge 0} \|\mathbf{x} - \mathbf{Y}\boldsymbol{\lambda}\|^2, \quad \hat{\mathbf{x}} = \mathbf{Y}\boldsymbol{\lambda}^*$$

The reconstructed energy is $R^2_+ = \|\hat{\mathbf{x}}\|^2$, the foundational anchor is $m = \max_j (\mathbf{x}^T \mathbf{y}_j)$, and multi-parent synergy is defined as $\Delta = R^2_+ - \max(m, 0)^2$.