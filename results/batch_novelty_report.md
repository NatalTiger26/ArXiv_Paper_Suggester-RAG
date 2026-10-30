# Batch Novelty Subspace Analysis Report

**Generated:** 2026-10-30T11:00:16.291341+00:00  
**Papers Evaluated:** 100  
**Success Rate:** 100 / 100 (100.0%)

## 1. Aggregate Geometric Metrics

| Metric | Definition | Cohort Mean | Interpretation |
| :--- | :--- | :--- | :--- |
| **m** | Cosine to closest parent | `0.8837` | Baseline anchor strength to closest single prior work |
| **Δ (Delta)** | Multi-parent cone synergy | `0.0789` | Energy gain from non-negative combination of multiple parents |
| **R²₊** | Cone reconstruction energy | `0.8618` | Proportion of embedding explained by the reference subspace |
| **√(1 - R²₊)** | Novelty residual gap | `0.3636` | Orthogonal departure distance from reference cone to unit circle |

## 2. Discovery Mode Distribution

| Discovery Classification | Paper Count | Percentage |
| :--- | :---: | :---: |
| **High Synergy Cone Blend** | 53 | 53.0% |
| **Moderate Synthesis** | 37 | 37.0% |
| **Linear / Incremental Extension** | 10 | 10.0% |

## 3. Top Synergistic Syntheses (Highest Multi-Parent Synergy Δ)
Papers that achieve dramatic innovation by non-linearly fusing multiple distinct research threads:

| arXiv ID | Title | Year | Citations | Δ (Synergy) | m (Anchor) | R²₊ | Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| [1705.10301](https://arxiv.org/abs/1705.10301) | Contextual Explanation Networks... | - | 92 | **0.1441** | 0.809 | 0.799 | High Synergy Cone Blend |
| [1905.04172](https://arxiv.org/abs/1905.04172) | On the Connection Between Adversarial Robustn... | 2019 | 184 | **0.1420** | 0.812 | 0.802 | High Synergy Cone Blend |
| [1811.05437](https://arxiv.org/abs/1811.05437) | Argumentation for Explainable Scheduling (Ful... | 2018 | 30 | **0.1400** | 0.844 | 0.852 | High Synergy Cone Blend |
| [1901.09392](https://arxiv.org/abs/1901.09392) | On the (In)fidelity and Sensitivity for Expla... | 2019 | 651 | **0.1334** | 0.839 | 0.838 | High Synergy Cone Blend |
| [1904.02057](https://arxiv.org/abs/1904.02057) | Interpreting Adversarial Examples by Activati... | - | 46 | **0.1278** | 0.819 | 0.799 | High Synergy Cone Blend |
| [1703.06585](https://arxiv.org/abs/1703.06585) | Learning Cooperative Visual Dialog Agents wit... | 2017 | 435 | **0.1253** | 0.862 | 0.868 | High Synergy Cone Blend |
| [1904.03367](https://arxiv.org/abs/1904.03367) | Reinforcement Learning with Attention that Wo... | - | 67 | **0.1200** | 0.688 | 0.593 | High Synergy Cone Blend |
| [1808.05054](https://arxiv.org/abs/1808.05054) | Shedding Light on Black Box Machine Learning ... | 2018 | 46 | **0.1185** | 0.854 | 0.848 | High Synergy Cone Blend |
| [1712.00840](https://arxiv.org/abs/1712.00840) | Visual Explanation by High-Level Abduction: O... | 2017 | 32 | **0.1156** | 0.883 | 0.895 | High Synergy Cone Blend |
| [1905.11833](https://arxiv.org/abs/1905.11833) | Interpreting and improving natural-language p... | - | 336 | **0.1147** | 0.851 | 0.839 | High Synergy Cone Blend |

## 4. Top Subspace Outliers / Paradigm Shifts (Highest Residual √(1 - R²₊))
Papers with large orthogonal departures that cannot be reconstructed from their references:

| arXiv ID | Title | Year | Citations | Novelty Residual | R²₊ | m (Anchor) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [1806.04398](https://arxiv.org/abs/1806.04398) | Attentive cross-modal paratope prediction... | 2018 | 67 | **0.6392** | 0.591 | 0.703 |
| [1904.03367](https://arxiv.org/abs/1904.03367) | Reinforcement Learning with Attention that Wo... | - | 67 | **0.6382** | 0.593 | 0.688 |
| [1004.4421](https://arxiv.org/abs/1004.4421) | Efficient Learning with Partially Observed At... | - | 95 | **0.5093** | 0.741 | 0.808 |
| [1706.06060](https://arxiv.org/abs/1706.06060) | Consistent feature attribution for tree ensem... | 2017 | 187 | **0.4896** | 0.760 | 0.844 |
| [1612.08795](https://arxiv.org/abs/1612.08795) | Provable learning of Noisy-or Networks... | 2016 | 28 | **0.4719** | 0.777 | 0.823 |
| [1711.02245](https://arxiv.org/abs/1711.02245) | Challenges in Disentangling Independent Facto... | 2017 | 53 | **0.4624** | 0.786 | 0.856 |
| [1811.02394](https://arxiv.org/abs/1811.02394) | DeepChannel: Salience Estimation by Contrasti... | 2018 | 31 | **0.4540** | 0.794 | 0.853 |
| [1812.06369](https://arxiv.org/abs/1812.06369) | Provable limitations of deep learning... | - | 46 | **0.4521** | 0.796 | 0.829 |
| [1904.02057](https://arxiv.org/abs/1904.02057) | Interpreting Adversarial Examples by Activati... | - | 46 | **0.4487** | 0.799 | 0.819 |
| [1705.10301](https://arxiv.org/abs/1705.10301) | Contextual Explanation Networks... | - | 92 | **0.4485** | 0.799 | 0.809 |

## 5. Top Foundationally Rooted Works (Highest Single Parent m)
Works that build directly and deeply on a primary ancestor paper:

| arXiv ID | Title | Year | Citations | m (Anchor) | Δ (Synergy) | Primary Parent Attribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| [1404.1530](https://arxiv.org/abs/1404.1530) | Provable Deterministic Leverage Score Sa... | 2014 | 88 | **1.0000** | 0.000 | `openalex:W2109818719` (λ=1.00) |
| [1801.02480](https://arxiv.org/abs/1801.02480) | Facial Attributes: Accuracy and Adversar... | 2018 | 67 | **0.9552** | 0.006 | `s2:e87a38f70420b2d4a11c0bbab55677ae139bcc1b` (λ=0.82) |
| [1810.12829](https://arxiv.org/abs/1810.12829) | Cross-Modal Attentional Context Learning... | - | 76 | **0.9438** | 0.056 | `1507.06821` (λ=0.30) |
| [1705.09425](https://arxiv.org/abs/1705.09425) | Hierarchical Cellular Automata for Visua... | 2017 | 61 | **0.9427** | 0.031 | `openalex:W1897243830` (λ=0.53) |
| [1709.06316](https://arxiv.org/abs/1709.06316) | Predicting Video Saliency with Object-to... | 2017 | 125 | **0.9400** | 0.044 | `s2:25c65c87968b94631d3dcca253478d9f50806be3` (λ=0.27) |
| [1812.09041](https://arxiv.org/abs/1812.09041) | A Multi-Task Neural Approach for Emotion... | - | 30 | **0.9390** | 0.047 | `s2:7502372bac32d9967eeb4bf68a430a74dadf748c` (λ=0.38) |
| [1710.11063](https://arxiv.org/abs/1710.11063) | Grad-CAM++: Improved Visual Explanations... | 2017 | 3,549 | **0.9380** | 0.047 | `s2:5582bebed97947a41e3ddd9bd1f284b73f1648c2` (λ=0.50) |
| [1901.05742](https://arxiv.org/abs/1901.05742) | A Temporal Attentive Approach for Video-... | - | 29 | **0.9372** | 0.041 | `1709.09930` (λ=0.17) |
| [1807.04855](https://arxiv.org/abs/1807.04855) | A feature agnostic approach for glaucoma... | 2018 | 193 | **0.9369** | 0.028 | `s2:4a8d000ef73ac5adf3317fc820cccc5332e236af` (λ=0.57) |
| [1904.04745](https://arxiv.org/abs/1904.04745) | Cross-Modal Self-Attention Network for R... | 2019 | 605 | **0.9362** | 0.040 | `1703.07939` (λ=0.38) |

---
### Methodology & Theoretical Foundations
Every paper embedding $\mathbf{x}$ is projected onto the non-negative convex cone of its cited references $\mathbf{Y} = [\mathbf{y}_1, \dots, \mathbf{y}_k]$ via Non-Negative Least Squares (NNLS):

$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \ge 0} \|\mathbf{x} - \mathbf{Y}\boldsymbol{\lambda}\|^2, \quad \hat{\mathbf{x}} = \mathbf{Y}\boldsymbol{\lambda}^*$$

The reconstructed energy is $R^2_+ = \|\hat{\mathbf{x}}\|^2$, the foundational anchor is $m = \max_j (\mathbf{x}^T \mathbf{y}_j)$, and multi-parent synergy is defined as $\Delta = R^2_+ - \max(m, 0)^2$.