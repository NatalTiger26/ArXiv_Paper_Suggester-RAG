# C. Top-1 ≠ expected (sample for human judgment)

Pool of mismatches: 52; sample n=20; seed=42

## 1. rev_13 (definition)
**Q:** What is meant by mechanistic interpretability for neural algorithmic reasoning?
**Expected:** [{'paper_id': '2501.16496', 'title': 'Open Problems in Mechanistic Interpretability'}]
**Top-1:** 2506.18852 — Mechanistic Interpretability Needs Philosophy (score=0.8727)
**Snippet:** How and why do artificial neural networks produce the outputs they do? Since the resurgence of deep learning approximately a decade ago, this question has driven various efforts to interpret and explain AI systems. Among these efforts, mechanistic interpretability (MI) has emerged as an increasingly

## 2. op_06 (paraphrase)
**Q:** Why might connecting generalization theories like the NTK to interpretability help MI?
**Expected:** [{'paper_id': '2501.16496', 'title': 'Open Problems in Mechanistic Interpretability'}, {'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2506.18852 — Mechanistic Interpretability Needs Philosophy (score=0.8157)
**Snippet:** One major hope for MI is that it can help to detect unsafe or misaligned model processing that is not evident from outward behaviour [Amodei, 2025]. Here, too, philosophy can make important contributions to MI research. While there are many forms of unsafe or misaligned model processing, 

> 10Philo

## 3. ntk_02 (definition)
**Q:** What happens to the NTK in the infinite-width limit?
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2106.10165 — The Principles of Deep Learning Theory (score=0.8428)
**Snippet:** and a special name: the **frozen NTK** . The frozen NTK controls the training dynamics in the infinite-width limit, which we will investigate in detail next chapter.<sup>1</sup> 

> 1Typically in the literature, the _neural tangent kernel_ or NTK refers to this deterministic infinitewidth NTK mean Θ

## 4. pad_3 (paraphrase)
**Q:** Explain the significance of the neural tangent kernel for understanding deep learning (variant 3).
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2406.04606 — Helpful or Harmful Data? Fine-tuning-free Shapley Attribution for Explaining Language Model Predictions (score=0.8264)
**Snippet:** The neural tangent kernel (NTK) theory is proposed to study the training dynamics of infinite-width neural networks (Jacot et al., 2018) , and has been extended to other applications, such as active learning (Hemachandra et al., 2023; Lau et al., 2024). Existing works (Arora et al., 2019; Liu et al.

## 5. cmp_01 (comparison)
**Q:** How do automated circuit discovery and open problems in MI relate to each other?
**Expected:** [{'paper_id': '2304.14997', 'title': 'Towards Automated Circuit Discovery for Mechanistic Interpretability'}, {'paper_id': '2501.16496', 'title': 'Open Problems in Mechanistic Interpretability'}]
**Top-1:** 2602.16823 — Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees (score=0.8421)
**Snippet:** **Circuit discovery.** Our work joins recent efforts on circuit discovery in MI (Olah et al., 2020; Elhage et al., 2021; Dunefsky et al., 2024), particularly those advancing _automated_ algorithms (Conmy et al., 2023; Hsu et al., 2024). Other relevant avenues include metrics for circuit faithfulness

## 6. rev_02 (open_problem)
**Q:** What safety motivations are given for mechanistic interpretability?
**Expected:** [{'paper_id': '2404.14082', 'title': 'Mechanistic Interpretability for AI Safety -- A Review'}]
**Top-1:** 2211.00593 — Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (score=0.8442)
**Snippet:** ossible model behaviors. Some researchers have argued that interpretability is critical for the safe deployment of advanced machine learning systems (Hendrycks & Mazeika, 2022). 

Work in mechanistic interpretability aims to discover, understand, and verify the algorithms that model weights implemen

## 7. acdc_07 (method)
**Q:** What are limitations of purely manual circuit analysis that automation aims to fix?
**Expected:** [{'paper_id': '2304.14997', 'title': 'Towards Automated Circuit Discovery for Mechanistic Interpretability'}]
**Top-1:** 2501.16496 — Open Problems in Mechanistic Interpretability (score=0.7864)
**Snippet:** 1. Can we improve on AI automated feature description and validation methods? 

   - a. Through automating the generation and testing of arbitrary hypotheses? 

   - b. Through describing differences between features? 

   - c. Through descriptions of how components interact? 

2. Can we improve on 

## 8. op_10 (open_problem)
**Q:** What open issues remain around validating that an interpretation is faithful?
**Expected:** [{'paper_id': '2501.16496', 'title': 'Open Problems in Mechanistic Interpretability'}, {'paper_id': '2407.13594', 'title': 'Validating Mechanistic Interpretations: An Axiomatic Approach'}]
**Top-1:** 2411.17992 — New Faithfulness-Centric Interpretability Paradigms for Natural Language Processing (score=0.8334)
**Snippet:** en removal cause out-of-distribution issues, what is a significant change, and what if multiple tokens need to be removed for the prediction to change (Chapter 3 and Chapter 4)? 

Besides the challenges of the faithfulness metric, there is also the fundamental question of when an interpretability me

## 9. op_04 (open_problem)
**Q:** How does sparse dictionary learning relate to neural network decomposition in MI?
**Expected:** [{'paper_id': '2501.16496', 'title': 'Open Problems in Mechanistic Interpretability'}]
**Top-1:** 2608.31084 — Verbalizing Multi-Token Concepts in LLMs (score=0.8068)
**Snippet:** Sparse autoencoders (SAEs) learn a dictionary that represents a hidden state as a sparse combination of learned feature directions (Huben et al., 2024). The feature vocabulary is the set of learned latent coordinates, and interpretability is studied by examining the examples and behavior associated 

## 10. op_11 (open_problem)
**Q:** What problems arise when features are not cleanly disentangled in neural networks?
**Expected:** [{'paper_id': '2501.16496', 'title': 'Open Problems in Mechanistic Interpretability'}]
**Top-1:** 2103.11251 — Interpretable Machine Learning: Fundamental Principles and 10 Grand Challenges (score=0.8312)
**Snippet:** latent spaces where concepts are more “pure” than those of a standard DNN, without hurting accuracy. With the disentangled latent space, one can answer questions about how 

29 

the network gradually learns concepts over layers (Chen et al., 2020), or one could interact with the model by intervenin

## 11. ntk_12 (method)
**Q:** What does the NTK say about generalization of over-parameterized networks?
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2201.02177 — Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (score=0.8364)
**Snippet:** The generalization of overparameterized neural networks has long been a source of interest to the machine learning community since it defies intuitions derived from classical learning theory. In this paper we show that training networks on small algorithmically generated datasets can reliably exhibi

## 12. acdc_08 (method)
**Q:** What kinds of transformer behaviors have been reverse-engineered by circuit-style analysis?
**Expected:** [{'paper_id': '2304.14997', 'title': 'Towards Automated Circuit Discovery for Mechanistic Interpretability'}]
**Top-1:** 2407.08734 — Transformer Circuit Faithfulness Metrics are not Robust (score=0.8271)
**Snippet:** **Circuit Analysis** . Circuit analysis is a form of post-hoc interpretability focused on understanding the full end-to-end learned algorithm responsible for some specified narrow behaviour. A circuit is a subgraph of the full computational graph of the model that (is alleged to) implement some prec

## 13. para_07 (paraphrase)
**Q:** Summarize why infinite width makes training dynamics more tractable.
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2106.10165 — The Principles of Deep Learning Theory (score=0.8259)
**Snippet:** raining error even further. 

Recall from §10.2 that, in order to fully train an infinite-width network in a single step, we needed to make a _second-order update_ of the form 



which we interpreted either _(i)_ as a _generalized training algorithm_ (10.19) optimizing the standard MSE loss (10.5),

## 14. ntk_03 (definition)
**Q:** How is the limiting NTK defined recursively for deep networks?
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2106.10165 — The Principles of Deep Learning Theory (score=0.8693)
**Snippet:** and a special name: the **frozen NTK** . The frozen NTK controls the training dynamics in the infinite-width limit, which we will investigate in detail next chapter.<sup>1</sup> 

> 1Typically in the literature, the _neural tangent kernel_ or NTK refers to this deterministic infinitewidth NTK mean Θ

## 15. pad_5 (paraphrase)
**Q:** Explain the significance of the neural tangent kernel for understanding deep learning (variant 5).
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2406.04606 — Helpful or Harmful Data? Fine-tuning-free Shapley Attribution for Explaining Language Model Predictions (score=0.8291)
**Snippet:** The neural tangent kernel (NTK) theory is proposed to study the training dynamics of infinite-width neural networks (Jacot et al., 2018) , and has been extended to other applications, such as active learning (Hemachandra et al., 2023; Lau et al., 2024). Existing works (Arora et al., 2019; Liu et al.

## 16. cmp_06 (comparison)
**Q:** Compare review-style surveys of MI with primary method papers on circuit discovery.
**Expected:** [{'paper_id': '2404.14082', 'title': 'Mechanistic Interpretability for AI Safety -- A Review'}, {'paper_id': '2304.14997', 'title': 'Towards Automated Circuit Discovery for Mechanistic Interpretability'}]
**Top-1:** 2602.16823 — Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees (score=0.8563)
**Snippet:** **Circuit discovery.** Our work joins recent efforts on circuit discovery in MI (Olah et al., 2020; Elhage et al., 2021; Dunefsky et al., 2024), particularly those advancing _automated_ algorithms (Conmy et al., 2023; Hsu et al., 2024). Other relevant avenues include metrics for circuit faithfulness

## 17. acdc_06 (identifier)
**Q:** What problem does arXiv 2304.14997 address?
**Expected:** [{'paper_id': '2304.14997', 'title': 'Towards Automated Circuit Discovery for Mechanistic Interpretability'}]
**Top-1:** 2604.18094 — Decision-Aware Attention Propagation for Vision Transformer Explainability (score=0.7829)
**Snippet:** Code is available at https://anonymous.4open.science/r/dap-395D

## 18. pad_1 (paraphrase)
**Q:** Explain the significance of the neural tangent kernel for understanding deep learning (variant 1).
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2406.04606 — Helpful or Harmful Data? Fine-tuning-free Shapley Attribution for Explaining Language Model Predictions (score=0.8275)
**Snippet:** The neural tangent kernel (NTK) theory is proposed to study the training dynamics of infinite-width neural networks (Jacot et al., 2018) , and has been extended to other applications, such as active learning (Hemachandra et al., 2023; Lau et al., 2024). Existing works (Arora et al., 2019; Liu et al.

## 19. id_05 (identifier)
**Q:** What is 2406.11779 about?
**Expected:** [{'paper_id': '2406.11779', 'title': 'Compact Proofs of Model Performance via Mechanistic Interpretability'}]
**Top-1:** 2508.06656 — ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering (score=0.7864)
**Snippet:** fmyIP 7 C <latexish1 _ b64="dA3rupKO72/PEZ8BXY>cVNSwMJ+TCRU 9 qLvDgozWkFHmG05QyjfnI 36 C <latexish1 _ b64="Z7pDW3BQ8VvmGykrL>AncTgJEOzF+I9jHRYdw u UNoXfPCSq2/K0M5 2 C <latexish1 _ b64="DzqvM3mfpHoW/YdcKQ>AB7XVNSwEJ2+TCRU9LgkZ8F G 05yjrunOIP 13<br>C <latexish1 _ b64="DrQJBuwKHC+dFjVgW8>A7cNSE3Uq/9L0o

## 20. pad_7 (paraphrase)
**Q:** Explain the significance of the neural tangent kernel for understanding deep learning (variant 7).
**Expected:** [{'paper_id': '1806.07572', 'title': 'Neural Tangent Kernel: Convergence and Generalization in Neural Networks'}]
**Top-1:** 2406.04606 — Helpful or Harmful Data? Fine-tuning-free Shapley Attribution for Explaining Language Model Predictions (score=0.8256)
**Snippet:** The neural tangent kernel (NTK) theory is proposed to study the training dynamics of infinite-width neural networks (Jacot et al., 2018) , and has been extended to other applications, such as active learning (Hemachandra et al., 2023; Lau et al., 2024). Existing works (Arora et al., 2019; Liu et al.
