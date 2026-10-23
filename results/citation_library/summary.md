# In-library citation analysis

Each **library_cite_count** is how many papers *in your library* list this work as a reference.

- Min count: **2**
- Ranked works: **168**
- Candidates not in library (have arXiv id): **14**

## Top overall

| # | Count | In lib | arXiv | Title |
|--:|------:|:------:|-------|-------|
| 1 | 8 | yes | 1202.2745 | Multi-column deep neural networks for image classification |
| 2 | 6 | yes | 1311.2901 | Visualizing and Understanding Convolutional Networks |
| 3 | 6 | yes | 1311.2524 | Rich Feature Hierarchies for Accurate Object Detection and Semantic Se |
| 4 | 6 |  | — | ImageNet: A large-scale hierarchical image database |
| 5 | 5 |  | 1312.6229 | OverFeat: Integrated Recognition, Localization and Detection using Con |
| 6 | 5 |  | — | Backpropagation Applied to Handwritten Zip Code Recognition |
| 7 | 5 |  | 1310.1531 | DeCAF: A Deep Convolutional Activation Feature for Generic Visual Reco |
| 8 | 4 |  | — | Gradient-based learning applied to document recognition |
| 9 | 4 |  | — | Deep Neural Networks for Acoustic Modeling in Speech Recognition: The  |
| 10 | 4 |  | — | Greedy Layer-Wise Training of Deep Networks |
| 11 | 4 |  | — | Improving the Fisher Kernel for Large-Scale Image Classification |
| 12 | 4 |  | — | Adaptive Subgradient Methods for Online Learning and Stochastic Optimi |
| 13 | 4 |  | — | Learning Multiple Layers of Features from Tiny Images |
| 14 | 4 |  | — | Context-Dependent Pre-Trained Deep Neural Networks for Large-Vocabular |
| 15 | 4 |  | — | A Neural Probabilistic Language Model |
| 16 | 4 |  | — | Understanding the difficulty of training deep feedforward neural netwo |
| 17 | 3 | yes | 1207.0580 | Improving neural networks by preventing co-adaptation of feature detec |
| 18 | 3 |  | — | Unbiased look at dataset bias |
| 19 | 3 |  | — | Algorithms for Hyper-Parameter Optimization |
| 20 | 3 |  | — | Learning representations by back-propagating errors |
| 21 | 3 |  | — | Tiled convolutional neural networks |
| 22 | 3 |  | — | Why Does Unsupervised Pre-training Help Deep Learning? |
| 23 | 3 |  | 1405.3531 | Return of the Devil in the Details: Delving Deep into Convolutional Ne |
| 24 | 3 |  | — | Random Search for Hyper-Parameter Optimization |
| 25 | 3 |  | — | Contractive Auto-Encoders: Explicit Invariance During Feature Extracti |
| 26 | 3 |  | — | What is the best multi-stage architecture for object recognition? |
| 27 | 3 |  | — | On Random Weights and Unsupervised Feature Learning |
| 28 | 3 |  | — | Long Short-Term Memory |
| 29 | 3 |  | — | Large Scale Distributed Deep Networks |
| 30 | 3 |  | — | A Stochastic Approximation Method |
| 31 | 3 |  | — | Learning hierarchical invariant spatio-temporal features for action re |
| 32 | 3 |  | — | Deep learning via Hessian-free optimization |
| 33 | 3 |  | — | Convolutional Learning of Spatio-temporal Features |
| 34 | 3 |  | 1312.6026 | How to Construct Deep Recurrent Neural Networks |
| 35 | 3 |  | — | A unified architecture for natural language processing: deep neural ne |
| 36 | 3 |  | — | Caltech-256 Object Category Dataset |
| 37 | 3 |  | — | Practical Variational Inference for Neural Networks |
| 38 | 3 |  | — | Visualizing Higher-Layer Features of a Deep Network |
| 39 | 3 |  | — | Deep Sparse Rectifier Neural Networks |
| 40 | 3 |  | — | The Importance of Encoding Versus Training with Sparse Coding and Vect |

## Top candidates to add (not in library)

| # | Count | arXiv | Title |
|--:|------:|-------|-------|
| 1 | 5 | 1312.6229 | OverFeat: Integrated Recognition, Localization and Detection using Con |
| 2 | 5 | 1310.1531 | DeCAF: A Deep Convolutional Activation Feature for Generic Visual Reco |
| 3 | 3 | 1405.3531 | Return of the Devil in the Details: Delving Deep into Convolutional Ne |
| 4 | 3 | 1312.6026 | How to Construct Deep Recurrent Neural Networks |
| 5 | 3 | 1112.6209 | Building high-level features using large scale unsupervised learning |
| 6 | 3 | 1211.5063 | On the difficulty of training recurrent neural networks |
| 7 | 3 | 1302.4389 | Maxout Networks |
| 8 | 3 | 1103.0398 | Natural Language Processing (Almost) from Scratch |
| 9 | 3 | 1010.3467 | Fast Inference in Sparse Coding Algorithms with Applications to Object |
| 10 | 3 | 1312.5402 | Some Improvements on Deep Convolutional Neural Network Based Image Cla |
| 11 | 2 | 1205.2662 | On Smoothing and Inference for Topic Models |
| 12 | 2 | 1211.4246 | What regularized auto-encoders learn from the data-generating distribu |
| 13 | 2 | 0912.1128 | How to Explain Individual Classification Decisions |
| 14 | 2 | 1306.1091 | Deep Generative Stochastic Networks Trainable by Backprop |

## Files

- Full ranking: `results/citation_library/citation_counts.csv`
- Candidates CSV: `results/citation_library/candidates_not_in_library.csv`
- arXiv ids (for ingest): `results/citation_library/candidates_arxiv_ids.txt`
- Importance corpus JSON: `results/citation_library/importance_corpus.json`

## Next steps

```bash
# Add top candidates into the main library
uv run python src/corpus_builder.py from-ids $(head -50 results/citation_library/candidates_arxiv_ids.txt | tr '\n' ' ')
# Or: merge importance_corpus.json manually / rebuild index
# Ingest full text for Q&A:
uv run python -m src.rag.cli ingest $(head -20 results/citation_library/candidates_arxiv_ids.txt | tr '\n' ' ')
```
