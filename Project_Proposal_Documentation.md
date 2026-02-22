# Graph-Based Social Network Analysis for Misinformation Detection

> **Project Proposal Documentation**

---

## 1. Abstract

The rapid spread of misinformation on social media platforms poses serious social and political risks. Traditional detection systems mainly analyze the **textual content** of posts, but misinformation often spreads through **coordinated social interactions** rather than through text alone.

This project proposes a **graph-based framework** that models social media as a network of interacting users to detect misinformation. Users are represented as **nodes**, and interactions such as reposts, replies, or mentions form **edges** (connections) between them. By extracting network features — such as community structures, influence patterns, and information propagation behavior — machine learning models can identify suspicious spreading patterns associated with misinformation campaigns.

The system aims to improve detection accuracy by **combining content and interaction patterns**, enabling earlier identification of coordinated misinformation networks.

---

## 2. Problem Statement

Most existing fake news detection systems focus on a single question:

> *"What is written in the post?"*

This project shifts the perspective to ask:

- **How** is the post spreading?
- **Who** is spreading it?
- **Does the spread pattern look suspicious?**

Misinformation often spreads through:

| Mechanism | Description |
|---|---|
| **Coordinated groups** | Organized clusters of accounts amplifying the same content |
| **Bot-like accounts** | Automated or semi-automated profiles designed to boost reach |
| **Highly connected hubs** | Influential accounts that serve as amplification points |
| **Sudden rapid resharing** | Abnormal spikes in sharing velocity compared to organic news |

**The core insight:** By analyzing *interaction patterns* rather than just text, we can detect misinformation that would otherwise evade content-based classifiers.

---

## 3. Core Concept — Social Media as a Graph

The foundation of this project is converting social media data into a **network (graph) structure**.

### 3.1 Graph Structure

| Element | Representation |
|---|---|
| **Nodes** | Users / Accounts |
| **Edges** | Interactions between users |

### 3.2 Types of Interactions (Edges)

- Retweets / Shares
- Mentions (`@user`)
- Replies
- Comments
- Following relationships

> **Example:** If User A retweets User B, an edge is created connecting them in the graph.

### 3.3 How Misinformation Differs Structurally

| Feature | Normal News | Misinformation |
|---|---|---|
| **Spread rate** | Gradual, organic | Rapid, bursty |
| **Sharer diversity** | Diverse set of users | Same group repeatedly shares |
| **Community structure** | Less clustering | Dense, suspicious communities |
| **Control** | Distributed | Few accounts control spread |

These **structural differences** are the detection signals this project leverages.

---

## 4. Novelty & Research Gaps Addressed

Based on a thorough literature review, the following key research gaps are identified, which this project aims to address:

### 4.1 Static vs. Dynamic Graphs
Most GNN-based fake-news models treat the interaction graph as **static**, even though misinformation cascades **evolve over time**. Temporal dynamics are often reduced to simple features (e.g., time to first 100 retweets) rather than full dynamic graph modeling. There is limited use of **temporal GNNs** or **diffusion-aware architectures** that explicitly model the sequence and timing of interactions.

### 4.2 Graph Construction Choices — Under-explored
Many works build graphs with **heuristic edge definitions** and simple weights, with little ablation on how graph topology choices affect performance. Trade-offs between:
- Different **node types** (users vs. posts vs. publishers)
- **Edge semantics** (retweet vs. reply vs. co-engagement)
- **Edge weighting schemes** (frequency, recency, rumor-spread rate)

...remain under-studied.

### 4.3 Limited Multimodal & Multi-View Integration
Classic methods often use *either* content *or* network, or concatenate hand-crafted features. Recent hybrid GPT-GNN and multimodal GNN models show that **explicitly combining text embeddings with graph structure yields sizable gains**, but principled fusion strategies (late vs. early fusion, attention over modalities) and explainable cross-modal reasoning are still emergent.

### 4.4 Poor Generalization Across Domains
Many models are trained and tested within a **single domain** (e.g., US politics or COVID-19), risking overfitting to topical cues. Cross-event and cross-language evaluations are still relatively sparse.

### 4.5 Label & Dataset Limitations
- Labeling often depends on **fact-checker platforms** (PolitiFact, Snopes), creating label bias and limited coverage.
- Graph datasets with full propagation trees and rich user metadata remain **relatively few**.
- Platform API restrictions make updated large-scale graph collection challenging.

### 4.6 Explainability & Coordinated Behavior Detection
Most GNN-based fake news detectors still act as **black boxes**. Coordinated inauthentic behavior (botnets, astroturfing) is usually **not explicitly modeled** as higher-order structures (dense near-cliques, repeated co-retweet patterns), leaving room for richer **graph motif and subgraph pattern analysis**.

---

## 5. Literature Review — Critical Analysis

### Paper 1: Graph-Theory SEIR Health Misinformation in Aging Populations
**Key Weaknesses Identified:**
- Weak empirical grounding — SEIR transmission probabilities lack specified priors, likelihoods, or validation metrics
- Insufficient dataset transparency — no description of size, timeframe, labeling pipeline, or age inference methodology
- Simulated interventions not tested against real-world platform policies
- Older adults are treated via overly generic parameters with no subgroup modeling
- Ethics, privacy, and governance are largely unaddressed despite targeting a vulnerable group

### Paper 2: Knowledge-Based Fake News Detection with BERT + GCN/GNN
**Key Weaknesses Identified:**
- Core GCN/GNN contribution is **not empirically realized** — only BERT and GPT-3 results are reported
- Twitter dataset is sparsely described and only available "on request"
- No ablation study to justify the contribution of SPO triples, sentiment, and topic components
- Multimodal (image + text) side is only sketched conceptually, not tested
- No analysis of scalability, streaming robustness, or adversarial perturbations

### Paper 3: Contextual + Sentiment + Social-Credibility Hybrid Fake News Detector
**Key Weaknesses Identified:**
- Frequently misclassifies **satire** and ambiguous news as fake
- Cannot reliably handle image/video-driven misinformation or meme-based campaigns
- Complex ensemble architecture (transformers, CNN, LSTM, GNN, RF, XGBoost) with **no XAI tooling**
- Only validated on English, news-centric datasets (LIAR, FakeNewsNet, BuzzFeedNews)
- No analysis of inference latency, scalability, or adversarial robustness

---

## 6. Datasets

### 6.1 FibVID — Fake News Diffusion Dataset (COVID-19 Infodemic, Twitter)
| Attribute | Details |
|---|---|
| **Claims** | 1,353 news claims (True/False from Snopes, PolitiFact, etc.) |
| **Tweets** | 221,253 related tweets/retweets |
| **Users** | 144,741 users (pseudonymized) |
| **Data** | Tweet IDs, user IDs, propagation depths, claim labels |
| **Graph potential** | Claim-level diffusion trees, user–user retweet graphs, user–claim bipartite graphs with temporal info |
| **Link** | [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9759652/) |

### 6.2 MiDe22 — Multi-Event Tweet Dataset for Misinformation Detection
| Attribute | Details |
|---|---|
| **Tweets** | 10,348 tweets (English & Turkish) |
| **Labels** | True / False / Other |
| **Events** | 40 events (COVID-19, Russia–Ukraine war, refugees, etc.) |
| **Engagement data** | Likes, replies, retweets, quotes, media indicators |
| **Graph potential** | User–tweet and user–user interaction graphs via engagement expansion |
| **Link** | [arXiv Paper](https://arxiv.org/html/2210.05401v2) |

### 6.3 COVID-19 Misinformation Twitter Dataset
| Attribute | Details |
|---|---|
| **Source** | GitHub-hosted dataset |
| **Content** | Tweet IDs, labels, misinformation categories |
| **Graph potential** | Retweet/mention/reply graph reconstruction via Twitter API |
| **Link** | [GitHub Repository](https://github.com/Gautamshahi/Misinformation_COVID-19) |

---

## 7. Technology Stack

| Component | Technology |
|---|---|
| **Graph Neural Networks** | GAT (Graph Attention Networks), GraphSAGE |
| **Text Embeddings** | BERT, RoBERTa |
| **Traditional ML Baselines** | Random Forest, XGBoost, MLP |
| **Graph Construction** | NetworkX, PyTorch Geometric |
| **Deep Learning Framework** | PyTorch |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Gephi |

---

## 8. Evaluation Metrics & Baselines

### 8.1 Metrics

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness; hybrid/GNN methods often achieve ~97% |
| **F1 Score (macro/weighted)** | Balances performance on both fake and real classes; reported 90–96% on Twitter15/16, PHEME, FakeNewsNet |
| **ROC AUC** | Ranking quality assessment; ATA-GNN reports 93–97% |
| **Precision & Recall (fake class)** | Critical when fake content is minority class |
| **AUCPR** | Recommended for rare/emerging events with severe class imbalance |
| **Detection Delay** | Time to flag relative to first post — valuable for campaign detection |

### 8.2 Evaluation Matrix (Baseline Comparisons)

| Model Type | Description |
|---|---|
| **Text-only** | BERT or RoBERTa fine-tuned on posts |
| **Network-only** | Graph features + Random Forest / XGBoost |
| **Simple Fusion** | Text features + graph features in an MLP |
| **Full GNN** | Heterogeneous GAT/GraphSAGE with text node features |

---

## 9. Methodology — How We Proceed

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Data Collection & Preprocessing                │
│  • Acquire datasets (FibVID, MiDe22, COVID-19 dataset)  │
│  • Clean, deduplicate, and normalize tweet data         │
│  • Extract user interaction metadata                    │
├─────────────────────────────────────────────────────────┤
│  Step 2: Graph Construction                             │
│  • Define nodes (users/accounts)                        │
│  • Define edges (retweets, mentions, replies, comments) │
│  • Assign edge weights (frequency, recency, etc.)       │
├─────────────────────────────────────────────────────────┤
│  Step 3: Feature Engineering                            │
│  • Extract graph features (degree, centrality, clusters)│
│  • Generate text embeddings (BERT/RoBERTa)              │
│  • Compute temporal features (spread velocity, delay)   │
├─────────────────────────────────────────────────────────┤
│  Step 4: Model Development                              │
│  • Implement baseline models (text-only, network-only)  │
│  • Build GNN models (GAT, GraphSAGE)                    │
│  • Design fusion strategy (text + graph features)       │
├─────────────────────────────────────────────────────────┤
│  Step 5: Training & Evaluation                          │
│  • Train on labeled datasets                            │
│  • Evaluate using Accuracy, F1, AUC, Precision, Recall  │
│  • Compare against baselines in evaluation matrix       │
├─────────────────────────────────────────────────────────┤
│  Step 6: Analysis & Reporting                           │
│  • Analyze structural patterns in detected misinfo      │
│  • Visualize suspicious communities and propagation     │
│  • Document findings and potential improvements         │
└─────────────────────────────────────────────────────────┘
```

---

## 10. References

1. *Advanced Text Analytics — Graph Neural Network for Fake News Detection in Social Media* — [arXiv](https://www.arxiv.org/pdf/2502.16157)
2. *A Graph Network Approach to Disinformation Detection in Social Media* — [ResearchGate](https://www.researchgate.net/publication/391973703)
3. *Context-Based Fake News Detection using Graph Based Approach: A COVID-19 Use-case* — [ACM DL](https://dl.acm.org/doi/10.1145/3729706.3729819)
4. *Fake news detection: A survey of graph neural network methods* — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036155/)
5. *Misinformation detection based on news dispersion* — [IEEE](https://ieeexplore.ieee.org/document/10167997)
6. Gangireddy et al., *Unsupervised Fake News Detection: A Graph-based Approach (GTUT)*, 2020 — [PDF](https://pureadmin.qub.ac.uk/ws/files/212663108/ht20_crc.pdf)
7. Spanakis group, *Fake News Detection on Twitter Using Propagation Patterns*, 2020 — [PDF](https://dke.maastrichtuniversity.nl/jerry.spanakis/wp-content/uploads/2020/11/10.1007@978-3-030-61841-4_MMGWJS.pdf)
8. Hiremath et al., *Analysis of Fake News Detection using Graph Neural Networks*, 2023 — [IEEE](https://ieeexplore.ieee.org/document/10405304/)
9. Gul et al., *Advancing fake news detection with graph neural network and deep learning techniques*, 2025 — [PDF](https://zuscholars.zu.ac.ae/cgi/viewcontent.cgi?article=8253&context=works)
10. Orugboh, Ezeogu & Juba, *A graph theory approach to modeling the spread of health misinformation in aging populations on social media platforms*, MJH, vol. 2, no. 1, Aug. 2025 — [Link](https://researchcorridor.org/index.php/mjh/article/view/503)
11. Arshad, Manzoor & Hassan, *Fake news detection via textual BERT embeddings and knowledge-aware graph neural networks*, MJH, vol. 1, no. 1, Aug. 2025 — [Link](https://gajcet.com/index.php/gajcet/article/view/16)
12. Bellam & Aakula, *Contextual and Sentiment-Based Hybrid Models for Detecting Fake News on Social Networks*, IEEE AIC 2025 — [IEEE](https://ieeexplore.ieee.org/abstract/document/11212091)

---

*Document prepared for Foundations of Data Science (FDS) course project proposal.*
