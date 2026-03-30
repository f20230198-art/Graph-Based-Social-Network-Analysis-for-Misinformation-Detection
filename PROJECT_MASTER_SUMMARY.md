# PROJECT MASTER SUMMARY
## Hybrid Misinformation Detection System — Complete Project Overview

> **Course:** Foundations of Data Science (FDS)  
> **Date Compiled:** March 1, 2026  
> **Sources:** All project documents synthesized into one reference

---

## TABLE OF CONTENTS

1. [Project Identity & Core Idea](#1-project-identity--core-idea)
2. [Problem Statement](#2-problem-statement)
3. [Key Innovation — Why This Approach](#3-key-innovation--why-this-approach)
4. [Research Gaps Addressed](#4-research-gaps-addressed)
5. [Dataset](#5-dataset)
6. [System Architecture — PropNet](#6-system-architecture--propnet)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Components](#8-model-components)
9. [Training Strategy](#9-training-strategy)
10. [Evaluation](#10-evaluation)
11. [Technology Stack](#11-technology-stack)
12. [Literature Review Summary](#12-literature-review-summary)
13. [Document Alignment Analysis](#13-document-alignment-analysis)
14. [References](#14-references)

---

## 1. Project Identity & Core Idea

| Field | Detail |
|---|---|
| **Title** | Graph-Based Social Network Analysis for Misinformation Detection (Hybrid Approach) |
| **Model Name** | **PropNet** (Propagation-aware Network) |
| **Target** | Detect misinformation by jointly analyzing *what is written* + *how it spreads* |
| **Expected Performance** | **87% F1-score** (7–12% improvement over text-only baselines) |

**In simple terms:**  
Most fake news detectors ask *"What does the post say?"*  
This project asks *"How is it spreading, who is spreading it, and does the pattern look suspicious?"*

---

## 2. Problem Statement

Traditional detection systems fail because they only analyze **text content**, missing:

| Propagation Signal | Why It Matters |
|---|---|
| **Coordinated groups** | Organized clusters amplifying the same content |
| **Bot-like accounts** | Automated profiles designed to boost reach |
| **Highly connected hubs** | Influential accounts as amplification points |
| **Sudden rapid resharing** | Abnormal velocity spikes vs organic news |

### How Misinformation Looks Different in Graphs

| Feature | Normal News | Misinformation |
|---|---|---|
| **Spread rate** | Gradual, organic | Rapid, bursty (high early velocity) |
| **Sharer diversity** | Diverse users | Same group repeatedly |
| **Community structure** | Less clustering | Dense, suspicious communities |
| **Structural virality** | Tree-like spread | Broadcast from few sources (lower V) |
| **Control** | Distributed | Few accounts control spread |

---

## 3. Key Innovation — Why This Approach

**Traditional text-only detectors fail because:**
- Misinformation evolves faster than models can adapt
- Sophisticated fake news mimics legitimate writing style
- Context-dependent claims are hard to verify from text alone
- Adversaries can evade linguistic patterns

**Our hybrid solution captures:**
- **Abnormal propagation patterns** — harder to fake than text
- **Coordination signals** from bot networks and brigading
- **Echo chamber dynamics** where misinformation thrives
- **Temporal anomalies** in spread velocity
- **Cross-modal interactions** between content and structure

**Paradigm shift:** Content analysis → **Network/Graph analysis + Content**

---

## 4. Research Gaps Addressed

| # | Gap | This Project's Response |
|---|---|---|
| **1** | Most GNN models use **static graphs**, ignoring temporal dynamics | Incorporates temporal propagation features (velocity, acceleration, burst count) |
| **2** | **Graph construction choices** are under-explored (node types, edge semantics, weighting) | Systematic heterogeneous graph with 5 edge types and ablation-ready design |
| **3** | **Limited multimodal integration** — only text or network, rarely fused properly | Attention-weighted fusion of RoBERTa text + HetGAT structural embeddings |
| **4** | **Poor cross-domain generalization** (politics model fails on health) | FakeNewsNet spans PolitiFact (politics) + GossipCop (entertainment) |
| **5** | **Label and dataset limitations** (API restrictions, fact-checker bias) | Uses verified FakeNewsNet with full propagation trees |
| **6** | **Explainability** of GNN detectors lacking; coordinated behavior not modeled | Co-retweet edges, coordination scores, Louvain community detection |

---

## 5. Dataset

### Primary: FakeNewsNet (Enhanced)

**Why FakeNewsNet over alternatives:**

| Criterion | FakeNewsNet | Twitter15/16 | PHEME | Weibo |
|---|---|---|---|---|
| **Scale** | 23,196 articles, 850K+ tweets | ~1,500 events | Limited | Large but Chinese |
| **Modality** | Content + full propagation trees | Propagation only | Limited events | No English |
| **Ground Truth** | Fact-checker verified | Crowdsourced | Mixed | Crowdsourced |
| **Availability** | Public, well-documented | Public | Public | Restricted |

**Dataset Composition:**

```
FakeNewsNet
├── PolitiFact (Politics & Elections)
│   ├── Fake: 1,056 articles
│   ├── Real: 1,760 articles
│   └── Cascades: ~300K tweets
│
└── GossipCop (Entertainment & Celebrity)
    ├── Fake: 5,323 articles
    ├── Real: 16,817 articles
    └── Cascades: ~550K tweets

TOTAL: 23,196 articles | 850K+ tweets | ~2.75 GB raw data
```

### Required Data Files

| File | Contents | Size |
|---|---|---|
| `news_content.json` | Article text, title, source, date, label | ~50 MB |
| `tweets.json` | Tweet text, engagement counts, timestamps | ~2 GB |
| `users.json` | Follower counts, verified status, account age | ~500 MB |
| `retweet_edges.csv` | Source → Target user, timestamp, news_id | ~200 MB |
| `follow_edges.csv` | Follower → Followee relationships | Varies |

### Data Preprocessing

**Text cleaning:**
- Normalize URLs → `[URL]`, mentions → `[USER]` (don't remove — preserve structure)
- Keep hashtags with `#` for semantic meaning
- Language filter: English only initially
- Min text length: 10 chars | Max: 1,000 chars

**Quality filters:**
- Min cascade size: 10 interactions | Max: 10,000 (cap viral outliers)
- Remove suspended/deleted users
- Min cascade depth: 2 (at least one reshare)
- Temporal window: 7 days after publication

---

## 6. System Architecture — PropNet

```
Input → [Text Branch]  → Text Embeddings     (768-d RoBERTa)
     → [Graph Branch] → Structural Embeddings (128-d HetGAT)
     → [Fusion Layer] → Combined Representation (128-d attention-weighted)
     → [Classifier]   → P(fake | content, structure)
```

**High-Level Diagram:**

```
Raw Post Text + Metadata        Interaction Graph
         │                              │
    ─────▼─────                   ──────▼──────
   TEXT BRANCH                   GRAPH BRANCH
  RoBERTa (768-d)              HetGAT Layer 1
  Sentiment (3-d)         (Multi-head Attention, 8 heads)
  Emotion    (6-d)              HetGAT Layer 2
  Linguistic(12-d)         (GraphSAGE Aggregation)
  Claim      (8-d)           Structural MLP (65-d→128-d)
  ────────────
  Concat: 797-d
         │                              │
         └────────────┬─────────────────┘
                      │
               FUSION LAYER
          Attention-weighted (128-d)
                      │
             CLASSIFIER HEAD
               128 → 64 → 2
                      │
              [P(real), P(fake)]
```

---

## 7. Feature Engineering

### 7.1 Text Features (797-d total)

| Feature Group | Dimension | What It Captures |
|---|---|---|
| **RoBERTa embedding** | 768-d | Semantic content, writing style |
| **Sentiment** (twitter-roberta) | 3-d | Negative / Neutral / Positive probabilities |
| **Emotion** (distilroberta) | 6-d | Anger, Disgust, Fear, Joy, Sadness, Surprise |
| **Linguistic style** | 12-d | Capitalization ratio, punctuation density, readability, etc. |
| **Claim-specific** (roberta-fake-news) | 8-d | Transfer features from other fake news datasets |
| **TOTAL** | **797-d** | |

**Key findings from research:**
- Fake news is **3–5× more emotionally polarized** than real news
- **Fear + Anger combination** → 94% precision for conspiracy theories
- Fake news targets **lower reading levels** (simpler vocabulary, shorter words)
- **Structural virality score (V)** is lower for fake news (broadcast pattern, not tree-like spread)

### 7.2 Structural Features (65-d total)

**Cascade-level (25-d):**
- Depth & breadth metrics (max/avg depth, branching factor)
- Temporal dynamics: velocity_early, velocity_peak, time_to_peak, acceleration, decay_rate, burst_count
- Structural virality V = (1/n) × Σᵢⱼ d(i,j)
- Audience size, participation ratio, reshare patterns

**User-level (20-d):**
- Profile: followers, friends, account age (all log-scaled), verified flag
- Behavior: posting frequency, retweet ratio, avg engagement received
- Network position: degree centrality, PageRank, betweenness, clustering coefficient
- Bot indicators: bot_score, coordination_score, amplification_factor

**Community-level (8-d):**
- Louvain community detection, modularity score
- Community concentration (Gini coefficient — echo chamber signal)
- Cross-community edges ratio, community polarization

**Temporal (12-d):**
- 24-hour activity vector, 7-day activity vector
- Synchronized burst score (coordination indicator)
- Inter-event time standard deviation

---

## 8. Model Components

### 8.1 Heterogeneous Graph Schema

**Node types:**
- `post` — news articles/tweets (features: text embeddings, metadata)
- `user` — accounts (features: profile + behavior features)

**Edge types (multi-relational):**

| Edge Type | Direction | Weight | Meaning |
|---|---|---|---|
| `retweet` | User → Post | 1.0 | Standard propagation |
| `reply` | User → Post | 0.8 | Reactive engagement |
| `quote` | User → Post | 1.2 | Active engagement |
| `mention` | User → User | 0.5 | Cross-user signal |
| `follow` | User → User | 0.3 | Weaker social tie |
| `co_retweet` | User ↔ User | dynamic | **Coordination signal** |

### 8.2 GNN Layers

**Layer 1 — HeteroConv with GATConv:**
- 8 attention heads, hidden dim 128
- Type-specific transformation per edge type
- Activation: LeakyReLU(0.2), Dropout 0.2

**Layer 2 — HeteroConv with SAGEConv:**
- Cross-type mean aggregation
- Hidden dim 128, Dropout 0.3

**Structural MLP:** 65 → 128 → 128 (ReLU, Dropout 0.3)

### 8.3 Fusion Layer

```
h_combined = α × h_text + (1−α) × h_struct
where α = sigmoid(W × [h_text || h_struct || h_text ⊙ h_struct])
```
Learnable attention weight balances text and structure contributions dynamically.

### 8.4 Classifier Head

```
128 → (ReLU, Dropout 0.5) → 64 → (ReLU, Dropout 0.3) → 2 → Softmax
Output: [P(real), P(fake)]
```

---

## 9. Training Strategy

| Parameter | Value | Rationale |
|---|---|---|
| **Text encoder LR** | 2e-5 | Lower for pretrained RoBERTa |
| **GNN LR** | 1e-3 | Higher for randomly initialized layers |
| **Optimizer** | AdamW | Weight decay for regularization |
| **Loss** | Weighted Cross-Entropy | Handle class imbalance (GossipCop: 3:1 real:fake) |
| **Batch size** | 32 (graph mini-batches) | Memory-constrained |
| **Max epochs** | 50 with early stopping | Patience = 10 epochs |
| **Scheduler** | ReduceLROnPlateau | Halve LR when val F1 plateaus |

**Split strategy:** Temporal split (train on older data, test on newer) — prevents data leakage via time.

---

## 10. Evaluation

### Metrics

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness; GNN hybrids ~97% |
| **F1-Score (macro/weighted)** | Balance across both classes — **primary metric** |
| **ROC AUC** | Ranking quality (reported 93–97% in comparable work) |
| **Precision & Recall (fake class)** | Critical when fake is minority class |
| **AUCPR** | For severe class imbalance scenarios |
| **Detection Delay** | Time to flag relative to first post — campaign detection value |

### Evaluation Matrix (Ablation)

| Model | What It Tests |
|---|---|
| 1. Text-only (RoBERTa fine-tuned) | Baseline: content only |
| 2. Network-only (graph features + XGBoost) | Baseline: structure only |
| 3. Simple Fusion (text + graph in MLP) | Naive combination |
| 4. **Full PropNet (HetGAT + RoBERTa fusion)** | **Our model** |

**Target:** PropNet should show clear incremental gains at each step.

---

## 11. Technology Stack

| Component | Technology |
|---|---|
| **Text Encoding** | RoBERTa (`roberta-base` via HuggingFace) |
| **Graph Neural Networks** | GAT, GraphSAGE (via PyTorch Geometric) |
| **Sentiment Model** | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **Emotion Model** | `j-hartmann/emotion-english-distilroberta-base` |
| **Graph Construction** | NetworkX, PyTorch Geometric (HeteroData) |
| **Community Detection** | Louvain algorithm |
| **Deep Learning Framework** | PyTorch |
| **Traditional ML Baselines** | Random Forest, XGBoost, MLP |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Gephi |
| **Linguistic Features** | textstat (Flesch-Kincaid), spaCy (NER) |

---

## 12. Literature Review Summary

### Papers Reviewed

**Paper 1 — Graph-Theory SEIR Health Misinformation in Aging Populations**  
*Orugboh, Ezeogu & Juba, MJH 2025*  
Key weaknesses: Weak empirical grounding, no validation metrics for SEIR probabilities, no subgroup modeling, ethics unaddressed.

**Paper 2 — Knowledge-Based Fake News Detection with BERT + GCN/GNN**  
*Arshad, Manzoor & Hassan, MJH 2025*  
Key weaknesses: GCN/GNN component never empirically tested (only BERT/GPT-3 results shown), no ablation study, multimodal side only conceptual.

**Paper 3 — Contextual + Sentiment + Social-Credibility Hybrid**  
*Bellam & Aakula, IEEE AIC 2025*  
Key weaknesses: Misclassifies satire, cannot handle image/video misinformation, no XAI tooling, only validated on English news datasets.

### Additional Key References

1. *Graph Neural Network for Fake News Detection* — arXiv 2502.16157
2. *Graph Network Approach to Disinformation* — ResearchGate 2025
3. *Context-Based Fake News Detection (COVID-19)* — ACM DL
4. *Fake news detection: GNN survey* — PMC
5. *Unsupervised Fake News Detection (GTUT)* — Gangireddy et al., 2020
6. *Fake News Detection via Propagation Patterns* — Spanakis group, 2020
7. *ATA-GNN Analysis* — Hiremath et al., IEEE 2023
8. *Advancing fake news detection with GNN* — Gul et al., 2025

---

## 13. Document Alignment Analysis

### Summary of All Documents

| Document | Type | Focus |
|---|---|---|
| `info.txt` | Project seed / brief | Graph-based detection concept, team references, paper links |
| `Project_Proposal_Documentation.md` | Formal proposal | Problem, methodology, datasets (FibVID/MiDe22), lit review, tech stack |
| `Presentation_Guidelines.md` | Slide guide | 2–3 slides: problem, approach, datasets (FibVID/MiDe22), gaps |
| `Complete_Production_Plan_Hybrid_Misinformation_Detection.md` | **NEW — Implementation plan** | Full production spec: FakeNewsNet, PropNet architecture, all code |
| `Hybrid_Misinformation_Detection_System_Complete_Guide.md` | **NEW — Technical guide** | Complete guide: architecture, training, deployment, 3,325 lines |

---

### Are the Original Docs Aligned with the New Files? ✅ YES — with one notable difference

**Strongly Aligned:**

| Aspect | Original Docs | New Files | Status |
|---|---|---|---|
| Core approach | GNN + text hybrid | GNN + text hybrid (PropNet) | ✅ Same |
| Why GNNs | Propagation patterns, not just text | Same reasoning | ✅ Same |
| Text encoder | BERT / RoBERTa | RoBERTa (`roberta-base`) | ✅ Same |
| GNN models | GAT, GraphSAGE | HetGAT + GraphSAGE | ✅ Evolved version |
| Research gaps | Static graphs, multimodal, explainability, cross-domain | Identical 6 gaps | ✅ Exact match |
| Tech stack | PyTorch, PyG, NetworkX | PyTorch, PyG, NetworkX | ✅ Same |
| Graph structure | Users as nodes, retweets/replies/mentions as edges | Same + co_retweet edge | ✅ Expanded |
| Lit review papers | Same 3 papers + references | Same papers referenced | ✅ Same |
| Evaluation metrics | Accuracy, F1, AUC, Precision/Recall | Same metrics | ✅ Same |

**Notable Difference — Dataset Evolution:**

| | Original Docs | New Files |
|---|---|---|
| **Primary Dataset** | FibVID (1,353 claims, COVID-19), MiDe22 (10,348 tweets, 40 events), COVID-19 Twitter dataset | **FakeNewsNet** (23,196 articles, 850K+ tweets, PolitiFact + GossipCop) |
| **Reason for change** | Proposal-level dataset selection | FakeNewsNet has full propagation trees — better fit for the hybrid architecture |
| **Impact** | Evolution, not contradiction | FakeNewsNet is a superset capability — richer data for the same goal |

> **Verdict:** The original documents are **fully in the same direction** as the new files. The new files are a natural evolution: the proposal-level ideas have been expanded into a complete, production-ready technical implementation. The dataset shift from FibVID/MiDe22 → FakeNewsNet is a deliberate upgrade motivated by richer propagation data, not a change in direction.

**The original proposal says:** "Build a GNN to detect misinformation from propagation patterns."  
**The new files say:** "Here's exactly how to build that GNN — with PropNet, FakeNewsNet, HetGAT, and complete training code."  
**They are the same project at different levels of detail.**

---

## 14. References

1. *Advanced Text Analytics — Graph Neural Network for Fake News Detection in Social Media* — [arXiv 2502.16157](https://www.arxiv.org/pdf/2502.16157)
2. *A Graph Network Approach to Disinformation Detection in Social Media* — [ResearchGate](https://www.researchgate.net/publication/391973703)
3. *Context-Based Fake News Detection using Graph Based Approach: A COVID-19 Use-case* — [ACM DL](https://dl.acm.org/doi/10.1145/3729706.3729819)
4. *Fake news detection: A survey of graph neural network methods* — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036155/)
5. *Misinformation detection based on news dispersion* — [IEEE](https://ieeexplore.ieee.org/document/10167997)
6. Gangireddy et al., *Unsupervised Fake News Detection: GTUT*, 2020 — [PDF](https://pureadmin.qub.ac.uk/ws/files/212663108/ht20_crc.pdf)
7. Spanakis group, *Fake News Detection on Twitter Using Propagation Patterns*, 2020 — [PDF](https://dke.maastrichtuniversity.nl/jerry.spanakis/wp-content/uploads/2020/11/10.1007@978-3-030-61841-4_MMGWJS.pdf)
8. Hiremath et al., *Analysis of Fake News Detection using GNNs*, 2023 — [IEEE](https://ieeexplore.ieee.org/document/10405304/)
9. Gul et al., *Advancing fake news detection with GNN and deep learning*, 2025 — [PDF](https://zuscholars.zu.ac.ae/cgi/viewcontent.cgi?article=8253&context=works)
10. Orugboh, Ezeogu & Juba, *Graph theory approach to health misinformation in aging populations*, MJH 2025 — [Link](https://researchcorridor.org/index.php/mjh/article/view/503)
11. Arshad, Manzoor & Hassan, *Fake news detection via BERT + knowledge-aware GNN*, MJH 2025 — [Link](https://gajcet.com/index.php/gajcet/article/view/16)
12. Bellam & Aakula, *Contextual and Sentiment-Based Hybrid Models for Detecting Fake News*, IEEE AIC 2025 — [IEEE](https://ieeexplore.ieee.org/abstract/document/11212091)
13. FibVID Dataset — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9759652/)
14. MiDe22 Dataset — [arXiv](https://arxiv.org/html/2210.05401v2)
15. COVID-19 Misinformation Twitter Dataset — [GitHub](https://github.com/Gautamshahi/Misinformation_COVID-19)
16. FakeNewsNet — [GitHub](https://github.com/KaiDMML/FakeNewsNet)

---

*This document synthesizes all five project files into a single master reference.*  
*Generated: March 1, 2026*
