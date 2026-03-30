# Graph-Based Social Network Analysis for Misinformation Detection

> **Course:** Foundations of Data Science (FDS)
> **Team:** Ashmit Dhown, Aaryan Gupta, Sachin P, Srivathsa H

---

## What This Project Is About (In Plain English)

Imagine someone shares a fake news article on social media. Within minutes, hundreds of accounts retweet it, comment on it, and spread it further. Most fake news detectors try to read the article and decide if the words sound fake. But clever misinformation is written to *look* real.

**Our approach is different.** Instead of only reading the text, we look at *how the post is spreading* across the network of users. We ask:

- **How fast** is it spreading? (Fake news often spreads in sudden bursts)
- **Who** is spreading it? (Often the same small group of accounts, or bots)
- **What does the network pattern look like?** (Fake news creates dense, suspicious clusters)

Think of it like detective work: instead of just reading a letter, you also check who delivered it, how many copies were made, and whether the same courier keeps showing up. That pattern tells you a lot.

---

## The Problem We Are Solving

Most existing fake news detection tools only analyze the **text content** of a post. This misses critical signals:

| What They Miss | Why It Matters |
|---|---|
| **Coordinated groups** | Organized clusters of accounts amplifying the same content |
| **Bot-like accounts** | Automated profiles designed to boost reach |
| **Influential hubs** | A few accounts controlling the entire spread |
| **Sudden rapid resharing** | Abnormal speed compared to how real news spreads |

**Real news vs. Fake news patterns:**

| Feature | Real News | Fake News |
|---|---|---|
| Spread speed | Gradual, organic | Rapid, bursty |
| Who shares it | Diverse, unrelated users | Same group over and over |
| Network shape | Spread out, tree-like | Dense clusters, few sources broadcasting |
| Control | Distributed among many | Controlled by a few accounts |

---

## Our Approach: Combining Text + Network Analysis

We build a system called **PropNet** (Propagation-aware Network) that combines two types of analysis:

### Branch 1: Text Analysis (What does the post say?)
- Uses **RoBERTa** (a language model) to understand the meaning of the text
- Detects emotional manipulation (fear, anger, outrage)
- Checks writing style (excessive caps, exclamation marks, low reading level)
- Analyzes sentiment (fake news is 3-5x more emotionally polarized)

### Branch 2: Network/Graph Analysis (How is it spreading?)
- Models social media as a **graph** (network):
  - **Nodes** = Users/accounts
  - **Edges** = Interactions (retweets, replies, mentions, follows)
- Extracts structural features:
  - How deep and wide the sharing cascade goes
  - Speed and acceleration of spread
  - Community structure (are the sharers all in one tight group?)
  - Bot detection signals (coordination scores, amplification patterns)
- Uses **Graph Neural Networks** (GAT + GraphSAGE) to learn from these patterns

### Fusion: Combining Both
The two branches feed into a **fusion layer** that learns to weight text vs. network signals depending on what's most informative for each case. The final output is a probability: how likely is this post to be misinformation?

```
Post Text ──> [RoBERTa + Sentiment + Style] ──> Text Features (797 dimensions)
                                                         |
                                                    [Fusion Layer] ──> Real or Fake?
                                                         |
Spread Network ──> [GAT + GraphSAGE + Features] ──> Graph Features (128 dimensions)
```

---

## What Makes This Project Novel

We address 6 specific gaps identified in existing research:

| # | Gap in Current Research | How We Address It |
|---|---|---|
| 1 | Most models treat networks as **static** (frozen in time) | We incorporate **temporal features** — speed, acceleration, burst patterns |
| 2 | **Graph construction choices** (what counts as a connection) are rarely studied | We use 6 different edge types with different weights, systematically |
| 3 | Text and network are rarely **combined properly** | We use attention-weighted fusion that learns the optimal balance |
| 4 | Models trained on one topic **fail on others** | We test across politics (PolitiFact) AND entertainment (GossipCop) |
| 5 | Datasets with full spread data are **limited** | We use FakeNewsNet with complete propagation trees |
| 6 | Most GNN detectors are **black boxes** | We detect coordinated behavior via community analysis and graph motifs |

---

## Dataset

### Primary: FakeNewsNet

| Detail | Value |
|---|---|
| **Total articles** | 23,196 news articles |
| **Total tweets** | 850,000+ tweets with full retweet trees |
| **Sources** | PolitiFact (politics) + GossipCop (entertainment) |
| **Fake articles** | 6,379 (1,056 PolitiFact + 5,323 GossipCop) |
| **Real articles** | 18,577 (1,760 PolitiFact + 16,817 GossipCop) |
| **Ground truth** | Verified by professional fact-checkers |
| **Why this dataset** | Has BOTH article content AND complete propagation trees — exactly what our hybrid model needs |

### How to Get the Dataset

**Step 1: Clone the FakeNewsNet repository**
```bash
git clone https://github.com/KaiDMML/FakeNewsNet.git
cd FakeNewsNet
```

**Step 2: Install requirements**
```bash
pip install -r requirements.txt
```

**Step 3: Download the data**
```bash
python download_data.py --dataset all
```

**Note:** Some tweet data may need "rehydration" (downloading the full tweet content using tweet IDs) via the Twitter/X API. Due to API restrictions, pre-collected archived versions may be available in the repository or through academic data sharing platforms.

### Additional Datasets (Referenced in Proposal)

These were considered during the proposal phase and may be used for supplementary validation:

| Dataset | Description | Link |
|---|---|---|
| **FibVID** | 1,353 COVID-19 claims, 221K tweets, 144K users | [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9759652/) |
| **MiDe22** | 10,348 tweets across 40 events (English + Turkish) | [arXiv](https://arxiv.org/html/2210.05401v2) |
| **COVID-19 Misinfo** | Tweet IDs + labels + misinformation categories | [GitHub](https://github.com/Gautamshahi/Misinformation_COVID-19) |

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Text Understanding** | RoBERTa (via HuggingFace) | Reads and understands post content |
| **Graph Neural Networks** | GAT, GraphSAGE (via PyTorch Geometric) | Learns from network spread patterns |
| **Sentiment Analysis** | twitter-roberta-base-sentiment | Detects emotional tone |
| **Emotion Detection** | distilroberta-emotion | Identifies anger, fear, etc. |
| **Graph Building** | NetworkX, PyTorch Geometric | Constructs the user interaction network |
| **Community Detection** | Louvain algorithm | Finds suspicious clusters |
| **Deep Learning** | PyTorch | Framework for training models |
| **ML Baselines** | Random Forest, XGBoost | Comparison models |
| **Data Processing** | Pandas, NumPy | Cleaning and organizing data |
| **Visualization** | Matplotlib, Gephi | Charts and network diagrams |

---

## How We Evaluate Success

### Metrics
| Metric | What It Measures |
|---|---|
| **Accuracy** | Overall correctness (target: ~97%) |
| **F1 Score** | Balance between catching fake news and not flagging real news (target: 87%+) |
| **ROC AUC** | How well the model ranks fake vs. real (target: 93-97%) |
| **Precision** | Of posts flagged as fake, how many actually are? |
| **Recall** | Of all fake posts, how many did we catch? |
| **Detection Delay** | How quickly can we flag misinformation after it first appears? |

### Comparison Plan (Ablation Study)
We test 4 model configurations to prove each component adds value:

| Model | What It Tests |
|---|---|
| **Text-only** (RoBERTa fine-tuned) | Baseline: just reading the content |
| **Network-only** (Graph features + XGBoost) | Baseline: just looking at spread patterns |
| **Simple Fusion** (Text + Graph in basic MLP) | Naive combination |
| **Full PropNet** (HetGAT + RoBERTa with attention fusion) | Our complete model |

**Expected result:** Each step should show improvement, with PropNet delivering 7-12% better F1 than text-only.

---

## Literature Review

We critically analyzed 3 key papers and identified their weaknesses:

### Paper 1: Graph-Theory SEIR Health Misinformation in Aging Populations
*Orugboh, Ezeogu & Juba, MJH 2025*
- SEIR model parameters lack proper validation
- Dataset not described (size, timeframe, how age was inferred)
- Simulated interventions never tested against real-world policies
- Older adults treated as one generic group
- Ethics and privacy concerns not addressed

### Paper 2: Knowledge-Based Fake News Detection with BERT + GCN/GNN
*Arshad, Manzoor & Hassan, MJH 2025*
- The GNN part (the main claimed contribution) was never actually tested — only BERT results shown
- Twitter dataset poorly described and not publicly available
- No ablation study to justify each component's value
- Multimodal features only described conceptually, not implemented

### Paper 3: Contextual + Sentiment + Social-Credibility Hybrid
*Bellam & Aakula, IEEE AIC 2025*
- Frequently misclassifies satire as fake news
- Cannot handle image/video misinformation
- Complex ensemble architecture with no explainability tooling
- Only tested on English news datasets

---

## References

1. *Graph Neural Network for Fake News Detection in Social Media* — arXiv 2502.16157
2. *A Graph Network Approach to Disinformation Detection* — ResearchGate 2025
3. *Context-Based Fake News Detection (COVID-19 Use-case)* — ACM DL
4. *Fake news detection: A survey of GNN methods* — PMC
5. *Misinformation detection based on news dispersion* — IEEE
6. Gangireddy et al., *Unsupervised Fake News Detection (GTUT)*, 2020
7. Spanakis group, *Fake News Detection via Propagation Patterns*, 2020
8. Hiremath et al., *Analysis of Fake News Detection using GNNs*, IEEE 2023
9. Gul et al., *Advancing fake news detection with GNN and deep learning*, 2025
10. Orugboh, Ezeogu & Juba, *Graph theory approach to health misinformation in aging populations*, MJH 2025
11. Arshad, Manzoor & Hassan, *Fake news detection via BERT + knowledge-aware GNN*, MJH 2025
12. Bellam & Aakula, *Contextual and Sentiment-Based Hybrid Models for Detecting Fake News*, IEEE AIC 2025
13. FakeNewsNet Dataset — [GitHub](https://github.com/KaiDMML/FakeNewsNet)

---

*Foundations of Data Science — Project Proposal*
