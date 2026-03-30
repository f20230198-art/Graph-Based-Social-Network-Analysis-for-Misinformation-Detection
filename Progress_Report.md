# Progress Report
## Graph-Based Social Network Analysis for Misinformation Detection

> **Team:** Ashmit Dhown, Aaryan Gupta, Sachin P, Srivathsa H
> **Course:** Foundations of Data Science (FDS)
> **Last Updated:** March 30, 2026

---

## How to Read This Document

This document tells the story of our project from start to present — what we did, why we did it, and how. It is written so that anyone, even without a technical background, can follow along step by step.

---

## Phase 1: Understanding the Problem

### What we did
We started by asking a simple question: **How do people detect fake news online today, and why isn't it working well enough?**

We read 12 research papers on the topic. We found that almost every existing system works the same way — it reads a social media post and tries to figure out if the words sound "fake." Think of it like a spell-checker, but for truthfulness.

### Why this matters
The problem is that clever misinformation is written to *look and sound exactly like real news*. So just reading the text isn't enough. We needed a different approach.

### What we discovered
We found that fake news and real news **spread differently** across social networks:
- **Fake news** spreads in sudden bursts, often pushed by the same small group of accounts
- **Real news** spreads gradually, shared by many different, unrelated people
- **Fake news** creates dense clusters (echo chambers) in the network
- **Real news** spreads more evenly across diverse communities

This was our key insight: **Don't just read the message — study the messenger and the delivery route.**

---

## Phase 2: Researching Existing Work (Literature Review)

### What we did
Each team member took on specific research papers to analyze critically. We didn't just summarize them — we identified their **weaknesses** so we could build something better.

### What we found (and why it shaped our project)

**Paper 1** (Sachin's review): A model for health misinformation in older adults. Problem: the math behind it was never properly tested against real data. The ethics of targeting vulnerable groups were ignored. *This taught us: always validate your model with real data and consider ethics.*

**Paper 2** (Sachin's review): A system claiming to combine text analysis (BERT) with graph neural networks. Problem: they never actually built or tested the graph part — they only showed text results! *This taught us: we must actually build and test every component we claim to use.*

**Paper 3** (Sachin's review): A complex hybrid system combining many techniques. Problem: it kept confusing satire (comedy news) with real fake news. It couldn't handle images or videos. It was a "black box" — nobody could explain why it made its decisions. *This taught us: our system needs to be explainable and handle edge cases.*

**Papers 4-12** (Ashmit's reviews): Covered graph neural networks, propagation patterns, and various detection methods. These gave us the technical foundation for our approach. *Key takeaway: combining text understanding with network analysis consistently outperforms using either alone.*

### Six gaps we identified in current research

1. Most models treat social networks as **frozen snapshots** — they ignore that misinformation spreads over time
2. Nobody carefully studies **what counts as a connection** between users (retweet? reply? follow?)
3. Text and network information are rarely **combined intelligently**
4. Models trained on political fake news **fail when applied to health or entertainment fake news**
5. Datasets with complete spread data are **scarce**
6. Most detection systems are **black boxes** — they can't explain their decisions

Our project was designed to address all six of these gaps.

---

## Phase 3: Choosing Our Dataset

### What we did
We evaluated multiple datasets to find the best fit for our hybrid approach (text + network analysis).

### The candidates we considered

| Dataset | What it contains | Limitation |
|---|---|---|
| **FibVID** | 1,353 COVID-19 claims, 221K tweets | Limited to one topic (COVID) |
| **MiDe22** | 10,348 tweets across 40 events | Smaller scale, limited propagation data |
| **COVID-19 Misinfo** | Tweet IDs with labels | Needs Twitter API to reconstruct the network |

### Why we chose FakeNewsNet

We ultimately chose **FakeNewsNet** because it was the only dataset that gave us *everything* we needed:
- **23,196 news articles** with full text (for the text analysis branch)
- **850,000+ tweets** with complete retweet trees (for the network analysis branch)
- **Two different domains** — politics (PolitiFact) and entertainment (GossipCop) — so we can test if our model works across topics
- **Fact-checker verified labels** — professional fact-checkers confirmed what's real and what's fake

This was a deliberate upgrade from our original proposal (which listed FibVID and MiDe22). FakeNewsNet is larger, richer, and better suited to the hybrid model we designed.

---

## Phase 4: Designing the System Architecture

### What we did
We designed a system called **PropNet** (Propagation-aware Network) that processes information through two parallel branches before combining them.

### How it works (step by step)

**Step 1 — Read the post (Text Branch)**
The system feeds the post's text into a language model called RoBERTa. Think of RoBERTa as a very advanced reader that converts text into a list of numbers representing its meaning. It also checks:
- Is the tone angry, fearful, or overly emotional? (Fake news is 3-5x more emotionally charged)
- Is the writing style suspicious? (ALL CAPS, lots of exclamation marks!!!, low vocabulary)
- Does it look like known fake news patterns?

**Step 2 — Map the spread (Graph Branch)**
The system builds a map (graph) of how the post spread across social media:
- Each user becomes a dot (node)
- Each interaction (retweet, reply, mention) becomes a line connecting dots (edge)
- It then analyzes the shape of this map using Graph Neural Networks (GNNs)
- It looks for: How fast did it spread? Are the sharers all in one tight group? Do any accounts look like bots?

**Step 3 — Combine and decide (Fusion Layer)**
The system takes the insights from both branches and combines them using a smart weighting system. Sometimes the text is more telling; sometimes the spread pattern is more telling. The system learns which to trust more for each specific case.

**Step 4 — Output a verdict**
The final output is two probabilities: how likely the post is to be **real** vs. **fake**.

### Why we designed it this way
- **Two branches** catch what either alone would miss
- **Attention-weighted fusion** is smarter than just averaging the two signals
- **Graph Neural Networks** can learn complex patterns in network data that traditional statistics miss
- **RoBERTa** is specifically good at understanding informal social media text

---

## Phase 5: Feature Engineering

### What we did
We defined exactly what information the system extracts from the data. Think of features as the "clues" the detective (our model) looks for.

### Text clues (797 total measurements per post)
- **Semantic meaning** (768 numbers from RoBERTa): What does the text actually say?
- **Sentiment** (3 numbers): Is the tone negative, neutral, or positive?
- **Emotion** (6 numbers): How much anger, disgust, fear, joy, sadness, or surprise?
- **Writing style** (12 numbers): Caps ratio, punctuation density, readability level, etc.
- **Fake-news-specific patterns** (8 numbers): How similar to known fake news?

### Network clues (65 total measurements per news cascade)
- **Cascade shape** (25 numbers): How deep, wide, and fast the spread is
- **User profiles** (20 numbers): Account age, follower count, posting behavior, bot indicators
- **Community structure** (8 numbers): Are sharers clustered in echo chambers?
- **Timing patterns** (12 numbers): Time-of-day patterns, synchronized bursts

---

## Phase 6: Building the Model

### What we did
We implemented the PropNet architecture with the following components:

**Text Branch:**
- RoBERTa-base as the main text encoder
- Separate sentiment and emotion models running in parallel
- Custom linguistic feature extractor

**Graph Branch:**
- Heterogeneous Graph with 6 types of connections (retweet, reply, quote, mention, follow, co-retweet)
- Layer 1: Graph Attention Network (GAT) — learns which connections matter most
- Layer 2: GraphSAGE — aggregates information from neighbors
- Structural feature MLP — processes the 65 network measurements

**Fusion:**
- Learnable attention weight that dynamically balances text vs. graph contributions

**Classifier:**
- Final layers that output real/fake probabilities

### Training plan
- **Text branch** learns slowly (learning rate 0.00002) because RoBERTa is already pre-trained
- **Graph branch** learns faster (learning rate 0.001) because it starts from scratch
- **Class imbalance handling** — GossipCop has 3x more real than fake articles, so we weight the loss function to compensate
- **Temporal split** — we train on older data, test on newer data (because in the real world, you can't use future information to detect today's fake news)
- **Early stopping** — if the model stops improving for 10 rounds, we stop training to prevent overfitting

---

## Phase 7: Evaluation Plan

### What we planned
We designed a 4-step comparison to prove that each component of our system adds value:

| Step | Model | Purpose |
|---|---|---|
| 1 | **Text-only** (just RoBERTa) | Baseline — what can text alone achieve? |
| 2 | **Network-only** (graph features + XGBoost) | Baseline — what can network patterns alone achieve? |
| 3 | **Simple Fusion** (text + graph in a basic combiner) | Does combining help at all? |
| 4 | **Full PropNet** (our complete system) | Does our smart fusion beat the naive approach? |

### Expected outcome
Based on comparable work in the literature, we expect:
- Text-only: ~80% F1 score
- Network-only: ~75% F1 score
- Simple fusion: ~82% F1 score
- **Full PropNet: ~87% F1 score** (7-12% improvement over text-only)

---

## Current Status

| Phase | Status |
|---|---|
| Problem Understanding | Complete |
| Literature Review | Complete (12 papers reviewed) |
| Dataset Selection | Complete (FakeNewsNet chosen) |
| System Design | Complete (PropNet architecture defined) |
| Feature Engineering | Complete (797-d text + 65-d structural features specified) |
| Model Implementation | In progress |
| Training & Evaluation | Upcoming |
| Analysis & Reporting | Upcoming |

---

## Key Decisions and Why We Made Them

| Decision | Why |
|---|---|
| **Shifted from FibVID/MiDe22 to FakeNewsNet** | FakeNewsNet has both full article text AND complete retweet propagation trees — the other datasets had one but not both |
| **Chose RoBERTa over BERT** | RoBERTa was trained on 10x more data and handles informal social media text better |
| **Used heterogeneous graphs (multiple edge types)** | Different interactions (retweet vs. reply vs. follow) carry different signals — lumping them together loses information |
| **Attention-weighted fusion over simple concatenation** | The model should learn when to trust text more vs. network more, rather than always weighting them equally |
| **Temporal train/test split** | Prevents "data leakage" — the model can't cheat by seeing future patterns during training |

---

*This document will be updated as the project progresses through implementation and evaluation.*
