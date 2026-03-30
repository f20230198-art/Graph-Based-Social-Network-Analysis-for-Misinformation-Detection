# Presentation Guidelines — Project Proposal
### Graph-Based Social Network Analysis for Misinformation Detection

> **Format:** 2–3 minimalistic slides only  
> **Purpose:** Concise, high-impact overview of the project proposal

---

## Slide Structure & Content Guide

---

### 📌 SLIDE 1 — Project Overview

| Section | Content |
|---|---|
| **Title** | **Graph-Based Social Network Analysis for Misinformation Detection** |
| **Problem** | Misinformation spreads through coordinated social interactions (bot networks, echo chambers, rapid resharing), but most detection systems only analyze text content — missing structural propagation patterns entirely. |
| **Novelty** | Instead of asking *"What does the post say?"*, we ask *"How is it spreading?"* — using **Graph Neural Networks (GNNs)** to model user interaction networks and detect suspicious propagation patterns (dense communities, hub-driven cascades, burst-like diffusion) that signal misinformation campaigns. |

**Speaker Notes:**
- Open by stating the *scale* of the misinformation problem (cite a stat if possible)
- Emphasize the paradigm shift: **content analysis → network/graph analysis**
- Highlight that this addresses gaps in current research: static graphs, lack of temporal modeling, poor cross-domain generalization

---

### 📌 SLIDE 2 — Approach & Technical Plan

| Section | Content |
|---|---|
| **Dataset** | **FibVID** (1,353 claims, 221K tweets, 144K users — COVID-19 diffusion trees), **MiDe22** (10,348 tweets across 40 events — multi-event), **COVID-19 Misinformation Twitter Dataset** (tweet IDs + labels + categories) |
| **Tech Stack** | **Graph Neural Networks** (GAT, GraphSAGE) · **Text Embeddings** (BERT/RoBERTa) · **ML Baselines** (Random Forest, XGBoost) · **Tools** (PyTorch, PyTorch Geometric, NetworkX) |
| **How We Proceed** | **1.** Collect & preprocess tweet data → **2.** Construct interaction graphs (users as nodes, retweets/mentions/replies as edges) → **3.** Extract graph features (centrality, clustering, propagation velocity) + text embeddings → **4.** Train GNN models (GAT/GraphSAGE) with fusion strategy → **5.** Evaluate against baselines using Accuracy, F1, AUC, Precision/Recall |

**Speaker Notes:**
- Briefly mention each dataset and what makes it suitable (propagation trees, multi-event coverage)
- Walk through the pipeline visually if possible (left-to-right flow diagram)
- Stress the **evaluation matrix**: text-only → network-only → simple fusion → full GNN — showing incremental value of graph features

---

### 📌 SLIDE 3 *(Optional — if 3 slides allowed)* — Key Research Gaps & Expected Impact

| Section | Content |
|---|---|
| **Research Gaps Addressed** | ① Most GNN models use **static graphs** — we incorporate temporal dynamics ② **Graph construction choices** (node types, edge semantics, weighting) are under-explored ③ **Explainability** of GNN detectors is lacking ④ **Cross-domain generalization** is poor (trained on politics, fails on health) |
| **Expected Impact** | Earlier detection of coordinated misinformation campaigns · Improved accuracy by combining content + network signals · Framework applicable across multiple social media platforms and event types |

**Speaker Notes:**
- Pick 2–3 strongest gaps to emphasize (don't overload)
- End with a strong closing statement about real-world applicability

---

## General Presentation Tips

- ✅ **Keep text minimal** — use keywords and short phrases, not paragraphs
- ✅ **Use visuals** — include a simple graph diagram showing nodes (users) and edges (interactions)
- ✅ **Consistent design** — use a clean, dark-mode or minimal template
- ✅ **Speak, don't read** — slides are prompts, the explanation comes from you
- ✅ **Time management** — aim for ~2 minutes per slide maximum
- ✅ **Flow:** Problem → Why it matters → How we solve it → What tools/data → Expected outcome

---

## Quick Reference — What Each Slide Covers

| Slide | Covers |
|---|---|
| **Slide 1** | Title, Problem, Novelty |
| **Slide 2** | Dataset, Tech Stack, Methodology (How we proceed) |
| **Slide 3** *(optional)* | Research Gaps, Expected Impact |

---

*Keep it sharp. Keep it visual. Let the graph tell the story.* 🎯
