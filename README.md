# Alone TV Show — Loadout Strategy Clustering via Text Representation

**Course:** BA820 Unsupervised & Unstructured Machine Learning
**Institution:** Boston University, Questrom School of Business
**Term:** Spring 2026
**Author:** Xincheng (George) Ding — Team A1_08

---

## Overview

Participants on the History Channel show *Alone* each choose 10 items from a fixed list before being dropped into the wilderness. This project asks:

> **Do contestants fall into discrete loadout "strategies," and do those strategies relate to how long they survive or why they exit?**

This is the M4 milestone of a semester-long project. M2 and M3 used a binary one-hot pivot (94 participants × 27 items) for loadout clustering and found weak, method-dependent structure. M4 tests whether **a text-based Bag-of-Words representation** of the same loadouts produces cleaner clusters and more interpretable archetypes.

## Data

| Table | Rows | Role |
|---|---|---|
| `survivalists.csv` | 94 participants | Outcomes: `days_lasted`, `medically_evacuated`, `reason_category` |
| `loadouts.csv` | 10 items per participant | Item-level loadout records |

Tables are joined on `name`. After join, 84 participants have complete loadout + outcome data (the M4 working sample).

## Motivation — why change representation

M3's controlled experiments showed that loadout clustering is **method-dependent**:

- Hierarchical + Jaccard (equipment-only) → silhouette collapses, most participants in one blob.
- K-Means + Euclidean (equipment-only) → moderate structure (silhouette ≈ 0.22).
- K-Means + Euclidean (equipment + standardized age) → silhouette drops slightly; age adds noise.
- ARI between hierarchical and K-Means under identical features ≈ 0.34 — assignments change materially when only the method changes.

Binary one-hot treats `"sleeping bag"` and `"sleeping pad"` as entirely unrelated dummy variables, even though they share the token `"sleeping"` and reflect the same underlying strategy (prepared sleeping setup). BoW preserves that token-level overlap.

## Method

### 1. Text representation

Each participant's 10 items are concatenated into a short "loadout document." Two vectorizations are compared:

- **Unigram BoW** (`CountVectorizer(ngram_range=(1,1))`) — ~37 tokens
- **Unigram + bigram BoW** (`CountVectorizer(ngram_range=(1,2))`) — 100+ features

### 2. Model selection

K-Means over `k = 2 … 8` for each representation. Selection criteria:

- **Silhouette score** (cohesion vs separation)
- **Elbow / inertia** (diminishing returns)
- **PCA projection** (2D structure check, variance explained)
- **ARI** (cross-method assignment agreement, from M3)

### 3. Outcome linkage

Once `k` is locked, clusters are linked to the three outcome variables:

- `days_lasted` — boxplot + medians
- `medically_evacuated` — rate by cluster
- `reason_category` — distribution by cluster
- Winner rate — % of cluster that won the season

## Results

### Representation comparison

| Representation | Best k | Silhouette | Δ vs one-hot |
|---|---|---|---|
| One-hot (M2/M3 baseline) | 3 | 0.222 | — |
| **Unigram BoW (M4)** | **3** | **0.253 → 0.261 on final run** | **+0.031** |
| Unigram + bigram BoW | 3 | lower across all k | worse |

Key finding: **BoW unigram beats one-hot at every k in [2, 8].** Bigrams consistently hurt silhouette — with only ~10 items per document, most bigrams appear 1–2 times and act as noise.

### k selection

`k = 3` locked. `k = 4` reproduced M3's imbalance problem (one cluster of only 13 participants, silhouette drops to 0.220). The data naturally supports 3 groups.

### Three archetypes (all 84 participants have positive per-sample silhouette — zero misassignments)

| Cluster | Archetype | Characteristic tokens | Size |
|---|---|---|---|
| 0 | **Hunting/Trapping + Utility** | bow/arrows, trapping wire, paracord, multitool, knife | 56% |
| 1 | **Shelter + Preparedness** | tarp, bivy, canteen, soap, rations, frying pan | ~25% |
| 2 | **Fire/Fishing + Knife-focused** | ferro rod, fishing rod, hammock, slingshot, stone | ~19% |

### Outcome linkage

**Survival duration.** Hunting/Trapping has the highest median `days_lasted` and contains the longest-duration outlier (~100 days). Hunting/Trapping also has the highest winner rate.

**Medical evacuation.** Fire/Fishing has the lowest med-evac rate (**0.13**); Hunting/Trapping and Shelter are both at **0.31**.

**Exit reason.** Exits concentrate differently by archetype:

- Hunting/Trapping → **70% medical/health** exits
- Shelter + Preparedness → **62–64% family/personal** exits
- Fire/Fishing → **62–64% family/personal** exits

Different archetypes appear to have different **exit mechanisms**, not just different exit timing.

## Key Takeaways

1. **Representation matters.** Switching from one-hot to BoW improved silhouette by +0.031 without changing the underlying algorithm — the features (and the similarity metric they imply) did the work.
2. **More expressivity ≠ better.** Bigrams hurt. With ~10 items per document, n-gram complexity needs to be matched to document length.
3. **Loadouts carry real information.** The three archetypes are not statistical artifacts — zero per-sample silhouette misassignments, and the outcome differences (winner rate, med-evac rate, exit reason) are directionally consistent across multiple metrics.
4. **Different strategies fail differently.** The 70% medical-exit concentration in Hunting/Trapping — but 62–64% family/personal concentration in the other two — suggests loadout choice shapes *why* people leave, not just *when*.

## Dead Ends & Iterations

- **Bigrams.** Feature count jumped from 37 to 100+ but silhouette dropped at every k. Most bigrams appear in 1–2 participants.
- **k = 4.** Attempted to match M3's choice; produced a 13-participant singleton cluster and lower silhouette (0.220 vs 0.253). Reinforced that the data supports 3 groups, not 4.
- **Age as a feature.** Tested in M3 (equipment + standardized age). Silhouette dropped. Age was not retained in M4.

## Limitations

- **Sample size.** n = 84 after the join. Subgroup outcome comparisons (especially Fire/Fishing, the smallest cluster) should be read as descriptive, not inferential.
- **Chi-square association between strategy and exit** was not statistically significant under some specifications in M3. M4's outcome differences are **directional** signals, not causal claims.
- **Item vocabulary is sparse.** Because each document is only ~10 tokens, richer representations (TF-IDF weighting, embeddings) did not have enough text to work with. BoW unigram was the sweet spot.

## Repository Structure

```
.
├── README.md                                        ← you are here
├── George_M4_Text_based_Representation_Refinement.ipynb   ← main M4 analysis
├── BA820_A1_08_M3.ipynb                             ← team M3 (methodology baseline)
├── Xincheng_Ding_M2_Alone.ipynb                     ← individual M2 (episode engagement)
└── data/
    ├── survivalists.csv
    ├── loadouts.csv
    └── episodes.csv
```

## Reproducibility

**Environment:**

```
Python 3.10+
pandas
numpy
scikit-learn          # KMeans, PCA, silhouette_score, CountVectorizer
matplotlib
seaborn
scipy                 # chi-square (used in M3)
```

**Run order:**

1. Place the three CSVs in `data/`.
2. Open `George_M4_Text_based_Representation_Refinement.ipynb` and run all cells top-to-bottom.
3. Outputs (silhouette curves, PCA scatter, cluster profiles, outcome linkage charts) regenerate inline.

The notebook is deterministic — `random_state=42` is set on every K-Means call.

## Skills & Tools Demonstrated

- **Unsupervised learning:** K-Means, hierarchical clustering, silhouette, ARI, elbow method
- **Text representation:** Bag-of-Words (CountVectorizer), unigram vs bigram tradeoffs
- **Dimensionality reduction & visualization:** PCA, per-sample silhouette plots
- **Methodology comparison:** Controlled experiments isolating method vs features
- **Outcome linkage:** Cluster × categorical outcome analysis, winner/evacuation rates

---

*This README documents the M4 milestone of the BA820 team project (Team A1_08) at Boston University Questrom. M4 is individual work; M2 and M3 components by teammates are cited where relevant.*
