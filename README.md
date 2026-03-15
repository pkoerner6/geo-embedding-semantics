# geo-embedding-semantics

## Task

**How much semantic structure do geo foundation model embeddings actually encode, and what can simple linear tools extract from it?**

- Time budget: ~5 hours
- Deadline: March 15th EOD
- Compute: CPU-only
- Data & models: Public assets only (AlphaEarth embeddings, Clay, or any open FM/embeddings)
- Deliverable: Notebook (or repo) + 1-page writeup

The core bet: geo embeddings encode rich semantics extractable with simple tools (cosine similarity, linear SVMs). The question is how true this is, where it holds, and where it breaks.

---

## Suggested Directions
The following directions were given as optional structure. 

1. **From chips to concepts**: Embeddings encode images or pixels, but we want semantics. How are they fused? Can you extract meaningful concepts (land cover, infrastructure, change patterns) from raw embeddings using simple tools? Where do such abstractions hold, and where do they break?
2. **How far do linear tools go?**: Take real embeddings and push linear operations (e.g. LSVM) until they fail. Where is the boundary between “linear is enough” and “you need something nonlinear”? What types of semantics are easy vs. hard for linear tools?
3. **What does reconstruction difficulty tell you?**: Many Geo-foundation models trained with reconstruction objectives embed a difficulty signal: some tiles are harder to reconstruct than others. Can you find or approximate this signal from embedding geometry alone? Is it useful — for filtering, anomaly detection, quality control, or something else?
4. **Indexing Earth**: We make embeddings for each raster, not composites, and for any instrument. This scales quickly to a hard problem in many aspects. A key one for semantic retrieval via cosines is indexing these vectors. What ideas do you have for a scale of  trillions of embeddings on cosine similarity, as  fast and cheap as possible?   

---

## Ideas

---

### Idea 1: Semantics vs. Visual Similarity

**Maps to:** From chips to concepts + How far do linear tools go?

**Question:** Do cosine nearest neighbors retrieve semantically similar tiles, or just visually similar ones?

This tests the foundational claim directly. Two tiles can look alike (same colors, textures) without sharing semantics (a beach vs. a salt flat), and two tiles can share semantics without looking alike (a dense European city vs. a South Asian city).

**Prior work:** [Blumenstiel et al. 2024](https://arxiv.org/abs/2403.02059) shows FM embeddings outperform RGB baselines for retrieval on BigEarthNet and ForestNet. However, no paper has done a controlled ablation comparing FM cosine neighbors against raw pixel/spectral baselines on the same semantic retrieval task. The gap between visual and semantic similarity as a function of scene type is open.

**Steps:**
1. Sample a diverse set of query tiles (e.g. solar farms, ports, deforestation fronts, urban grids).
2. Retrieve top-k cosine neighbors from a large pool.
3. Evaluate: do neighbors share the semantic category of the query, or just surface appearance?
4. Compare against a plain visual baseline (raw spectral vectors or JPEG features) to isolate the FM's contribution.
5. Identify failure cases: where does retrieval return visually similar but semantically wrong results?

**Extension:** Use a linear SVM on embeddings to classify land cover. Then deliberately stress-test it: mixed-use tiles, unusual viewpoints, cross-geographic examples (same semantic class, different appearance). This probes the "how far do linear tools go?" question directly.

---

### Idea 2: Semantic Arithmetic

**Maps to:** From chips to concepts

**Question:** Do geo embeddings support semantic vector arithmetic? For example: `city - buildings + water ≈ port`

This is the `king - man + woman = queen` analogy applied to geo space.

**Prior work:** [Tile2Vec (Jean et al., AAAI 2019)](https://arxiv.org/abs/1805.02855) demonstrated qualitative geo analogies with a CNN encoder, showing e.g. `rural NYC - rural SF + urban SF ≈ urban NYC`. However, these results are illustrative and not benchmarked, and the model is a shallow CNN from 2019. No paper has tested whether modern transformer-based FMs (Clay, SatCLIP, ViT-MAE) preserve this property.

**Steps:**
1. Define semantic directions as the difference between class mean embeddings (e.g. mean("industrial") - mean("forest")).
2. Apply arithmetic: take a query tile, add/subtract semantic directions, retrieve the nearest neighbor.
3. Check whether the result shifts semantically as expected (removing "vegetation" direction should retrieve a more built-up version).
4. Test compositionality: can two directions be added at once?

**Caveat:** Hard to evaluate without ground truth. Focus on qualitative examples and visualizations.

---

### Idea 3: Is "Difficulty" a Semantic Property?

**Maps to:** What does reconstruction difficulty tell you?

**Question:** Does the ELLE signal work because embeddings directly encode training difficulty or because difficulty correlates with semantic region membership?

**Background:** The ELLE finding ([blog post](https://devlogs.lgnd.ai/posts/2026-03-01-self-aware-embeddings/)) is that a simple linear probe `predicted_loss = w · embedding + b` predicts a model's pretraining loss with high correlation (r ≈ 0.89 for ViT-MAE). Two competing explanations:

- **Hypothesis A (semantic region effect):** Hard tiles cluster in semantically complex regions (cluttered scenes, unusual textures). The probe learns "complex urban direction -> high loss." It is a soft semantic classifier, not a difficulty detector. Loss is predictable because difficulty is a semantic property.
- **Hypothesis B (direct encoding):** The embedding encodes fine-grained within-category difficulty. Even among all "urban" tiles, the probe distinguishes a cluttered intersection from a simple grid street.

**Prior work:** The ELLE signal is described in a blog post. The ELLE notebook validates the signal across modalities but never tests these two hypotheses. 

**Steps:**
1. Reproduce the ELLE probe on a geo dataset: Clay or ViT-MAE embeddings with a proxy for loss (reconstruction error, or JPEG file size as a crude approximation).
2. Cluster embeddings by semantic class (land cover labels if available, or k-means as proxy).
3. **Within-cluster test:** For each cluster, run the ELLE probe on samples inside the cluster only. Does the probe still predict relative loss within the cluster?
   - If yes: Hypothesis B -> fine-grained difficulty is encoded beyond semantics.
   - If no: Hypothesis A -> the probe is essentially a semantic classifier, and loss is a side effect.
4. Plot per-cluster r² to see which semantic regions drive the signal.

**Why this test is clean:** Holding semantic category constant removes the confound. Any residual predictive signal must come from something other than "what type of scene is this."

**Application:** If Hypothesis A holds, high-loss embeddings can be used as a signal for semantically complex or ambiguous tiles -> useful for data quality control and filtering edge cases.

---

### Idea 4: Temporal Trajectory of Change

**Maps to:** From chips to concepts (change patterns)

**Question:** When a region undergoes land-use change (e.g. forest to agriculture), is the transition in embedding space smooth or discontinuous? Does the post-transition embedding retain a "memory" of the prior state?

**Prior work:** [TESSERA (2025)](https://arxiv.org/abs/2506.20380) and [Google Satellite Embedding V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) both provide annual per-pixel embeddings. Google's documentation explicitly mentions using the angle between yearly embeddings for change detection. However, no paper has analyzed the geometry of these trajectories during change events: smoothness, step-function behavior, or whether post-transition embeddings cluster with their new or prior class.

**Steps:**
1. Find a region with documented land-use change (Global Forest Watch or a known deforestation site).
2. Embed the same tile at multiple time steps (before, during, after change).
3. Plot the trajectory in 2D (PCA or UMAP). Is it a smooth path or a step?
4. Compare: do post-transition embeddings cluster with other tiles of the same new type, or do they remain outliers, retaining a signature of their history?

---


## Background Notes

**Tile boundaries:** Tiles are fixed crops, semantics may straddle boundaries. A tile that is 50% urban and 50% water has an ambiguous embedding. Worth examining how similarity behaves at semantic boundaries vs. tile centers.

**Patch vs. pixel embeddings:** Patch (CLS) embeddings are paragraph summaries -> strong semantic signal, no spatial detail. Pixel embeddings are word-level -> precise, but context is local. The right choice depends on the task: semantic retrieval favors patch embeddings; segmentation and change detection favor pixel embeddings.

**Intrinsic dimensionality:** Geo embedding vectors are likely over-parameterized. The real information content may live in far fewer dimensions. Estimating intrinsic dimension would reveal how many independent semantic axes the model actually uses and which dimensions are redundant. This has direct implications for compression and indexing.

**Multi-semantic tiles:** A tile may contain multiple semantics (e.g. urban + coastal). Which dominates? Are secondary semantics still recoverable? This matters for retrieval -> similarity in one semantic dimension may mask dissimilarity in another. 

**The ELLE signal source:** SatCLIP (contrastive, location-based pretraining) achieves r = 0.96 on the ELLE probe. This is a model with no reconstruction objective, it organizes embedding space purely by semantic/geographic similarity. The fact that it also predicts loss well is circumstantial evidence for Hypothesis A (difficulty as a semantic property) in Idea 3.
