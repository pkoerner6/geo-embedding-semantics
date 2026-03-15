# Embeddings

## What Are Embeddings?

*Source: [Foundations of Vector Retrieval](https://arxiv.org/pdf/2401.09350)*

An embedding is a real-valued vector in R^d that represents an object. The core design goal is simple: similar objects should map to nearby vectors, according to some distance function. Before deep learning, these vectors were hand-crafted feature collections (e.g. bag-of-words for text, pixel histograms for images). Since the rise of transformer-based models, the term "embedding" refers specifically to representations learned end-to-end by a neural network, where the structure of the vector space is shaped by the training objective rather than by manual feature engineering.

The choice of dimensionality $d$ and distance function defines the geometry of the embedding space. Both need to be consistent: the similarity function used at retrieval time should match the objective used during training.


## Earth Embeddings

*Source: [Earth Embeddings: Towards AI-centric Representations of our Planet (Klemmer et al.)](https://arxiv.org/abs/2306.04738)*

Earth embeddings are vector representations of geospatial locations at specific points in space and time. They compress multi-modal geospatial data (satellite imagery, radar, climate, elevation, geotagged text) into a single dense vector per location, encoding the similarities and differences between places according to their local characteristics.

Earth embedding models come in two forms:

- **Explicit models:** Extract embeddings from raw data associated with a location (e.g. a satellite image chip). `embedding ~ E(image_at_location)`
- **Implicit models:** Return embeddings from coordinates alone, with location-specific information encoded directly in the model weights. `embedding ~ E(lat, lon)`

A key property: Earth embeddings organize space by functional similarity rather than geographic proximity. For example, urban New York and urban Delhi may be closer in embedding space than urban New York and rural Arkansas, because the model captures what a place is like, not where it is.

Earth embeddings serve four main functions:
1. **Compression:** Distill high-dimensional multi-modal data into a small vector.
2. **Fusion:** Combine data from multiple modalities (optical, radar, text) into a joint representation.
3. **Interpolation:** Implicit models produce continuous representations, so embeddings can be queried for any location or time, including unobserved ones.
4. **Interoperability:** Embeddings can serve as location tokens that plug into other foundation models or text-based interfaces.

**Open challenge:** Current Earth embeddings are largely uninterpretable. We do not know which dimensions encode which ground conditions, or whether specific semantics are captured at all. This limits their use in decision-support contexts.


## How Geo Semantics Are Learned

*Source: [Why Can't Google Maps Find Grass? (Bruno Sanchez-Andrade Nuno, LinkedIn)](https://www.linkedin.com/pulse/why-cant-google-maps-find-grass-bruno-sanchez-andrade-nu%C3%B1o-f7q0f/)*

Satellite images are made of pixels, each with several spectral bands. A 256x256 Sentinel-2 image with 13 bands contains roughly 850,000 numbers. A well-trained foundation model compresses this into ~768 numbers (Clay) while retaining most of the information needed for downstream tasks. This compression factor is what makes embeddings practical at scale.

Three mechanisms drive the learning of geo semantics:

1. **Pixel values as anchors:** Unlike text, where embeddings start from random initializations, geo models start with a linear projection of actual pixel values. The embedding is grounded in physical measurements from the start.

2. **Self-attention context:** Each patch embedding encodes not just the pixels within it, but also their relationship to neighboring patches. A patch of asphalt in a parking lot will differ from a patch of asphalt on a highway, because the surrounding context is different.

3. **Masked reconstruction:** Models like Clay and ViT-MAE are trained to reconstruct an image from only 30% of its visible patches. This forces the model to learn spatial statistics and contextual patterns -> simple regions (deserts, open water) are easy; complex or unique regions are hard. This asymmetry is why training difficulty correlates with semantic complexity.

The model has an encoder-decoder "U" shape. The encoder compresses the image into a narrow embedding bottleneck; the decoder reconstructs the input from it. After training, the decoder is discarded and the encoder is reused for downstream tasks. Because the embedding already contains compressed semantic information, downstream decoders (for e.g. biomass prediction or land cover segmentation) are much lighter than full end-to-end pipelines.


## Key Challenges for Geo Embeddings

*Source: [Why Can't Google Maps Find Grass? (Bruno Sanchez-Andrade Nuno, LinkedIn)](https://www.linkedin.com/pulse/why-cant-google-maps-find-grass-bruno-sanchez-andrade-nu%C3%B1o-f7q0f/)*

**Semantic colocation vs. polysemy.** In text, a single word like "bank" carries multiple meanings and appears in isolation as a token. In Earth imagery, we never observe an isolated concept. Every patch includes both the focal semantic (e.g. "house") and its surroundings (crops, roads, water). The model must infer concepts from patterns in context, not from discrete tokens. This makes Earth semantics harder to isolate than text semantics.

**No negative encoding.** Embeddings encode what is present, not what is absent. "Not a house" is not representable. Similarly, opposite concepts are not necessarily in opposite directions in embedding space.

**Semantic specificity.** Cosine similarity is a global measure, it compares entire embedding vectors. If you want similarity with respect to a specific semantic axis (e.g. vegetation density only, ignoring built-up area), you need directional projection onto that axis before computing similarity.

**Multi-semantic tiles.** A single tile may contain multiple distinct semantics. Which one dominates the embedding? Can secondary semantics still be recovered? This is an open question.

**Tile boundaries.** Tiles are fixed crops. Semantics can straddle boundaries -> a tile that is 50% urban and 50% water will have an ambiguous embedding. This matters for retrieval: two tiles may be semantically different but have similar embeddings because they both contain the same mixture of classes.


## Visualizing Embeddings

Embedding spaces are high-dimensional and cannot be inspected directly. Common approaches:

- **PCA:** Linear projection that retains maximum variance. Good for understanding the global structure of a dataset.
- **t-SNE:** Non-linear projection that preserves local distances. Clusters emerge clearly, but global layout is not meaningful.
- **UMAP:** Similar to t-SNE but better preserves global structure. Faster for large datasets.
- **RGB colorization:** Assign the first three PCA components to RGB channels. Produces spatially interpretable false-color maps where semantically similar regions share similar colors.


## Vector Retrieval

*Source: [Foundations of Vector Retrieval](https://arxiv.org/pdf/2401.09350)*

Vector retrieval is the problem of finding the k vectors in a database most similar to a query vector. Three common formulations:

**k-Nearest Neighbor (k-NN):** Minimize L2 distance. Finds the k points geometrically closest to the query in Euclidean space.

**k-Maximum Cosine Similarity (k-MCS):** Minimize angular distance, equivalent to maximizing the cosine of the angle between vectors. This is the standard for normalized embeddings, where the length of a vector carries no semantic meaning and only direction matters.

**k-Maximum Inner Product Search (k-MIPS):** Maximize the raw dot product. Generalizes both k-NN and k-MCS. Harder to solve because inner product is not a proper metric.

None of these can be solved exactly and efficiently in high dimensions. For small databases, exhaustive search (compute distance to every point) is feasible at $O(|X| * d * \log k)$. For large databases, approximate methods are necessary.

**Approximate Nearest Neighbor (ANN):** Accept a $(1 + \epsilon)$ approximation factor. Return a point whose distance is at most $(1 + \epsilon)$ times the true nearest neighbor distance. This enables sublinear query time through data structures like HNSW, IVF-PQ (FAISS), or LSH.

At the scale of trillions of embeddings (e.g. AlphaEarth's 1.4 trillion footprints), even ANN methods require careful engineering: product quantization to compress vectors by $16-32x$, hierarchical indexes, and semantic partitioning to reduce the effective search space.


## Linear SVMs

A Support Vector Machine (SVM) with a linear kernel finds the maximum-margin hyperplane separating two classes in the embedding space. For embeddings, this is particularly useful because it tests a strong claim: that the semantic distinction of interest is linearly separable in the embedding space.

If a linear SVM achieves high accuracy on a semantic classification task (e.g. forest vs. urban), it means the embedding already organizes the data such that a single hyperplane separates the classes -> no nonlinear transformation needed. If a linear SVM fails but a nonlinear one (e.g. RBF kernel) succeeds, the distinction exists in the embedding space but requires a nonlinear boundary.

This makes linear SVMs a clean diagnostic tool: they measure how explicitly a concept is encoded, not just whether it is recoverable at all.
