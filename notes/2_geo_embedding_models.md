# Geo Foundation Model Embeddings

## Overview

Three major open models represent different design philosophies for embedding the Earth's surface.

| Model | Input | Output | Core idea |
|---|---|---|---|
| AlphaEarth Foundations | Multi-sensor time series + coordinates | 64-dim per-pixel embedding | Global continuous embedding field |
| Clay | Satellite image chip | 768-dim patch embedding | MAE-based image foundation model |
| SatCLIP | Geographic coordinates | Location embedding | Contrastive location-image alignment |

All three produce embeddings that can be used for downstream tasks with simple linear methods. The key differences are in what the embedding represents (a pixel, a tile, or a place), and how it was trained.


## AlphaEarth Foundations (Google DeepMind)

*Sources: [DeepMind blog](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/), [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL), [Google Earth Medium post](https://medium.com/google-earth/ai-powered-pixels-introducing-googles-satellite-embedding-dataset-31744c1f4650), [AEF paper preprint](https://arxiv.org/abs/2507.22291)*

AlphaEarth Foundations (AEF) is designed as a universal Earth representation. Its goal is to produce one semantically rich embedding per pixel, covering all terrestrial land and coastal waters at 10-meter resolution.

**What it produces:** The Satellite Embedding dataset on Google Earth Engine contains annual 64-dimensional embeddings for every 10-meter pixel globally, going back to 2017. With over 1.4 trillion embedding footprints per year, it is one of the largest datasets of its kind. Each embedding summarizes a full calendar year of observations for that pixel.

**Inputs:** AEF assimilates data from many sources simultaneously: Sentinel-1 SAR, Sentinel-2 multispectral, Landsat 8/9 (multispectral, panchromatic, thermal), GEDI canopy height, GLO-30 elevation, ERA5 climate reanalysis, ALOS PALSAR-2, GRACE gravity, and geotagged text from Wikipedia and GBIF species occurrence records. This multi-source fusion means embeddings generally overcome common data quality issues such as clouds, scan lines, or missing acquisitions.

**Continuous time:** A key innovation is that AEF treats time as a continuous variable, not discrete frames. It uses an implicit decoder: 

`f(embedding, time, sensor_parameters) -> predicted observation` 

This means the model can synthesize what a location would look like at any time, even between observations, and can produce embeddings for arbitrary time windows. This is the first EO featurization approach to support continuous time in this way.

**Architecture (STP encoder):** AEF uses a Space-Time-Precision (STP) encoder. Each block contains three parallel operators:
- **Space operator:** Vision Transformer attention over spatial locations at 1/16 resolution. Captures long-range spatial relationships.
- **Time operator:** Attention across frames in time at 1/8 resolution. Captures temporal dynamics and seasonal patterns.
- **Precision operator:** CNN layers at $1/2$ resolution. Preserves fine local spatial detail that global attention would lose.

These three operators process data at different scales simultaneously, then exchange information via learned resampling. This multi-scale design captures both global context and local precision while keeping compute tractable.

**Training setup:** AEF uses a teacher-student setup. A large teacher model produces target embeddings via implicit reconstruction; a smaller student model learns to match it (knowledge distillation). A third text alignment model aligns embeddings with Wikipedia and GBIF text, injecting ecological and semantic knowledge. Training used over 3 billion image frames from 5 million global locations.

**Key properties of the released embeddings:**
- Unit-length (L2-normalized), distributed on the unit sphere. Cosine similarity is the correct distance measure.
- Temporally consistent: the same location in different years produces comparable embeddings, so year-to-year angle change indicates surface change.
- Linearly composable: embeddings can be aggregated across pixels or transformed with vector arithmetic and still retain semantic meaning.
- Each pixel embedding captures context beyond the pixel itself -> a parking lot pixel and a highway pixel look similar spectrally but have distinct embeddings because the surrounding context differs.


## Clay

*Source: [Clay documentation](https://clay-foundation.github.io/model/index.html)*

Clay is an open-source satellite image foundation model based on a Vision Transformer Masked Autoencoder (ViT-MAE). Unlike AlphaEarth, it operates at the patch level rather than pixel level.

**What it produces:** 768-dimensional patch embeddings. A standard chip of $256x256$ pixels is split into 8x8 patches; each patch produces one embedding. A single embedding for the whole chip is obtained by averaging patch embeddings.

**Inputs:** Clay v1.5 handles inputs from multiple sensors (Sentinel-2, Landsat, Sentinel-1 SAR, NAIP, LINZ, MODIS) with any number of bands. The model takes image chips alongside metadata: wavelengths per band, ground sampling distance (GSD), latitude/longitude, and time step (week/hour).

**Architecture:** Four components work together:
- *Dynamic Embedding Block:* Generates patch tokens from the image bands and their wavelengths.
- *Position Encoding:* Encodes spatial and temporal metadata (lat/lon, time, GSD), injecting geographic and temporal context into each token.
- *Masked Autoencoder (MAE):* A ViT-based encoder reconstructs 70% masked patches from 30% visible ones. This reconstruction loss accounts for 95% of training loss.
- *DINOv2 teacher:* A self-distillation objective (5% of loss) encourages the embeddings to be structurally organized beyond reconstruction.

**Training:** Trained on 70 million globally distributed $256x256$ chips, sampled to match global land use/land cover statistics.

**Usage:** Clay embeddings are available on [Hugging Face](https://huggingface.co/made-with-clay/Clay) and [Source Cooperative](https://source.coop/clay/clay-model-v0-embeddings) (training data embeddings).

**Limitations:** Training covers only land and coastal waters. No poles, no open ocean, no nighttime data, no explicit extreme events. At most 6 temporal samples per location.


## SatCLIP

*Source: [SatCLIP paper](https://arxiv.org/abs/2311.17179)*

SatCLIP is a CLIP-style model that aligns geographic coordinates with satellite imagery via contrastive learning. Where Clay embeds image tiles and AlphaEarth builds a global pixel field, SatCLIP embeds *locations*.

**Training objective:** For each (location, image) pair, SatCLIP trains a location encoder and an image encoder to produce similar embeddings for matching pairs and dissimilar embeddings for non-matching pairs.

**What it captures:** The resulting location embedding summarizes the average visual characteristics of a place -> its typical land cover, terrain, climate signature, and ecological context. It does not encode temporal change or specific events.

**Practical strength:** Given only a latitude/longitude coordinate, SatCLIP produces a meaningful embedding immediately, without requiring an image. This is useful for spatial prediction tasks where coordinates are available but imagery may not be.

**Limitation:** Because it encodes stable geographic characteristics rather than specific observations, SatCLIP cannot detect change events or distinguish between a location before and after a transformation (e.g. forest to agriculture).


## Patch vs. Pixel Embeddings

Models like Clay, SatMAE, and Prithvi use patch-level embeddings (typically $8x8$ or $16x16$ pixels per token). AlphaEarth produces pixel-level embeddings. The tradeoff:

**Patch embeddings**
- Massive compute savings: Transformer self-attention scales with $O(tokens^2)$. Going from $1M$ pixels to $4K$ patches is a $~250x$ reduction in tokens.
- Stronger high-level semantics: A patch covering a city block encodes richer contextual meaning than a single asphalt pixel.
- Fits standard ViT/MAE architectures directly.
- Downside: Loses spatial precision within a patch. A 50% road / 50% building patch has a blended embedding that represents neither cleanly.

**Pixel embeddings**
- Precise geographic meaning: Each vector maps to an exact geographic coordinate.
- Compatible with GIS workflows, segmentation masks, and per-pixel analysis.
- Handles heterogeneous patches without mixing semantics.
- Downside: Computationally expensive. A $1024x1024$ image produces $~1M$ pixel tokens vs. $~4K$ patch tokens.

A useful analogy: patch embeddings are paragraph summaries; pixel embeddings are word-level tokens. Both are useful depending on the task: patch for semantic retrieval and scene classification, pixel for segmentation, change detection, and object extraction.

Interestingly, patch embeddings often produce strong semantic representations despite blending pixels, because compression + context + self-supervision naturally push them toward concept-level representations. Semantics emerge from patterns across space, not individual pixels.


## Hierarchical Embeddings

Rather than choosing one spatial scale, hierarchical models represent multiple scales simultaneously. Different scales encode different types of meaning:

| Scale | Example semantic |
|---|---|
| Pixel | Surface material (asphalt, grass, water) |
| Patch | Object or land-use unit (building, crop field, road) |
| Region | Landscape structure (residential zone, river corridor) |
| Scene | Global context (coastal city, agricultural valley) |

Examples of hierarchical EO models:
- **Prithvi (IBM/NASA):** Multi-scale transformer features used for flood detection, wildfire monitoring, crop analysis.
- **SatMAE:** Hierarchical feature extraction via intermediate transformer layers.
- **Swin Transformer:** Hierarchical window attention ($4x4$ -> $8x8$ -> $16x16$ patches). Used widely in EO segmentation.
- **Feature Pyramid Networks (FPN):** Produce embeddings at multiple resolutions $(1/4, 1/8, 1/16, 1/32)$ for segmentation pipelines.

Hierarchical approaches are particularly useful when the task requires reasoning at both fine (object boundary) and coarse (land cover class) levels simultaneously.
