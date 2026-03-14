# 


AlphaEarth Foundations, Clay, and SatCLIP all try to learn vector embeddings of the Earth’s surface, but they represent three different design philosophies for geospatial foundation models.

Model	Core idea
AlphaEarth Foundations (AEF)	A global embedding field representing the planet across space and time
Clay	A satellite image foundation model producing embeddings from image chips
SatCLIP	A CLIP-style location encoder linking coordinates to satellite imagery
All three produce Earth embeddings, but the input representation and training objective differ substantially.


AlphaEarth learns a continuous embedding field over Earth.
Instead of embedding individual images, it models a function:
(lat,lon,time,sensors)→embedding

Clay is a foundation model for satellite imagery, similar to how ViT or MAE models embed images.
Instead of embedding coordinates, it embeds image tiles.
satellite image chip→embedding
Typical workflow:
Chip satellite imagery
Feed into Clay
Get embeddings for downstream tasks


atCLIP learns location embeddings using contrastive learning.
It aligns:
satellite images
geographic coordinates
during training.
(lat,lon)↔satellite imagery
The model then produces embeddings that represent the characteristics of a location (climate, urbanization, terrain, etc.).



## AlphaEarth embeddings


source start: https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/ 
AlphaEarth Foundations, an artificial intelligence (AI) model that functions like a virtual satellite. It accurately and efficiently characterizes the planet’s entire terrestrial land and coastal waters by integrating huge amounts of Earth observation data into a unified digital representation, or "embedding," that computer systems can easily process. This allows the model to provide scientists with a more complete and consistent picture of our planet's evolution, helping them make more informed decisions on critical issues like food security, deforestation, urban expansion, and water resources.
To accelerate research and unlock use cases, we are now releasing a collection of AlphaEarth Foundations’ annual embeddings as the Satellite Embedding dataset in Google Earth Engine. 

First, it combines volumes of information from dozens of different public sources— optical satellite images, radar, 3D laser mapping, climate simulations, and more. It weaves all this information together to analyse the world's land and coastal waters in sharp, 10x10 meter squares, allowing it to track changes over time with remarkable precision.

Second, it makes this data practical to use. The system's key innovation is its ability to create a highly compact summary for each square. These summaries require 16 times less storage space than those produced by other AI systems that we tested and dramatically reduces the cost of planetary-scale analysis.
This breakthrough enables scientists to do something that was impossible until now: create detailed, consistent maps of our world, on-demand. Whether they are monitoring crop health, tracking deforestation, or observing new construction, they no longer have to rely on a single satellite passing overhead. They now have a new kind of foundation for geospatial data.

Powered by AlphaEarth Foundations, the Satellite Embedding dataset in Google Earth Engine is one of the largest of its kind with over 1.4 trillion embedding footprints per year. https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content#description

source end: https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/


source start: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content#description

The Google Satellite Embedding dataset is a global, analysis-ready collection of learned geospatial embeddings. Each 10-meter pixel in this dataset is a 64-dimensional representation, or "embedding vector," that encodes temporal trajectories of surface conditions at and around that pixel as measured by various Earth observation instruments and datasets, over a single calendar year. Unlike conventional spectral inputs and indices, where bands correspond to physical measurements, embeddings are feature vectors that summarize relationships across multi-source, multi-modal observations in a less directly interpretable, but more powerful way.

The dataset covers terrestrial land surfaces and shallow waters, including intertidal and reef zones, inland waterways, and coastal waterways. Coverage at the poles is limited by satellite orbits and instrument coverage.

The collection is composed of images covering approximately 163,840 meters by 163,840 meters, and each image has 64 bands {A00, A01, …, A63}, one for each axis of the 64D embedding space. All bands should be used for downstream analysis as they collectively refer to a 64D coordinate in the embedding space and are not independently interpretable.

All images are generated in their local Universal Transverse Mercator projection as indicated by the UTM_ZONE property, and have system:time_start and system:time_end properties that reflect the calendar year summarized by the embeddings; for example, an embedding image for 2021 will have a system:start_time equal to ee.Date('2021-01-01 00:00:00') and a system:end_time equal to ee.Date('2022-01-01 00:00:00').

The embeddings are unit-length, meaning they have a magnitude of 1 and do not require any additional normalization, and are distributed across the unit sphere, making them well-suited for use with clustering algorithms and tree-based classifiers. The embedding space is also consistent across years, and embeddings from different years can be used for condition change detection by considering the dot product or angle between two embedding vectors. Furthermore, the embeddings are designed to be linearly composable, i.e., they can be aggregated to produce embeddings at coarser spatial resolutions or transformed with vector arithmetic, and still retain their semantic meaning and distance relationships.

The Satellite Embedding dataset was produced by AlphaEarth Foundations, a geospatial embedding model that assimilates multiple datastreams including optical, radar, LiDAR, and other sources (Brown, Kazmierski, Pasquarella et al., 2025; preprint available here).

Because representations are learned across many sensors and images, embedding representations generally overcome common issues such as clouds, scan lines, sensor artifacts, or missing data, providing seamless analysis-ready features that can be directly substituted for other Earth Observation image sources in classification, regression, and change detection analyses.

source end: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content#description



source start: https://medium.com/google-earth/ai-powered-pixels-introducing-googles-satellite-embedding-dataset-31744c1f4650

assimilates observations across diverse sources of geospatial information, including optical and thermal imagery from Sentinel-2 and Landsat satellites, radar data that can see through clouds, 3D measurements of surface properties, global elevation models, climate information, gravity fields, and descriptive text. Unlike traditional deep learning models that require users to fine-tune weights and run their own inference on clusters of high-end computers, AlphaEarth Foundations was designed to produce information-rich, 64-dimensional geospatial “embeddings” that are suitable for use with Earth Engine’s built-in machine learning classifiers and other pixel-based analysis.

We’ve run AlphaEarth Foundations at scale to produce a global dataset of precomputed, analysis-ready embeddings at a 10-meter resolution for each year back to 2017. While this may look like any standard Earth Engine Image Collection, we’ve effectively packed AI-powered feature extraction into every pixel, and you can use these embedding “images” in place of more conventional image composites and engineered features like spectral indices and harmonic fits. The best part is that embedding layers are analysis-ready; no need for atmospheric correction, cloud masking, spectral transformations, speckle filtering, or other featurization techniques — just superior results at reduced effort and complexity.

The geospatial embeddings generated by AlphaEarth Foundations are learned across diverse data sources from the Earth Engine Data Catalog and geo-temporally located text labels. The model uses a self-supervised approach that enables learning from many types of data at once without hand-annotated training data. By assimilating information across multiple sources and modes of description, including Sentinel-1 C-Band SAR, multi-spectral Sentinel-2, and multi-spectral, panchromatic, and thermal observations from Landsat 8 and Landsat 9, GEDI Raster Canopy Height metrics, GLO-30 DEM, ERA5-Land Reanalysis Monthly Aggregates, ALOS PALSAR-2 ScanSAR, GRACE monthly mass grids, and several text sources, AlphaEarth Foundations is able to learn a more compact representation of pixel properties and semantics.

AlphaEarth Foundations was trained on over 3 billion individual image frames sampled from over 5 million locations globally. By treating satellite images of a given location over time as if they were frames in a video, the model is able to learn across space, time, and mode of measurement to produce embeddings that capture spatial context and preserve temporal trajectories. This means every embedding vector in the Satellite Embedding dataset provides a highly compact, yet semantically rich representation of surface conditions for every 10-meter pixel (100 square-meter) area of Earth’s terrestrial land surface. Each 10-meter pixel’s embedding also captures information about the area around that pixel, so that areas that appear very similar when considered in isolation, e.g., the asphalt surfaces of a parking lot or a freeway, will have quite distinct embeddings. And in the case of our GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL collection, embeddings summarize a full year of image acquisitions, which means they include seasonal signals, like vegetation phenology or seasonal snow cover, and other within-year change events.

Similarity-based comparisons also work through time, and can be used for embedding-powered change detection and stability monitoring. The AlphaEarth Foundations embedding space was designed to be temporally consistent, so relatively stable locations should have similar embedding vectors across years in the dataset, while year-to-year changes in embedding vectors for a given location are indicative of changes in surface properties, environmental conditions, and/or their temporal dynamics. By computing the angle between annual embedding vectors for different years, you can monitor for long-term stability and catastrophic change, and begin to explore and understand the drivers of these changes.

source end: https://medium.com/google-earth/ai-powered-pixels-introducing-googles-satellite-embedding-dataset-31744c1f4650 


Visual app to investiagte similarity of embeddings: https://earthengine-ai.projects.earthengine.app/view/embedding-similarity-search#year=2024;zoom=15;lon=-114.77545392172516;lat=36.04996230797995;clicked=true;



Source start Paper: AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data
a number of innovations; namely, we employ an adaptive decoding
scheme that considers time and sensor parame-
tersascontinuousvariablesinanimplicitdecoder
with associated losses, a spatially dense infor-
mation time-bottleneck, time-conditional summa-
rization,andspatially-precisealignmentwithgeo-
tagged text from Wikipedia articles joined with
locations from Global Biodiversity Information
Facility (GBIF) species occurrence records. To
the best of our knowledge, AEF is the first EO fea-
turization approach to support continuous time
Additionally we introduce a challenging evalu-
ation suite composed of high-quality reference
data that attempts to faithfully replicate realis-
tic mapping scenarios. 


Satellite data comes from many sensors:
different satellites
different resolutions
different spectral bands
different observation times
Most previous systems treat these as separate discrete inputs.

But Earth observation is really continuous.
time is continuous
sensors vary continuously
viewing angles vary
atmospheric conditions vary

Use an implicit decoder that takes these as continuous variables.
Conceptually: embedding + time + sensor parameters -> predicted observation
So the model learns: f(location_embedding, time, sensor_settings)
This lets it simulate what the Earth would look like at any time and from any sensor, not just the training images.


he model can ingest multiple Earth observation data sources. For each sensor i: N_i = number of images (frames); C_i = number of channels (bands). Each image also has a timestamp t_j. All images are resampled to the same spatial resolution so the pixels line up.

Support period = the time span covered by the input images

Valid period = can ask the model to summarize any time interval you want. Even if there are no satellite images exactly during those months.

Interpolation
The requested interval lies inside the support period, but there may be no images exactly in that window.

Extrapolation
You ask for a time outside the observed data.

The embedding is trained so that a decoder can reconstruct Earth variables from it. So the embedding captures the temporal trajectory of these variables.
Meaning it encodes:
how things changed during that time period
Not just a static snapshot.

Source end Paper



Embeddings are per pixel! 

In ML, a bottleneck forces the model to compress information.
Example pipeline:
satellite images
      ↓
compressed embedding
      ↓
reconstruction
The model must store the important information in the embedding.
Key twist: enforce the bottleneck across time. Meaning: The model must summarize a time series of observations into a compact spatial embedding. All that information must compress into one representation for that location.
This forces the model to learn: seasonal dynamics, vegetation cycles, long-term patterns, without storing raw imagery.

Time-conditional summarization: Instead of storing a single static representation, the model learns something like:
embedding(location) + time → representation_at_time
AEF treats time as a continuous variable.

Spatially precise alignment with geotagged Wikipedia + GBIF: 
They inject semantic knowledge about places using text and biodiversity data.
Wikipedia
Many articles describe locations:
mountains
ecosystems
parks
cities
rivers
These contain descriptions of geography and ecology.

GBIF contains millions of species observations, each with:
species name
latitude
longitude
time

hey align Earth embeddings with these datasets.
So a location embedding learns relationships like:
this area → rainforest species
this area → alpine plants
this area → urban mammals
This injects biological and ecological knowledge into the representation.
So the model doesn't just learn pixels, it learns meaningful ecological structure.

Continuous time support: Meaning the model can represent Earth at any timestamp, not just discrete images. embedding(location, t), where t is any real number


Summary: They improved Earth-representation learning by:
Treating time and sensor properties as continuous inputs instead of discrete categories
Compressing long satellite time series into spatial embeddings
Allowing the representation to vary smoothly over time
Injecting semantic ecological knowledge using Wikipedia and biodiversity records
Supporting continuous time modeling for Earth observation
Evaluating the system with realistic mapping benchmarks

Can think of the system as learning a Google-Maps-like latent model of Earth where you can ask:
what is here?
how does it change over time?
what species live here?
what would it look like from a different satellite?
All encoded in those 64-dimensional embeddings.


The system:
Takes many satellite images from different sensors and times
Compresses them into a small embedding
That embedding summarizes a specific time interval you ask for
Even if there are no images exactly in that interval
So the model learns a temporal representation of Earth that you can query for any time window.


the system must:
keep very precise local detail (small spatial patterns like field boundaries)
understand long-distance relationships
across space (large regions)
across time (seasonal cycles)
do this efficiently (Earth-scale data)

--> designed a special encoder called STP (Space Time Precision).


Instead of one big network, each block contains three parallel operators:
Operator	What it focuses on
Space	long-range spatial relationships
Time	temporal relationships
Precision	fine local spatial details
So each block processes the data in three complementary ways simultaneously.

space operator: This is basically Vision Transformer attention over spatial locations at low resolution (L/16).

time operator: Instead of attending across pixels, it attends across frames in time. Run at moderate spatial resolution (L/8).

precision operator: a classic CNN layer to preserve fine spatial detail. Convolutions are perfect for this because they focus on local neighborhoods. This operator runs at higher resolution (L/2).

This is a multi-scale architecture.
Operator	Resolution	Purpose
Space	L/16	global context
Time	L/8	temporal relationships
Precision	L/2	local detail
This trick massively reduces compute while still capturing global + local information.


Each frame has a timestamp t_j .
They convert it into a sinusoidal time encoding.
This is the same trick used in Transformers for positions.
Example representation:
time → sinusoidal embedding

mention spatial pyramid exchanges. This means the three operators share information across scales. This is done using learned resampling (upsampling/downsampling layers).

The teacher–student training setup
They train three models together.
1. Teacher model
Large model that produces the best embeddings.
Uses implicit decoders to reconstruct data.
2. Student model
Same architecture as teacher.
Learns to imitate the teacher.
This makes inference faster.
This is called knowledge distillation.
3. Text alignment model
This aligns embeddings with text data.
Sources include:
Wikipedia
biodiversity datasets (GBIF)
This gives the embeddings semantic meaning.

end up with a global grid of embeddings.
They call this:
embedding fields
Think of it like:
Earth map
each pixel = 64-dim vector
Those vectors contain:
ecological signals
land cover
climate dynamics
seasonal behavior

The STP design combines the strengths of three major approaches:
Method	Strength
Transformers	global relationships
temporal attention	time modeling
CNNs	spatial precision
AEF explicitly uses all three simultaneously.



#### Clay

Source: https://clay-foundation.github.io/model/index.html


dimesnions: 768

not pixel level embeddings, are patch-level embeddings (typically 256 x 256 pixels, compressed into a single high-dimensional vector)


Clay’s model takes satellite imagery, along with information about location and time, as an input, and outputs embeddings, which are mathematical representations of a given area at a certain time on Earth’s surface. It uses a Vision Transformer architecture adapted to understand geospatial and temporal relations on Earth Observation data. The model is trained via Self-supervised learning (SSL) using a Masked Autoencoder (MAE) method.

The Clay model can be used in three main ways:

Generate semantic embeddings for any location and time. You can use embeddings for a variety of tasks, including to:
Find features: Locate objects or features, such as surface mines, aquaculture, or concentrated animal feeding operations.
Fine-tune the model for downstream tasks such as classification, regression, and generative tasks. Fine-tuning the model takes advantage of its pre-training to more efficiently classify types, predict values, or detect change than from-scratch methods. Embeddings can also be used to do the following, which require fine-tuning:
Classify types or predict values of interest: Identify the types or classes of a given feature, such as crop type or land cover, or predict values of variables of interest, such as above ground biomass or agricultural productivity.
Detect changes over time: Find areas that have experienced changes such as deforestation, wildfires, destruction from human conflict, flooding, or urban development.
This can be done by training a downstream model to take embeddings as input and output predicted classes/values. This could also include fine-tuning model weights to update the embeddings themselves.
Use the model as a backbone for other models.

lay v1.5 is our MAE-based model designed to handle inputs from a variety of satellite sensors, including Sentinel-2, Landsat, Sentinel-1 SAR, LINZ, NAIP and MODIS. It supports inputs of any size and any number of bands.

Components of Clay v1.5:

Dynamic Embedding Block: This component generates patches for the chips from the number of bands and their wavelengths, which are then fed into the masked autoencoder (MAE).
Position Encoding: This component encodes spatial and temporal information by adding positional encoding to the model. This encoding is scaled according to the Ground Sampling Distance (GSD) and is combined with location information (latitude/longitude) and time step (week/hour).
Masked Autoencoder (MAE): A VIT-based MAE is used to reconstruct the sensor data for all input bands. This contributes to 95% of the total loss, known as the reconstruction loss.
Teacher: DINOv2 is used as a teacher to compute the representation loss, which accounts for the remaining 5% of the total loss.

The pre-trained model can process stacks of geospatial data from different sensors with various resolutions and bands, and output vector embeddings. During pre-training, the model processes stacks of chips from different sensors along with metadata such as wavelengths, GSD, latitude/longitude, and time step. The task involves capturing spatial, temporal, and spectral information about Earth and representing these relationships in high-dimensional space. Each resulting embedding represents a specific area of Earth at a particular time.

Clay v1.5 was trained on 70 million globally distributed chips of size 156x256, collected according to the land use/land cover (LULC) statistics of the globe. The training was conducted on AWS using 20 g6.48xlarge instances for ~100 epochs in Sep 2024.


Limitations and biases:

Training data for this model only covers land and coastal waters.
We only train on a very small sample of the source archives, both in terms of spatial coverage and time.
We do not train on the poles, and we do not train on open ocean, nor ocean nor atmospheric volumetric data.
We do not train on night time data.
We do not explicitly include extreme events in the training data.
We only train at most 6 different times per location.



The website is madewithclay.org.
The Clay model code lives on GitHub (https://github.com/Clay-foundation/model). License: Apache-2.0. The latest release is v1.5
The Clay model weights on Hugging Face (https://huggingface.co/made-with-clay/Clay). License: Apache-2.0.
The Clay documentation lives on this site (https://clay-foundation.github.io/model/index.html). License: CC-BY.
We release the embeddings of the used training data on Source Cooperative (https://source.coop/clay/clay-model-v0-embeddings). License: ODC-BY.



#### Alpha Earth Foundations vs Clay

Pixel level embeddings vs patch embeddings

AlphaEarth Foundations is designed primarily as a universal Earth representation for mapping tasks. The key idea: Every location on Earth should have a semantic vector representation. To achieve this, the model produces one embedding per pixel.

Why pixel-level is useful
1. Precise geographic meaning
Each pixel embedding corresponds to an exact geographic coordinate.
That means you can directly attach meaning to:
a building corner
a road segment
a coastline
a crop row
This is critical for: mapping, segmentation, object extraction, monitoring change

2. Direct compatibility with GIS layers
Pixel embeddings align naturally with:
raster datasets
segmentation masks
geospatial grids

You can do things like:
pixel embedding → classify land cover
pixel embedding → detect building footprint
pixel embedding → detect flood extent
without needing extra spatial interpolation.

3. Better for heterogeneous areas
Within a single 16×16 patch you might have:
half road
half building
some vegetation
Patch embeddings blend these together.
Pixel embeddings preserve mixed scenes.

4. Enables universal per-location representation
AlphaEarth aims for something like:
embedding(latitude, longitude, time)
This turns the Earth into a semantic vector field, which is powerful for:
retrieval
change detection
monitoring

The downside
Pixel embeddings are computationally heavy.

Example:
1024×1024 image
Patch tokens (16×16 patches):
64 × 64 = 4096 tokens
Pixel tokens:
1,048,576 tokens
So pixel embeddings require huge memory and compute, which is why they are less common.



Models like Clay, SatMAE, Prithvi, etc., follow the Vision Transformer design.

Instead of pixels, they treat an image as tokens = patches.
Why this is attractive
1. Massive compute savings
Transformers scale roughly with:

O(tokens²)
Reducing tokens by 100–1000× dramatically reduces compute.

2. Better semantic context per token
Each patch embedding sees a larger area, so the vector captures richer context.
Example:
patch embedding → "urban block with roads"
pixel embedding → "asphalt"
Patch embeddings often produce stronger high-level semantics.

3. Scales to global datasets
Foundation models trained on millions of satellite tiles need efficient tokenization.
Patch tokens allow:
larger batch sizes
longer training
bigger models

4. Matches existing vision architectures
Patch tokens fit directly into:
Vision Transformers
masked autoencoders
contrastive learning pipelines
Which accelerates research and adoption.

The tradeoff in one sentence
Pixel embeddings → spatial precision
Patch embeddings → semantic efficiency

Semantics often emerge from patterns across space, not individual pixels.
Aggregating many pixels into a patch averages away noise.
What remains are stable spatial patterns
But in practice they learned very strong semantics, sometimes even outperforming CNN features.
This turned out to be because of:
compression + context + self-supervision
Patch embeddings often become semantic primitives because they:
compress information
see enough context
remove noise
interact with neighboring patches
Together, that naturally pushes them toward concept-level representations.




#### SatCLIP

Source Paper: SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery

Satellite Contrastive Location-Image Pretrain-
ing (SatCLIP). This global, general-purpose geographic lo-
cation encoder learns an implicit representation of locations
by matching CNN and ViT inferred visual patterns of openly
available satellite imagery with their geographic coordinates.
The resulting SatCLIP location encoder efficiently summa-
rizes the characteristics of any given location for convenient
use in downstream tasks.


CLIP-style contrastive objective:
location encoder(lat, lon)
image encoder(image)

→ maximize similarity of matching pairs
Strength
Embeddings can be generated from coordinates alone
Good for spatial prediction tasks
Weakness
Encodes average visual patterns of a location
Limited temporal modeling.

What you can do with it
You can:
Predict properties of locations
Cluster regions with similar environments
Estimate variables like:
climate
land cover
population density
biodiversity
crop suitability

SatCLIP embeddings represent the average appearance of a place, not specific events over time.

SatCLIP would struggle because:
it doesn’t encode temporal change
it encodes stable geographic patterns

Learned: “what kind of place is this?”




#### Hierarchical embeddins (multi-scale representations)

This is where things get really interesting.
Instead of choosing one scale, hierarchical models represent multiple spatial scales simultaneously.
pixel → patch → region → scene
Each level captures different semantics.


Imagine a satellite image.
Pixel level
Represents surface materials:
roof
grass
water
asphalt
Patch level
Represents objects or land-use units:
building
road segment
crop field
tree cluster
Region level
Represents landscape structure:
residential neighborhood
industrial zone
river corridor
forest stand
Scene level
Represents global context:
coastal city
agricultural valley
mountain forest


Models that use hierarchical embeddings
Several modern EO models incorporate this idea.

1. Prithvi (IBM + NASA)
Uses multi-scale features from transformer blocks.
Captures:
local textures
+ medium land patterns
+ global scene context
Used for:
flood detection
wildfire monitoring
crop analysis


2. SatMAE
Masked autoencoder with multi-scale feature extraction.
Intermediate transformer layers encode:
fine features → coarse semantics

3. Swin Transformer (many EO models use it)
Uses hierarchical windows:
4×4 patches
→ merged to
8×8
→ 16×16
This creates a pyramid similar to CNN feature pyramids.

4. Feature Pyramid Networks (FPN)
Common in segmentation pipelines.
Produces embeddings at multiple resolutions:
1/4 resolution
1/8
1/16
1/32




### Semantics
Semantics are meaningful real-world concepts encoded in the embedding. An embedding is just a vector. On its own, it is a list of numbers. “Semantics” means those numbers are not arbitrary: they may contain information like:
- forest vs. cropland vs. water
- urban density
- roads, airports, ports, solar farms
So the idea is that a fixed embedding from a geo foundation model may already organize the world by meaningful concepts, and we may be able to recover those concepts with cheap downstream methods like cosine similarity, PCA, clustering, or a linear SVM.
