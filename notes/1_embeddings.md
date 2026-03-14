# Embeddings


## Embeddings in general

Begin Source: Foundations of Vector Retrieval (https://arxiv.org/pdf/2401.09350)

In the past, this vector representation of an object was nothing more than
a collection of its features. Every feature described some facet of the object
(for example, the color intensity of a pixel in a photograph) as a continuous
or discrete value. The idea was that, while individual features describe only a
small part of the object, together they provide sufficiently powerful statistics
about the object and its properties for the machine learnt model to act on.
The features that led to the vector representation of an object were gen-
erally hand-crafted functions. E.g. bag of words

The advent of deep learning and, in particular, Transformer-based mod-
els [Vaswani et al., 2017] brought about vector representations that are be-
yond the elementary formation above. The resulting representation is often,
as a single entity, referred to as an embedding, instead of a “feature vector,”
though the underlying concept remains unchanged: an object is encoded as
a real d-dimensional vector, a point in Rd.

must first ask ourselves what goal we are trying
to achieve by turning objects into vectors. It turns out, we often intend for
the vector representation of two similar objects to be “close” to each other
according to some well-defined distance function.

That is the structure we desire: Similarity in the vector space must imply
similarity between objects. So, as we engineer features to be extracted from
an object, or design a protocol to learn a model to produce embeddings of
data, we must choose the dimensionality dof the target space (a subset of Rd) along with a distance function δ(·,·). Together, these define an inner product
or metric space.


End Source: Foundations of Vector Retrieval



## Visualising embeddings

PCA (retain variability of data with least dimensions) or tSNE (keep distances between points while reducing dimensions), or UMAP (similar to tSNE but tries to keep global distributions more intact than tSNE)

Can assign RGB to three of the embedding dimensions or first do dimensionality reduction and then do that

Others(??)

## Text vs image embeddings
there is a fixed vocabulary for text embeddings, right? 
Not for image embeddings?


## Earth embeddings

Source: Earth Embeddings: Towards AI-centric Representations of our Planet (Klemmer et al.)

Earth embeddings provide a
unified and accessible vector representation of local geographic characteristics. They fuse dif-
ferentgeospatialdatasourcesacrosstimeandspace,compresshighly-correlatedrawgeospatial
data into one dense representation, can be used to guide interpolation between data obser-
vations, and can serve as a universal location token for foundation models.
This provides a
powerful alternative to existing geospatial workflows that rely on heterogeneous data, hard-
to-acquire expertise, and significant computation by the user: embeddings instead provide
convenient representations, easily adaptable for numerous downstream tasks. We posit that
Earthembeddingsredefinegeospatialanalytics, transformingitfromfragmented, task-specific
modeling into a coherent, generalizable framework for AI.

Geospatial data are discrete snapshots acquired to capture information about the conditions and
continuous dynamics of our planet. These data range from Earth observation or meterological
imagery captured by hundreds of in-orbit satellites to human-generated social media posts and
photos, tagged with geospatial metadata. At first glance, these data sources may appear com-
pletely different, but they share a fundamental commonality: all geospatial data are indexed in
space and time. Just as a satellite image provides information about a particular geographic area
at a particular point in time, so does a geotagged image or text description uploaded to point-
of-interest databases like Open Street Map or social media platforms. Geospatial data are widely
used to support modeling and decision making: Notable examples include disaster response,
weather forecasting, transportation management, forest and wildfire management,
water quality management, and disease spread modeling.

Enabled by the key strengths of
AI methods to represent complex data in flexible and generalizable formats, Earth embeddings
are vector representations of geospatial data at specific locations in space and time
(see Figure 1) that allow us to capture the similarities and differences between locations according
to their local features. They are produced by Earth embedding models, which process data
inputs into the vector shape of embeddings.

Earth embedding vectors emb are produced by a family of embedding functions E that
map continuous location inputs (i.e., longitude, latitude with optionally elevation, and
time) into a d-dimensional vector space:
E : (location) −→emb ∈Rd
. (1)
E, also referred to as embedding models, come in two forms: First, as explicit models,
extracting embeddings from raw data (e.g. satellite imagery) associated with a location
(emb∼Eexplicit(datalocation)). Examples for this include vision models combined with a
globalsatelliteimagedatabase. Second, asimplicit models, returningembeddingsfromonly
location inputs (emb∼Eimplicit(location)). Here, location-specific information is stored
directly withing the weights of e.g. a location encoder network Eimplicit.

Similar to how text or image embeddings capture semantic relationships, Earth embeddings
map places and times that share similar properties closer together in embedding space. For
example, an embedding of urban New York City may resemble that of urban Delhi more than that
of rural Arkansas, reflecting functional rather than geographic similarity. Likewise, embeddings
across time capture temporal variability—summer and winter embeddings in equatorial Costa
Rica may be nearly identical, while those in highly seasonal Alaska may differ strongly—enabling
quantitative comparison of environmental change and similarity across space and time.

The applications of Earth embeddings include
image geo-localization, poverty mapping, detecting small-scale and artisanal mining,
and satellite image super-resolution.



The Central Functions of Earth Embeddings
Current Earth embedding approaches are designed to serve one or more of four central functions,
as illustrated in Figure 2:
1. Compression: Earth embeddings can be interpreted as data compression, as they distill high-
dimensional, location-specific, multi-modal data such as satellite images or geotagged text data into a smaller vector format
2. Fusion: Earth embeddings fuse different geospatial data modalities–e.g.,images and text data– into a joint embedding space, leveraging their shared location and time information.
3. Interpolation: Some Earth embeddings–those obtained via location encoders–are available in
continuous space and time. They are based on neural implicit representations (see section 4.1),
which learn a smooth surface in embedding space by interpolating between seen training lo-
cations. As such, any location (in continuous space and time) may be queried from location
encoder models
4. Interoperability: Earth embeddings can be used to inject geospatial contextual informa-
tion into other AI foundation models. They can be seen as interoperable “location tokens”
(Figure 2), allowing users to interact with Earth embeddings using text prompts directly, ask-
ing queries such as “What are similar locations to the location (lon, lat)?” and to augment
current text-based assistants and foundation models efficiently with location information.

Explainable and interpretable Earth embeddings: Embeddings obtained by cur-
rent deep networks often lack interpretability. That is, we don’t know which part of
an Earth embedding represents a given ground condition, such as a tree or a building–or
if the embedding captures these ground conditions at all. This impedes our ability to
use Earth embeddings to understand interactions or explore drivers of variations, such
as seasonal versus geographic distances. Black-box methods complicate decision making, a general challenge with modern AI. As such, improving Earth embedding inter-
pretability (e.g., understanding which input modality contributes to which part of the
Earth embedding) can help to make them more attractive for integration into decision-
support and real-world deployment settings.

Learning planetary processes with Earth embeddings: It has long been a dream
of geospatial researchers to create a digital version of our planet, a “digital twin”.
Earth embeddings may offer a step towards this dream: as they learn to fuse and com-
press multi-modal geospatial data into an succinct representation of the planet within
their vectors,


## Image / geo embeddigns

Every day, satellites capture information-rich images and measurements, providing scientists and experts with a nearly real-time view of our planet. While this data has been incredibly impactful, its complexity, multimodality and refresh rate creates a new challenge: connecting disparate datasets and making use of them all effectively.


Image Embeddings (from https://www.linkedin.com/pulse/why-cant-google-maps-find-grass-bruno-sanchez-andrade-nuño-f7q0f/?trackingId=poYK51JvQtGgABqTU2gvVA%3D%3D --> need to reformulate and summarize the below text):
"Now let’s look at those images like a computer. Images are made of pixels, each pixel is made of three bands (red, green, and blue), and each band is digitally encoded using 8 bits, which allows for numbers from 0 to 255. In our case, each image is 256x256 (width and height) x3 (red, green, blue), that’s 200k numbers, and each pixel is 1 byte, or 8bits (from 0 to 255). So this image comes to  ~200KB (or 200K numbers, where each is anything from 0 to 255). There’s a lot you can do with that many numbers. What if I told you we can reduce any image to ~1KB (a thousand numbers, from 0 to 255), yet retain most of the information?"

Here's the catch, embeddings are not an image, but a list of numbers.

Despite having 0.3% of the size, AI tools can retrieve very similar results than without AI. Here's some examples of tests we've run with Clay: +90% of the same biomass estimate with embeddings than using full images. Or +90% of the land cover maps. Or detect more than 90% of the aquaculture locations. 

But there's even more. It takes 100s, thousands, even 10.000s times less time to recreate these outputs with embeddings than with the full image.

Images are pixels, and only the interpretations of those values and patterns define semantics. Embeddings encode directly those abstractions, as numbers. This means they already embed a lot of the computations one otherwise needs to detect things. Embeddings make retrieving and computing semantics extremely fast, partly because they have part of the computation already baked in.


REconstrucction: Embeddings are not the point of the AI for Earth model, but their utility and modular nature have drawn a lot of attention to them as separate assets. Embeddings are the highly abstracted summaries of the input data, and the narrow neck of the model,  that it uses to perform the task it is asked. These models tend to have a "U" shape with a wide input and output, and a narrow middle point that serves as the choke point to force the model to learn by abstracting least most useful semantics. 

The AI model takes an image of some size (width and height, say 512x512 pixels) with several bands (say red, green, blue). On each band we typically have 8 bits, so a number from 0 (black) to 255 (white). So in total in our case we have 256x256x3 = 200K dimensions. At the end of the encoding process we will have figured out summarizing the entire image into just 768 dimensions. That's quite impressive! A factor .4% of the size, yet contains most of the information. This is even more impressive when we have 13 bands, like on Sentinel-2 satellite images, when then the ratio is 0.01%. (TODO why does sentinel-2 have 13 bands?)

The image gets split into chunks, of size 8x8 in the case of Clay. These chunks of images are the units of the embeddings. We actually create embeddings for each chunk and then make the average of all the chunks to create the final embedding for the whole image. Why the average? There is no hard rule, and we can certainly improve on this aspect. It seems very crude to me to just average them all. 

But why averaging them at all? Because a Transformer-based model learns to embed each chunk not only the semantics within, but also in context of all the chunks around it, and their relative position (this is called "self attention", and that idea proved so powerful, that the paper that introduced this is called "Attention is all you need"). This means that embedding of a patch will also include semantics outside of itself. This makes it really powerful for some applications, but also confusing when you only care about what’s within a specific context.

After the embedding, there is usually a decoder that mirrors the encoder to bring back up the reconstruction of the same input image. The difference between that reconstruction and the input is literally the "loss" to minimize. The model looks at what changes help the loss go down, and it slowly updates all the millions of parameters to make this loss as small as possible.

After the task is finished, you can also replace the decoder with another architecture whose output is for example the amount of biomass in the image (regression problem), or the land cover class (segmentation problem), … Because you already have an encoder (or embeddings), these decoders are much lighter, faster and flexible than traditional methods where each output requires building a whole pipeline starting from the input image.

How are Earth Semantics learned?

I believe Earth semantics are learned into embeddings through 3 main mechanisms:

The actual value of the pixels: In text, embeddings start literally with random numbers. In visual transformers, like Clay and other Transformer based models, embeddings start with a linear projection of the actual pixel values. Hence embeddings are actually rooted on the ground, not floating around without anchors. I think this means that Earth embeddings across model runs are much more similar than text embeddings.
The context around them: This is exactly the same mechanism in all other Transformer-based models. The value of an embedding depends on the value of the embeddings around them, and their relative position. In our case the context is strictly limited to the size of the image. This means that the unit of embedding is the image; patches see other patches but only within the same artificially tiled bounds of the image. They cannot usually see patches beyond their image, even if Earth is continuous. The only way such model learns across images is through the metadata of latitude and longitude.
Masking: The task we ask the model to solve is to reconstruct an input image after compressing it into an embedding of much smaller size, but to make the task harder and the learning by context more strong, we actually mask out up to 70% of the image, so the model needs to extrapolate how to fill out the missing parts with access to only 30% of the image. This is obvious in some cases (like deserts), very relevant in others ( a highway across the image), and impossible in some cases (an isolated static boat in the water).

The way the model learns is also affected by other factors, for example how many images we use before allowing the model to update the way it makes the guesses ("backpropagation of model weights") to achieve high scores in our task (with stochastic gradient descent). If we update the model with every image, the learning will be very noisy and bumpy, trying to learn from all errors, even those from very rare cases. If we update the model after averaging the errors of too many examples, we will improve very smoothly, but missing many opportunities to pay attention to more rare but still common examples.

The implication of this is that when working with embeddings, angles between vectors are much more relevant than “straight” distances, like euclidean. These “flat” distances might be useful when working very locally, but since the embedding space has such defined overall shape, doing such metrics at global scales tells you about the topology shape more than it tells you about the semantics. In other words, you don’t measure the distance between Boston and Madrid going through Earth, you measure on the surface of Earth.


So far we've gone from images to embeddings of semantics. We still cannot search for the word "grass" even though we now know that the semantics for "grass" is encoded in the embeddings. In practice, embeddings are written in mathematics, not human language. This is not a problem, since there are several approaches we can take to bridge those embeddings to text. One we've tried that has success in some cases are literally forcing them to align: For each image of Earth we pull the information from a normal map (we use Open Street Map), things like "road here", "house there", "lake here". We then use the embedding of the image, and create an encoder that makes a random encoding of the description of the text. Then we ask the text encoder to learn to tweak the text embedding so that it's the same as the embedding of the image.  Thus, we can go from image to text and vice versa. We can even take an image, make the image embedding, find the closest text embedding that describes it, and then the closest text embeddings that describe images whose descriptions yield closest embeddings. A bit of a roundabout, but essentially a similarity search based on map descriptions.


How do we find similar images? As we’ve seen above 782 dimensions are many more than we can get an intuition of clusters, let alone relationships between clusters. It is therefore hard to even conceptualize how to operate with them. Is the average embedding of water and desert a beach? It seemed so, and if I check it does, but why? Why is the midpoint of those semantics the expected one? I don’t know. I suspect that in that case it is not a new concept but having both concepts in the image, just like the midpoint of the tree and parking lot is a parking lot with trees. But why is not something else completely random? Is like going from an extremely poor village to a very rich one and expecting to see the suburbs. It seems too good, and unpredictable.

What's even more crazy is that these semantics, operate with highly abstracted concepts. We can retrieve land cover classes, or find floods, or biomass within the image... We are so early in understanding Earth embeddings.

I believe part of the challenge in understanding Earth semantics is that they inherit known properties of other types of embeddings (like text embeddings) but they are also unique in other ways. One of them is what I called polysemy versus semantic colocation:

One of the biggest differences between text embeddings and Earth embeddings is how we deal with cases of embeddings of concepts that must contain different meanings.

In text, the word "bank" could mean where you put your money, or the side of a river, or a group of fish. A word with many meanings (polysemy) is really common. This is very common. In our case, the embedding of the word "bank" needs to encode all those meanings. 

But on Earth data, we have a different problem, and I think we have not yet figured out how to solve this. In the example, all 4 images contain the semantic "house" in different contexts (desert in California, crops in France, soil in Mongolia and water in Maldives). We can split images, in fact the model does, but we will never have a unit that just has the concept "house", which then will carry the core of the concept (with or without different meanings). With Earth data we both have absolute anchors on the actual pixels of the locations, and relative anchors of the information around them. We never have "words" isolated, or tokens. It is always patterns, and their surroundings. From that the model must learn the concept of "house", and "Water", and "crops", ... Semantics of Earth images are more deeply rooted in both pixels and context, than semantics of text where words can live isolated in the abstract, in fact we split text by them (or sub-words, tokens). Moreover, I believe that this colocation has a very small variability. That is, most things tend to be close to few other things. E.g. houses and roads, not houses and corals. This makes learning to reconstruct Earth locations with embeddings easy, but isolating Earth semantics more difficult.

Let's consider the case where we want to find "houses". If we pick one image with a house and see what other images are closest, we might also see one with "desert" on it. If I include all but the mongolian Yurt in the bottom right, we might average out the surroundings of the house, but will also reinforce the idea that houses are only squares, and we'll miss the circular yurt. In essence, I believe it is hard to define precisely semantics in Earth data, and fundamentally different than text.

One approach we follow is to search both with positive and negative examples. If we take an image of a house surrounded by grass as a positive example, and then give it a negative example of an image with only grass, we are much closer to the concept of house, without reinforcing the specific houses in the examples.

> Negative examples in embeddings means to stay as far away as possible from that point, just like a positive example is to stay as close as possible. We must be careful to remember that a negative example is not an example of the opposite concept (if that exists). Embeddings of opposite concepts are not necessarily in opposite locations in the embedding space. E.g. A person might consider water and desert opposites, but in the embedding space they might actually be close to each other. Also worth noting that embeddings cannot encode negative concepts (e.g. "not a house"). Embeddings are abstractions of the data, and therefore cannot encode the specific ways data might be missing. 

Because of this, a while ago I tried to increase the quality of our similarity search by doing what I called "semantic pruning". Basically use the few available examples of a semantic to find out which dimensions of the embedding are more important for that semantic, and drop the rest. This, in theory, would make similarity searches on fewer dimensions faster and cleaner. It's quite simple to do that: I took the few examples I had available and I fit a Random Forest classifier (this method basically picks random dimensions and random thresholds to divide the data into ever smaller buckets, and the answer is the random choices that yield the most accurate buckets with the right labels). This method also tells you what dimensions are most important ("feature importance", or what bucket choice splits the data most accurately towards the labels). Since Random Forest is very fast, we can filter out dimensions after every example given, and repeat the process. Long story short, it yielded no improvements in overall speed or quality.

We know Earth embeddings are extremely useful, and we also know we don’t yet know how to work with them well.


(end from https://www.linkedin.com/pulse/why-cant-google-maps-find-grass-bruno-sanchez-andrade-nuño-f7q0f/?trackingId=poYK51JvQtGgABqTU2gvVA%3D%3D)




## Semantics
Semantics are meaningful real-world concepts encoded in the embedding. An embedding is just a vector. On its own, it is a list of numbers. “Semantics” means those numbers are not arbitrary: they may contain information like:
- forest vs. cropland vs. water
- urban density
- roads, airports, ports, solar farms
So the idea is that a fixed embedding from a geo foundation model may already organize the world by meaningful concepts, and we may be able to recover those concepts with cheap downstream methods like cosine similarity, PCA, clustering, or a linear SVM.



## Vector Retrieval

Begin Source: Foundations of Vector Retrieval (https://arxiv.org/pdf/2401.09350)

Mathematically, “recalling information” translates to finding vectors that are
most similar to a query vector. The query vector represents what we wish
to know more about, or recall information for. So, if we have a particular
question in mind, the query is the vector representation of that question. If
we wish to know more about an event, our query is that event expressed as
a vector. If we wish to predict the function of a protein, perhaps we may
learn a thing or two from known proteins that have a similar structure to the
one in question, making a vector representation of the structure of our new
protein a query.

Similarity is then a function of two vectors, quantifying how similar two
vectors are. It may, for example, be based on the Euclidean distance between
the query vector and a database vector, where similar vectors have a smaller
distance. Or it may instead be based on the inner product between two vec-
tors. Or their angle. Whatever function we use to measure similarity between
pieces of data defines the structure of a database.
Finding k vectors from a database that have the highest similarity to a
query vector is known as the top-k retrieval problem. When similarity is
based on the Euclidean distance, the resulting problem is known as near-
est neighbor search. Inner product for similarity leads to a problem known
as maximum inner product search. Angular distance gives maximum cosine
similarity search. These are mathematical formulations of the mechanism we
called “recalling information.

Finding the most similar vectors to a query vector is easy when the
database is small or when time is not of the essence: We can simply com-
pare every vector in the database with the query and sort them by similarity.
When the database grows large and the time budget is limited, as is often
the case in practice, a na¨ıve, exhaustive comparison of a query with database
vectors is no longer realistic. That is where vector retrieval algorithms
become relevant.

Research continues to date. In
fact, how we do vector retrieval today faces a stress-test as databases grow
orders of magnitude larger than ever before. None of the existing methods,
for example, proves easy to scale to a database of billions of high-dimensional
vectors, or a database whose records change frequently.





#### top k retrieval

Given a distance function δ(·
,·), we
wish to pre-process a collection of data points X⊂Rd in time that is poly-
nomial in |X|and d, to form a data structure (the “index”) whose size is
polynomial in |X|and d, so as to efficiently solve the following in time o(|X|d)
for an arbitrary query q∈Rd:
(k)
arg min
u∈X
δ(q,u).


#### k neirest neighbor

In
Nearest Neighbor search, we find the data point whose L2 distance to the
query point is minimal


Nearest Neighbor Search
In many cases, the distance function is derived from a proper metric where
non-negativity, coincidence, symmetry, and triangle inequality hold for δ. A
clear example of this is the L2 distance: δ(u,v) = ∥u−v∥2. The resulting
problem, illustrated for a toy example in Figure 1.2(a), is known as k-Nearest
Neighbors (k-NN) search:
(k)
arg min
∥q−u∥2 =
u∈X
(k)
arg min
∥q−u∥2
2.


#### Cosine Similarity
In Maximum Cosine Similarity
search, we instead find the point whose angular distance to the query point
is minimal

The distance function may also be the angular distance between vectors,
which is again a proper metric. The resulting minimization problem can be
stated as follows, though its equivalent maximization problem (involving the
cosine of the angle between vectors) is perhaps more recognizable:
(k)
arg min
u∈X
1−
⟨q,u⟩
∥q∥2∥u∥2
=
(k)
arg max
u∈X
⟨q,u⟩
∥u∥2
.
The latter is referred to as the k-Maximum Cosine Similarity (k-MCS) prob-
lem. Note that, because the norm of the query point, ∥q∥2, is a constant in
the optimization problem, it can simply be discarded; the resulting distance
function is rank-equivalent to the angular distance.


#### Maximum inner product search

In Maximum Inner
Product Search, we find a vector that maximizes the inner product with the
query vector. This can be understood as letting the hyperplane orthogonal to
the query point sweep the space towards the origin; the first vector to touch
the sweeping plane is the maximizer of inner product.

Both of the above (knn search and cosine similarity search) are special instances of a
more general problem known as
k-Maximum Inner Product Search (k-MIPS):
(k)
arg max
⟨q,u⟩. (1.4)
u∈X

This is easy to see for k-MCS: If, in a pre-processing step, we L2-normalized
all vectors in Xso that uis transformed to u′
= u/∥u∥2, then ∥u′∥2 = 1 and
therefore Equation (1.3) reduces to Equation (1.4).

In a sense, then, it is sufficient to solve the k-MIPS problem as it is the
umbrella problem for much of vector retrieval. Unfortunately, k-MIPS is a
much harder problem than the other variants. That is because inner product
is not a proper metric.

Saying one problem is harder than another neither implies that we cannot
approach the harder problem, nor does it mean that the “easier” problem
is easy to solve. In fact, none of these variants of vector retrieval (k-NN, k-
MCS, and k-MIPS) can be solved exactly and efficiently in high dimensions.
Instead, we must either accept that the solution would be inefficient (in terms
of space- or time-complexity), or allow some degree of error.

The first case of solving the problem exactly but inefficiently is uninterest-
ing: If we are looking to find the solution for k= 1, for example, it is enough
to compute the distance function for every vector in the collection and the
query, resulting in linear complexity.
When k>1, the total time complexity
is O(|X|dlog k), where |X|is the size of the collection. So it typically makes
more sense to investigate the second strategy of admitting error.

That argument leads naturally to the class of ϵ-approximate vector re-
trieval problems. This idea can be formalized rather easily for the special
case where k = 1: The approximate solution for the top-1 retrieval is satis-
factory so long as the vector u returned by the algorithm is at most (1 + ϵ)
factor farther than the optimal vector u∗, according to δ(·
,·) and for some
arbitrary ϵ>0:
δ(q,u) ≤(1 + ϵ)δ(q,u∗).

The formalism above extends to the more general case where k>1 in an
obvious way: a vector uis a valid solution to the ϵ-approximate top-kproblem
if its distance to the query point is at most (1 + ϵ) times the distance to the
k-th optimal vector.



End Source: Foundations of Vector Retrieval 



#### linear SVMs


SVM is a supervised learning algorithm primarily used for classification and regression. The key principle of SVM is to find the optimal hyperplane that maximizes the margin between different classes in high-dimensional space.

How to use them??