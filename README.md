# geo-embedding-semantics


## Task
%% TODO phrase as question, can't mention the company LGND!!!!

Time budget: Budget ~5 hours at most.
Deadline: March 15th EOD
Compute: CPU-only is fine. No large GPU runs expected.
Data & models: Public assets only (AlphaEarth embeddings, Clay, or any open FM/embeddings).
AI use: We encourage responsible use of AI tools. Please briefly state what you used, how and why.
Deliverable: A notebook (or repo) + a 1-page writeup. Hacky code is fine.

The prompt
Bet that geo embeddings encode rich semantics which can be extracted with simple tools (cosine similarity, linear SVMs) and are one of the most strategic components of our impact and commercial thesis.

Your task: Explore any angle of the above idea - and show us how to extract, analyse and use semantics of geo embeddings.

You choose the area, the model, and the question. The scope is deliberately open.

Suggested Directions
You are not required to follow any of these. They exist to help you scope if you want structure. Pick one, combine several, or ignore them entirely. These represent some of the challenges this JD might be asked to contribute to or own.
- From chips to concepts: Embeddings encode images or pixels, but we want semantics. How are they fused? Can you extract meaningful concepts (land cover, infrastructure, change patterns) from raw embeddings using simple tools? Where do such abstractions hold, and where do they break?
- How far do linear tools go? Take real embeddings and push linear operations (e.g. LSVM) until they fail. Where is the boundary between “linear is enough” and “you need something nonlinear”? What types of semantics are easy vs. hard for linear tools?
- What does reconstruction difficulty tell you? Many Geo-foundation models trained with reconstruction objectives embed a difficulty signal: some tiles are harder to reconstruct than others. Can you find or approximate this signal from embedding geometry alone? Is it useful — for filtering, anomaly detection, quality control, or something else?
- Indexing Earth. We make embeddings for each raster, not composites, and for any instrument. This scales quickly to a hard problem in many aspects. A key one for semantic retrieval via cosines is indexing these vectors. What ideas do you have for a scale of  trillions of embeddings on cosine similarity, as  fast and cheap as possible?  

What We’re Evaluating
We are not looking for anything specifically or expect a fully polished output. We are reading your work to understand how you think and how you work: Embedding intuition, pragmatism, communication, strategic thinking, day-one usefulness.



### Ideas

They assume: geo embeddings encode rich semantics which can be extracted with simple tools (cosine similarity, linear SVMs) --> First investigate if this stands

What about adversarial attack -> are embeddings of same satellite tile with small perturbations close?

Test if the same item in different form and lighning and region results in similar embeddings (e.g. solar farms, lakes in different shapes and colors)

Basically I want to test if the embeddings just show similar looking pixels / salteite tiles or if they actually have learned the semantics 

Can we do something like the language embeddings example: King + women = queen?



Interesting notes:
- ELLE: Embeddings Linearly contain their Loss Estimate: https://devlogs.lgnd.ai/posts/2026-03-01-self-aware-embeddings/ (CLS embeddings are sequence embeddings)
--> Foundation model embeddings secretly contain information about how hard the input was for the model during training. --> called the Embeddings Linearly contain their Loss Estimate (ELLE) signal
--> key idea: If you take the embedding of an input (like the CLS embedding from a transformer) you can linearly predict the model’s training loss for that input.
--> trained a simple linear regression with L2 regularization on training samples (embedding, pretrained_loss) `predicted_loss = w · embedding + b`. Can get the pretraining loss with no decoder needed, no forward pass through the full model objective.


-> is there a semantic of difficult? Are the difficult embeddings grouped in some way? Instead of the explanation "embeddings encode training difficulty" can't it be that maybe difficulty itself is a semantic property that occupies regions of embedding space. the alternative hypothesis is that samples that are difficult share semantic properties, e.g. hard images might include cluttered scenes, occlusion, unusual viewpoints, complex textures. These are visual semantics. If that's true, then predicting loss becomes easy because: embedding → semantic region → typical loss. The model would not encoding loss explicitly -> it's encoding the underlying structure that causes loss.
How could this be tested?
-> can I use this to detect the edge cases / difficult items where the emantics break down? (ceveat: Correlation is population-level. Good for dataset A harder than dataset B, less reliable for sample 183 harder than sample 184)



What about the issue of semantics for defined cut images. The squares / images we embed don't necessarily have all information to caryy semantics. 

Also what about averaging about all squares in an image (is this still done this way)


What about the time aspect? Is it possible to trace an image of a region that undergoes change (e.g. from forest to agricultural land)? Is this a smooth transition or is there a jump? Do the embeddings of the new agricultiural land contain some "history", meaning are they different because they recently transition from forest to agriculture?


Can we have an LLM write text about what is seen in an image and use this somehow? Either embed both and use for training or use it for evaluation somehow?


How do we distinguish between semantics. An image might have multiple semantics, which one dominates? Ar ethe others still present?

Can the embeddings be used for predictions? E.g. predicting deforestation in certain locations? Climate change?

Similarity alone is not enough, we want similarity with respect to certain semantics. How can this be done??

Can you assign semantics to certain regions in the embedding space?? Can you do this systematically? kmeans is not ideal, because you have to set k. Is there a way to identify centroids of clusters and assign them (e.g. with LLM), with the ability to increase centroids. Is this even possible given we have not a finit token vocabulary?



What is the difference between pixel-level embeddings are patch level embeddings? meso-scale semantics (the model learns “what kind of place this patch is.” -> You lose fine spatial detail inside the patch, but gain strong semantic meaning.) vs fine spatial semantics (Each pixel embedding represents “what is at this exact location.” This allows:
precise segmentation
boundary detection
sub-patch heterogeneity
object outlines
But pixel embeddings are:
much larger
often less context-aware per vector (context must be aggregated))

Another way to think about it
Patch embeddings are like paragraph summaries.
Pixel embeddings are like word-level tokens.
Both are useful depending on the task.




Current Earth embedding models aren't optimized for compression. Their vectors contain redundant dimensions, meaning the real information content is much lower than the number of features in the embedding. Can we test this? How many independent degrees of freedom are there and what are they?

Intrinsic dimension = how many independent degrees of freedom the data really has.