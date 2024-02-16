# Colbert

This is a playground to use [ColBERT](https://github.com/stanford-futuredata/ColBERT)
It does not use [RAGatouille](https://github.com/bclavie/RAGatouille)

The problem with RAGatouille are
* It does not expose all the ColBERT configurations.
* It is difficult directly integrate with LangChain's Embeddings class

## Gotcha
The latest version of `colbert-ai==0.2.19` or its dependencies require `pyarraow==14.0.0`

Install `faiss-gpu` on CUDA


## what's in the repo
* A LangChain Embedding class - [ColbertEmbedding](langchain/libs/community/langchain_community/embeddings/colbert.py). This class runs compute in the local host. If CUDA is available, it can take advantage of GPU computes. Therefore, the `faiss-gpu` module is required on GPU.
* Indexing and encoding example to test on GPU.

## High dimensional embedding
ColBERT generates a two dimensional matrix of vector, as supposed to an array of float in the common vector that is supported by most vector store. Therefore, the LangChain compatible [ColBertEmbedding](langchain/libs/community/langchain_community/embeddings/colbert.py) added a step to transform two dimensional metrics to one dimension array. The current implementation offers these strategy: 
1. Flat (default): Flattening the two-dimensional array into a one-dimensional array is a straightforward approach. However, this method can significantly increase the dimensionality of the vector and may not be practical or meaningful, especially since the order of flattened elements loses the spatial relationship inherent in the two-dimensional structure.
2. Average: To average the embeddings across tokens to produce a single vector that represents the entire text. This approach reduces the N×D matrix to a one-dimensional D-dimensional vector. While this method loses some granularity, it preserves the overall semantic meaning of the text. 
3. Principle Component Analysis (PCA): a common linear dimension reduction method
4. Max pooling: it takes the maximum value across each dimension of the embeddings, resulting in a single 
D-dimensional vector.
5. Min-pooling works similarly but takes the minimum values, potentially capturing different aspects of the semantic space.

TODO: Implement the dimensional reduction as a call back so that a user can implement custom aggregation.

I have not done any performance measurement over these transform methods. However, it's a common sense flatten a 2d metrics retains better granularity than Average.

# Next Step

## Performance and configuration
* Query performance of transformed one dimensional embedding
* Configuration parameters of ColBERTConfig (chunk size, nbits, kmeans_nitters, nranks on CUDA, bsize, rerank?)
* Two dimensioned index query is implemented in the index.py. Yet I need to measure the relevancy  rank.
