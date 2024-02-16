import colbert
from colbert import Indexer, Searcher
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.modeling.checkpoint import Checkpoint
import itertools
from typing import List
import torch

# from cassandra.concurrent import execute_concurrent_with_args
# from db import DB

#
# An example to use the ColBERT library to index a collection of passages and encode them into a set of embeddings.
#

from datasets import load_dataset
import multiprocessing

dataset = 'lifestyle'
datasplit = 'dev'

collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

queries_dataset = load_dataset("colbertv2/lotte", dataset)
queries = [x['query'] for x in queries_dataset['search_' + datasplit]]

f'Loaded {len(queries)} queries and {len(collection):,} passages'

print(queries[10])
print()
print(collection[1])
print()

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300 # truncate passages at 300 tokens
max_id = 100

class NormalizationCategory():
    FLAT = "flat"
    AVERAGE = "average"
    PCA = "pca"
    MAX_POOLING = "max_pooling"
    MIN_POOLING = "min_pooling"

def normalize_tensor(embeddings, category):
    if category == NormalizationCategory.FLAT:
        return embeddings.flatten()
    
    elif category == NormalizationCategory.AVERAGE:
        return embeddings.mean(axis=0)
    
    elif category == NormalizationCategory.PCA:
        from sklearn.decomposition import PCA
        # PCA to reduce the dimensionality to 2 for demonstration
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)

        # The PCA result is two-dimensional; to use it as a one-dimensional key,
        # we can either select one of the components or further process it (e.g., flatten, though it's already low-dimensional)
        pca_flattened = pca_result.flatten()
        return pca_flattened
    
    elif category == NormalizationCategory.MAX_POOLING:
        return torch.max(embeddings, dim=0).values

    elif category == NormalizationCategory.MIN_POOLING:
        return torch.min(embeddings, dim=0).values

    else:
        raise ValueError(f"Unknown normalization category {category}")

def normalize_list(embeddings, category) -> List[float]:
    tensor_list = normalize_tensor(embeddings, category)
    return [t.item() for t in tensor_list]

# convert tensor to list[float]
# a squashed tensor can have multiple dimensions
def tensor_to_list(tensor_list)->List[float]:
    # float_list = [t.item() for t in tensor_list]
    if tensor_list.numel() > 1:
        return [item for tensor in tensor_list for item in tensor.numpy().tolist()]

index_name = f'{dataset}.{datasplit}.{nbits}bits'

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Only necessary if you're packaging your script with PyInstaller or similar
    checkpoint = 'colbert-ir/colbertv2.0'

    with Run().context(RunConfig(nranks=4, experiment='notebook')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=100, nbits=2, kmeans_niters=4, checkpoint=checkpoint, index_bsize=1)
                # kmeans_niters specifies the number of iterations of k-means clustering; 
                # 4 is a good and fast default.
                # Consider larger numbers for small datasets.
        print(config)

        ckp = Checkpoint(config.checkpoint, colbert_config=config)

        encoder = CollectionEncoder(config=config, checkpoint=ckp)
        collections=[]
        collections.append(collection[4])
        embeddings, count = encoder.encode_passages(collection[:4])
        print(f"embedding count {count}")

        print(embeddings.shape)

        # split up embeddings by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [embeddings[start:start+count] for start, count in zip(start_indices, count)]
        size = len(embeddings_by_part)
        for part, embedding in enumerate(embeddings_by_part):
            # print(f"embedding part {part} shape {embeddings_by_part}")
            # print(f"Inserted {len(embedding)} shape {embedding.shape} embeddings for part {part}")
            norm = normalize_list(embedding, NormalizationCategory.MIN_POOLING)
            print(f"embedding part {part} norm {norm}")
            #for i, e in enumerate(embedding):
                #print(f"embedding {i} shape {e.shape}")
                # print(f"embedding {i} embedding {e}")
                    
        print(f'Loaded embeddings {part} parts and {len(embedding)} embeddings. end')
