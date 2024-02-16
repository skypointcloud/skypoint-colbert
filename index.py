import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

from datasets import load_dataset
import multiprocessing
import torch

if torch.cuda.is_available():
    print("This machine has GPU support.")
else:
    print("warning CUDA is not detected. GPU is required for this test.")

dataset = 'lifestyle'
datasplit = 'dev'

collection_dataset = load_dataset("colbertv2/lotte_passages", dataset, trust_remote_code=True)
collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

queries_dataset = load_dataset("colbertv2/lotte", dataset)
queries = [x['query'] for x in queries_dataset['search_' + datasplit]]


nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 100 # truncate passages at 300 tokens

index_name = f'{dataset}.{datasplit}.{nbits}bits'

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Only necessary if you're packaging your script with PyInstaller or similar
    checkpoint = 'colbert-ir/colbertv2.0'

    with Run().context(RunConfig(nranks=4, experiment='notebook')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=100, nbits=2, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering;
                                                                                # 4 is a good and fast default.
                                                                                # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection[:17], overwrite=True)

        print(config)
        print(indexer.get_index())