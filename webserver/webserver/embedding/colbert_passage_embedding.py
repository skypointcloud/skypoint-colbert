from typing import Any, Dict, List
import itertools
import torch # it should part of colbert dependencies
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.modeling.checkpoint import Checkpoint

class NormalizationCategory():
    FLAT = "flat"
    AVERAGE = "average"
    PCA = "pca"
    MAX_POOLING = "max_pooling"
    MIN_POOLING = "min_pooling"

#
# Utility functions
# 
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



class ColbertEmbeddings(Embeddings):
    """
    Colbert embeddings model.

    The embedding runs locally and requires the colbert library to be installed.
    
    Example:
    Currently the pyarrow module requires a specific version to be installed.

    pip uninstall pyarrow && pip install pyarrow==14.0.0
    pip install colbert-ai==0.2.19
    pip torch

    To take advantage of GPU, please install faiss-gpu
    """

    colbert_config: ColBERTConfig
    checkpoint: Checkpoint
    encoder: CollectionEncoder

    # these are default values aligned with the colbert library
    __doc_maxlen: int = 220,
    __nbits: int = 1,
    __kmeans_niters: int = 4,
    __nranks: int = 1,
    __index_bsize: int = 64,

    # TODO: expose these values
    # these are default values aligned with the colbert library
    __resume: bool = False,
    __similarity: str = 'cosine',
    __bsize: int = 32,
    __accumsteps: int = 1,
    __lr: float = 0.000003,
    __maxsteps: int = 500000,
    __nway: int = 2,
    __use_ib_negatives: bool = False,
    __reranker: bool = False,
    __is_cuda: bool = False

    @classmethod
    @root_validator()
    def validate_environment(self, values: Dict) -> Dict:
        """Validate colbert and its dependency is installed."""
        try:
            from colbert import Indexer
        except ImportError as exc:
            raise ImportError(
                "Could not import colbert library. "
                "Please install it with `pip install colbert`"
            ) from exc
        
        try:
            import torch
            if torch.cuda.is_available():
                self.__is_cuda = True
                try:
                    import faiss
                except ImportError as e:
                    raise ImportError(
                        "Could not import faiss library. "
                        "Please install it with `pip install faiss-gpu`"
                    ) from e
                    
        except ImportError as exc:
            raise ImportError(
                "Could not import torch library. "
                "Please install it with `pip install torch`"
            ) from exc
        
        return values

    
    def __init__(
            self,
            checkpoint: str = "colbert-ir/cobertv2.0", 
            doc_maxlen: int = 220,
            nbits: int = 1,
            kmeans_niters: int = 4,
            nranks: int = 1,
            normalization_category: str = NormalizationCategory.FLAT,
            **data: Any,
    ):
        self.colbert_config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            nranks=nranks,
            checkpoint='colbert-ir/colbertv2.0' # checkpoint,
        )
        self.__doc_maxlen = doc_maxlen
        self.__nbits = nbits
        self.__kmeans_niters = kmeans_niters
        self.__nranks = nranks
        print("creating checkpoint")
        self.checkpoint = Checkpoint(self.colbert_config.checkpoint, colbert_config=self.colbert_config)
        self.encoder = CollectionEncoder(config=self.colbert_config, checkpoint=self.checkpoint)
        self.normalization_category = normalization_category


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        if self.__is_cuda:
            return self.encode_on_cuda(texts)
        else:
            return self.encode(texts)


    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        collections=[]
        collections.append(text)
        embeddings, count = self.encoder.encode_passages(collections)
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [embeddings[start:start+count] for start, count in zip(start_indices, count)]
        for part, embedding in enumerate(embeddings_by_part):
            norm = normalize_list(embedding, self.normalization_category)
            # return since the list size is one
            return norm

        # norm = normalize_list(embeddings_by_part[0], self.normalization_category)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode the given texts."""
        # embeddings is a tensor of shape (n, 128) n is the number of tokens in the total passage
        # count is the number of tokens in each passage
        embeddings, count = self.encoder.encode_passages(texts)

        # the starting index of each passage or text
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        # embeddings_by_part is a list of tensors, each tensor is a passage
        embeddings_by_part = [embeddings[start:start+count] for start, count in zip(start_indices, count)]
        
        rc = []
        for part, embedding in enumerate(embeddings_by_part):
            # each embedding is a tensor of shape (n, 128) n is the number of tokens in the passage
            # shape {embedding.shape}, len(embedding) is the number of tokens in the passage
            norm = normalize_list(embedding, self.normalization_category)
            rc.append(norm)

        return rc

    def encode_on_cuda(self, texts: List[str]) -> List[List[float]]:
        with Run().context(RunConfig(nranks=self.__nranks, experiment='notebook')):  # nranks specifies the number of GPUs to use
            config = ColBERTConfig(
                doc_maxlen=self.__doc_maxlen,
                nbits=self.__nbits,
                kmeans_niters=self.__kmeans_niters,
                checkpoint=self.checkpoint,
                index_bsize=1)
            ckp = Checkpoint(config.checkpoint, colbert_config=config)

            encoder = CollectionEncoder(config=config, checkpoint=ckp)
            embeddings, count = encoder.encode_passages(texts)
            rc = []

            # split up embeddings by counts, a list of the number of tokens in each passage
            start_indices = [0] + list(itertools.accumulate(count[:-1]))
            embeddings_by_part = [embeddings[start:start+count] for start, count in zip(start_indices, count)]
            size = len(embeddings_by_part)
            for part, embedding in enumerate(embeddings_by_part):
                norm = normalize_list(embedding, NormalizationCategory.FLAT)
                print(f"embedding part {part} ") #norm {norm}")
                rc.append(norm)

            return rc

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)