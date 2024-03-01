from typing import Any, Dict, List
import itertools
import torch # it should part of colbert dependencies
from .token_embedding import TokenEmbeddings, PerTokenEmbeddings, PassageEmbeddings
from langchain_core.pydantic_v1 import Extra, root_validator


from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.modeling.checkpoint import Checkpoint



class ColbertTokenEmbeddings(TokenEmbeddings):
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
            checkpoint: str = "colbert-ir/colbertv2.0", 
            doc_maxlen: int = 220,
            nbits: int = 1,
            kmeans_niters: int = 4,
            nranks: int = 1,
            **data: Any,
    ):
        self.colbert_config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            nranks=nranks,
            checkpoint=checkpoint,
        )
        self.__doc_maxlen = doc_maxlen
        self.__nbits = nbits
        self.__kmeans_niters = kmeans_niters
        self.__nranks = nranks
        print("creating checkpoint")
        self.checkpoint = Checkpoint(self.colbert_config.checkpoint, colbert_config=self.colbert_config)
        self.encoder = CollectionEncoder(config=self.colbert_config, checkpoint=self.checkpoint)


    def embed_documents(self, texts: List[str]) -> List[TokenEmbeddings]:
        """Embed search docs."""
        if self.__is_cuda:
            return self.encode_on_cuda(texts)
        else:
            return self.encode(texts)


    def embed_query(self, text: str, title: str) -> PassageEmbeddings:
        """Embed query text."""
        collections=[]
        collections.append(text)
        embeddings, count = self.encoder.encode_passages(collections)
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [embeddings[start:start+count] for start, count in zip(start_indices, count)]

        perToken = PerTokenEmbeddings(title, text)
        collectionEmbd = PassageEmbeddings(title, text)
        for __part, embedding in enumerate(embeddings_by_part):
            perToken.add_embeddings(embedding)

        return collectionEmbd



    def encode(self, texts: List[str]) -> List[PassageEmbeddings]:
        # collection = Collection(texts)
        # batches = collection.enumerate_batches(rank=Run().rank)
        ''' 
        config = ColBERTConfig(
            doc_maxlen=self.__doc_maxlen,
            nbits=self.__nbits,
            kmeans_niters=self.__kmeans_niters,
            checkpoint=self.checkpoint,
            index_bsize=1)
        ckp = Checkpoint(config.checkpoint, colbert_config=config)
        encoder = CollectionEncoder(config=self.config, checkpoint=self.checkpoint)
        '''
        embeddings, count = self.encoder.encode_passages(texts)

        collectionEmbds = []
        # split up embeddings by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [embeddings[start:start+count] for start, count in zip(start_indices, count)]
        size = len(embeddings_by_part)
        for part, embedding in enumerate(embeddings_by_part):
            collectionEmbd = PassageEmbeddings(text=texts[part])
            pid = collectionEmbd.id_str()
            token_id = 0
            for __part, perTokenEmbedding in enumerate(embedding):
                perToken = PerTokenEmbeddings(parent_id=pid, id=token_id)
                perToken.add_embeddings(perTokenEmbedding.tolist())
                print(f"    token embedding part {__part} parent id {pid}")
                collectionEmbd.add_token_embeddings(perToken)
                token_id += 1
            collectionEmbds.append(collectionEmbd)
            print(f"embedding part {part} collection id {pid}, collection size {len(collectionEmbd.get_all_token_embeddings())}")

        return collectionEmbds

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)