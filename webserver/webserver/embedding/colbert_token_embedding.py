# type: ignore
from typing import Any, Dict, List, Union

import torch
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.tokenization import QueryTokenizer
from langchain_core.pydantic_v1 import root_validator
from torch import Tensor

from .token_embedding import PassageEmbeddings, TokenEmbeddings


def calculate_query_maxlen(tokens: List[List[str]]) -> int:
    max_token_length = max(len(inner_list) for inner_list in tokens)
    offset = max_token_length % 2
    return max_token_length + offset


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
    __doc_maxlen: int = (220,)
    __nbits: int = (1,)
    __kmeans_niters: int = (4,)
    __nranks: int = (1,)
    __index_bsize: int = (64,)

    # TODO: expose these values
    # these are default values aligned with the colbert library
    __resume: bool = (False,)
    __similarity: str = ("cosine",)
    __bsize: int = (32,)
    __accumsteps: int = (1,)
    __lr: float = (0.000003,)
    __maxsteps: int = (500000,)
    __nway: int = (2,)
    __use_ib_negatives: bool = (False,)
    __reranker: bool = (False,)
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

        return values

    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        doc_maxlen: int = 220,
        nbits: int = 1,
        kmeans_niters: int = 4,
        nranks: int = -1,
        query_maxlen: int = 32,
        **data: Any,
    ):
        self.__cuda = torch.cuda.is_available()
        total_visible_gpus = 0
        if self.__cuda:
            self.__cuda_device_count = torch.cuda.device_count()
            self.__cuda_device_name = torch.cuda.get_device_name()
            print(f"nrank {nranks}")
            if nranks < 1:
                nranks = self.__cuda_device_count
            if nranks > 1:
                total_visible_gpus = self.__cuda_device_count
            print(
                f"run on {self.__cuda_device_count} gpus and visible {total_visible_gpus} gpus embeddings on {nranks} gpus"
            )
        else:
            if nranks < 1:
                nranks = 1

        with Run().context(RunConfig(nranks=nranks)):
            if self.__cuda:
                torch.cuda.empty_cache()
            self.colbert_config = ColBERTConfig(
                doc_maxlen=doc_maxlen,
                nbits=nbits,
                kmeans_niters=kmeans_niters,
                nranks=nranks,
                checkpoint=checkpoint,
                query_maxlen=query_maxlen,
                gpus=total_visible_gpus,
            )
        self.__doc_maxlen = doc_maxlen
        self.__nbits = nbits
        self.__kmeans_niters = kmeans_niters
        self.__nranks = nranks
        self.checkpoint = Checkpoint(
            self.colbert_config.checkpoint, colbert_config=self.colbert_config
        )
        self.encoder = CollectionEncoder(
            config=self.colbert_config, checkpoint=self.checkpoint
        )
        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.__cuda = torch.cuda.is_available()
        if self.__cuda:
            self.checkpoint = self.checkpoint.cuda()

    def embed_query(self, text: str) -> Tensor:
        """Embed query text."""
        pass

    def encode_queries(
        self,
        query: Union[str, List[str]],
        full_length_search: bool = False,
        query_maxlen: int = -1,
    ):
        queries = query if type(query) is list else [query]
        bsize = 128 if len(queries) > 128 else None
        tokens = self.query_tokenizer.tokenize(queries)
        if query_maxlen == -1:
            query_maxlen = calculate_query_maxlen(tokens)
            import logging

            logging.info("query_maxlen = %s", query_maxlen)

        self.checkpoint.query_tokenizer.query_maxlen = query_maxlen
        Q = self.checkpoint.queryFromText(
            queries,
            bsize=bsize,
            to_cpu=(not self.__is_cuda),
            full_length_search=full_length_search,
        )

        return Q

    def encode_query(
        self,
        query: str,
        full_length_search: bool = False,
        query_maxlen: int = -1,
    ):
        Q = self.encode_queries(query, full_length_search, query_maxlen=query_maxlen)
        return Q[0]

    def embed_documents(
        self, texts: List[str], title: str = ""
    ) -> List[PassageEmbeddings]:
        """Embed search docs."""
        return self.encode(texts, title)


def get_colbert_embeddings() -> ColbertTokenEmbeddings:
    """
    Get colbert embeddings.
    """
    colbert = ColbertTokenEmbeddings(doc_maxlen=220, nbits=1, kmeans_niters=4)
    return colbert
