#
# this is a base class for single token based embedding

from abc import ABC, abstractmethod
from typing import Any, List
import uuid

from langchain_core.runnables.config import run_in_executor

class PerTokenEmbeddings():

    __embeddings: List[float]

    def __init__(
            self,
            id: int,
            parent_id: int,
        ):
        self.id = id
        self.parent_id = parent_id
        self.__embeddings = []

    def add_embeddings(self, embeddings: List[float]): 
        self.__embeddings.append(embeddings)

class PassageEmbeddings():
    __token_embeddings: List[PerTokenEmbeddings]
    __text: str
    __id: uuid.UUID

    def __init__(
            self,
            text: str,
            id: uuid.UUID = None,
        ):
        #self.token_ids = token_ids
        self.__text = text
        self.__token_embeddings = []
        if id is None:
            self.__id = uuid.uuid4()
        else:
            self.__id = id

    def token_size(self):
        return len(self.token_ids)

    def __len__(self):
        return len(self.embeddings)

    def id_str(self):
        return str(self.__id)

    def add_token_embeddings(self, token_embeddings: PerTokenEmbeddings):
        self.__token_embeddings.append(token_embeddings)

    def get_token_embeddings(self, token_id: int) -> PerTokenEmbeddings:
        for token in self.__token_embeddings:
            if token.token_id == token_id:
                return token
        return None
    
    def get_all_token_embeddings(self) -> List[PerTokenEmbeddings]:
        return self.__token_embeddings

    def get_text(self):
        return self.__text
    
#
# This is the base class for token based embedding
# ColBERT token embeddings is an example of a class that inherits from this class
class TokenEmbeddings(ABC):
    """Interface for token embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[PassageEmbeddings]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> PassageEmbeddings:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)