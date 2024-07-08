# type: ignore
# this is a base class for single token based embedding

import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class PerTokenEmbeddings:
    __embeddings: List[float]

    def __init__(
        self,
        token_id: int,
        part: int,
        parent_id: uuid.UUID,
        title: str = "",
    ):
        self.token_id = token_id
        self.parent_id_token = parent_id
        self.__embeddings = []
        self.title = title
        self.part_token = part

    def add_embeddings(self, embeddings: List[float]) -> None:
        self.__embeddings = embeddings

    def get_embeddings(self) -> List[float]:
        return self.__embeddings

    def id(self) -> int:
        return self.token_id

    def parent_id(self) -> uuid.UUID:
        return self.parent_id_token

    def part(self) -> int:
        return self.part_token


class PassageEmbeddings:
    __token_embeddings: List[PerTokenEmbeddings]
    __text: str
    __title: str
    __id: uuid.UUID

    def __init__(
        self,
        text: str,
        title: str = "",
        part: int = 0,
        id: uuid.UUID = None,
        model: str = "colbert-ir/colbertv2.0",
        dim: int = 128,
    ):
        # self.token_ids = token_ids
        self.__text = text
        self.__token_embeddings = []
        if id is None:
            self.__id = uuid.uuid4()
        else:
            self.__id = id
        self.__model = model
        self.__dim = dim
        self.__title = title
        self.__part = part

    def model(self) -> str:
        return self.__model

    def dim(self) -> int:
        return self.__dim

    def title(self) -> str:
        return self.__title

    def id(self) -> uuid.UUID:
        return self.__id

    def part(self) -> int:
        return self.__part

    def add_token_embeddings(self, token_embeddings: PerTokenEmbeddings) -> None:
        self.__token_embeddings.append(token_embeddings)

    def get_token_embeddings(self, token_id: int) -> Optional[PerTokenEmbeddings]:
        for token in self.__token_embeddings:
            if token.token_id == token_id:
                return token
        return None

    def get_all_token_embeddings(self) -> List[PerTokenEmbeddings]:
        return self.__token_embeddings

    def get_text(self) -> str:
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
