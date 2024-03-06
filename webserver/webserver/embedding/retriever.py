from embedding import ColbertTokenEmbeddings

from embedding import AstraDB
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import BaseModel
from torch import tensor
from typing import List
import torch

# max similarity between a query vector and a list of embeddings
# The function returns the highest similarity score (i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.
# The function iterates over each embedding vector (e) in the embeddings.
# For each e, it performs a dot product operation (@) with the query vector (qv).
# The dot product of two vectors is a measure of their similarity. In the context of embeddings,
# a higher dot product value usually indicates greater similarity.
# The max function then takes the highest value from these dot product operations.
# Essentially, it's picking the embedding vector that has the highest similarity to the query vector qv.
def maxsim(qv, embeddings, is_cuda: bool=False):
    if is_cuda:
        # Assuming qv and embeddings are PyTorch tensors
        qv = qv.to('cuda')  # Move qv to GPU
        embeddings = [e.to('cuda') for e in embeddings]  # Move all embeddings to GPU
    return max(qv @ e for e in embeddings)


class ColbertAstraRetriever(BaseRetriever):
    astra: AstraDB
    colbertEmbeddings: ColbertTokenEmbeddings
    verbose: bool
    is_cuda: bool=False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        astraDB: AstraDB,
        colbertEmbeddings: ColbertTokenEmbeddings,
        verbose: bool=False,
        **kwargs
    ):
        # initialize pydantic base model
        super().__init__(astra=astraDB, colbertEmbeddings=colbertEmbeddings, verbose=verbose, **kwargs)
        self.astra = astraDB
        self.colbertEmbeddings = colbertEmbeddings
        self.verbose = verbose
        self.is_cuda = torch.cuda.is_available()

    def retrieve(self, query: str, k: int=5):
        if k > 10:
            raise ValueError("k cannot be greater than 10")

        query_encodings = self.colbertEmbeddings.encode_query(query)

        # find the most relevant documents
        docparts = set()
        for qv in query_encodings:
            # per token based retrieval
            rows = self.astra.session.execute(self.astra.query_colbert_ann_stmt, [list(qv)])
            docparts.update((row.title, row.part) for row in rows)
        # score each document
        scores = {}
        for title, part in docparts:
            # find all the found parts so that we can do max similarity search
            rows = self.astra.session.execute(self.astra.query_colbert_parts_stmt, [title, part])
            embeddings_for_part = [tensor(row.bert_embedding) for row in rows]
            # score based on The function returns the highest similarity score
            #(i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.
            scores[(title, part)] = sum(maxsim(qv, embeddings_for_part, self.is_cuda) for qv in query_encodings)
        # load the source chunk for the top k documents
        docs_by_score = sorted(scores, key=scores.get, reverse=True)[:k]
        answers = []
        rank = 1
        for title, part in docs_by_score:
            rs = self.astra.session.execute(self.astra.query_part_by_pk_stmt, [title, part])
            score = scores[(title, part)]
            answers.append({'title': title, 'score': score.item(), 'rank': rank, 'body': rs.one().body})
            rank=rank+1
        # clean up on tensor memory on GPU
        del scores
        return answers

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        answers = self.retrieve(query)
        documents = [Document(metadata={'title': d['title'], 'score': d['score'], 'rank': d['rank']}, page_content=d['body']) for d in answers]
        return documents