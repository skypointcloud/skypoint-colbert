from embedding import ColbertEmbeddings
from embedding import AstraDB
from torch import tensor

# max similarity between a query vector and a list of embeddings
# The function returns the highest similarity score (i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.
# The function iterates over each embedding vector (e) in the embeddings.
# For each e, it performs a dot product operation (@) with the query vector (qv).
# The dot product of two vectors is a measure of their similarity. In the context of embeddings,
# a higher dot product value usually indicates greater similarity.
# The max function then takes the highest value from these dot product operations.
# Essentially, it's picking the embedding vector that has the highest similarity to the query vector qv.
def maxsim(qv, embeddings):
    return max(qv @ e for e in embeddings)

class ColbertAstraRetriever:
    def __init__(
        self,
        astraDB: AstraDB,
        colbertEmbeddings: ColbertEmbeddings,
        verbose: bool=False
    ):
        self.astra = astraDB
        self.colbert = colbertEmbeddings
        self.verbose = verbose

    def retrieve(self, query: str, k: int=5):
        if k > 10:
            raise ValueError("k cannot be greater than 10")

        query_encodings = self.colbert.encode_query(query)

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
            scores[(title, part)] = sum(maxsim(qv, embeddings_for_part) for qv in query_encodings)
        # load the source chunk for the top k documents
        docs_by_score = sorted(scores, key=scores.get, reverse=True)[:k]
        answers = []
        rank = 1
        for title, part in docs_by_score:
            rs = self.astra.session.execute(self.astra.query_part_by_pk_stmt, [title, part])
            score = scores[(title, part)]
            answers.append({'title': title, 'score': score.item(), 'rank': rank, 'body': rs.one().body})
            rank=rank+1
        return answers
