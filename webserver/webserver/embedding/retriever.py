from embedding import ColbertEmbeddings
from embedding import AstraDB
from torch import tensor

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

        query_encodings = self.colbert.encode_query(query)

        # find the most relevant documents
        docparts = set()
        for qv in query_encodings:
            rows = self.astra.session.execute(self.astra.query_colbert_ann_stmt, [list(qv)])
            docparts.update((row.title, row.part) for row in rows)
        # score each document
        scores = {}
        for title, part in docparts:
            rows = self.astra.session.execute(self.astra.query_colbert_parts_stmt, [title, part])
            embeddings_for_part = [tensor(row.bert_embedding) for row in rows]
            scores[(title, part)] = sum(maxsim(qv, embeddings_for_part) for qv in query_encodings)
        # load the source chunk for the top 5
        docs_by_score = sorted(scores, key=scores.get, reverse=True)[:k]
        L = []
        for title, part in docs_by_score:
            rs = self.astra.session.execute(self.astra.query_part_by_pk_stmt, [title, part])
            L.append({'title': title, 'body': rs.one().body})
        return L

