from fastapi import FastAPI, Query
from typing import List
from embedding import ColbertEmbeddings, NormalizationCategory
import torch
import os
import uvicorn

#
# This is a web server that provides an API to embed text using the ColBERT model.
# Limitations:
# - It only supports the ColBERT model.
# - It creates new Checkpoint and CollectionEncoder objects for each request.
#

app = FastAPI()

# Health status
@app.get("/health")
def read_root():
    return {"status": "ok"}

# Is cuda available?
@app.get("/cuda")
def is_cuda():
    if torch.cuda.is_available():
        return {"is_cuda": True}
    else:
        return {"is_cuda": False}

# Define a route with a parameter (item_id).
# The function will return a dictionary with the item_id value.
@app.get("/v1/embedding/colbert")
def embedding(
    texts: List[str], 
    doc_maxlen: int = Query(220),
    nbits: int = Query(1),
    kmeans_niters: int = Query(4),
    nranks: int = Query(1),
    normalization_category: str = Query(NormalizationCategory.FLAT),
):
    colbert = ColbertEmbeddings(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        kmeans_niters=kmeans_niters,
        nranks=nranks,
        normalization_category=normalization_category,
    )
    if len(texts) == 0:
        return {"error": "No text to embed"}
    
    if len(texts) == 1:
        result = colbert.embed_query(texts[0])
        rc = []
        rc.append(result)
        return {"embeddings": rc}
    
    return {"embeddings": colbert.embed_documents(texts)}

if __name__ == "__main__":
    this_port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=this_port)