from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        self.documents = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.documents)

    def retrieve(self, query, top_k=5):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[i], scores[i]) for i in ranked_indices]
