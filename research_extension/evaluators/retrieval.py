
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    retrieved_ids: list
    similarity_scores: list
    ground_truth_id: int
    rank: int | None


class RetrievalEvaluator:

    def __init__(self, index, embedding_model, contexts):
        """
        Parameters
        ----------
        index : faiss.Index

        embedding_model : SentenceTransformer

        contexts : list[str]
        """

        self.index = index
        self.model = embedding_model
        self.contexts = contexts

    def retrieve_topk(
        self,
        query,
        k=5
    ):
        """
        Retrieve the Top-k most similar contexts.

        Parameters
        ----------
        query : str

        k : int

        Returns
        -------
        RetrievalResult
        """

        # Generate query embedding
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        # Convert to float32 for FAISS
        embedding = embedding.astype("float32")

        # Search FAISS
        scores, ids = self.index.search(
            embedding,
            k
        )

        return RetrievalResult(
            retrieved_ids=ids[0].tolist(),
            similarity_scores=scores[0].tolist(),
            ground_truth_id=-1,
            rank=None
        )
