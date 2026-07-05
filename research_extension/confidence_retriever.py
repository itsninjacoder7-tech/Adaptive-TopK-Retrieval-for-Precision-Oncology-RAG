
from research_extension.analysis.similarity_analysis import SimilarityAnalyzer


class ConfidenceAwareRetriever:

    def __init__(

        self,

        retriever,

        confidence_model,

        high_threshold=0.85,

        medium_threshold=0.60

    ):

        self.retriever = retriever

        self.model = confidence_model

        self.analyzer = SimilarityAnalyzer()

        self.high_threshold = high_threshold

        self.medium_threshold = medium_threshold

    def retrieve(

        self,

        query

    ):

        retrieval = self.retriever.retrieve_topk(

            query,

            k=10

        )

        stats = self.analyzer.analyze(

            retrieval.similarity_scores

        )

        feature_order = self.model.features

        features = [

            stats[f]

            for f in feature_order

        ]

        import pandas as pd

        df = pd.DataFrame(

            [features],

            columns=feature_order

        )

        rcs = self.model.predict_probability(

            df

        )[0]

        if rcs >= self.high_threshold:

            k = 3

        elif rcs >= self.medium_threshold:

            k = 5

        else:

            k = 10

        return {

            "RCS": rcs,

            "selected_k": k,

            "context_ids":

                retrieval.retrieved_ids[:k],

            "scores":

                retrieval.similarity_scores[:k]
        }
