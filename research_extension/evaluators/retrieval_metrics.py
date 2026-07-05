
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:

    recall_at_1: float

    recall_at_5: float

    recall_at_10: float

    hit: float

    mrr: float


class RetrievalMetricsEvaluator:

    """
    Compute retrieval metrics from the
    rank of the correct context.
    """

    def evaluate(
        self,
        rank
    ):

        if rank is None:

            return RetrievalMetrics(

                recall_at_1=0,

                recall_at_5=0,

                recall_at_10=0,

                hit=0,

                mrr=0
            )

        return RetrievalMetrics(

            recall_at_1=float(rank <= 1),

            recall_at_5=float(rank <= 5),

            recall_at_10=float(rank <= 10),

            hit=1.0,

            mrr=1.0 / rank
        )
