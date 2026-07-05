
from dataclasses import dataclass


@dataclass
class RankResult:

    ground_truth_id: int

    rank: int | None

    found: bool


class RankEvaluator:

    """
    Computes the position of the correct context
    inside retrieved contexts.
    """

    def evaluate(
        self,
        ground_truth_id,
        retrieved_ids
    ):

        for rank, context_id in enumerate(retrieved_ids, start=1):

            if context_id == ground_truth_id:

                return RankResult(

                    ground_truth_id=ground_truth_id,

                    rank=rank,

                    found=True
                )

        return RankResult(

            ground_truth_id=ground_truth_id,

            rank=None,

            found=False
        )
