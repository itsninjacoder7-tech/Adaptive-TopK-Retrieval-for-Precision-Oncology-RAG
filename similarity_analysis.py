
import numpy as np


class SimilarityAnalyzer:

    def analyze(self, similarity_scores):

        scores = np.array(similarity_scores)

        top1 = scores[0]
        top2 = scores[1]
        top3 = scores[2]

        return {

            "top1": top1,
            "top2": top2,
            "top3": top3,

            "gap12": top1 - top2,
            "gap13": top1 - top3,
            "gap23": top2 - top3,

            "mean": scores.mean(),
            "std": scores.std(),

            "min": scores.min(),
            "max": scores.max(),

            "range": scores.max() - scores.min(),

            "cv": scores.std() / scores.mean()
        }
