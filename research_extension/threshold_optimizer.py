
import numpy as np
import pandas as pd


class ThresholdOptimizer:

    def __init__(self):

        self.best_high = None
        self.best_medium = None
        self.best_score = -1

    def evaluate_policy(

        self,

        dataframe,

        high,

        medium

    ):

        recall = []

        retrieved = []

        mrr = []

        for _, row in dataframe.iterrows():

            score = row["CRCS"]

            if score >= high:

                k = 3

            elif score >= medium:

                k = 5

            else:

                k = 10

            retrieved.append(k)

            rank = row["rank"]

            if pd.isna(rank):

                recall.append(0)
                mrr.append(0)

            else:

                recall.append(int(rank <= k))

                if rank <= k:
                    mrr.append(1/rank)
                else:
                    mrr.append(0)

        recall = np.mean(recall)

        avg_k = np.mean(retrieved)

        mrr = np.mean(mrr)

        return {

            "recall": recall,

            "avg_k": avg_k,

            "mrr": mrr

        }

    def search(

        self,

        dataframe

    ):

        results = []

        for high in np.arange(0.50,0.96,0.02):

            for medium in np.arange(0.20,high,0.02):

                metrics = self.evaluate_policy(

                    dataframe,

                    high,

                    medium

                )

                score = (

                    metrics["recall"]*100

                    -

                    metrics["avg_k"]

                )

                results.append({

                    "high": high,

                    "medium": medium,

                    "score": score,

                    **metrics

                })

        results = pd.DataFrame(results)

        best = results.sort_values(

            "score",

            ascending=False

        ).iloc[0]

        self.best_high = best["high"]

        self.best_medium = best["medium"]

        self.best_score = best["score"]

        return results
