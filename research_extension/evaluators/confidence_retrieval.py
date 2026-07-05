
import pandas as pd


class ConfidenceRetrievalEvaluator:

    def evaluate(self, dataframe, thresholds):

        results = []

        for threshold in thresholds:

            predicted_k = []

            recall = []

            mrr = []

            for _, row in dataframe.iterrows():

                # Confidence policy
                if row["RCS"] >= threshold:
                    k = 3
                else:
                    k = 10

                predicted_k.append(k)

                rank = row["rank"]

                if pd.isna(rank):

                    recall.append(0)
                    mrr.append(0)

                else:

                    recall.append(int(rank <= k))
                    mrr.append(1 / rank if rank <= k else 0)

            results.append({

                "threshold": threshold,

                "avg_k": sum(predicted_k) / len(predicted_k),

                "recall": sum(recall) / len(recall),

                "mrr": sum(mrr) / len(mrr)

            })

        return pd.DataFrame(results)
