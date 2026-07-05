
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


class AdaptiveTopKModel:

    def __init__(self):

        self.model = RandomForestClassifier(

            n_estimators=200,
            random_state=42,
            n_jobs=-1

        )

    def train(self, df):

        feature_columns = [

            "top1",
            "top2",
            "top3",

            "gap12",
            "gap13",
            "gap23",

            "mean",
            "std",
            "min",
            "max",

            "range",
            "cv"
        ]

        X = df[feature_columns]

        y = df["best_k"]

        groups = df["statement_id"]

        splitter = GroupShuffleSplit(

            n_splits=1,

            test_size=0.2,

            random_state=42

        )

        train_idx, test_idx = next(

            splitter.split(

                X,

                y,

                groups

            )
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        self.model.fit(

            X_train,

            y_train

        )

        predictions = self.model.predict(

            X_test
        )

        return {

            "accuracy": accuracy_score(

                y_test,

                predictions

            ),

            "classification_report": classification_report(

                y_test,

                predictions

            ),

            "confusion_matrix": confusion_matrix(

                y_test,

                predictions

            ),

            "feature_importance":

                dict(

                    zip(

                        feature_columns,

                        self.model.feature_importances_

                    )

                )
        }
