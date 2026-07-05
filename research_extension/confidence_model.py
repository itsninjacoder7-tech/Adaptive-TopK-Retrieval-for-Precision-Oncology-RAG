
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


class ConfidenceModel:

    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        self.features = [

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

    def train(self, df):

        X = df[self.features]
        y = df["confidence"]
        groups = df["statement_id"]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42
        )

        train_idx, test_idx = next(
            splitter.split(X, y, groups)
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(X_test)

        probabilities = self.model.predict_proba(X_test)[:, 1]

        return {

            "accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "roc_auc":
                roc_auc_score(
                    y_test,
                    probabilities
                ),

            "classification_report":
                classification_report(
                    y_test,
                    predictions
                ),

            "confusion_matrix":
                confusion_matrix(
                    y_test,
                    predictions
                ),

            "probabilities":
                probabilities,

            "y_test":
                y_test.reset_index(drop=True),

            "feature_importance":
                dict(
                    zip(
                        self.features,
                        self.model.feature_importances_
                    )
                )
        }

    def predict_confidence(self, dataframe):

        return self.model.predict_proba(
            dataframe[self.features]
        )[:, 1]
