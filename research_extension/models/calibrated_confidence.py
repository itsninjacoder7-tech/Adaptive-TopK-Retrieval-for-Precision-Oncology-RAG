
import joblib

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV


class CalibratedConfidenceModel:

    def __init__(self):

        self.features = None

        self.model = None

    def train(

        self,

        dataframe,

        target_column,

        feature_columns

    ):

        self.features = feature_columns

        X = dataframe[feature_columns]

        y = dataframe[target_column]

        rf = RandomForestClassifier(

            n_estimators=300,

            random_state=42,

            class_weight="balanced"

        )

        self.model = CalibratedClassifierCV(

            estimator=rf,

            method="isotonic",

            cv=5

        )

        self.model.fit(X, y)

    def predict(self, dataframe):

        X = dataframe[self.features]

        return self.model.predict(X)

    def predict_probability(self, dataframe):

        X = dataframe[self.features]

        return self.model.predict_proba(X)[:,1]

    def save(self, path):

        joblib.dump(

            {

                "model": self.model,

                "features": self.features

            },

            path

        )

    def load(self, path):

        obj = joblib.load(path)

        self.model = obj["model"]

        self.features = obj["features"]
