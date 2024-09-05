import os
import re

import joblib
import orjson as json
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import LocalOutlierFactor


class Base:
    def __init__(
        self,
        n_neighbors=20,
        contamination=0.05,
    ):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.vectorizer = HashingVectorizer(
            n_features=2**20,
        )

    def name(self):
        raise NotImplementedError

    def fit(self, data):
        try:
            X_train, _ = self.load_and_preprocess(data, fit=True)
            print("Fitting the model...")
            self.model.fit(X_train)
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

    def fill_anomaly_data(self, anomalies_df, df_test):
        raise NotImplementedError

    def predict(self, data):
        try:
            X_test, df_test = self.load_and_preprocess(data, fit=False)
            print("Predicting anomalies...")
            anomalies = self.model.predict(X_test)

            anomalies_df = df_test[anomalies == -1]
            return self.fill_anomaly_data(anomalies_df, df_test)
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

    def is_target_event(self, alert):
        raise NotImplementedError

    def process_alerts(self, alerts, fit=False):
        raise NotImplementedError

    def filter_line(self, line):
        raise NotImplementedError

    def load_and_preprocess(self, data, keep_original=False, fit=False):
        alerts = []
        if isinstance(data, str):
            with open(data, encoding="utf-8", errors="replace") as json_file:
                for line in json_file:
                    try:
                        if self.filter_line(line):
                            continue

                        alert = json.loads(line)
                        if self.is_target_event(alert):
                            alerts.append(alert)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
        elif isinstance(data, dict):
            if self.is_target_event(data):
                alerts.append(data)
        else:
            raise ValueError("Input data must be a filepath or a DataFrame")

        return self.process_alerts(alerts, fit=fit)

    @staticmethod
    def model_path(model_dir, name):
        return os.path.join(model_dir, f"{name}_model.pkl")

    def save_model(self, model_dir):
        joblib.dump(
            {
                "model": self.model,
                "vectorizer": self.vectorizer,
            },
            Base.model_path(model_dir, self.name()),
        )

    @staticmethod
    def load_model():
        raise NotImplementedError
