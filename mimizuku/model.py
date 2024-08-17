import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from .utils import anonymize_path, safe_transform


class Mimizuku:
    def __init__(
        self, n_estimators=1000, random_state=42, max_samples=0.8, contamination=0.01
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples=max_samples,
            contamination=contamination,
        )
        self.tfidf_vectorizer_path = TfidfVectorizer(stop_words=None)
        self.tfidf_vectorizer_directory = TfidfVectorizer(stop_words=None)
        self.le_hostname = LabelEncoder()

    def load_and_preprocess(self, data, fit=False, keep_original=False):
        alerts = []
        if isinstance(data, str):
            with open(data, encoding="utf-8", errors="replace") as json_file:
                for line in json_file:
                    try:
                        alert = json.loads(line)
                        if (
                            # change file messages
                            re.match(r"55[0-9]", str(alert["rule"]["id"]))
                        ):
                            alerts.append(alert)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
        elif isinstance(data, pd.DataFrame):
            alerts = data.to_dict("records")
        else:
            raise ValueError("Input data must be a filepath or a DataFrame")

        processed_data = []
        for alert in alerts:
            if "syscheck" not in alert:
                continue
            path = alert.get("syscheck", {}).get("path")
            directory = os.path.dirname(path) if path else "N/A"

            row = {
                "hostname": re.sub("[0-9]", "*", alert.get("agent", {}).get("name")),
                "path": anonymize_path(path).replace("/", " "),
                "directory": directory.replace("/", " "),
                "id": alert.get("rule", {}).get("id"),
            }

            if keep_original:
                row.update(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                        "original_directory": directory,
                    }
                )

            processed_data.append(row)

        df = pd.DataFrame(processed_data)

        if len(df) == 0:
            raise ValueError("No alerts were found in the input data")

        if fit:
            df["hostname_encoded"] = self.le_hostname.fit_transform(df["hostname"])
            df["path_encoded"] = self.tfidf_vectorizer_path.fit_transform(
                df["path"]
            ).toarray()
            df["directory_encoded"] = self.tfidf_vectorizer_directory.fit_transform(
                df["directory"]
            ).toarray()
        else:
            df["hostname_encoded"] = safe_transform(self.le_hostname, df["hostname"])
            df["path_encoded"] = self.tfidf_vectorizer_path.transform(
                df["path"]
            ).toarray()
            df["directory_encoded"] = self.tfidf_vectorizer_directory.transform(
                df["directory"]
            ).toarray()
        X = df[
            [
                "hostname_encoded",
                "path_encoded",
                "directory_encoded",
            ]
        ]

        X.columns = X.columns.astype(str)
        return X, df

    def fit(self, data):
        try:
            X_train, _ = self.load_and_preprocess(data, fit=True)
            self.model.fit(X_train)
        except ValueError as e:
            print(f"Error: {e}")

    def predict(self, data):
        try:
            X_test, df_test = self.load_and_preprocess(
                data, fit=False, keep_original=True
            )
            anomalies = self.model.predict(X_test)
            anomalies_df = df_test[anomalies == -1]
            return anomalies_df[
                [
                    "original_hostname",
                    "original_path",
                    "original_directory",
                ]
            ]
        except ValueError as e:
            print(f"Error: {e}")

    def save_model(self, model_path):
        joblib.dump(
            {
                "model": self.model,
                "tfidf_vectorizer_path": self.tfidf_vectorizer_path,
                "tfidf_vectorizer_directory": self.tfidf_vectorizer_directory,
                "le_hostname": self.le_hostname,
            },
            model_path,
        )

    @staticmethod
    def load_model(model_path):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku()
        mimizuku.model = saved_objects["model"]
        mimizuku.tfidf_vectorizer_path = saved_objects["tfidf_vectorizer_path"]
        mimizuku.tfidf_vectorizer_directory = saved_objects[
            "tfidf_vectorizer_directory"
        ]
        mimizuku.le_hostname = saved_objects["le_hostname"]
        return mimizuku
