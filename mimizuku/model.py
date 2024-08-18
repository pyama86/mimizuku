import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor


class Mimizuku:
    def __init__(self, n_neighbors=15, contamination=0.05):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination, novelty=True
        )
        self.tfidf_vectorizer = TfidfVectorizer()

    def load_and_preprocess(self, data, fit=False, keep_original=False):
        alerts = []
        if isinstance(data, str):
            with open(data, encoding="utf-8", errors="replace") as json_file:
                for line in json_file:
                    try:
                        alert = json.loads(line)
                        if "syscheck" in alert and re.match(
                            r"55[0-9]", str(alert["rule"]["id"])
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
            syscheck = alert.get("syscheck", {})
            path = syscheck.get("path")
            size_after = syscheck.get("size_after", 0)
            uid_after = syscheck.get("uid_after", 0)
            event = syscheck.get("event", "")

            row = {
                "hostname": alert.get("agent", {}).get("name"),
                "path": path,
                "size_after": size_after,
                "uid_after": uid_after,
                "event": event,
            }

            if keep_original:
                row.update(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                    }
                )

            processed_data.append(row)

        df = pd.DataFrame(processed_data)

        if len(df) == 0:
            raise ValueError("No alerts were found in the input data")

        combined_text = df["hostname"] + " " + df["path"] + " " + df["event"]
        if fit:
            tfidf_features = self.tfidf_vectorizer.fit_transform(combined_text)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(combined_text)

        size_uid_features = df[["size_after", "uid_after"]].values

        X = np.hstack([tfidf_features.toarray(), size_uid_features])
        return X, df

    def fit(self, data):
        try:
            X_train, _ = self.load_and_preprocess(data, fit=True)
            self.model.fit(X_train)
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

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
                ]
            ]
        except ValueError as e:
            if "No alerts were found" in str(e):
                pass
            else:
                raise e

    def save_model(self, model_path):
        joblib.dump(
            {
                "model": self.model,
                "tfidf_vectorizer": self.tfidf_vectorizer,
            },
            model_path,
        )

    @staticmethod
    def load_model(model_path):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku()
        mimizuku.model = saved_objects["model"]
        mimizuku.tfidf_vectorizer = saved_objects["tfidf_vectorizer"]
        return mimizuku
