import hashlib
import re

import joblib
import numpy as np
import orjson as json
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

hex_to_int = {char: idx for idx, char in enumerate("0123456789abcdef")}


def md5_to_vector(md5_hash):
    return [hex_to_int[char] for char in md5_hash]


class Mimizuku:
    def __init__(self, n_neighbors=20, contamination=0.05, ignore_files=[]):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.event_encoder = LabelEncoder()
        self.ignore_files = ignore_files

    def is_target_event(self, alert):
        return (
            "syscheck" in alert
            and re.match(r"55[0-9]", str(alert["rule"]["id"]))
            and int(alert["rule"]["level"]) > 0
            and all(
                not alert["syscheck"]["path"].startswith(ignore_file)
                for ignore_file in self.ignore_files
            )
        )

    def load_and_preprocess(self, data, fit=False, keep_original=False):
        alerts = []
        if isinstance(data, str):
            with open(data, encoding="utf-8", errors="replace") as json_file:
                for line in json_file:
                    try:
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

        vectorized_data = []
        original_data = []
        for alert in alerts:
            syscheck = alert.get("syscheck", {})
            path = syscheck.get("path")
            event = syscheck.get("event", "")
            filename_vector = md5_to_vector(
                hashlib.md5(
                    re.sub(
                        r"(\.[a-zA-Z0-9]+)\..*", r"\1.*", re.sub("[0-9]", "", path)
                    ).encode()
                ).hexdigest()
            )

            combined_vector = np.hstack([filename_vector])
            vectorized_data.append(combined_vector)

            if keep_original:
                original_data.append(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                        "original_event": event,
                    }
                )

        if len(vectorized_data) == 0:
            raise ValueError("No alerts were found in the input data")

        X = csr_matrix(np.array(vectorized_data))
        df = pd.DataFrame(original_data)
        return X, df

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

    def predict(self, data):
        try:
            X_test, df_test = self.load_and_preprocess(
                data, fit=False, keep_original=True
            )
            print("Predicting anomalies...")
            anomalies = self.model.predict(X_test)
            anomalies_df = df_test[anomalies == -1]
            return anomalies_df[
                [
                    "original_hostname",
                    "original_path",
                    "original_event",
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
                "event_encoder": self.event_encoder,
            },
            model_path,
        )

    @staticmethod
    def load_model(model_path, ignore_files=[]):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku(ignore_files=ignore_files)
        mimizuku.model = saved_objects["model"]
        mimizuku.event_encoder = saved_objects["event_encoder"]
        return mimizuku
