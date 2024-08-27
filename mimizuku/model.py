import re

import joblib
import orjson as json
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder


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
        self.vectorizer = HashingVectorizer(
            n_features=2**20,
        )

    def replace_temp_strings(self, path):
        pattern = r"[a-f0-9]{7,40}"
        modified_path = re.sub(pattern, "", path)
        modified_path = re.sub(r"[\d]+", "", modified_path)
        print(f"modified_path: {modified_path}")
        return modified_path

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

    def filename_to_vector(self, filename):
        return self.vectorizer.transform([filename])

    def load_and_preprocess(self, data, keep_original=False, fit=False):
        alerts = []
        if isinstance(data, str):
            with open(data, encoding="utf-8", errors="replace") as json_file:
                for line in json_file:
                    try:
                        if not re.search(r'"id":\s*"55[0-9]"', line):
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

        original_data = []
        filenames = []
        for alert in alerts:
            syscheck = alert.get("syscheck", {})
            path = syscheck.get("path")
            event = syscheck.get("event", "")
            filenames.append(
                re.sub(
                    r"[\.\-_/]",
                    " ",
                    self.replace_temp_strings(path),
                )
            )

            if keep_original:
                original_data.append(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                        "original_event": event,
                    }
                )

        if len(filenames) == 0:
            raise ValueError("No alerts were found in the input data")

        if fit:
            X = self.vectorizer.fit_transform(filenames)
        else:
            X = self.vectorizer.transform(filenames)

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
            X_test, df_test = self.load_and_preprocess(data, keep_original=True)
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
                "vectorizer": self.vectorizer,
            },
            model_path,
        )

    @staticmethod
    def load_model(model_path, ignore_files=[]):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku(ignore_files=ignore_files)
        mimizuku.model = saved_objects["model"]
        mimizuku.event_encoder = saved_objects["event_encoder"]
        mimizuku.vectorizer = saved_objects["vectorizer"]
        return mimizuku
