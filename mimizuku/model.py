import json
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder


class Mimizuku:
    def __init__(self, n_neighbors=20, contamination=0.05, ignore_files=[]):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination, novelty=True
        )
        self.hostname_vectorizer = TfidfVectorizer()
        self.filename_vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.directory_vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.event_encoder = LabelEncoder()
        self.ignore_files = ignore_files

    def extract_time_features(self, timestamp):
        if timestamp is None:
            return -1, -1
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.hour, dt.minute
        except ValueError:
            return -1, -1

    def is_target_event(self, alert):
        return (
            "syscheck" in alert
            and re.match(r"55[0-9]", str(alert["rule"]["id"]))
            and int(alert["rule"]["level"]) > 0
            # pathがignore_filesのパターンから開始していないこと
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

        processed_data = []
        for alert in alerts:
            syscheck = alert.get("syscheck", {})
            path = syscheck.get("path")
            directory = os.path.dirname(path)
            filename = os.path.basename(path)
            event = syscheck.get("event", "")
            timestamp = alert.get("timestamp")
            hour, minute = self.extract_time_features(timestamp)

            row = {
                "hostname": re.sub("[0-9]", "", alert.get("agent", {}).get("name")),
                "filename": re.sub("[0-9]", "", filename),
                "directory": re.sub("[0-9]", "", directory),
                "event": event,
                "hour": hour,
                "minute": minute,
            }

            if keep_original:
                row.update(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                        "original_event": event,
                    }
                )

            processed_data.append(row)

        df = pd.DataFrame(processed_data)

        if len(df) == 0:
            raise ValueError("No alerts were found in the input data")

        if fit:
            hostname_features = self.hostname_vectorizer.fit_transform(df["hostname"])
            filename_features = self.filename_vectorizer.fit_transform(df["filename"])
            directory_features = self.directory_vectorizer.fit_transform(
                df["directory"]
            )
            df["event_encoded"] = self.event_encoder.fit_transform(df["event"])
        else:
            hostname_features = self.hostname_vectorizer.transform(df["hostname"])
            filename_features = self.filename_vectorizer.transform(df["filename"])
            directory_features = self.directory_vectorizer.transform(df["directory"])
            df["event_encoded"] = self.event_encoder.transform(df["event"])

        time_features = df[["hour", "minute"]].values
        event_features = df[["event_encoded"]].values

        X = hstack(
            [
                hostname_features,
                filename_features,
                directory_features,
                event_features,
                time_features,
            ]
        )

        df.drop(
            columns=[
                "hostname",
                "filename",
                "directory",
                "event",
                "hour",
                "minute",
                "event_encoded",
            ],
            inplace=True,
        )
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
                "hostname_vectorizer": self.hostname_vectorizer,
                "filename_vectorizer": self.filename_vectorizer,
                "directory_vectorizer": self.directory_vectorizer,
                "event_encoder": self.event_encoder,
            },
            model_path,
        )

    @staticmethod
    def load_model(model_path, ignore_files=[]):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku(ignore_files=ignore_files)
        mimizuku.model = saved_objects["model"]
        mimizuku.hostname_vectorizer = saved_objects["hostname_vectorizer"]
        mimizuku.filename_vectorizer = saved_objects["filename_vectorizer"]
        mimizuku.directory_vectorizer = saved_objects["directory_vectorizer"]
        mimizuku.event_encoder = saved_objects["event_encoder"]
        return mimizuku
