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
    def __init__(self, n_estimators=100, contamination=0.01, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.tfidf_vectorizer = TfidfVectorizer()
        self.le_hostname = LabelEncoder()
        self.le_path = LabelEncoder()
        self.le_directory = LabelEncoder()
        self.le_event = LabelEncoder()
        self.le_user_id = LabelEncoder()
        self.le_command = LabelEncoder()
        self.le_cwd = LabelEncoder()

    def load_and_preprocess(self, data, fit=False, keep_original=False):
        alerts = []
        if isinstance(data, str):
            with open(data) as json_file:
                for line in json_file:
                    try:
                        alert = json.loads(line)
                        if (
                            # Wazuh messages
                            re.match(r"2[0-9]{2}", str(alert["rule"]["id"]))
                            # Access control messages
                            or re.match(r"25[0-9]{2}", str(alert["rule"]["id"]))
                            # Cron messages
                            or re.match(r"28[0-9]{2}", str(alert["rule"]["id"]))
                            # Kernel messages
                            or re.match(r"51[0-9]{2}", str(alert["rule"]["id"]))
                            # Su messages
                            or re.match(r"53[0-9]{2}", str(alert["rule"]["id"]))
                            # Sudo messages
                            or re.match(r"54[0-9]{2}", str(alert["rule"]["id"]))
                            # Add user messages
                            or re.match(r"59[0-9]{2}", str(alert["rule"]["id"]))
                            # Auditd messages
                            or re.match(r"807[0-9]{2}", str(alert["rule"]["id"]))
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
            execve_args = " ".join(
                [
                    value
                    for key, value in sorted(
                        alert.get("data", {}).get("audit", {}).get("execve", {}).items()
                    )
                    if re.match(r"a[0-9]+", key)
                ]
            ).strip()

            path = alert.get("syscheck", {}).get("path")
            directory = os.path.dirname(path) if path else "N/A"
            row = {
                "hostname": re.sub("[0-9]", "*", alert.get("agent", {}).get("name")),
                "path": anonymize_path(path),
                "directory": directory,
                "event": alert.get("syscheck", {}).get("event"),
                "user_id": alert.get("data", {}).get("audit", {}).get("auid"),
                "command": alert.get("data", {}).get("audit", {}).get("command"),
                "args": execve_args,
                "cwd": alert.get("data", {}).get("audit", {}).get("cwd"),
                "id": alert.get("id"),
            }

            if keep_original:
                row.update(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                        "original_directory": directory,
                        "original_event": alert.get("syscheck", {}).get("event"),
                        "original_user_id": alert.get("data", {})
                        .get("audit", {})
                        .get("auid"),
                        "original_command": alert.get("data", {})
                        .get("audit", {})
                        .get("command"),
                        "original_args": execve_args,
                        "original_cwd": alert.get("data", {})
                        .get("audit", {})
                        .get("cwd"),
                    }
                )

            processed_data.append(row)

        df = pd.DataFrame(processed_data)

        if fit:
            df["hostname_encoded"] = self.le_hostname.fit_transform(df["hostname"])
            df["path_encoded"] = self.le_path.fit_transform(df["path"])
            df["directory_encoded"] = self.le_directory.fit_transform(df["directory"])
            df["event_encoded"] = self.le_event.fit_transform(df["event"])
            df["user_id_encoded"] = self.le_user_id.fit_transform(df["user_id"])
            df["command_encoded"] = self.le_command.fit_transform(df["command"])
            df["cwd_encoded"] = self.le_cwd.fit_transform(df["cwd"])
            args_tfidf = self.tfidf_vectorizer.fit_transform(df["args"]).toarray()
        else:
            df["hostname_encoded"] = safe_transform(self.le_hostname, df["hostname"])
            df["path_encoded"] = safe_transform(self.le_path, df["path"])
            df["directory_encoded"] = safe_transform(self.le_directory, df["directory"])
            df["event_encoded"] = safe_transform(self.le_event, df["event"])
            df["user_id_encoded"] = safe_transform(self.le_user_id, df["user_id"])
            df["command_encoded"] = safe_transform(self.le_command, df["command"])
            df["cwd_encoded"] = safe_transform(self.le_cwd, df["cwd"])
            args_tfidf = self.tfidf_vectorizer.transform(df["args"]).toarray()

        X = pd.concat(
            [
                df[
                    [
                        "id",
                        "hostname_encoded",
                        "path_encoded",
                        "directory_encoded",
                        "event_encoded",
                        "user_id_encoded",
                        "command_encoded",
                        "cwd_encoded",
                    ]
                ],
                pd.DataFrame(args_tfidf, index=df.index),
            ],
            axis=1,
        )

        X.columns = X.columns.astype(str)
        return X, df

    def fit(self, data):
        X_train, _ = self.load_and_preprocess(data, fit=True)
        self.model.fit(X_train)

    def predict(self, data):
        X_test, df_test = self.load_and_preprocess(data, fit=False, keep_original=True)
        anomalies = self.model.predict(X_test)
        anomalies_df = df_test[anomalies == -1]
        return anomalies_df[
            [
                "original_hostname",
                "original_path",
                "original_directory",
                "original_event",
                "original_user_id",
                "original_command",
                "original_args",
                "original_cwd",
            ]
        ]

    def save_model(self, model_path):
        joblib.dump(
            {
                "model": self.model,
                "tfidf_vectorizer": self.tfidf_vectorizer,
                "le_hostname": self.le_hostname,
                "le_path": self.le_path,
                "le_directory": self.le_directory,
                "le_event": self.le_event,
                "le_user_id": self.le_user_id,
                "le_command": self.le_command,
                "le_cwd": self.le_cwd,
            },
            model_path,
        )

    @staticmethod
    def load_model(model_path):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku()
        mimizuku.model = saved_objects["model"]
        mimizuku.tfidf_vectorizer = saved_objects["tfidf_vectorizer"]
        mimizuku.le_hostname = saved_objects["le_hostname"]
        mimizuku.le_path = saved_objects["le_path"]
        mimizuku.le_directory = saved_objects["le_directory"]
        mimizuku.le_event = saved_objects["le_event"]
        mimizuku.le_user_id = saved_objects["le_user_id"]
        mimizuku.le_command = saved_objects["le_command"]
        mimizuku.le_cwd = saved_objects["le_cwd"]
        return mimizuku
