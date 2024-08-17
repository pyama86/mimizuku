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
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer_args = TfidfVectorizer()
        self.tfidf_vectorizer_path = TfidfVectorizer(stop_words=None)
        self.tfidf_vectorizer_directory = TfidfVectorizer(stop_words=None)
        self.tfidf_vectorizer_cwd = TfidfVectorizer(stop_words=None)
        self.tfidf_vectorizer_command = TfidfVectorizer()

        self.le_hostname = LabelEncoder()
        self.le_user_id = LabelEncoder()

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
                    anonymize_path(value)
                    for key, value in sorted(
                        alert.get("data", {}).get("audit", {}).get("execve", {}).items()
                    )
                    if re.match(r"a[0-9]+", key)
                ]
            ).strip()

            path = alert.get("syscheck", {}).get("path")
            directory = os.path.dirname(path) if path else "N/A"
            cwd = alert.get("data", {}).get("audit", {}).get("cwd") or "N/A"

            row = {
                "hostname": re.sub("[0-9]", "*", alert.get("agent", {}).get("name")),
                "path": anonymize_path(path).replace("/", " "),
                "directory": directory.replace("/", " "),
                "user_id": alert.get("data", {}).get("audit", {}).get("auid", ""),
                "command": alert.get("data", {}).get("audit", {}).get("command", ""),
                "args": execve_args,
                "cwd": cwd.replace("/", " "),
                "id": alert.get("rule", {}).get("id"),
            }

            if keep_original:
                row.update(
                    {
                        "original_hostname": alert.get("agent", {}).get("name"),
                        "original_path": path,
                        "original_directory": directory,
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

        if len(df) == 0:
            raise ValueError("No alerts were found in the input data")

        if fit:
            df["hostname_encoded"] = self.le_hostname.fit_transform(df["hostname"])
            df["user_id_encoded"] = self.le_user_id.fit_transform(df["user_id"])
            args_tfidf = self.tfidf_vectorizer_args.fit_transform(df["args"]).toarray()
            path_tfidf = self.tfidf_vectorizer_path.fit_transform(df["path"]).toarray()
            directory_tfidf = self.tfidf_vectorizer_directory.fit_transform(
                df["directory"]
            ).toarray()
            cwd_tfidf = self.tfidf_vectorizer_cwd.fit_transform(df["cwd"]).toarray()
            command_tfidf = self.tfidf_vectorizer_command.fit_transform(
                df["command"]
            ).toarray()
        else:
            df["hostname_encoded"] = safe_transform(self.le_hostname, df["hostname"])
            df["user_id_encoded"] = safe_transform(self.le_user_id, df["user_id"])
            args_tfidf = self.tfidf_vectorizer_args.transform(df["args"]).toarray()
            path_tfidf = self.tfidf_vectorizer_path.transform(df["path"]).toarray()
            directory_tfidf = self.tfidf_vectorizer_directory.transform(
                df["directory"]
            ).toarray()
            cwd_tfidf = self.tfidf_vectorizer_cwd.transform(df["cwd"]).toarray()
            command_tfidf = self.tfidf_vectorizer_command.transform(
                df["command"]
            ).toarray()

        X = pd.concat(
            [
                df[
                    [
                        "id",
                        "hostname_encoded",
                        "user_id_encoded",
                    ]
                ],
                pd.DataFrame(args_tfidf, index=df.index),
                pd.DataFrame(path_tfidf, index=df.index),
                pd.DataFrame(directory_tfidf, index=df.index),
                pd.DataFrame(cwd_tfidf, index=df.index),
                pd.DataFrame(command_tfidf, index=df.index),
            ],
            axis=1,
        )

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
                    "original_user_id",
                    "original_command",
                    "original_args",
                    "original_cwd",
                ]
            ]
        except ValueError as e:
            print(f"Error: {e}")

    def save_model(self, model_path):
        joblib.dump(
            {
                "model": self.model,
                "tfidf_vectorizer_args": self.tfidf_vectorizer_args,
                "tfidf_vectorizer_path": self.tfidf_vectorizer_path,
                "tfidf_vectorizer_directory": self.tfidf_vectorizer_directory,
                "tfidf_vectorizer_cwd": self.tfidf_vectorizer_cwd,
                "tfidf_vectorizer_command": self.tfidf_vectorizer_command,
                "le_hostname": self.le_hostname,
                "le_user_id": self.le_user_id,
            },
            model_path,
        )

    @staticmethod
    def load_model(model_path):
        saved_objects = joblib.load(model_path)
        mimizuku = Mimizuku()
        mimizuku.model = saved_objects["model"]
        mimizuku.tfidf_vectorizer_args = saved_objects["tfidf_vectorizer_args"]
        mimizuku.tfidf_vectorizer_path = saved_objects["tfidf_vectorizer_path"]
        mimizuku.tfidf_vectorizer_directory = saved_objects[
            "tfidf_vectorizer_directory"
        ]
        mimizuku.tfidf_vectorizer_cwd = saved_objects["tfidf_vectorizer_cwd"]
        mimizuku.tfidf_vectorizer_command = saved_objects["tfidf_vectorizer_command"]
        mimizuku.le_hostname = saved_objects["le_hostname"]
        mimizuku.le_user_id = saved_objects["le_user_id"]
        return mimizuku
